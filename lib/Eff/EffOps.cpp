//===- EffOps.cpp - Eff dialect ops ----------------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "Eff/EffOps.h"

#include <functional>

#include "Eff/EffDialect.h"

#include "mlir/Conversion/ConvertToEmitC/ToEmitCInterface.h"
#include "mlir/Conversion/ConvertToLLVM/ToLLVMInterface.h"
#include "mlir/Dialect/Bufferization/IR/BufferizableOpInterface.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/TypeUtilities.h"
#include "mlir/IR/Value.h"
#include "mlir/Interfaces/FunctionImplementation.h"
#include "mlir/Transforms/InliningUtils.h"
#include "llvm/ADT/APFloat.h"
#include "llvm/ADT/MapVector.h"
#include "llvm/ADT/STLExtras.h"
#include "mlir/IR/DialectImplementation.h"

using namespace mlir;
using namespace mlir::eff;

// TODO: force handle and body areas to end on yield


//===----------------------------------------------------------------------===//
// CallOp
//===----------------------------------------------------------------------===//

LogicalResult CallOp::verifySymbolUses(SymbolTableCollection &symbolTable) {
  // Check that the callee attribute was specified.
  auto fnAttr = (*this)->getAttrOfType<FlatSymbolRefAttr>("callee");
  if (!fnAttr)
    return emitOpError("requires a 'callee' symbol reference attribute");
  FuncOp fn = symbolTable.lookupNearestSymbolFrom<FuncOp>(*this, fnAttr);
  if (!fn)
    return emitOpError() << "'" << fnAttr.getValue()
                         << "' does not reference a valid function";

  // Verify that the operand and result types match the callee.
  auto fnType = fn.getFunctionType();
  if (fnType.getNumInputs() != getNumOperands())
    return emitOpError("incorrect number of operands for callee");

  for (unsigned i = 0, e = fnType.getNumInputs(); i != e; ++i)
    if (getOperand(i).getType() != fnType.getInput(i))
      return emitOpError("operand type mismatch: expected operand type ")
             << fnType.getInput(i) << ", but provided "
             << getOperand(i).getType() << " for operand number " << i;

  if (fnType.getNumResults() != getNumResults())
    return emitOpError("incorrect number of results for callee");

  for (unsigned i = 0, e = fnType.getNumResults(); i != e; ++i)
    if (getResult(i).getType() != fnType.getResult(i)) {
      auto diag = emitOpError("result type mismatch at index ") << i;
      diag.attachNote() << "      op result types: " << getResultTypes();
      diag.attachNote() << "function result types: " << fnType.getResults();
      return diag;
    }

  // Verify that each call has effects attribute
  auto callEffs = (*this)->getAttrOfType<ArrayAttr>("effects");
  if (!callEffs)
    return emitOpError("requires a 'effects' attribute attribute");

  // Verify that the caller has all effects that the callee has
  for (auto &x : fn->getAttrOfType<ArrayAttr>("effects")) {
    // Every effect on the function definition must appear on this call too
    auto fnEff = llvm::dyn_cast<FlatSymbolRefAttr>(x);
    assert(fnEff);

    bool found = false;
    for (auto &y : callEffs) {
      auto callEff = llvm::dyn_cast<FlatSymbolRefAttr>(y);
      if (fnEff == callEff) {
        found = true;
        break;
      }
    }
    if (!found)
      return emitOpError("calls to a function with the effect ") << fnEff << " but the call does not specify this effect";
  }

  return success();
}

LogicalResult CallOp::verify() {
  // Verify that each call has effects attribute
  auto callEffs = (*this)->getAttrOfType<ArrayAttr>("effects");
  if (!callEffs)
    return emitOpError("requires a 'effects' attribute attribute");

  // COND: Each effect of this call needs to be handled properly.
  for (auto &x : callEffs) {
    auto callEff = llvm::dyn_cast<FlatSymbolRefAttr>(x).getValue();

    // Find the nearest parent handler
    bool found = false;
    auto handleOp = (*this)->getParentOfType<HandleOp>();
    while (handleOp && !found) {
      // Check what effect is handled by this handler
      if (handleOp.getSymName() == callEff) {
        found = true;
        // TODO: debug print?
        break;
      }
      // If not matching, then find the nearest handleOp and continue
      handleOp = handleOp->getParentOfType<HandleOp>();
    }
    if (found)
      continue;

    // If handlers don't handle this, then the function signature should
    auto funcOp = (*this)->getParentOfType<FuncOp>();
    assert(funcOp); // this should be in some function

    // On this function, check its effects
  for (auto &y : funcOp->getAttrOfType<ArrayAttr>("effects")) {
    auto funcEff = llvm::dyn_cast<FlatSymbolRefAttr>(y).getValue();
      if (funcEff == callEff) {
        found = true;
        break;
      }
    }

    // If not found, this effect is not handled by a handler or on the function
    // signature
    if (!found)
      return emitOpError("effect ") << callEff << " is not handled or in function's signature";

  }

  return success();
}

FunctionType CallOp::getCalleeType() {
  return FunctionType::get(getContext(), getOperandTypes(), getResultTypes());
}


//===----------------------------------------------------------------------===//
// DoEffect
//===----------------------------------------------------------------------===//

LogicalResult DoEffectOp::verify() {

  // Get symbol
  auto eff = getCallee();

    // Find the nearest parent handler
    bool found = false;
    auto handleOp = (*this)->getParentOfType<HandleOp>();
    while (handleOp && !found) {
      // Check what effect is handled by this handler
      if (handleOp.getSymName() == eff) {
        found = true;
        // TODO: debug print?
        break;
      }
      // If not matching, then find the nearest handleOp and continue
      handleOp = handleOp->getParentOfType<HandleOp>();
    }
    if (found)
      return success();

    // If handlers don't handle this, then the function signature should
    auto funcOp = (*this)->getParentOfType<FuncOp>();
    assert(funcOp); // this should be in some function

    // On this function, check its effects
  for (auto &y : funcOp->getAttrOfType<ArrayAttr>("effects")) {
    auto funcEff = llvm::dyn_cast<FlatSymbolRefAttr>(y).getValue();
      if (funcEff == eff) {
        found = true;
        break;
      }
    }

    // If not found, this effect is not handled by a handler or on the function
    // signature
    if (!found)
      return emitOpError("effect ") << eff << " is not handled or in function's signature";

  return success();
  }


LogicalResult DoEffectOp::verifySymbolUses(SymbolTableCollection &symbolTable) {


   return success();
 }


//===----------------------------------------------------------------------===//
// FuncOp
//===----------------------------------------------------------------------===//

FuncOp FuncOp::create(Location location, StringRef name, FunctionType type,
                      ArrayRef<NamedAttribute> attrs) {
  OpBuilder builder(location->getContext());
  OperationState state(location, getOperationName());
  FuncOp::build(builder, state, name, type, attrs);
  return cast<FuncOp>(Operation::create(state));
}
FuncOp FuncOp::create(Location location, StringRef name, FunctionType type,
                      Operation::dialect_attr_range attrs) {
  SmallVector<NamedAttribute, 8> attrRef(attrs);
  return create(location, name, type, llvm::ArrayRef(attrRef));
}
FuncOp FuncOp::create(Location location, StringRef name, FunctionType type,
                      ArrayRef<NamedAttribute> attrs,
                      ArrayRef<DictionaryAttr> argAttrs) {
  FuncOp func = create(location, name, type, attrs);
  func.setAllArgAttrs(argAttrs);
  return func;
}

void FuncOp::build(OpBuilder &builder, OperationState &state, StringRef name,
                   FunctionType type,
                   ArrayRef<NamedAttribute> attrs,
                   ArrayRef<DictionaryAttr> argAttrs) {
  state.addAttribute(SymbolTable::getSymbolAttrName(),
                     builder.getStringAttr(name));
  state.addAttribute(getFunctionTypeAttrName(state.name), TypeAttr::get(type));
  state.attributes.append(attrs.begin(), attrs.end());
  state.addRegion();

  if (argAttrs.empty())
    return;
  assert(type.getNumInputs() == argAttrs.size());
  call_interface_impl::addArgAndResultAttrs(
      builder, state, argAttrs, /*resultAttrs=*/{},
      getArgAttrsAttrName(state.name), getResAttrsAttrName(state.name));
}

ParseResult FuncOp::parse(OpAsmParser &parser, OperationState &result) {
  auto buildFuncType =
      [](Builder &builder, ArrayRef<Type> argTypes, ArrayRef<Type> results,
         function_interface_impl::VariadicFlag,
         std::string &) { return builder.getFunctionType(argTypes, results); };

  return function_interface_impl::parseFunctionOp(
      parser, result, /*allowVariadic=*/false,
      getFunctionTypeAttrName(result.name), buildFuncType,
      getArgAttrsAttrName(result.name), getResAttrsAttrName(result.name));
}

void FuncOp::print(OpAsmPrinter &p) {
  function_interface_impl::printFunctionOp(
      p, *this, /*isVariadic=*/false, getFunctionTypeAttrName(),
      getArgAttrsAttrName(), getResAttrsAttrName());
}

/// Clone the internal blocks from this function into dest and all attributes
/// from this function to dest.
void FuncOp::cloneInto(FuncOp dest, IRMapping &mapper) {
  // Add the attributes of this function to dest.
  llvm::MapVector<StringAttr, Attribute> newAttrMap;
  for (const auto &attr : dest->getAttrs())
    newAttrMap.insert({attr.getName(), attr.getValue()});
  for (const auto &attr : (*this)->getAttrs())
    newAttrMap.insert({attr.getName(), attr.getValue()});

  auto newAttrs = llvm::to_vector(llvm::map_range(
      newAttrMap, [](std::pair<StringAttr, Attribute> attrPair) {
        return NamedAttribute(attrPair.first, attrPair.second);
      }));
  dest->setAttrs(DictionaryAttr::get(getContext(), newAttrs));

  // Clone the body.
  getBody().cloneInto(&dest.getBody(), mapper);
}

/// Create a deep copy of this function and all of its blocks, remapping
/// any operands that use values outside of the function using the map that is
/// provided (leaving them alone if no entry is present). Replaces references
/// to cloned sub-values with the corresponding value that is copied, and adds
/// those mappings to the mapper.
FuncOp FuncOp::clone(IRMapping &mapper) {
  // Create the new function.
  FuncOp newFunc = cast<FuncOp>(getOperation()->cloneWithoutRegions());

  // If the function has a body, then the user might be deleting arguments to
  // the function by specifying them in the mapper. If so, we don't add the
  // argument to the input type vector.
  if (!isExternal()) {
    FunctionType oldType = getFunctionType();

    unsigned oldNumArgs = oldType.getNumInputs();
    SmallVector<Type, 4> newInputs;
    newInputs.reserve(oldNumArgs);
    for (unsigned i = 0; i != oldNumArgs; ++i)
      if (!mapper.contains(getArgument(i)))
        newInputs.push_back(oldType.getInput(i));

    /// If any of the arguments were dropped, update the type and drop any
    /// necessary argument attributes.
    if (newInputs.size() != oldNumArgs) {
      newFunc.setType(FunctionType::get(oldType.getContext(), newInputs,
                                        oldType.getResults()));

      if (ArrayAttr argAttrs = getAllArgAttrs()) {
        SmallVector<Attribute> newArgAttrs;
        newArgAttrs.reserve(newInputs.size());
        for (unsigned i = 0; i != oldNumArgs; ++i)
          if (!mapper.contains(getArgument(i)))
            newArgAttrs.push_back(argAttrs[i]);
        newFunc.setAllArgAttrs(newArgAttrs);
      }
    }
  }

  /// Clone the current function into the new one and return it.
  cloneInto(newFunc, mapper);
  return newFunc;
}
FuncOp FuncOp::clone() {
  IRMapping mapper;
  return clone(mapper);
}

LogicalResult HandleOp::verify() {
  return success();
}

LogicalResult HandleOp::verifySymbolUses(::mlir::SymbolTableCollection &symbolTable) {
  auto handlerSig = llvm::dyn_cast<FunctionType>(getEffSigAttr().getValue());
  auto fnAttr = getSymNameAttr();
  DefineOp def = symbolTable.lookupNearestSymbolFrom<DefineOp>(*this, fnAttr);
  if (!def)
    return emitOpError() << "'" << fnAttr.getValue()
                         << "' does not reference a valid defined effect";

  // The handler's arguments should match the effect's arguments
  auto effArgs = def.getFunctionType().getInputs();


  // The continue op should have 1 + result types operand values.
  if (handlerSig.getNumInputs() != effArgs.size())
    return emitOpError("has ")
           << handlerSig.getNumInputs() + 1 << " parameters, but enclosing effect (\""
           << fnAttr << "\") returns " << effArgs.size() << " (extra operand for continuation needed)";


  // The rest of the arguments should match the effect type signature
  for (unsigned i = 0, e = effArgs.size(); i != e; ++i)
    if (handlerSig.getInput(i) != effArgs[i])
      return emitError() << "type of handler parameter" << i+1 << " ("
                         << handlerSig.getInput(i)
                         << ") doesn't match effect argument type ("
                         << effArgs[i] << ")"
                         << " in effect \"" << fnAttr << "\"";

  return success();

}


void HandleOp::build(OpBuilder &builder, OperationState &state, SymbolRefAttr name,
                   FunctionType type,
                   ArrayRef<NamedAttribute> attrs,
                   ArrayRef<DictionaryAttr> argAttrs) {

  // Store effect name and type
  state.addAttribute(getSymNameAttrName(state.name), name);
  state.addAttribute(getEffSigAttrName(state.name), TypeAttr::get(type));

  // Add existing attrs
  state.attributes.append(attrs.begin(), attrs.end());

  // Add handler region
  auto handlerRegion = state.addRegion();
  // Handler region should have one more arg than effect type
  assert(type.getNumInputs() + 1 == argAttrs.size());

  auto nonEmptyAttrsFn = [](DictionaryAttr attrs) {
    return attrs && !attrs.empty();
  };
  // Convert the specified array of dictionary attrs (which may have null
  // entries) to an ArrayAttr of dictionaries.
  auto getArrayAttr = [&](ArrayRef<DictionaryAttr> dictAttrs) {
    SmallVector<Attribute> attrs;
    for (auto &dict : dictAttrs)
      attrs.push_back(dict ? dict : builder.getDictionaryAttr({}));
    return builder.getArrayAttr(attrs);
  };

  // Add the attributes to the operation arguments.
  if (llvm::any_of(argAttrs, nonEmptyAttrsFn))
    state.addAttribute(getArgAttrsAttrName(state.name), getArrayAttr(argAttrs));

  // Add body region
  auto bodyRegion = state.addRegion();

}


ParseResult HandleOp::parse(OpAsmParser &parser, OperationState &result) {

  SmallVector<OpAsmParser::Argument> entryArgs;
  SmallVector<DictionaryAttr> resultAttrs;
  SmallVector<Type> resultTypes;
  auto &builder = parser.getBuilder();

  // Parse the name as a symbol.
  StringAttr nameAttr;
  if (parser.parseSymbolName(nameAttr, SymbolTable::getSymbolAttrName(),
                             result.attributes))
    return failure();

  // Parse the function signature.
  SMLoc signatureLocation = parser.getCurrentLocation();
  auto listRes = parser.parseCommaSeparatedList(
      OpAsmParser::Delimiter::Paren, [&]() -> ParseResult {
        // Parse argument name if present.
        OpAsmParser::Argument argument;
        auto argPresent = parser.parseOptionalArgument(
            argument, /*allowType=*/true, /*allowAttrs=*/true);
        if (argPresent.has_value()) {
          if (failed(argPresent.value()))
            return failure(); // Present but malformed.

          // Reject this if the preceding argument was missing a name.
          if (!entryArgs.empty() && entryArgs.back().ssaName.name.empty())
            return parser.emitError(argument.ssaName.location,
                                    "expected type instead of SSA identifier");

        } else {
          argument.ssaName.location = parser.getCurrentLocation();
          // Otherwise we just have a type list without SSA names.  Reject
          // this if the preceding argument had a name.
          if (!entryArgs.empty() && !entryArgs.back().ssaName.name.empty())
            return parser.emitError(argument.ssaName.location,
                                    "expected SSA identifier");

          NamedAttrList attrs;
          if (parser.parseType(argument.type) ||
              parser.parseOptionalAttrDict(attrs) ||
              parser.parseOptionalLocationSpecifier(argument.sourceLoc))
            return failure();
          argument.attrs = attrs.getDictionary(parser.getContext());
        }
        entryArgs.push_back(argument);
        return success();
      });

    if (listRes)
      return failure();


  // DEBUG: check return type
  auto resultattrsize= resultAttrs.size();
  auto resulttypessize = resultTypes.size();

  // Get argument types from entry args
  SmallVector<Type> argTypes;
  argTypes.reserve(entryArgs.size());
  for (auto &arg : entryArgs)
    argTypes.push_back(arg.type);

  // Ensure first argument is a continuation
  if (argTypes.size() < 1)
    return parser.emitError(signatureLocation)
        << "first argument of continuation type is needed";
  if (!isa<ContinuationType>(argTypes[0]))
    return parser.emitError(signatureLocation)
        << "first argument is not of type Continuation";

  // Store eff type (removing first arg)
  argTypes.erase(argTypes.begin());
  result.addAttribute(getEffSigAttrName(result.name), TypeAttr::get(
    builder.getFunctionType(argTypes, resultTypes)
    ));

  // If function attributes are present, parse them.
  NamedAttrList parsedAttributes;
  SMLoc attributeDictLocation = parser.getCurrentLocation();
  if (parser.parseOptionalAttrDictWithKeyword(parsedAttributes))
    return failure();

  // Disallow attributes that are inferred from elsewhere in the attribute
  // dictionary.
  for (StringRef disallowed :
       { SymbolTable::getSymbolAttrName(),
        getEffSigAttrName(result.name).getValue()}) {
    if (parsedAttributes.get(disallowed))
      return parser.emitError(attributeDictLocation, "'")
             << disallowed
             << "' is an inferred attribute and should not be specified in the "
                "explicit attribute dictionary";
  }
  result.attributes.append(parsedAttributes);


  // Add the attributes to the function arguments.
  assert(resultAttrs.size() == resultTypes.size());
    StringAttr argAttrsName = getArgAttrsAttrName(result.name);
   StringAttr resAttrsName = getResAttrsAttrName(result.name);
  // Convert the specified array of dictionary attrs (which may have null
  // entries) to an ArrayAttr of dictionaries.
  SmallVector<DictionaryAttr> argAttrs;
  for (const auto &arg : entryArgs)
    argAttrs.push_back(arg.attrs);

  auto nonEmptyAttrsFn = [](DictionaryAttr attrs) {
    return attrs && !attrs.empty();
  };
  auto getArrayAttr = [&](ArrayRef<DictionaryAttr> dictAttrs) {
    SmallVector<Attribute> attrs;
    for (auto &dict : dictAttrs)
      attrs.push_back(dict ? dict : builder.getDictionaryAttr({}));
    return builder.getArrayAttr(attrs);
  };

  // Add the attributes to the operation arguments.
  if (llvm::any_of(argAttrs, nonEmptyAttrsFn))
    result.addAttribute(argAttrsName, getArrayAttr(argAttrs));

  // Add the attributes to the operation results.
  if (llvm::any_of(resultAttrs, nonEmptyAttrsFn))
    result.addAttribute(resAttrsName, getArrayAttr(resultAttrs));

  // Parse the handler region
  auto *handler = result.addRegion();
  SMLoc loc = parser.getCurrentLocation();
    if (failed(parser.parseRegion(*handler, entryArgs, false)))
      return failure();

  // Function body was parsed, make sure its not empty.
    if (handler->empty())
      return parser.emitError(loc, "expected non-empty body for handler");

  // Parse body

  auto *body = result.addRegion();
  SMLoc bodyLoc = parser.getCurrentLocation();
  if (failed(parser.parseRegion(*body, {}, false)))
    return failure();
  if (body->empty())
    return parser.emitError(bodyLoc, "expected non-empty body");

  if (succeeded(parser.parseOptionalColon())) {
    call_interface_impl::parseFunctionResultList(parser, resultTypes,
                                                           resultAttrs);
    result.addTypes(resultTypes);
  }

  return success();

}

void HandleOp::print(OpAsmPrinter &p) {

  auto typeAttrName = getEffSigAttrName();
  auto argAttrsName = getArgAttrsAttrName();
  auto resAttrsName = getResAttrsAttrName();

  // Print the operation and the function name.
  auto funcName =
      (*this)->getAttrOfType<StringAttr>(SymbolTable::getSymbolAttrName())
          .getValue();
  p << ' ';
  p.printSymbolName(funcName);
  auto argTypes = (*this).getHandler().getArgumentTypes();
  ArrayRef<Type> resultTypes = this->getEffSig().getResults();
  call_interface_impl::printFunctionSignature(
    p, argTypes, this->getArgAttrsAttr(), false, resultTypes,
    this->getResAttrsAttr(), &this->getRegion(0),
    /*printEmptyResult=*/false);
  function_interface_impl::printFunctionAttributes(
      p, *this, {typeAttrName, argAttrsName, resAttrsName});
  // Print the body if this is not an external function.

  Region &handler = this->getHandler();
  if (!handler.empty()) {
    p << ' ';
    p.printRegion(handler, /*printEntryBlockArgs=*/false,
                  /*printBlockTerminators=*/true);
  }
  Region &body = this->getBody();
  if (!body.empty()) {
    p << ' ';
    p.printRegion(body, /*printEntryBlockArgs=*/false,
                  /*printBlockTerminators=*/true);
  }

  if (this->getNumResults() > 0) {
    p << " : " << this->getResultTypes();
  }
}

// void HandleOp::build(OpBuilder &builder, OperationState &state, SignatureType effect) {
//    // Store effect
//    state.addAttribute(getEffectAttrName(state.name), TypeAttr::get(effect));
//
//    // create region with handler bb that matches effect signature
//    auto handlerRegion = state.addRegion();
//    Block *handlerEntryBlock = new Block();
//    handlerRegion->push_back(handlerEntryBlock);
//    // add continuation as argument
//    ContinuationType contTy = ContinuationType::get(builder.getContext());
//    handlerEntryBlock->addArgument(contTy, state.location);
//    for (auto &arg : effect.getFn().getInputs()) {
//       handlerEntryBlock->addArgument(arg, state.location) ;
//    }
//
//    // auto bodyRegion = state.addRegion();
//    // bodyRegion->t
//    // han
//    //
//    //
//    // state.attributes.append(attrs.begin(), attrs.end());
//    //
//    // if (argAttrs.empty())
//    //   return;
//    // assert(type.getNumInputs() == argAttrs.size());
//    // call_interface_impl::addArgAndResultAttrs(
//    //     builder, state, argAttrs, /*resultAttrs=*/{},
//    //     getArgAttrsAttrName(state.name), getResAttrsAttrName(state.name));
//  }

//===----------------------------------------------------------------------===//
// ReturnOp
//===----------------------------------------------------------------------===//

LogicalResult ReturnOp::verify() {
  auto function = cast<FuncOp>((*this)->getParentOp());

  // The operand number and types must match the function signature.
  const auto &results = function.getFunctionType().getResults();
  if (getNumOperands() != results.size())
    return emitOpError("has ")
           << getNumOperands() << " operands, but enclosing function (@"
           << function.getName() << ") returns " << results.size();

  for (unsigned i = 0, e = results.size(); i != e; ++i)
    if (getOperand(i).getType() != results[i])
      return emitError() << "type of return operand " << i << " ("
                         << getOperand(i).getType()
                         << ") doesn't match function result type ("
                         << results[i] << ")"
                         << " in function @" << function.getName();

  return success();

}

//===----------------------------------------------------------------------===//
// ContinueOp
//===----------------------------------------------------------------------===//

LogicalResult ContinueOp::verifySymbolUses(SymbolTableCollection &symbolTable) {

  // Check that the callee attribute was specified.
  auto handler = cast<HandleOp>((*this)->getParentOp());
  auto fnAttr = handler.getSymNameAttr();

  DefineOp def = symbolTable.lookupNearestSymbolFrom<DefineOp>(*this, fnAttr);
  if (!def)
    return emitOpError() << "'" << fnAttr.getValue()
                         << "' does not reference a valid defined effect";

  // Verify that the
  auto effResults = def.getFunctionType().getResults();

  // The continue op should have 1 + result types operand values.
  if (getNumOperands() != effResults.size() + 1)
    return emitOpError("has ")
           << getNumOperands() << " operands, but enclosing effect (\""
           << handler.getSymName() << "\") returns " << effResults.size() << " (extra operand for continuation needed)";


  // The rest of the arguments should match the effect type signature
  for (unsigned i = 0, e = effResults.size(); i != e; ++i)
    if (getOperand(i+1).getType() != effResults[i])
      return emitError() << "type of continuation operand " << i << " ("
                         << getOperand(i+1).getType()
                         << ") doesn't match function result type ("
                         << effResults[i] << ")"
                         << " in effect \"" << fnAttr << "\"";

  return success();
}

LogicalResult ContinueOp::verify() {

  auto handler = cast<HandleOp>((*this)->getParentOp());
  // Assert that size is greater than 0
  if (getNumOperands() < 1)
    return emitOpError("needs at least 1 operand for the continuation");
  // The first value should be a continuation type
  if (!isa<ContinuationType>(getOperand(0).getType())) {
    return emitOpError() << "The first operand must be a continuation type.";
  }

  // TODO: this implementation forces each continue's continuation to come from
  //       the most immediate parent handler that defines it.
  // The first value should be exactly the same as the initial argument
  if (getOperand(0) != handler.getHandler().getArgument(0)) {
    return emitError() << "The continuation must originate from this op's immediate parent.";
  }
  return success();
}

//===----------------------------------------------------------------------===//
// YieldOp
//===----------------------------------------------------------------------===//

LogicalResult YieldOp::verify() {
  auto handler = cast<HandleOp>((*this)->getParentOp());

  // The operand number and types must match the results of the handler.
  const auto &results = handler.getResults().getTypes();
  if (getNumOperands() != results.size())
    return emitOpError("has ")
           << getNumOperands() << " operands, but handler returns " << results.size();

  for (unsigned i = 0, e = results.size(); i != e; ++i)
    if (getOperand(i).getType() != results[i])
      return emitError() << "type of return operand " << i << " ("
                         << getOperand(i).getType()
                         << ") doesn't match handler result type ("
                         << results[i] << ")";

  return success();
}


// TODO: define op in symbol table

//===----------------------------------------------------------------------===//
// TableGen'd op method definitions
//===----------------------------------------------------------------------===//

#define GET_OP_CLASSES
#include "Eff/EffOps.cpp.inc"
