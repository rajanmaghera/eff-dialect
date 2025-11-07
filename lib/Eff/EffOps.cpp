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


//===----------------------------------------------------------------------===//
// CallOp
//===----------------------------------------------------------------------===//

// TODO: move to symbol table version?

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
  for (auto &x : fn.getEffects()) {
    // Every effect on the function definition must appear on this call too
    auto fnEff = llvm::dyn_cast<TypeAttr>(x).getValue();
    assert(fnEff);

    bool found = false;
    for (auto &y : callEffs) {
      auto callEff = llvm::dyn_cast<TypeAttr>(y).getValue();
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
    auto callEff = llvm::dyn_cast<SignatureType>(llvm::dyn_cast<TypeAttr>(x).getValue());
    assert(callEff);

    // Find the nearest parent handler
    bool found = false;
    auto handleOp = (*this)->getParentOfType<HandleOp>();
    while (handleOp && !found) {
      // Check what effect is handled by this handler
      if (handleOp.getEffect() == callEff) {
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
    for (auto &y: funcOp.getEffects()) {
      auto funcEff= llvm::dyn_cast<SignatureType>(llvm::dyn_cast<TypeAttr>(y).getValue());
      assert(funcEff);
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


SignatureType DoEffectOp::getEffectSig() {
   return SignatureType::get(getCallee(), FunctionType::get(getContext(), getOperandTypes(), getResultTypes()));
 }


LogicalResult DoEffectOp::verify() {

  // Get the effect signature being performed
  auto eff = getEffectSig();

    // Find the nearest parent handler
    bool found = false;
    auto handleOp = (*this)->getParentOfType<HandleOp>();
    while (handleOp && !found) {
      // Check what effect is handled by this handler
      if (handleOp.getEffect() == eff) {
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
    for (auto &y: funcOp.getEffects()) {
      auto funcEff= llvm::dyn_cast<SignatureType>(llvm::dyn_cast<TypeAttr>(y).getValue());
      assert(funcEff);
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
   // TODO: check that effect type exists somewhere?
   // auto fnAttr = (*this)->getAttrOfType<FlatSymbolRefAttr>("callee");
   // if (!fnAttr)
   //   return emitOpError("requires a 'callee' symbol reference attribute");
   // FuncOp fn = symbolTable.lookupNearestSymbolFrom<FuncOp>(*this, fnAttr);
   // if (!fn)
   //   return emitOpError() << "'" << fnAttr.getValue()
   //                        << "' does not reference a valid function";

   // Verify that the operand and result types match the callee.
   // auto fnType = fn.getFunctionType();
   // if (fnType.getNumInputs() != getNumOperands())
   //   return emitOpError("incorrect number of operands for callee");
   //
   // for (unsigned i = 0, e = fnType.getNumInputs(); i != e; ++i)
   //   if (getOperand(i).getType() != fnType.getInput(i))
   //     return emitOpError("operand type mismatch: expected operand type ")
   //            << fnType.getInput(i) << ", but provided "
   //            << getOperand(i).getType() << " for operand number " << i;
   //
   // if (fnType.getNumResults() != getNumResults())
   //   return emitOpError("incorrect number of results for callee");
   //
   // for (unsigned i = 0, e = fnType.getNumResults(); i != e; ++i)
   //   if (getResult(i).getType() != fnType.getResult(i)) {
   //     auto diag = emitOpError("result type mismatch at index ") << i;
   //     diag.attachNote() << "      op result types: " << getResultTypes();
   //     diag.attachNote() << "function result types: " << fnType.getResults();
   //     return diag;
   //   }

   return success();
 }


//===----------------------------------------------------------------------===//
// FuncOp
//===----------------------------------------------------------------------===//

FuncOp FuncOp::create(Location location, StringRef name, FunctionType type,
                      ArrayRef<SignatureType> effects,
                      ArrayRef<NamedAttribute> attrs) {
  OpBuilder builder(location->getContext());
  OperationState state(location, getOperationName());
  FuncOp::build(builder, state, name, type, effects, attrs);
  return cast<FuncOp>(Operation::create(state));
}
FuncOp FuncOp::create(Location location, StringRef name, FunctionType type,
ArrayRef<SignatureType> effects,
                      Operation::dialect_attr_range attrs) {
  SmallVector<NamedAttribute, 8> attrRef(attrs);
  return create(location, name, type, effects, llvm::ArrayRef(attrRef));
}
FuncOp FuncOp::create(Location location, StringRef name, FunctionType type,
                      ArrayRef<SignatureType> effects,
                      ArrayRef<NamedAttribute> attrs,
                      ArrayRef<DictionaryAttr> argAttrs) {
  FuncOp func = create(location, name, type, effects, attrs);
  func.setAllArgAttrs(argAttrs);
  return func;
}

void FuncOp::build(OpBuilder &builder, OperationState &state, StringRef name,
                   FunctionType type,
                   ArrayRef<SignatureType> effects,
                   ArrayRef<NamedAttribute> attrs,
                   ArrayRef<DictionaryAttr> argAttrs) {
  state.addAttribute(SymbolTable::getSymbolAttrName(),
                     builder.getStringAttr(name));
  state.addAttribute(getFunctionTypeAttrName(state.name), TypeAttr::get(type));
   auto a = llvm::map_to_vector<8>(effects, [](SignatureType v) -> Attribute {
     return TypeAttr::get(v);
   });
   state.addAttribute(getEffectsAttrName(state.name), builder.getArrayAttr(a));
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

void HandleOp::build(OpBuilder &builder, OperationState &state, SignatureType effect) {
   // Store effect
   state.addAttribute(getEffectAttrName(state.name), TypeAttr::get(effect));

   // create region with handler bb that matches effect signature
   auto handlerRegion = state.addRegion();
   Block *handlerEntryBlock = new Block();
   handlerRegion->push_back(handlerEntryBlock);
   // add continuation as argument
   ContinuationType contTy = ContinuationType::get(builder.getContext());
   handlerEntryBlock->addArgument(contTy, state.location);
   for (auto &arg : effect.getFn().getInputs()) {
      handlerEntryBlock->addArgument(arg, state.location) ;
   }

   // auto bodyRegion = state.addRegion();
   // bodyRegion->t
   // han
   //
   //
   // state.attributes.append(attrs.begin(), attrs.end());
   //
   // if (argAttrs.empty())
   //   return;
   // assert(type.getNumInputs() == argAttrs.size());
   // call_interface_impl::addArgAndResultAttrs(
   //     builder, state, argAttrs, /*resultAttrs=*/{},
   //     getArgAttrsAttrName(state.name), getResAttrsAttrName(state.name));
 }

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

LogicalResult ContinueOp::verify() {
  auto handler = cast<HandleOp>((*this)->getParentOp());
  const auto &effResults = handler.getEffect().getFn().getResults();

  // The continue op should have 1 + result types operand values.
  if (getNumOperands() != effResults.size() + 1)
    return emitOpError("has ")
           << getNumOperands() << " operands, but enclosing effect (\""
           << handler.getEffect().getName() << "\") returns " << effResults.size() << " (extra operand for continuation needed)";

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

  // The rest of the arguments should match the effect type signature
  for (unsigned i = 0, e = effResults.size(); i != e; ++i)
    if (getOperand(i+1).getType() != effResults[i])
      return emitError() << "type of continuation operand " << i << " ("
                         << getOperand(i+1).getType()
                         << ") doesn't match function result type ("
                         << effResults[i] << ")"
                         << " in effect \"" << handler.getEffect().getName() << "\"";

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
//===----------------------------------------------------------------------===//
// TableGen'd op method definitions
//===----------------------------------------------------------------------===//

#define GET_OP_CLASSES
#include "Eff/EffOps.cpp.inc"
