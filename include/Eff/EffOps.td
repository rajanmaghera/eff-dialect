//===- EffOps.td - Eff dialect ops -------------------------*- tablegen -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef EFF_OPS
#define EFF_OPS

include "Eff/EffTypes.td"
include "mlir/Interfaces/InferTypeOpInterface.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/IR/EnumAttr.td"
include "mlir/IR/OpAsmInterface.td"
include "mlir/IR/SymbolInterfaces.td"
include "mlir/Interfaces/CallInterfaces.td"
include "mlir/Interfaces/ControlFlowInterfaces.td"
include "mlir/Interfaces/FunctionInterfaces.td"

//===----------------------------------------------------------------------===//
// CallOp
//===----------------------------------------------------------------------===//

def CallOp : Eff_Op<"call",
    [CallOpInterface, MemRefsNormalizable,
     DeclareOpInterfaceMethods<SymbolUserOpInterface>]> {
  let summary = "call operation";
  let description = [{
    Call regular function
  }];

  let arguments = (ins
    FlatSymbolRefAttr:$callee,
    Variadic<AnyType>:$operands,
    TypedArrayAttrBase<TypeAttrOf<Eff_SigType>, "type array">:$effects,
    OptionalAttr<DictArrayAttr>:$arg_attrs,
    OptionalAttr<DictArrayAttr>:$res_attrs,
    UnitAttr:$no_inline
  );

  let results = (outs Variadic<AnyType>);

  let builders = [
    OpBuilder<(ins "FuncOp":$callee, CArg<"ValueRange", "{}">:$operands, CArg<"ArrayRef<::mlir::eff::SignatureType>", "{}">:$effects), [{
      $_state.addOperands(operands);
      $_state.addAttribute("callee", SymbolRefAttr::get(callee));
      $_state.addTypes(callee.getFunctionType().getResults());
      $_state.addAttribute(getEffectsAttrName($_state.name), $_builder.getArrayAttr(::llvm::map_to_vector<8>(effects, [](::mlir::eff::SignatureType v) -> ::mlir::Attribute { return ::mlir::TypeAttr::get(v); })));
    }]>,
    OpBuilder<(ins "SymbolRefAttr":$callee, "TypeRange":$results,
      CArg<"ValueRange", "{}">:$operands, CArg<"ArrayRef<::mlir::eff::SignatureType>", "{}">:$effects), [{
      $_state.addOperands(operands);
      $_state.addAttribute("callee", callee);
      $_state.addTypes(results);
      $_state.addAttribute(getEffectsAttrName($_state.name), $_builder.getArrayAttr(::llvm::map_to_vector<8>(effects, [](::mlir::eff::SignatureType v) -> ::mlir::Attribute { return ::mlir::TypeAttr::get(v); })));
    }]>,
    OpBuilder<(ins "StringAttr":$callee, "TypeRange":$results,
      CArg<"ValueRange", "{}">:$operands, CArg<"ArrayRef<::mlir::eff::SignatureType>", "{}">:$effects), [{
      build($_builder, $_state, SymbolRefAttr::get(callee), results, operands, effects);
    }]>,
    OpBuilder<(ins "StringRef":$callee, "TypeRange":$results,
      CArg<"ValueRange", "{}">:$operands, CArg<"ArrayRef<::mlir::eff::SignatureType>", "{}">:$effects), [{
      build($_builder, $_state, StringAttr::get($_builder.getContext(), callee),
            results, operands, effects);
    }]>,
    OpBuilder<(ins "TypeRange":$results, "FlatSymbolRefAttr":$callee,
      CArg<"ValueRange", "{}">:$operands, CArg<"ArrayRef<::mlir::eff::SignatureType>", "{}">:$effects), [{
      build($_builder, $_state, callee, results, operands, effects);
    }]>,
    OpBuilder<(ins "TypeRange":$results, "StringAttr":$callee,
      CArg<"ValueRange", "{}">:$operands, CArg<"ArrayRef<::mlir::eff::SignatureType>", "{}">:$effects), [{
      build($_builder, $_state, callee, results, operands, effects);
    }]>,
    OpBuilder<(ins "TypeRange":$results, "StringRef":$callee,
      CArg<"ValueRange", "{}">:$operands, CArg<"ArrayRef<::mlir::eff::SignatureType>", "{}">:$effects), [{
      build($_builder, $_state, callee, results, operands, effects);
    }]>];

  let extraClassDeclaration = [{
    FunctionType getCalleeType();

    /// Get the argument operands to the called function.
    operand_range getArgOperands() {
      return {arg_operand_begin(), arg_operand_end()};
    }

    MutableOperandRange getArgOperandsMutable() {
      return getOperandsMutable();
    }

    operand_iterator arg_operand_begin() { return operand_begin(); }
    operand_iterator arg_operand_end() { return operand_end(); }

    /// Return the callee of this operation.
    CallInterfaceCallable getCallableForCallee() {
      return (*this)->getAttrOfType<SymbolRefAttr>("callee");
    }

    /// Set the callee for this operation.
    void setCalleeFromCallable(CallInterfaceCallable callee) {
      (*this)->setAttr("callee", cast<SymbolRefAttr>(callee));
    }
  }];

  let assemblyFormat = [{
    $callee `(` $operands `)` attr-dict `:` functional-type($operands, results)
  }];
}


//===----------------------------------------------------------------------===//
// DoEffectOp
//===----------------------------------------------------------------------===//

def DoEffectOp: Eff_Op<"do_effect",
    [CallOpInterface, MemRefsNormalizable,
     DeclareOpInterfaceMethods<SymbolUserOpInterface>]> {
  let summary = "do effect operation";
  let description = [{
    Call or perform an effect
  }];

  let arguments = (ins
    FlatSymbolRefAttr:$callee,
    Variadic<AnyType>:$operands,
    OptionalAttr<DictArrayAttr>:$arg_attrs,
    OptionalAttr<DictArrayAttr>:$res_attrs,
    UnitAttr:$no_inline
  );

  let results = (outs Variadic<AnyType>);

  let builders = [
    OpBuilder<(ins "SymbolRefAttr":$callee, "TypeRange":$results,
      CArg<"ValueRange", "{}">:$operands), [{
      $_state.addOperands(operands);
      $_state.addAttribute("callee", callee);
      $_state.addTypes(results);
    }]>,
    OpBuilder<(ins "StringAttr":$callee, "TypeRange":$results,
      CArg<"ValueRange", "{}">:$operands), [{
      build($_builder, $_state, SymbolRefAttr::get(callee), results, operands);
    }]>,
    OpBuilder<(ins "StringRef":$callee, "TypeRange":$results,
      CArg<"ValueRange", "{}">:$operands), [{
      build($_builder, $_state, StringAttr::get($_builder.getContext(), callee),
            results, operands);
    }]>,
    OpBuilder<(ins "TypeRange":$results, "FlatSymbolRefAttr":$callee,
      CArg<"ValueRange", "{}">:$operands), [{
      build($_builder, $_state, callee, results, operands);
    }]>,
    OpBuilder<(ins "TypeRange":$results, "StringAttr":$callee,
      CArg<"ValueRange", "{}">:$operands), [{
      build($_builder, $_state, callee, results, operands);
    }]>,
    OpBuilder<(ins "TypeRange":$results, "StringRef":$callee,
      CArg<"ValueRange", "{}">:$operands), [{
      build($_builder, $_state, callee, results, operands);
    }]>];

  let extraClassDeclaration = [{

    ::mlir::eff::SignatureType getEffectSig();

    /// Get the argument operands to the called function.
    operand_range getArgOperands() {
      return {arg_operand_begin(), arg_operand_end()};
    }

    MutableOperandRange getArgOperandsMutable() {
      return getOperandsMutable();
    }

    operand_iterator arg_operand_begin() { return operand_begin(); }
    operand_iterator arg_operand_end() { return operand_end(); }

    /// Return the callee of this operation.
    CallInterfaceCallable getCallableForCallee() {
      return (*this)->getAttrOfType<SymbolRefAttr>("callee");
    }

    /// Set the callee for this operation.
    void setCalleeFromCallable(CallInterfaceCallable callee) {
      (*this)->setAttr("callee", cast<SymbolRefAttr>(callee));
    }
  }];

  let assemblyFormat = [{
    $callee `(` $operands `)` attr-dict `:` functional-type($operands, results)
  }];

  let hasVerifier = 1;
}

//===----------------------------------------------------------------------===//
// ConstantOp
//===----------------------------------------------------------------------===//

def ConstantOp : Eff_Op<"constant",
    [ConstantLike, Pure,
     DeclareOpInterfaceMethods<SymbolUserOpInterface>,
     DeclareOpInterfaceMethods<OpAsmOpInterface, ["getAsmResultNames"]>]> {
  let summary = "constant";
  let description = [{
    The `eff.constant` operation produces an SSA value from a symbol reference
    to a `eff.func` operation
  }];

  let arguments = (ins FlatSymbolRefAttr:$value);
  let results = (outs AnyType);
  let assemblyFormat = "attr-dict $value `:` type(results)";

  let extraClassDeclaration = [{
    /// Returns true if a constant operation can be built with the given value
    /// and result type.
    static bool isBuildableWith(Attribute value, Type type);
  }];

  let hasFolder = 1;
}


//===----------------------------------------------------------------------===//
// HandleOp
//===----------------------------------------------------------------------===//

def HandleOp: Eff_Op<"handle", [
  AffineScope, AutomaticAllocationScope, NoTerminator
]> {
  let summary = "A scoped effect handler for a single effect";
  let description = [{
  }];

  let arguments = (ins TypeAttrOf<Eff_SigType>:$effect);
  let regions = (region AnyRegion:$handler, AnyRegion:$body);

  let builders = [OpBuilder<(ins
    "::mlir::eff::SignatureType":$effect)
  >];
  let extraClassDeclaration = [{

  }];
  let hasCustomAssemblyFormat = 0;
}

//===----------------------------------------------------------------------===//
// FuncOp
//===----------------------------------------------------------------------===//

def FuncOp : Eff_Op<"func", [
  AffineScope, AutomaticAllocationScope,
  FunctionOpInterface, IsolatedFromAbove, OpAsmOpInterface
]> {
  let summary = "An operation with a name containing a single `SSACFG` region";
  let description = [{
    Operations within the function cannot implicitly capture values defined
    outside of the function, i.e. Functions are `IsolatedFromAbove`. All
    external references must use function arguments or attributes that establish
    a symbolic connection (e.g. symbols referenced by name via a string
    attribute like SymbolRefAttr). An external function declaration (used when
    referring to a function declared in some other module) has no body. While
    the MLIR textual form provides a nice inline syntax for function arguments,
    they are internally represented as “block arguments” to the first block in
    the region.

    Only dialect attribute names may be specified in the attribute dictionaries
    for function arguments, results, or the function itself.

  }];

  let arguments = (ins SymbolNameAttr:$sym_name,
                       TypeAttrOf<FunctionType>:$function_type,
                       TypedArrayAttrBase<TypeAttrOf<Eff_SigType>, "type array">:$effects,
                       OptionalAttr<StrAttr>:$sym_visibility,
                       OptionalAttr<DictArrayAttr>:$arg_attrs,
                       OptionalAttr<DictArrayAttr>:$res_attrs,
                       UnitAttr:$no_inline);
  let regions = (region AnyRegion:$body);

  let builders = [OpBuilder<(ins
    "StringRef":$name, "FunctionType":$type, CArg<"ArrayRef<::mlir::eff::SignatureType>", "{}">:$effects,
    CArg<"ArrayRef<NamedAttribute>", "{}">:$attrs,
    CArg<"ArrayRef<DictionaryAttr>", "{}">:$argAttrs)
  >];
  let extraClassDeclaration = [{
    static FuncOp create(Location location, StringRef name, FunctionType type,
                         ArrayRef<::mlir::eff::SignatureType> effects = {},
                         ArrayRef<NamedAttribute> attrs = {});
    static FuncOp create(Location location, StringRef name, FunctionType type,
                         ArrayRef<::mlir::eff::SignatureType> effects,
                         Operation::dialect_attr_range attrs);
    static FuncOp create(Location location, StringRef name, FunctionType type,
                         ArrayRef<::mlir::eff::SignatureType> effects,
                         ArrayRef<NamedAttribute> attrs,
                         ArrayRef<DictionaryAttr> argAttrs);
    /// Create a deep copy of this function and all of its blocks, remapping any
    /// operands that use values outside of the function using the map that is
    /// provided (leaving them alone if no entry is present). If the mapper
    /// contains entries for function arguments, these arguments are not
    /// included in the new function. Replaces references to cloned sub-values
    /// with the corresponding value that is copied, and adds those mappings to
    /// the mapper.
    FuncOp clone(IRMapping &mapper);
    FuncOp clone();

    /// Clone the internal blocks and attributes from this function into dest.
    /// Any cloned blocks are appended to the back of dest. This function
    /// asserts that the attributes of the current function and dest are
    /// compatible.
    void cloneInto(FuncOp dest, IRMapping &mapper);

    //===------------------------------------------------------------------===//
    // FunctionOpInterface Methods
    //===------------------------------------------------------------------===//

    /// Returns the region on the current operation that is callable. This may
    /// return null in the case of an external callable object, e.g. an external
    /// function.
    ::mlir::Region *getCallableRegion() { return isExternal() ? nullptr : &getBody(); }

    /// Returns the argument types of this function.
    ArrayRef<Type> getArgumentTypes() { return getFunctionType().getInputs(); }

    /// Returns the result types of this function.
    ArrayRef<Type> getResultTypes() { return getFunctionType().getResults(); }

    //===------------------------------------------------------------------===//
    // OpAsmOpInterface Methods
    //===------------------------------------------------------------------===//

    /// Allow the dialect prefix to be omitted.
    static StringRef getDefaultDialect() { return "eff"; }

    //===------------------------------------------------------------------===//
    // SymbolOpInterface Methods
    //===------------------------------------------------------------------===//

    bool isDeclaration() { return isExternal(); }
  }];
  let hasCustomAssemblyFormat = 1;
}

//===----------------------------------------------------------------------===//
// ReturnOp
//===----------------------------------------------------------------------===//

def ReturnOp : Eff_Op<"return", [Pure, HasParent<"FuncOp">,
                                MemRefsNormalizable, ReturnLike, Terminator]> {
  let summary = "Function return operation";
  let description = [{
    The `eff.return` operation represents a return operation within a function.
    The operation takes variable number of operands and produces no results.
    The operand number and types must match the signature of the function
    that contains the operation.

  }];

  let arguments = (ins Variadic<AnyType>:$operands);

  let builders = [OpBuilder<(ins), [{
    build($_builder, $_state, {});
  }]>];

  let assemblyFormat = "attr-dict ($operands^ `:` type($operands))?";
  let hasVerifier = 1;
}

#endif // EFF_OPS
