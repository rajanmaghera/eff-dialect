//===- EffDialect.cpp - Eff dialect ---------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "Eff/EffDialect.h"
#include "Eff/EffOps.h"
#include "Eff/EffTypes.h"

#include "mlir/Conversion/ConvertToEmitC/ToEmitCInterface.h"
#include "mlir/Conversion/ConvertToLLVM/ToLLVMInterface.h"
#include "mlir/Dialect/Bufferization/IR/BufferizableOpInterface.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Value.h"
#include "mlir/Interfaces/FunctionImplementation.h"
#include "mlir/Transforms/InliningUtils.h"
#include "llvm/ADT/APFloat.h"
#include "llvm/ADT/MapVector.h"
#include "llvm/ADT/STLExtras.h"
#include "mlir/IR/DialectImplementation.h"

using namespace mlir;
using namespace mlir::eff;

#include "Eff/EffOpsDialect.cpp.inc"


//===----------------------------------------------------------------------===//
// EffType
//===----------------------------------------------------------------------===//

/**
 * An effect type also encodes its name to differentiate different
 * effects that may have the same name.
 */
struct ::mlir::eff::detail::EffTypeStorage : public TypeStorage {

    using KeyTy = std::pair<StringRef, FunctionType>;

    EffTypeStorage(StringRef name, FunctionType funcSignature) : name(name), funcSignature(funcSignature) {}

    bool operator==(const KeyTy &key) const {
        return key == KeyTy(name, funcSignature);
    }

    static llvm::hash_code hashKey(const KeyTy &key) {
        return llvm::hash_value(key.first);
    }

    static KeyTy getKey(StringRef name, FunctionType funcSignature) {
        return KeyTy(name, funcSignature);
    }

    static EffTypeStorage *construct(TypeStorageAllocator &allocator, const KeyTy &key) {
        return new (allocator.allocate<EffTypeStorage>()) EffTypeStorage(key.first, key.second);
    }

    StringRef name;
    FunctionType funcSignature;
};


EffType EffType::get(StringRef name, FunctionType funcSig)
{
    return Base::get(funcSig.getContext(), name, funcSig);
}

LogicalResult EffType::verifyConstructionInvariants(
 Location loc, StringRef name, FunctionType funcSig
) {
    // TODO(rajanmaghera): only allow names in the format "xxx.yyy"

    // Name must not be empty
    if (name.empty())
        return mlir::emitError(loc) << "name cannot be empty";

    return success();
}

StringRef EffType::getHandlerName() {
    return getImpl()->name;
}

FunctionType EffType::getHandlerSignature() {
    return getImpl()->funcSignature;
}


//===----------------------------------------------------------------------===//
// Eff dialect.
//===----------------------------------------------------------------------===//

void EffDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "Eff/EffOps.cpp.inc"
      >();
  declarePromisedInterface<ConvertToEmitCPatternInterface, EffDialect>();
  declarePromisedInterface<DialectInlinerInterface, EffDialect>();
  declarePromisedInterface<ConvertToLLVMPatternInterface, EffDialect>();
  declarePromisedInterfaces<bufferization::BufferizableOpInterface, CallOp,
                            FuncOp, ReturnOp, DoEffectOp>();
  addTypes<EffType>();
}

/// Materialize a single constant operation from a given attribute value with
/// the desired resultant type.
Operation *EffDialect::materializeConstant(OpBuilder &builder, Attribute value,
                                            Type type, Location loc) {
  if (ConstantOp::isBuildableWith(value, type))
    return ConstantOp::create(builder, loc, type,
                              llvm::cast<FlatSymbolRefAttr>(value));
  return nullptr;
}
