//===- EffDialect.cpp - Eff dialect ---------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "Eff/EffTypes.h"
#include "Eff/EffDialect.h"
#include "Eff/EffOps.h"

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
// Eff dialect.
//===----------------------------------------------------------------------===//

void EffDialect::initialize() {
    registerTypes();
    addOperations<
#define GET_OP_LIST
#include "Eff/EffOps.cpp.inc"
      >();
  declarePromisedInterface<ConvertToEmitCPatternInterface, EffDialect>();
  declarePromisedInterface<DialectInlinerInterface, EffDialect>();
  declarePromisedInterface<ConvertToLLVMPatternInterface, EffDialect>();
  declarePromisedInterfaces<bufferization::BufferizableOpInterface, CallOp,
                            FuncOp, ReturnOp, DoEffectOp>();
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
