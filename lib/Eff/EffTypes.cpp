//===- EffTypes.cpp - Eff dialect types -----------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "Eff/EffTypes.h"

#include "Eff/EffDialect.h"
#include "mlir/IR/DialectImplementation.h"
#include "llvm/ADT/TypeSwitch.h"
#include "mlir/Dialect/Bufferization/IR/BufferizableOpInterface.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Interfaces/FunctionImplementation.h"
#include "mlir/Transforms/InliningUtils.h"

using namespace mlir;
using namespace mlir::eff;

#define GET_TYPEDEF_CLASSES
#include "Eff/EffOpsTypes.cpp.inc"

void EffDialect::registerTypes() {
  addTypes<
#define GET_TYPEDEF_LIST
#include "Eff/EffOpsTypes.cpp.inc"
      >();
}


