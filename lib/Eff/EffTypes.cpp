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

//===----------------------------------------------------------------------===//
// Parser for effect types.
//===----------------------------------------------------------------------===//

Type EffDialect::parseType(DialectAsmParser &parser) const {
    // Parse effect in form
    // eff-type ::= `effect` `<` @name `:` func-type `>`

    if (parser.parseKeyword("effect") || parser.parseLess())
        return Type();

    // Parse first name
    StringAttr name1;
    FunctionType func;
    if (parser.parseSymbolName(name1) || parser.parseColonType(func) || parser.parseGreater()) {
        parser.emitError(parser.getCurrentLocation(), "invalid definition of an effect type");
        return Type();
    }

    return EffType::get(name1, func);
}

void EffDialect::printType(Type type, DialectAsmPrinter &printer) const {
    auto effType =  llvm::cast<EffType>(type);
    printer << "effect<@" << effType.getHandlerName() << " : " << effType.getHandlerSignature() << ">";
}
