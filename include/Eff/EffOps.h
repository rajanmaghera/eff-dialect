//===- EffOps.h - Eff dialect ops -------------------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef EFF_EFFOPS_H
#define EFF_EFFOPS_H

#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/Interfaces/InferTypeOpInterface.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Interfaces/CallInterfaces.h"
#include "mlir/Interfaces/ControlFlowInterfaces.h"
#include "mlir/Interfaces/FunctionInterfaces.h"
#include "mlir/Bytecode/BytecodeOpInterface.h"
#include "mlir/IR/Builders.h"

#include "Eff/EffTypes.h"

#define GET_OP_CLASSES
#include "Eff/EffOps.h.inc"


namespace llvm {

    /// Allow stealing the low bits of FuncOp.
    template <>
    struct PointerLikeTypeTraits<mlir::eff::FuncOp> {
        static inline void *getAsVoidPointer(mlir::eff::FuncOp val) {
            return const_cast<void *>(val.getAsOpaquePointer());
        }
        static inline mlir::eff::FuncOp getFromVoidPointer(void *p) {
            return mlir::eff::FuncOp::getFromOpaquePointer(p);
        }
        static constexpr int numLowBitsAvailable = 3;
    };
} // namespace llvm

#endif // EFF_EFFOPS_H
