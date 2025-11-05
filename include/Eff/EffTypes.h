//===- EffTypes.h - Eff dialect types --------------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef EFF_EFFTYPES_H
#define EFF_EFFTYPES_H

#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/BuiltinAttributes.h"

#define GET_TYPEDEF_CLASSES
#include "Eff/EffOpsTypes.h.inc"

#endif // EFF_EFFTYPES_H
