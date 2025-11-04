//===- EffPasses.h - Eff passes  --------------------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
#ifndef EFF_EFFPASSES_H
#define EFF_EFFPASSES_H

#include "Eff/EffDialect.h"
#include "Eff/EffOps.h"
#include "mlir/Pass/Pass.h"
#include <memory>

namespace mlir {
namespace eff {
#define GEN_PASS_DECL
#include "Eff/EffPasses.h.inc"

#define GEN_PASS_REGISTRATION
#include "Eff/EffPasses.h.inc"
} // namespace eff
} // namespace mlir

#endif
