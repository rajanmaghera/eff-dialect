//===- EffPsss.td - Eff dialect passes ---------------------*- tablegen -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef EFF_PASS
#define EFF_PASS

include "mlir/Pass/PassBase.td"

def EffLowerToFunc: Pass<"eff-lower-to-func", "::mlir::ModuleOp"> {
  let summary = "Lower eff dialect to func dialect.";
}

#endif // EFF_PASS
