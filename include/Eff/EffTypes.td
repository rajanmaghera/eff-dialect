//===- EffTypes.td - Eff dialect types ---------------------*- tablegen -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef EFF_TYPES
#define EFF_TYPES

include "mlir/IR/AttrTypeBase.td"
include "Eff/EffDialect.td"

//===----------------------------------------------------------------------===//
// Eff type definitions
//===----------------------------------------------------------------------===//

def Eff_EffType :
    DialectType<Eff_Dialect, CPred<"isa<EffType>($_self)">,
                "Effect type">;

def Eff_Type: AnyTypeOf<[FunctionType, Eff_EffType]>;

#endif // EFF_TYPES
