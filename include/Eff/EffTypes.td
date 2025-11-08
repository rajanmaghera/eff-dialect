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

// TODO: maybe add return type to continuation (to make it easier to track when
//  continuing

class EffDialect_Type<string name, string typeMnemonic, list<Trait> traits = []>
    : TypeDef<Eff_Dialect, name, traits> {
  let mnemonic = typeMnemonic;
}

def Eff_ContType : EffDialect_Type<"Continuation", "cont"> {
    let summary = "Effect continuation type";
}

#endif // EFF_TYPES
