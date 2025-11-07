//===- EffDialect.td - Eff dialect -------------------------*- tablegen -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef EFF_DIALECT
#define EFF_DIALECT

include "mlir/IR/OpBase.td"

//===----------------------------------------------------------------------===//
// Eff dialect definition.
//===----------------------------------------------------------------------===//

def Eff_Dialect : Dialect {
    let name = "eff";
    let summary = "An effect dialect for MLIR";
    let description = [{
        An effect dialect for MLIR. Course project for CSC2126.
    }];
    let cppNamespace = "::mlir::eff";

    let useDefaultTypePrinterParser = 1;
    let extraClassDeclaration = [{
        void registerTypes();
    }];
}

//===----------------------------------------------------------------------===//
// Base eff operation definition.
//===----------------------------------------------------------------------===//

class Eff_Op<string mnemonic, list<Trait> traits = []> :
        Op<Eff_Dialect, mnemonic, traits>;

#endif // EFF_DIALECT
