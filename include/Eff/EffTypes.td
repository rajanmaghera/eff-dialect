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

def FunctionTypeParameter :
    TypeParameter<"::mlir::FunctionType", "A function type param">;

// TODO: move to purely attributes for effect signatures

class EffDialect_Type<string name, string typeMnemonic, list<Trait> traits = []>
    : TypeDef<Eff_Dialect, name, traits> {
  let mnemonic = typeMnemonic;
}

def Eff_SigType : EffDialect_Type<"Signature", "sig"> {
    let summary = "Effect signature type";

    let parameters = (ins
        StringRefParameter<>:$name,
        FunctionTypeParameter:$fn
    );

    let assemblyFormat = "`<` $name `:` $fn `>`";

    let builders = [
        AttrBuilderWithInferredContext<(ins "const ::llvm::StringRef &":$name,
                                            "const ::mlir::FunctionType &":$fn), [{
          return $_get(fn.getContext(), name, fn);
        }]>
      ];

   let genVerifyDecl = 1;
   let skipDefaultBuilders = 1;
}

def Eff_ContType : EffDialect_Type<"Continuation", "cont"> {
    let summary = "Effect continuation type";
}

#endif // EFF_TYPES
