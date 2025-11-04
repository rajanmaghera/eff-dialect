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

namespace mlir {
    class PatternRewriter;
} // namespace mlir

namespace mlir {
    namespace eff {
        class EffType;
        namespace detail {
            struct EffTypeStorage;
        }
    }
}


#define GET_TYPEDEF_CLASSES
#include "Eff/EffOpsTypes.h.inc"

namespace mlir {
    namespace eff {

        class EffType : public Type::TypeBase<EffType, Type, detail::EffTypeStorage> {

        public:
            using Base::Base;

            static EffType get(StringRef name, FunctionType funcSig);

            static LogicalResult verifyConstructionInvariants(
              Location loc, StringRef name, FunctionType funcSig
            );

            StringRef getHandlerName();

            FunctionType getHandlerSignature();

            static constexpr StringLiteral name = "eff.effect";

        };
    }
}



#endif // EFF_EFFTYPES_H
