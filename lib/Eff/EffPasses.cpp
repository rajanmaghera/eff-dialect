//===- EffPasses.cpp - Eff passes -----------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Rewrite/FrozenRewritePatternSet.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinDialect.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/ValueRange.h"
#include "mlir/Support/LLVM.h"

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"
#include "llvm/CodeGen/StackMaps.h"
#include <algorithm>
#include <utility>


#include "Eff/EffPasses.h"


  namespace {
    using namespace mlir;
    using namespace mlir::eff;

    struct FuncOpLowering : public OpConversionPattern<eff::FuncOp> {
      using OpConversionPattern<eff::FuncOp>::OpConversionPattern;

      LogicalResult
      matchAndRewrite(eff::FuncOp op, OpAdaptor adaptor,
                      ConversionPatternRewriter &rewriter) const final {

        auto func = mlir::func::FuncOp::create(rewriter, op.getLoc(), op.getName(),
                                               op.getFunctionType());
        rewriter.inlineRegionBefore(op.getRegion(), func.getBody(), func.end());
        rewriter.eraseOp(op);
        return success();
      }
    };

    struct CallOpLowering : public OpConversionPattern<eff::CallOp> {
      using OpConversionPattern<eff::CallOp>::OpConversionPattern;
      LogicalResult
      matchAndRewrite(eff::CallOp op, OpAdaptor adaptor,
        ConversionPatternRewriter &rewriter
        ) const final {
        auto call = mlir::func::CallOp::create(rewriter, op.getLoc(), op.getCallee(), op.getResultTypes(), op.getOperands());
        rewriter.replaceOp(op, call);
        return success();
      }
    };


    struct ConstantOpLowering : public OpConversionPattern<eff::ConstantOp> {
      using OpConversionPattern<eff::ConstantOp>::OpConversionPattern;
      LogicalResult
      matchAndRewrite(eff::ConstantOp op, OpAdaptor adaptor,
        ConversionPatternRewriter &rewriter
        ) const final {
        auto constant = mlir::func::ConstantOp::create(rewriter, op.getLoc(), op->getResultTypes(), op.getValue());
        rewriter.replaceOp(op, constant);
        return success();
      }
    };

    struct ReturnOpLowering : public OpConversionPattern<eff::ReturnOp> {
      using OpConversionPattern<eff::ReturnOp>::OpConversionPattern;
      LogicalResult
      matchAndRewrite(eff::ReturnOp op, OpAdaptor adaptor,
        ConversionPatternRewriter &rewriter
        ) const final {
        auto ret = mlir::func::ReturnOp::create(rewriter, op.getLoc(), op.getOperands());
        rewriter.replaceOp(op, ret);
        return success();
      }
    };

  } // namespace


namespace mlir::eff {

#define GEN_PASS_DEF_EFFLOWERTOFUNC
#include "Eff/EffPasses.h.inc"

    namespace {
      class EffLowerToFunc
     : public impl::EffLowerToFuncBase<EffLowerToFunc> {
      public:
        using impl::EffLowerToFuncBase<
            EffLowerToFunc>::EffLowerToFuncBase;
        void runOnOperation() final {

          ConversionTarget target(getContext());
          target.addLegalDialect<affine::AffineDialect, BuiltinDialect,
                                 arith::ArithDialect, func::FuncDialect,
                                 memref::MemRefDialect>();
          target.addIllegalDialect<eff::EffDialect>();

          RewritePatternSet patterns(&getContext());
          patterns.add<CallOpLowering, ConstantOpLowering, ReturnOpLowering, FuncOpLowering>(&getContext());


          if (failed(
                  applyPartialConversion(getOperation(), target, std::move(patterns))))
            signalPassFailure();
        }
      };
    } // namespace
  } // namespace mlir::eff




