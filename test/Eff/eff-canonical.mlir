// RUN: eff-opt %s | eff-opt | FileCheck %s

module {

  eff.define @io.merge: (i64, i64) -> i64

    // CHECK-LABEL: eff.func @thefunc(%arg0: i64, %arg1: i64) -> i64 attributes {effects = [@io.merge]}
  eff.func @thefunc(%arg0: i64, %arg1: i64) -> i64 attributes {effects = [@io.merge]} {
    %0 = arith.addi %arg0, %arg1 : i64
    %1 = do_effect @io.merge(%0, %0) : (i64, i64) -> i64
    %2 = arith.addi %0, %1 : i64
    %c42_i64 = arith.constant 42 : i64
    %3 = arith.addi %2, %c42_i64 : i64
    return %3 : i64
  }
  // CHECK-LABEL: eff.func @thedup(%arg0: i64) -> i64 attributes {effects = []}
  eff.func @thedup(%arg0: i64) -> i64 attributes {effects = []} {
    %1 = eff.handle @io.merge(%k: !eff.cont, %arg2: i64, %arg3: i64) {
        %2 = arith.addi %arg2, %arg3 : i64
        %3 = eff.continue %k, %2: (!eff.cont, i64) -> i64
        eff.yield %2 : i64
    } {
        %0 = eff.call @thefunc(%arg0, %arg0) {effects = [@io.merge]} : (i64, i64) -> i64
        eff.yield %0 : i64
    } : i64
    return %1 : i64
  }
}