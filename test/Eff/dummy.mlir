// RUN: eff-opt %s | eff-opt | FileCheck %s

module {
    // CHECK-LABEL: eff.func @bar()
    eff.func @bar() -> i32 attributes { effects = [] } {
        %0 = arith.constant 1 : i32
        // CHECK: return %{{.*}} : i32
        eff.return %0 : i32
    }

    // CHECK-LABEL: eff.func @eff_types(%arg0: !eff.sig<"custom" : () -> i32>)
    eff.func @eff_types(%arg0: !eff.sig<"custom" : () -> i32>) attributes { effects = [] } {
        eff.return
    }
}
