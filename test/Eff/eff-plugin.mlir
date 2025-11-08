// UNSUPPORTED: system-windows
// RUN: mlir-opt %s --load-dialect-plugin=%eff_libs/EffPlugin%shlibext --pass-pipeline="builtin.module()" | FileCheck %s

module {
    // CHECK-LABEL: eff.define @custom : () -> i32
    eff.define @custom : () -> i32

    // CHECK-LABEL: eff.func @eff_types() attributes {effects = [@custom]}
    eff.func @eff_types() attributes {effects = [@custom]} {
        eff.return
    }
}
