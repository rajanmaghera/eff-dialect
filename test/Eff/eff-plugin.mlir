// UNSUPPORTED: system-windows
// RUN: mlir-opt %s --load-dialect-plugin=%eff_libs/EffPlugin%shlibext --pass-pipeline="builtin.module(eff-lower-to-func)" | FileCheck %s

module {
  // CHECK-LABEL: func @eff_types(%arg0: !eff.effect<@custom : () -> i32>)
  func.func @eff_types(%arg0: !eff.effect<@custom : () -> i32>) {
    return
  }
}
