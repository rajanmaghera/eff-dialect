// UNSUPPORTED: system-windows
// RUN: mlir-opt %s --load-pass-plugin=%eff_libs/EffPlugin%shlibext --pass-pipeline="builtin.module(eff-lower-to-func)" | FileCheck %s

module {
  // TODO lowering test

  // CHECK-LABEL: @bar
  func.func @bar() {
    return
  }

  func.func @abar() {
    return
  }
}
