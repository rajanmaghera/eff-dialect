// RUN: not eff-opt %s 2>&1 | FileCheck %s

module {
    eff.define @sym : () -> ()
    // CHECK-LABEL: error
    eff.define @sym : (i32) -> ()
}
