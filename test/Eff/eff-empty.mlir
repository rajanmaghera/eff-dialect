// RUN: eff-opt %s | eff-opt | FileCheck %s
// XFAIL: *

module {
    // CHECK: error: name cannot be empty
    eff.func @eff_types(%arg0: !eff.sig<"" : () -> i32>) attributes { effects = [] } {
        eff.return
    }
}
