// RUN: not eff-opt %s 2>&1 | FileCheck %s

module {
    // CHECK: error: name cannot be empty
    eff.func @eff_types(%arg0: !eff.sig<"" : () -> i32>) attributes { effects = [] } {
        eff.return
    }
}
