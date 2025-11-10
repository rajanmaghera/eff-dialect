// RUN: not eff-opt %s 2>&1 | FileCheck %s

module {

  eff.define @io.modify : () -> ()
  eff.define @io.reade: (i64) -> ()
  eff.define @io.reader: (i64) -> ()
  eff.define @io.read: () -> ()

  eff.func @func1() attributes {effects = [@io.reader]} {
    eff.handle @io.modify(%arg1: !eff.cont) {
        eff.yield
    } {
    eff.handle @io.reader(%arg2: !eff.cont, %arg10: i64) {
        eff.yield
    } {
    eff.handle @io.modify(%arg3: !eff.cont) {
        eff.yield
    } {
    eff.handle @io.modify(%arg4: !eff.cont) {
        eff.yield
    } {
        %0 = arith.constant 42 : i64
        // CHECK-LABEL: error: 'eff.do' op effect io.something is not handled or in function's signature
        eff.do @io.something(%0) : (i64) -> ()
        eff.yield
    }
        eff.yield
    }
        eff.yield
    }
        eff.yield
    }
    return
  }

}