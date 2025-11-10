// RUN: eff-opt %s | eff-opt | FileCheck %s

module {

  eff.define @io.modify : () -> ()
  eff.define @io.modify2 : (i64) -> ()
  eff.define @io.something : (i64) -> ()

  // CHECK-LABEL: eff.func @func1() attributes {effects = []}
  eff.func @func1() attributes {effects = []} {
    eff.handle @io.modify(%arg1: !eff.cont) {
        eff.yield
    } {
    eff.handle @io.something(%arg2: !eff.cont, %arg10: i64) {
        eff.yield
    } {
    eff.handle @io.modify(%arg3: !eff.cont) {
        eff.yield
    } {
    eff.handle @io.modify(%arg4: !eff.cont) {
        eff.yield
    } {
        %0 = arith.constant 42 : i64
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

  // CHECK-LABEL: eff.func @func2() attributes {effects = [@io.something]}
  eff.func @func2() attributes {effects = [@io.something]} {
    eff.handle @io.modify(%arg1: !eff.cont) {
        eff.yield
    } {
    eff.handle @io.modify2(%arg2: !eff.cont, %arg10: i64) {
        eff.yield
    } {
    eff.handle @io.modify(%arg3: !eff.cont) {
        eff.yield
    } {
    eff.handle @io.modify(%arg4: !eff.cont) {
        eff.yield
    } {
        %0 = arith.constant 42 : i64
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