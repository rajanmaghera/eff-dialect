// RUN: eff-opt %s | eff-opt | FileCheck %s

module {

  // CHECK-LABEL: eff.func @func1() attributes {effects = []}
  eff.func @func1() attributes {effects = []} {
    "eff.handle"() <{effect = !eff.sig<"io.modify" : () -> ()>}> ({
    ^bb0(%arg1: !eff.cont):
    }, {
    "eff.handle"() <{effect = !eff.sig<"io.something" : (i64) -> ()>}> ({
    ^bb1(%arg2: !eff.cont, %arg10: i64):
    }, {
    "eff.handle"() <{effect = !eff.sig<"io.modify" : () -> ()>}> ({
    ^bb2(%arg3: !eff.cont):
    }, {
    "eff.handle"() <{effect = !eff.sig<"io.modify" : () -> ()>}> ({
    ^bb3(%arg4: !eff.cont):
    }, {
        %0 = arith.constant 42 : i64
        do_effect "io.something"(%0) : (i64) -> ()
        eff.yield
    }): () -> ()
        eff.yield
    }): () -> ()
        eff.yield
    }): () -> ()
        eff.yield
    }): () -> ()
    return
  }

  // CHECK-LABEL: eff.func @func2() attributes {effects = [!eff.sig<"io.something" : (i64) -> ()>]}
  eff.func @func2() attributes {effects = [!eff.sig<"io.something" : (i64) -> ()>]} {
    "eff.handle"() <{effect = !eff.sig<"io.modify" : () -> ()>}> ({
    ^bb0(%arg1: !eff.cont):
    }, {
    "eff.handle"() <{effect = !eff.sig<"io.modify_2" : (i64) -> ()>}> ({
    ^bb1(%arg2: !eff.cont, %arg10: i64):
    }, {
    "eff.handle"() <{effect = !eff.sig<"io.modify" : () -> ()>}> ({
    ^bb2(%arg3: !eff.cont):
    }, {
    "eff.handle"() <{effect = !eff.sig<"io.modify" : () -> ()>}> ({
    ^bb3(%arg4: !eff.cont):
    }, {
        %0 = arith.constant 42 : i64
        do_effect "io.something"(%0) : (i64) -> ()
        eff.yield
    }): () -> ()
        eff.yield
    }): () -> ()
        eff.yield
    }): () -> ()
        eff.yield
    }): () -> ()
    return
  }
}