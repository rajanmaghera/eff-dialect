// RUN: not eff-opt %s 2>&1 | FileCheck %s

module {

  eff.func @func() attributes {effects = []} {
    "eff.handle"() <{effect = !eff.sig<"io.modify" : () -> ()>}> ({
    ^bb0(%arg1: !eff.cont):
    }, {
    "eff.handle"() <{effect = !eff.sig<"io.modify" : () -> ()>}> ({
    ^bb1(%arg2: !eff.cont):
    }, {
    "eff.handle"() <{effect = !eff.sig<"io.modify" : () -> ()>}> ({
    ^bb2(%arg3: !eff.cont):
    }, {
    "eff.handle"() <{effect = !eff.sig<"io.modify" : () -> ()>}> ({
    ^bb3(%arg4: !eff.cont):
    }, {
        %0 = arith.constant 42 : i64
        // CHECK-LABEL: error: 'eff.do_effect' op effect '!eff.sig<"io.read" : (i64) -> ()>' is not handled or in function's signature
        do_effect "io.read"(%0) : (i64) -> ()
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

  eff.func @func2() attributes {effects = [!eff.sig<"io.reader" : (i64) -> ()>]} {
    "eff.handle"() <{effect = !eff.sig<"io.modify" : () -> ()>}> ({
    ^bb0(%arg1: !eff.cont):
    }, {
    "eff.handle"() <{effect = !eff.sig<"io.modify" : (i64) -> ()>}> ({
    ^bb1(%arg2: !eff.cont):
    }, {
    "eff.handle"() <{effect = !eff.sig<"io.read" : () -> ()>}> ({
    ^bb2(%arg3: !eff.cont):
    }, {
    "eff.handle"() <{effect = !eff.sig<"io.modify" : () -> ()>}> ({
    ^bb3(%arg4: !eff.cont):
    }, {
        %0 = arith.constant 42 : i64
        // CHECK-LABEL: error: 'eff.do_effect' op effect '!eff.sig<"io.reade" : (i64) -> ()>' is not handled or in function's signature
        do_effect "io.reade"(%0) : (i64) -> ()
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