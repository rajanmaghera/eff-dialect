# RUN: %python %s pybind11 | FileCheck %s
# RUN: %python %s nanobind | FileCheck %s

import sys
from mlir_eff.ir import *
from mlir_eff.dialects import builtin as builtin_d

if sys.argv[1] == "pybind11":
    from mlir_eff.dialects import eff_pybind11 as eff_d
elif sys.argv[1] == "nanobind":
    from mlir_eff.dialects import eff_nanobind as eff_d
else:
    raise ValueError("Expected either pybind11 or nanobind as arguments")


with Context():
    eff_d.register_dialect()
    module = Module.parse(
        """
    %0 = arith.constant 2 : i32
    eff.return %0 : i32
    """
    )
    # CHECK: %[[C:.*]] = arith.constant 2 : i32
    # CHECK: eff.return %[[C]] : i32
    print(str(module))
