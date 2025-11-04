add_mlir_dialect(EffOps eff)
add_mlir_doc(EffDialect EffDialect Eff/ -gen-dialect-doc)
add_mlir_doc(EffOps EffOps Eff/ -gen-op-doc)

set(LLVM_TARGET_DEFINITIONS EffPasses.td)
mlir_tablegen(EffPasses.h.inc --gen-pass-decls)
add_public_tablegen_target(MLIREffPassesIncGen)
