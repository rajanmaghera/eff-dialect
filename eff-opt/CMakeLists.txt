set(LIBS
    MLIRArithDialect
    MLIROptLib
    MLIRRegisterAllDialects
    MLIRRegisterAllPasses
    MLIREff
  )
add_llvm_executable(eff-opt eff-opt.cpp)

llvm_update_compile_flags(eff-opt)
target_link_libraries(eff-opt PRIVATE ${LIBS})

mlir_check_all_link_libraries(eff-opt)
