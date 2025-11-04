llvm_canonicalize_cmake_booleans(
        MLIR_ENABLE_BINDINGS_PYTHON
)

configure_lit_site_cfg(
        ${CMAKE_CURRENT_SOURCE_DIR}/lit.site.cfg.py.in
        ${CMAKE_CURRENT_BINARY_DIR}/lit.site.cfg.py
        MAIN_CONFIG
        ${CMAKE_CURRENT_SOURCE_DIR}/lit.cfg.py
)

set(EFF_TEST_DEPENDS
        FileCheck count not
        eff-capi-test
        eff-opt
        eff-translate
        )
if(NOT WIN32)
  list(APPEND EFF_TEST_DEPENDS EffPlugin)
endif()
if(MLIR_ENABLE_BINDINGS_PYTHON)
  list(APPEND EFF_TEST_DEPENDS EffPythonModules)
endif()

add_lit_testsuite(check-eff "Running the eff regression tests"
        ${CMAKE_CURRENT_BINARY_DIR}
        DEPENDS ${EFF_TEST_DEPENDS}
        )
set_target_properties(check-eff PROPERTIES FOLDER "Tests")

add_lit_testsuites(EFF ${CMAKE_CURRENT_SOURCE_DIR} DEPENDS ${EFF_TEST_DEPENDS})

add_subdirectory(CAPI)
