include(AddMLIRPython)

# Specifies that all MLIR packages are co-located under the `mlir_eff`
# top level package (the API has been embedded in a relocatable way).
add_compile_definitions("MLIR_PYTHON_PACKAGE_PREFIX=${MLIR_PYTHON_PACKAGE_PREFIX}.")


################################################################################
# Sources
################################################################################

declare_mlir_python_sources(EffPythonSources)

declare_mlir_dialect_python_bindings(
  ADD_TO_PARENT EffPythonSources
  ROOT_DIR "${CMAKE_CURRENT_SOURCE_DIR}/mlir_eff"
  TD_FILE dialects/EffOps.td
  SOURCES
    dialects/eff_pybind11.py
    dialects/eff_nanobind.py
  DIALECT_NAME eff)


declare_mlir_python_extension(EffPythonSources.Pybind11Extension
  MODULE_NAME _effDialectsPybind11
  ADD_TO_PARENT EffPythonSources
  SOURCES
    EffExtensionPybind11.cpp
  PRIVATE_LINK_LIBS
    LLVMSupport
  EMBED_CAPI_LINK_LIBS
    EffCAPI
  PYTHON_BINDINGS_LIBRARY pybind11
)

declare_mlir_python_extension(EffPythonSources.NanobindExtension
  MODULE_NAME _effDialectsNanobind
  ADD_TO_PARENT EffPythonSources
  SOURCES
    EffExtensionNanobind.cpp
  PRIVATE_LINK_LIBS
    LLVMSupport
  EMBED_CAPI_LINK_LIBS
    EffCAPI
  PYTHON_BINDINGS_LIBRARY nanobind
)


################################################################################
# Common CAPI
################################################################################

add_mlir_python_common_capi_library(EffPythonCAPI
  INSTALL_COMPONENT EffPythonModules
  INSTALL_DESTINATION "${MLIR_BINDINGS_PYTHON_INSTALL_PREFIX}/_mlir_libs"
  OUTPUT_DIRECTORY "${MLIR_BINARY_DIR}/${MLIR_BINDINGS_PYTHON_INSTALL_PREFIX}/_mlir_libs"
  RELATIVE_INSTALL_ROOT "../../../.."
  DECLARED_SOURCES
    EffPythonSources
    # TODO: Remove this in favor of showing fine grained registration once
    # available.
    MLIRPythonExtension.RegisterEverything
    MLIRPythonSources.Core
    MLIRPythonSources.Dialects.builtin
)

################################################################################
# Instantiation of all Python modules
################################################################################

add_mlir_python_modules(EffPythonModules
  ROOT_PREFIX "${MLIR_BINARY_DIR}/${MLIR_BINDINGS_PYTHON_INSTALL_PREFIX}"
  INSTALL_PREFIX "${MLIR_BINDINGS_PYTHON_INSTALL_PREFIX}"
  DECLARED_SOURCES
    EffPythonSources
    # TODO: Remove this in favor of showing fine grained registration once
    # available.
    MLIRPythonExtension.RegisterEverything
    MLIRPythonSources.Core
    MLIRPythonSources.Dialects.builtin
  COMMON_CAPI_LINK_LIBS
    EffPythonCAPI
  )
