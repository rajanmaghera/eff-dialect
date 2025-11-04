add_llvm_library(EffPlugin
        # BUILDTREE_ONLY is only for testing purposes
        MODULE BUILDTREE_ONLY
        eff-plugin.cpp

        DEPENDS
        MLIREff
        PLUGIN_TOOL
        mlir-opt

        LINK_LIBS
        MLIREff
        )
target_include_directories(
        EffPlugin
        PRIVATE
        "${EFF_BINARY_DIR}/include"
)
