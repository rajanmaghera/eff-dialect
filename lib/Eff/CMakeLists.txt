add_mlir_dialect_library(MLIREff
        EffTypes.cpp
        EffDialect.cpp
        EffOps.cpp
        EffPasses.cpp

        ADDITIONAL_HEADER_DIRS
        ${PROJECT_SOURCE_DIR}/include/Eff

        DEPENDS
        MLIREffOpsIncGen
        MLIREffPassesIncGen

        LINK_LIBS PUBLIC
        MLIRIR
        MLIRInferTypeOpInterface
        MLIRFuncDialect
        )
