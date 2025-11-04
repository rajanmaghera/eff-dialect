add_mlir_public_c_api_library(EffCAPI
  Dialects.cpp
  LINK_LIBS PUBLIC
  MLIREff
)
