# eff-dialect

A prototype effect handler dialect in MLIR.

## Building

Install pre-requisites.
```shell
sudo apt install build-essentials cmake python3 zlib1g-dev ninja-build
```

Obtain the LLVM repo.
```shell
# use a pinned version
wget -O "llvm-project.zip" "https://github.com/llvm/llvm-project/archive/2b135b931338a57c38d9c4a34ffdd59877ba82d6.zip"
unzip llvm-project
# or use the latest LLVM
git clone https://github.com/llvm/llvm-project
```

Build MLIR.
```shell
pushd llvm-project
mkdir build && cd build
cmake -G Ninja ../llvm \
    -DLLVM_ENABLE_PROJECTS=mlir \
    -DLLVM_TARGETS_TO_BUILD="Native" \
    -DCMAKE_BUILD_TYPE=Release \ # or Debug or RelWithDebInfo
    -DLLVM_ENABLE_ASSERTIONS=ON \
    -DCMAKE_C_COMPILER=clang \ # optional
    -DCMAKE_CXX_COMPILER=clang++ \ # optional
    -DLLVM_ENABLE_LLD=ON \ # optional
    -DLLVM_CCACHE_BUILD=ON  # optional
cmake --build .
export MLIR_BUILD_DIR="$(pwd)"
popd
```

Clone this repo.
```shell
git clone https://github.com/rajanmaghera/eff-dialect
```

Initialize build directory.
```shell
cd eff-dialect && mkdir build && cd build
cmake -G Ninja .. \
    -DMLIR_DIR="$MLIR_BUILD_DIR/lib/cmake/mlir" \
    -DLLVM_EXTERNAL_LIT="$MLIR_BUILD_DIR/bin/llvm-lit" \
    -DCMAKE_BUILD_TYPE=Release \ # or Debug or RelWithDebInfo
```

Build documentation from TableGen.
```shell
cmake --build . --target mlir-doc
```

Build and run tests.
```shell
cmake --build . --target check-eff
```

Format all files.
```shell
find . -regex '.*\\.\\(cpp\\|h\\)' | xargs clang-format -i
```
