#!/bin/bash

# ============================================================================ #
# Copyright (c) 2022 - 2023 NVIDIA Corporation & Affiliates.                   #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

# This script is intended to be called from the github workflows.

LLVM_INSTALL_PREFIX=${1}
BUILD_TYPE=${2}
CC=${3}
CXX=${4}
LAUNCHER=${5}

mkdir build
cd build

cmake .. \
  -G Ninja \
  -DCMAKE_INSTALL_PREFIX=../install \
  -DCMAKE_BUILD_TYPE=$BUILD_TYPE \
  -DCMAKE_C_COMPILER=$CC \
  -DCMAKE_CXX_COMPILER=$CXX \
  -DCMAKE_C_COMPILER_LAUNCHER=$LAUNCHER \
  -DCMAKE_CXX_COMPILER_LAUNCHER=$LAUNCHER \
  -DCMAKE_EXPORT_COMPILE_COMMANDS=ON \
  -DMLIR_DIR=$LLVM_INSTALL_PREFIX/lib/cmake/mlir \
  -DLLVM_DIR=$LLVM_INSTALL_PREFIX/lib/cmake/llvm \
  -DLLVM_EXTERNAL_LIT=$(which lit) \
  -DLLVM_ENABLE_ASSERTIONS=ON \
  -DCUDAQ_ENABLE_PYTHON=ON

cmake --build . --target install
