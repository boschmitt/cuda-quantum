#!/bin/bash

# ============================================================================ #
# Copyright (c) 2022 - 2023 NVIDIA Corporation & Affiliates.                   #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

# This script is intended to be called from the github workflows.

BUILD_TYPE=${1:-"Release"}
CC=${2:-"clang"}
CXX=${3:-"clang++"}
LAUNCHER=${4:-""}

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
  -DMLIR_DIR=/opt/llvm/lib/cmake/mlir \
  -DLLVM_DIR=/opt/llvm/lib/cmake/llvm \
  -DLLVM_EXTERNAL_LIT=$(which lit) \
  -DLLVM_ENABLE_ASSERTIONS=ON \
  -DCUDAQ_ENABLE_PYTHON=ON

cmake --build . --target install
