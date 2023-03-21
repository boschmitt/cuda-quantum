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

git submodule update --init --recursive --recommend-shallow tpls/llvm

mkdir -p tpls/llvm/build
mkdir -p tpls/llvm/install
cd tpls/llvm/build
cmake ../llvm \
  -G Ninja \
  -DCMAKE_BUILD_TYPE=$BUILD_TYPE \
  -DCMAKE_C_COMPILER=$CC \
  -DCMAKE_CXX_COMPILER=$CXX \
  -DCMAKE_INSTALL_PREFIX=../install \
  -DLLVM_BUILD_EXAMPLES=OFF \
  -DLLVM_ENABLE_ASSERTIONS=ON \
  -DLLVM_ENABLE_BINDINGS=OFF \
  -DLLVM_ENABLE_LLD=ON \
  -DLLVM_ENABLE_OCAMLDOC=OFF \
  -DLLVM_ENABLE_PROJECTS='clang;mlir' \
  -DLLVM_INSTALL_UTILS=ON \
  -DLLVM_OPTIMIZED_TABLEGEN=ON \
  -DLLVM_TARGETS_TO_BUILD="host"

cmake --build . --target install -- -j$(nproc)
