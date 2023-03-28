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

LLVM_PROJECTS="clang;mlir"

# LLVM library components
LLVM_COMPONENTS="cmake-exports;llvm-headers;llvm-libraries"

# Clang library components
LLVM_COMPONENTS+=";clang-cmake-exports;clang-headers;clang-libraries;clang-resource-headers"

# MLIR library components
LLVM_COMPONENTS+=";mlir-cmake-exports;mlir-headers;mlir-libraries"

# Tools / Utils
LLVM_COMPONENTS+=";llvm-config;clang-format;llc;clang;mlir-tblgen;FileCheck;count;not"

# Clone LLVM fast
LLVM_SHA=$(git rev-parse @:./tpls/llvm)
cd tpls/llvm
git init
git remote add origin https://github.com/llvm/llvm-project
git fetch --depth=1 origin $LLVM_SHA
git reset --hard FETCH_HEAD

mkdir build
mkdir -p install/llvm

# Configure and build
cd build
cmake ../llvm \
  -G Ninja \
  -DCMAKE_BUILD_TYPE=$BUILD_TYPE \
  -DCMAKE_C_COMPILER=$CC \
  -DCMAKE_CXX_COMPILER=$CXX \
  -DCMAKE_C_COMPILER_LAUNCHER=$LAUNCHER \
  -DCMAKE_CXX_COMPILER_LAUNCHER=$LAUNCHER \
  -DCMAKE_INSTALL_PREFIX=../install/llvm \
  -DLLVM_ENABLE_PROJECTS=$LLVM_PROJECTS \
  -DLLVM_DISTRIBUTION_COMPONENTS=$LLVM_COMPONENTS \
  -DLLVM_BUILD_EXAMPLES=OFF \
  -DLLVM_ENABLE_ASSERTIONS=ON \
  -DLLVM_ENABLE_BINDINGS=OFF \
  -DLLVM_ENABLE_LLD=ON \
  -DLLVM_ENABLE_OCAMLDOC=OFF \
  -DLLVM_INSTALL_UTILS=ON \
  -DLLVM_OPTIMIZED_TABLEGEN=ON \
  -DLLVM_TARGETS_TO_BUILD="host"

cmake --build . --target install-distribution-stripped
