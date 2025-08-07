# ============================================================================ #
# Copyright (c) 2022 - 2025 NVIDIA Corporation & Affiliates.                   #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-type-limits")

add_cudaq_tool(cudaq-opt cudaq-opt.cpp)
target_link_libraries(cudaq-opt
  PUBLIC
  MLIRLLVMDialect
  MLIROptLib
  MLIRIR
  MLIRFuncDialect
  MLIRFuncInlinerExtension
  MLIRArithDialect

  CCDialect
  QuakeDialect
  OptCodeGen
  OptTransforms
  CUDAQSupport
)

mlir_check_all_link_libraries(cudaq-opt)
export_executable_symbols_for_plugins(cudaq-opt)