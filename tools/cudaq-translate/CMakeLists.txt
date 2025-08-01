# ============================================================================ #
# Copyright (c) 2022 - 2025 NVIDIA Corporation & Affiliates.                   #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-type-limits")

add_cudaq_tool(cudaq-translate cudaq-translate.cpp)

target_link_libraries(cudaq-translate
  PUBLIC
  MLIRIR
  MLIRParser
  MLIRPass
  MLIRTranslateLib
  MLIRSupport
  MLIROptLib
  MLIRExecutionEngine
  MLIRTransforms
  MLIRTargetLLVMIRExport
  MLIRLLVMCommonConversion
  MLIRFuncInlinerExtension

  # Translation libraries.
  MLIRBuiltinToLLVMIRTranslation
  MLIRLLVMToLLVMIRTranslation

  CCDialect
  OptCodeGen
  OptTransforms
  QuakeDialect
)

mlir_check_all_link_libraries(cudaq-translate)
