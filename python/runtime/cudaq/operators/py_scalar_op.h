/****************************************************************-*- C++ -*-****
 * Copyright (c) 2022 - 2025 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include <nanobind/nanobind.h>

namespace nb = nanobind;

namespace cudaq {
/// @brief Wrapper function for exposing the bindings of `cudaq::spin`
/// and `cudaq::spin_op` to python.
void bindScalarWrapper(nb::module_ &mod);
} // namespace cudaq
