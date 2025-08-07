/****************************************************************-*- C++ -*-****
 * Copyright (c) 2022 - 2025 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/
#include <nanobind/nanobind.h>

#include "utils/LinkedLibraryHolder.h"

namespace nb = nanobind;

namespace cudaq {
/// @brief Bind `cudaq.MeasureCounts` to python.
void bindMeasureCounts(nb::module_ &mod);
} // namespace cudaq
