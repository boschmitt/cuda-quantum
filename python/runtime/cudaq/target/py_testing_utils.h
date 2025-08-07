/****************************************************************-*- C++ -*-****
 * Copyright (c) 2022 - 2025 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#pragma once

#include <nanobind/nanobind.h>

namespace nb = nanobind;

namespace cudaq {

class LinkedLibraryHolder;

/// @brief Bind test utilities needed for mock QPU QIR profile simulation
void bindTestUtils(nb::module_ &mod, LinkedLibraryHolder &holder);

} // namespace cudaq
