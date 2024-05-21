/*******************************************************************************
 * Copyright (c) 2022 - 2024 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "CUDAQTestUtils.h"
#include <cudaq/qis/qarith.h>

#ifndef CUDAQ_BACKEND_DM

//===----------------------------------------------------------------------===//
// Helpers
//===----------------------------------------------------------------------===//

/// Returns the largerst signed integer representable using `bitwidth` bits.
constexpr static int max_int(std::size_t bitwidth) {
  return (1 << (bitwidth - 1)) - 1;
}

/// Returns the smallest signed integer representable by `bitwidth` bits.
constexpr static int min_int(std::size_t bitwidth) {
  return -(1 << (bitwidth - 1));
}

//===----------------------------------------------------------------------===//

CUDAQ_TEST(QubitArithTester, checkLoad) {
  constexpr std::size_t kNUM_QUBITS = 3;

  for (int v = min_int(kNUM_QUBITS); v < max_int(kNUM_QUBITS); ++v) {
    cudaq::qvector<2> qs(kNUM_QUBITS);
    cudaq::load(qs, v);
    ASSERT_EQ(mz_int(qs), v);
  }
}

#endif
