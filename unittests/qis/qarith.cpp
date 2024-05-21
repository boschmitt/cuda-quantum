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

static int convert(int value, std::size_t bitwidth) {
  int max_int = (1 << (bitwidth - 1)) - 1;
  value &= (1 << bitwidth) - 1;
  if (value <= max_int)
    return value;

  return -((1 << bitwidth) - value);
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

CUDAQ_TEST(QubitArithTester, checkInPlaceIncrement) {
  constexpr std::size_t kNUM_QUBITS = 4;

  cudaq::qvector<2> qs(kNUM_QUBITS);
  cudaq::load(qs, min_int(kNUM_QUBITS));
  for (int v = min_int(kNUM_QUBITS) + 1; v < max_int(kNUM_QUBITS); ++v) {
    cudaq::increment(qs);
    ASSERT_EQ(mz_int(qs), v);
  }
}

CUDAQ_TEST(QubitArithTester, checkInPlaceDecrement) {
  constexpr std::size_t kNUM_QUBITS = 4;

  cudaq::qvector<2> qs(kNUM_QUBITS);
  cudaq::load(qs, max_int(kNUM_QUBITS));
  for (int v = max_int(kNUM_QUBITS); v-- > min_int(kNUM_QUBITS);) {
    cudaq::decrement(qs);
    ASSERT_EQ(mz_int(qs), v);
  }
}

CUDAQ_TEST(QubitArithTester, checkInPlaceAdd) {
  constexpr std::size_t kNUM_QUBITS = 4;

  for (int a = min_int(kNUM_QUBITS); a < max_int(kNUM_QUBITS); ++a) {
    for (int b = min_int(kNUM_QUBITS); b < max_int(kNUM_QUBITS); ++b) {
      cudaq::qvector<2> qa(kNUM_QUBITS);
      cudaq::qvector<2> qb(kNUM_QUBITS);
      cudaq::qubit carry;

      cudaq::load(qa, a);
      cudaq::load(qb, b);

      cudaq::add(qa, qb, carry);
      ASSERT_EQ(mz_int(qa), a);
      ASSERT_EQ(mz_int(qb), convert(a + b, kNUM_QUBITS))
          << "a + b : " << a << " + " << b;
    }
  }
}

CUDAQ_TEST(QubitArithTester, checkInPlaceSub) {
  constexpr std::size_t kNUM_QUBITS = 4;

  for (int a = min_int(kNUM_QUBITS); a < max_int(kNUM_QUBITS); ++a) {
    for (int b = min_int(kNUM_QUBITS); b < max_int(kNUM_QUBITS); ++b) {
      cudaq::qvector<2> qa(kNUM_QUBITS);
      cudaq::qvector<2> qb(kNUM_QUBITS);
      cudaq::qubit carry;

      cudaq::load(qa, a);
      cudaq::load(qb, b);

      cudaq::sub(qa, qb, carry);
      ASSERT_EQ(mz_int(qa), a);
      ASSERT_EQ(mz_int(qb), convert(b - a, kNUM_QUBITS))
          << "b - a : " << b << " - " << a;
    }
  }
}

#endif
