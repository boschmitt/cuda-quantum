/****************************************************************-*- C++ -*-****
 * Copyright (c) 2022 - 2024 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#pragma once

#include "cudaq/qis/internal/adder.h"
#include "cudaq/qis/qubit_qis.h"
#include "cudaq/qis/qview.h"

namespace cudaq {

/// Measures a set of qubits, and convert its result into an integer.
///
/// Note: The qview is interpreted as a signed interger in two's complement
/// form.
inline int mz_int(qview<2> veq) {
  std::vector<measure_result> results = mz(veq);

  int result = 0;
  // Check whether the result is a negative integer. (Note: in two's complement
  // a number is negative if its LSB is one.)
  if (results.back()) {
    result = ~result;
    for (std::size_t i = 0, end = results.size(); i < end; ++i)
      if (!results[i])
        result ^= (1 << i);
  } else {
    for (std::size_t i = 0, end = results.size(); i < end; ++i)
      if (results[i])
        result ^= (1 << i);
  }

  return result;
}

/// Loads an integer value into a set of qubits that is in the zero state.
///
/// Note: Effectively this kernel applies an in-place XOR operation between the
/// set of qubits and a constant, `value`. Thus, the integer value can only be
/// correctly loaded into the set, if the qubits are in the zero state.
inline void load(qview<2> veq, int value) {
  // TODO: make sure the value fits the vector.
  for (std::size_t i = 0, end = veq.size(); i < end; ++i) {
    if (value & (1 << i))
      x(veq[i]);
  }
}

/// Interprets the sets of qubtis, `a`, as an integer and do in-place increment
/// a += 1.
inline void increment(qview<2> qubits) {
  const std::size_t size = qubits.size();
  for (std::size_t i = 1; i < size; ++i)
    cudaq::control([](qubit &q) { x(q); }, qubits.front(size - i),
                   qubits[size - i]);
  // x<ctrl>(qubits.front(size - i), qubits[size - i]);
  x(qubits[0]);
}

/// Interprets the sets of qubtis, `a`, as an integer and do in-place decrement
/// a -= 1.
inline void decrement(qview<2> qubits) {
  // We implement decrement in terms of incrementing the one's complement form.
  x(qubits); // Broadcast
  increment(qubits);
  x(qubits); // Broadcast
}

/// Interprets the two sets of qubtis, `a` and `b`, as integers and do in-place
/// addition, b += a.
inline void add(qview<2> a, qview<2> b, qubit &carry) {
  internal::carry_ripple_adder_ttk(a, b, carry);
}

inline void two_complement(qview<2> qubits) {
  x(qubits); // Broadcast
  increment(qubits);
}

/// Interprets the two sets of qubtis, `a` and `b`, as integers and do in-place
/// addition, b -= a.
inline void sub(qview<2> a, qview<2> b, qubit &carry) {
  two_complement(a);
  internal::carry_ripple_adder_ttk(a, b, carry);
  two_complement(a);
}

} // namespace cudaq
