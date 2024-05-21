/****************************************************************-*- C++ -*-****
 * Copyright (c) 2022 - 2024 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#pragma once

#include "cudaq/qis/modifiers.h"
#include "cudaq/qis/qubit_qis.h"
#include "cudaq/qis/qudit.h"
#include "cudaq/qis/qview.h"

namespace cudaq::internal {

/// Ripple-Carry approach with depth O(n). (Implements b += a.)
///
/// Takahashi, Yasuhiro, and Noboru Kunihiro. "A fast quantum circuit for
/// addition with few qubits." Quantum Information & Computation 8.6 (2008):
/// 636-649.
inline void __qpu__ carry_ripple_adder_ttk(cudaq::qview<2> a, cudaq::qview<2> b,
                                           qubit &carry) {
  assert(a.size() == b.size());
  int n = a.size();

  // Do this so the construction is the same as the paper
  std::vector<cudaq::qubit *> a_carry;
  for (qubit &q : a)
    a_carry.push_back(&q);
  a_carry.push_back(&carry);

  // Step 1
  for (int i = 1; i < n; ++i)
    x<ctrl>(*a_carry[i], b[i]);

  // Step 2
  for (int i = n; i-- > 1;)
    x<ctrl>(*a_carry[i], *a_carry[i + 1]);

  // Step 3
  for (int i = 0; i < n; ++i)
    x<ctrl>(*a_carry[i], b[i], *a_carry[i + 1]);

  // Step 4
  for (int i = n; i-- > 1;) {
    x<ctrl>(*a_carry[i], b[i]);
    x<ctrl>(*a_carry[i - 1], b[i - 1], *a_carry[i]);
  }

  // Step 5
  for (int i = 1; i < n - 1; ++i)
    x<ctrl>(*a_carry[i], *a_carry[i + 1]);

  // Step 6
  for (int i = 0; i < n; ++i)
    x<ctrl>(*a_carry[i], b[i]);
}

} // namespace cudaq::internal
