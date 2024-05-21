/****************************************************************-*- C++ -*-****
 * Copyright (c) 2022 - 2024 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#pragma once

#include "cudaq/qis/qubit_qis.h"
#include "cudaq/qis/qview.h"

namespace cudaq::internal {

// Modified TTK ripple-carry: it computes only the carry out of a `add`
// operation, and it assumes a carry-in or 1.
inline void __qpu__ carry1_ttk(cudaq::qview<2> a, cudaq::qview<2> b,
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
  // Here we simulate the existance of a carry-in that is set to |1>
  x(*a_carry[0], b[0]);
  x<ctrl>(*a_carry[0], b[0], *a_carry[1]);
  x(*a_carry[1]);

  for (int i = 1; i < n; ++i)
    x<ctrl>(*a_carry[i], b[i], *a_carry[i + 1]);

  // Cleanup step 3 (not the carry)
  for (int i = n; i-- > 2;)
    x<ctrl>(*a_carry[i - 1], b[i - 1], *a_carry[i]);

  x(*a_carry[1]);
  x<ctrl>(*a_carry[0], b[0], *a_carry[1]);
  x(*a_carry[0], b[0]);

  // Cleanup step 2 (not the carry)
  for (int i = 1; i < n - 1; ++i)
    x<ctrl>(*a_carry[i], *a_carry[i + 1]);

  // Cleanup step 1
  for (int i = 1; i < n; ++i)
    x<ctrl>(*a_carry[i], b[i]);
}

} // namespace cudaq::internal
