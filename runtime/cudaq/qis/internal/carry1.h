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
  //assert(a.size() == b.size());
  int n = a.size();

  // Step 1
  for (int i = 1; i < n; ++i)
    x<ctrl>(a[i], b[i]);

  // Step 2
  x<ctrl>(a.back(), carry);
  for (int i = n - 1; i-- > 1;)
    x<ctrl>(a[i], a[i + 1]);

  // Step 3
  // Here we simulate the existance of a carry-in that is set to |1>
  x(a[0], b[0]);
  x<ctrl>(a[0], b[0], a[1]);
  x(a[1]);

  for (int i = 1; i < n - 1; ++i)
    x<ctrl>(a[i], b[i], a[i + 1]);
  x<ctrl>(a.back(), b.back(), carry);

  // Cleanup step 3 (not the carry)
  for (int i = n; i-- > 2;)
    x<ctrl>(a[i - 1], b[i - 1], a[i]);

  x(a[1]);
  x<ctrl>(a[0], b[0], a[1]);
  x(a[0], b[0]);

  // Cleanup step 2 (not the carry)
  for (int i = 1; i < n - 1; ++i)
    x<ctrl>(a[i], a[i + 1]);

  // Cleanup step 1
  for (int i = 1; i < n; ++i)
    x<ctrl>(a[i], b[i]);
}

} // namespace cudaq::internal
