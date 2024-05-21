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

namespace cudaq {

inline void __qpu__ equal(qubit &a, qubit &b, qubit &result) {
  x<ctrl>(a, b, result);

  x(a);
  x(b);
  x<ctrl>(a, b, result);
  x(a);
  x(b);
}

inline void __qpu__ not_equal(qubit &a, qubit &b, qubit &result) {
  x(a);
  x<ctrl>(a, b, result);
  x(a);

  x(b);
  x<ctrl>(a, b, result);
  x(b);
}

inline void __qpu__ and_(qubit &a, qubit &b, qubit &result) {
  x<ctrl>(a, b, result);
}

inline void __qpu__ or_(qubit &a, qubit &b, qubit &result) {
  x(a, b); // Broadcast
  x<ctrl>(a, b, result);
  x(a, b); // Broadcast
  x(result);
}

} // namespace cudaq
