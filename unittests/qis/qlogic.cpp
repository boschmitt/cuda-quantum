/*******************************************************************************
 * Copyright (c) 2022 - 2024 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "CUDAQTestUtils.h"
#include <cudaq/qis/qarith.h>
#include <cudaq/qis/qlogic.h>

#ifndef CUDAQ_BACKEND_DM

CUDAQ_TEST(QubitLogicTester, checkEqual) {
  for (auto i = 0; i < 4; ++i) {
    cudaq::qvector<2> qs(2);
    cudaq::qubit result;
    cudaq::load(qs, i);

    cudaq::equal(qs[0], qs[1], result);
    ASSERT_EQ(mz(result), (i == 0) || (i == 3));
  }
}

CUDAQ_TEST(QubitLogicTester, checkNotEqual) {
  for (auto i = 0; i < 4; ++i) {
    cudaq::qvector<2> qs(2);
    cudaq::qubit result;
    cudaq::load(qs, i);

    cudaq::not_equal(qs[0], qs[1], result);
    ASSERT_EQ(mz(result), (i != 0) && (i != 3));
  }
}

CUDAQ_TEST(QubitLogicTester, checkAnd) {
  for (auto i = 0; i < 4; ++i) {
    cudaq::qvector<2> qs(2);
    cudaq::qubit result;
    cudaq::load(qs, i);

    cudaq::and_(qs[0], qs[1], result);
    ASSERT_EQ(mz(result), i == 3);
  }
}

CUDAQ_TEST(QubitLogicTester, checkOr) {
  for (auto i = 0; i < 4; ++i) {
    cudaq::qvector<2> qs(2);
    cudaq::qubit result;
    cudaq::load(qs, i);

    cudaq::or_(qs[0], qs[1], result);
    ASSERT_EQ(mz(result), i != 0);
  }
}

#endif
