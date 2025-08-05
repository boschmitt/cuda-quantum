/*******************************************************************************
 * Copyright (c) 2022 - 2025 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include <complex>
#include <nanobind/stl/complex.h>
#include <nanobind/ndarray.h>
#include <nanobind/operators.h>
#include <nanobind/stl/vector.h>

#include "cudaq/operators.h"
#include "py_helpers.h"
#include "py_super_op.h"

namespace cudaq {

void bindSuperOperatorWrapper(nb::module_ &mod) {
  auto super_op_class = nb::class_<super_op>(mod, "SuperOperator");

  super_op_class
      .def(nb::init<>(), "Creates a default instantiated super-operator. A "
                         "default instantiated "
                         "super-operator means a no action linear map.")
      .def_static(
          "left_multiply",
          static_cast<super_op (*)(const cudaq::product_op<cudaq::matrix_handler> &)>(
              &super_op::left_multiply),
          "Creates a super-operator representing a left "
          "multiplication of the operator to the density matrix.")
      .def_static(
          "right_multiply",
          static_cast<super_op (*)(const cudaq::product_op<cudaq::matrix_handler> &)>(
              &super_op::right_multiply),
          "Creates a super-operator representing a right "
          "multiplication of the operator to the density matrix.")
      .def_static(
          "left_right_multiply",
          static_cast<super_op (*)(const cudaq::product_op<cudaq::matrix_handler> &,
                            const cudaq::product_op<cudaq::matrix_handler> &)>(
              &super_op::left_right_multiply),
          "Creates a super-operator representing a simultaneous left "
          "multiplication of the first operator operand and right "
          "multiplication of the second operator operand to the "
          "density matrix.")

      .def_static(
          "left_multiply",
          static_cast<super_op (*)(const cudaq::sum_op<cudaq::matrix_handler> &)>(
              &super_op::left_multiply),
          "Creates a super-operator representing a left "
          "multiplication of the operator to the density matrix. The sum is "
          "distributed into a linear combination of super-operator actions.")
      .def_static(
          "right_multiply",
          static_cast<super_op (*)(const cudaq::sum_op<cudaq::matrix_handler> &)>(
              &super_op::right_multiply),
          "Creates a super-operator representing a right "
          "multiplication of the operator to the density matrix. The sum is "
          "distributed into a linear combination of super-operator actions.")
      .def_static(
          "left_right_multiply",
          static_cast<super_op (*)(const cudaq::sum_op<cudaq::matrix_handler> &,
                            const cudaq::sum_op<cudaq::matrix_handler> &)>(
              &super_op::left_right_multiply),
          "Creates a super-operator representing a simultaneous left "
          "multiplication of the first operator operand and right "
          "multiplication of the second operator operand to the "
          "density matrix. The sum is distributed into a linear combination of "
          "super-operator actions.")
      .def(
          "__iter__",
          [](super_op &self) {
            nb::list result;
            for (auto it = self.begin(); it != self.end(); ++it) {
              result.append(*it);
            }
            return result;
          },
          "Loop through each term of the super-operator.")
      .def(nb::self += nb::self);
}

} // namespace cudaq
