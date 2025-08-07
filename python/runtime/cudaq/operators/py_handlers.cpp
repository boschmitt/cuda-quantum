/*******************************************************************************
 * Copyright (c) 2022 - 2025 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include <complex>
#include <nanobind/stl/complex.h>
#include <nanobind/stl/function.h>
#include <nanobind/ndarray.h>
#include <nanobind/operators.h>
#include <nanobind/stl/vector.h>
#include <nanobind/stl/string.h>

#include "cudaq/operators.h"
#include "py_handlers.h"
#include "py_helpers.h"

namespace cudaq {

void bindPauli(nb::module_ mod) {
  nb::enum_<pauli>(mod, "Pauli",
                   "An enumeration representing the types of Pauli matrices.")
      .value("X", pauli::X)
      .value("Y", pauli::Y)
      .value("Z", pauli::Z)
      .value("I", pauli::I);
}

void bindOperatorHandlers(nb::module_ &mod) {
  using matrix_callback = std::function<complex_matrix(
      const std::vector<int64_t> &, const parameter_map &)>;

  nb::class_<matrix_handler>(mod, "MatrixOperatorElement")
      .def_prop_ro(
          "id",
          [](const matrix_handler &self) { return self.to_string(false); },
          "Returns the id used to define and instantiate the operator.")
      .def_prop_ro("degrees", &matrix_handler::degrees,
                             "Returns a vector that lists all degrees of "
                             "freedom that the operator targets.")
      .def_prop_ro("parameters",
                             &matrix_handler::get_parameter_descriptions,
                             "Returns a dictionary that maps each parameter "
                             "name to its description.")
      .def_prop_ro("expected_dimensions",
                             &matrix_handler::get_expected_dimensions,
                             "The number of levels, that is the dimension, for "
                             "each degree of freedom "
                             "in canonical order that the operator acts on. A "
                             "value of zero or less "
                             "indicates that the operator is defined for any "
                             "dimension of that degree.")
      .def(nb::init<std::size_t>(),
           "Creates an identity operator on the given target.")
      .def("__init__", [](matrix_handler *self, std::string operator_id,
                          std::vector<std::size_t> degrees) {
            new (self) matrix_handler(std::move(operator_id), std::move(degrees));
          },
          nb::arg("id"), nb::arg("degrees"),
          "Creates the matrix operator with the given id acting on the given "
          "degrees of "
          "freedom. Throws a runtime exception if no operator with that id "
          "has been defined.")
      .def(nb::init<const matrix_handler &>(), "Copy constructor.")
      .def("__eq__", &matrix_handler::operator==)
      .def("to_string", &matrix_handler::to_string, nb::arg("include_degrees"),
           "Returns the string representation of the operator.")
      .def(
          "to_matrix",
          [](const matrix_handler &self, dimension_map &dimensions,
             const parameter_map &params) {
            auto cmat = self.to_matrix(dimensions, params);
            return details::cmat_to_numpy(cmat);
          },
          nb::arg("dimensions") = dimension_map(),
          nb::arg("parameters") = parameter_map(),
          "Returns the matrix representation of the operator.")
      .def(
          "to_matrix",
          [](const matrix_handler &self, dimension_map &dimensions,
             const nb::kwargs &kwargs) {
            auto cmat = self.to_matrix(dimensions,
                                       details::kwargs_to_param_map(kwargs));
            return details::cmat_to_numpy(cmat);
          },
          nb::arg("dimensions") = dimension_map(), nb::arg("kwargs"),
          "Returns the matrix representation of the operator.")

      // tools for custom operators
      .def_static(
          "_define",
          [](std::string operator_id, std::vector<int64_t> expected_dimensions,
             const matrix_callback &func, bool overwrite,
             const nb::kwargs &kwargs) {
            // we need to make sure the python function that is stored in
            // the static dictionary containing the operator definitions
            // is properly cleaned up - otherwise python will hang on exit...
            auto atexit = nb::module_::import_("atexit");
            atexit.attr("register")(nb::cpp_function([operator_id]() {
              matrix_handler::remove_definition(operator_id);
            }));
            if (overwrite)
              matrix_handler::remove_definition(operator_id);
            matrix_handler::define(
                std::move(operator_id), std::move(expected_dimensions), func,
                details::kwargs_to_param_description(kwargs));
          },
          nb::arg("operator_id"), nb::arg("expected_dimensions"),
          nb::arg("callback"), nb::arg("overwrite") = false, nb::arg("kwargs"),
          "Defines a matrix operator with the given name and dimensions whose"
          "matrix representation can be obtained by invoking the given "
          "callback function.");

  nb::class_<boson_handler>(mod, "BosonOperatorElement")
      .def_prop_ro(
          "target", &boson_handler::target,
          "Returns the degree of freedom that the operator targets.")
      .def_prop_ro("degrees", &boson_handler::degrees,
                             "Returns a vector that lists all degrees of "
                             "freedom that the operator targets.")
      .def(nb::init<std::size_t>(),
           "Creates an identity operator on the given target.")
      .def(nb::init<const boson_handler &>(), "Copy constructor.")
      .def("__eq__", &boson_handler::operator==)
      .def("to_string", &boson_handler::to_string, nb::arg("include_degrees"),
           "Returns the string representation of the operator.")
      .def(
          "to_matrix",
          [](const boson_handler &self, dimension_map &dimensions,
             const parameter_map &params) {
            auto cmat = self.to_matrix(dimensions, params);
            return details::cmat_to_numpy(cmat);
          },
          nb::arg("dimensions") = dimension_map(),
          nb::arg("parameters") = parameter_map(),
          "Returns the matrix representation of the operator.")
      .def(
          "to_matrix",
          [](const boson_handler &self, dimension_map &dimensions,
             const nb::kwargs &kwargs) {
            auto cmat = self.to_matrix(dimensions,
                                       details::kwargs_to_param_map(kwargs));
            return details::cmat_to_numpy(cmat);
          },
          nb::arg("dimensions") = dimension_map(), nb::arg("kwargs"),
          "Returns the matrix representation of the operator.");

  nb::class_<fermion_handler>(mod, "FermionOperatorElement")
      .def_prop_ro(
          "target", &fermion_handler::target,
          "Returns the degree of freedom that the operator targets.")
      .def_prop_ro("degrees", &fermion_handler::degrees,
                             "Returns a vector that lists all degrees of "
                             "freedom that the operator targets.")
      .def(nb::init<std::size_t>(),
           "Creates an identity operator on the given target.")
      .def(nb::init<const fermion_handler &>(), "Copy constructor.")
      .def("__eq__", &fermion_handler::operator==)
      .def("to_string", &fermion_handler::to_string, nb::arg("include_degrees"),
           "Returns the string representation of the operator.")
      .def(
          "to_matrix",
          [](const fermion_handler &self, dimension_map &dimensions,
             const parameter_map &params) {
            auto cmat = self.to_matrix(dimensions, params);
            return details::cmat_to_numpy(cmat);
          },
          nb::arg("dimensions") = dimension_map(),
          nb::arg("parameters") = parameter_map(),
          "Returns the matrix representation of the operator.")
      .def(
          "to_matrix",
          [](const fermion_handler &self, dimension_map &dimensions,
             const nb::kwargs &kwargs) {
            auto cmat = self.to_matrix(dimensions,
                                       details::kwargs_to_param_map(kwargs));
            return details::cmat_to_numpy(cmat);
          },
          nb::arg("dimensions") = dimension_map(), nb::arg("kwargs"),
          "Returns the matrix representation of the operator.");

  nb::class_<spin_handler>(mod, "SpinOperatorElement")
      .def_prop_ro(
          "target", &spin_handler::target,
          "Returns the degree of freedom that the operator targets.")
      .def_prop_ro("degrees", &spin_handler::degrees,
                             "Returns a vector that lists all degrees of "
                             "freedom that the operator targets.")
      .def(nb::init<std::size_t>(),
           "Creates an identity operator on the given target.")
      .def(nb::init<const spin_handler &>(), "Copy constructor.")
      .def("__eq__", &spin_handler::operator==)
      .def("as_pauli", &spin_handler::as_pauli,
           "Returns the Pauli representation of the operator.")
      .def("to_string", &spin_handler::to_string, nb::arg("include_degrees"),
           "Returns the string representation of the operator.")
      .def(
          "to_matrix",
          [](const spin_handler &self, dimension_map &dimensions,
             const parameter_map &params) {
            auto cmat = self.to_matrix(dimensions, params);
            return details::cmat_to_numpy(cmat);
          },
          nb::arg("dimensions") = dimension_map(),
          nb::arg("parameters") = parameter_map(),
          "Returns the matrix representation of the operator.")
      .def(
          "to_matrix",
          [](const spin_handler &self, dimension_map &dimensions,
             const nb::kwargs &kwargs) {
            auto cmat = self.to_matrix(dimensions,
                                       details::kwargs_to_param_map(kwargs));
            return details::cmat_to_numpy(cmat);
          },
          nb::arg("dimensions") = dimension_map(), nb::arg("kwargs"),
          "Returns the matrix representation of the operator.");
}

void bindHandlersWrapper(nb::module_ &mod) {
  bindPauli(mod);
  bindOperatorHandlers(mod);
}

} // namespace cudaq
