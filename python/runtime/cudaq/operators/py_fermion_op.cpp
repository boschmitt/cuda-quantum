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
#include <nanobind/stl/string.h>
#include <nanobind/stl/set.h>

#include "cudaq/operators.h"
#include "cudaq/operators/serialization.h"
#include "py_fermion_op.h"
#include "py_helpers.h"

namespace cudaq {

void bindFermionModule(nb::module_ &mod) {
  // Binding the functions in `cudaq::fermion` as `_pycudaq` submodule
  // so it's accessible directly in the cudaq namespace.
  auto fermion_submodule = mod.def_submodule("fermion");
  fermion_submodule.def(
      "empty", &fermion_op::empty,
      "Returns sum operator with no terms. Note that a sum with no terms "
      "multiplied by anything still is a sum with no terms.");
  fermion_submodule.def(
      "identity", []() { return fermion_op::identity(); },
      "Returns product operator with constant value 1.");
  fermion_submodule.def(
      "identity",
      [](std::size_t target) { return fermion_op::identity(target); },
      nb::arg("target"),
      "Returns an identity operator on the given target index.");
  fermion_submodule.def(
      "identities",
      [](std::size_t first, std::size_t last) {
        return fermion_op_term(first, last);
      },
      nb::arg("first"), nb::arg("last"),
      "Creates a product operator that applies an identity operation to all "
      "degrees of "
      "freedom in the open range [first, last).");
  fermion_submodule.def(
      "create", &fermion_op::create<fermion_handler>, nb::arg("target"),
      "Returns a fermionic creation operator on the given target index.");
  fermion_submodule.def(
      "annihilate", &fermion_op::annihilate<fermion_handler>, nb::arg("target"),
      "Returns a fermionic annihilation operator on the given target index.");
  fermion_submodule.def(
      "number", &fermion_op::number<fermion_handler>, nb::arg("target"),
      "Returns a fermionic number operator on the given target index.");
  fermion_submodule.def(
      "canonicalized",
      [](const fermion_op_term &orig) {
        return fermion_op_term::canonicalize(orig);
      },
      "Removes all identity operators from the operator.");
  fermion_submodule.def(
      "canonicalized",
      [](const fermion_op_term &orig, const std::set<std::size_t> &degrees) {
        return fermion_op_term::canonicalize(orig, degrees);
      },
      "Expands the operator to act on all given degrees, applying identities "
      "as needed. "
      "The canonicalization will throw a runtime exception if the operator "
      "acts on any degrees "
      "of freedom that are not included in the given set.");
  fermion_submodule.def(
      "canonicalized",
      [](const fermion_op &orig) { return fermion_op::canonicalize(orig); },
      "Removes all identity operators from the operator.");
  fermion_submodule.def(
      "canonicalized",
      [](const fermion_op &orig, const std::set<std::size_t> &degrees) {
        return fermion_op::canonicalize(orig, degrees);
      },
      "Expands the operator to act on all given degrees, applying identities "
      "as needed. "
      "If an empty set is passed, canonicalizes all terms in the sum to act on "
      "the same "
      "degrees of freedom.");
}

void bindFermionOperator(nb::module_ &mod) {

  auto fermion_op_class = nb::class_<fermion_op>(mod, "FermionOperator");
  auto fermion_op_term_class =
      nb::class_<fermion_op_term>(mod, "FermionOperatorTerm");

  fermion_op_class
      .def(
          "__iter__",
          [](fermion_op &self) {
            nb::list items;
            for (auto it = self.begin(); it != self.end(); ++it) {
              items.append(*it);
            }
            return items;
          },
          "Loop through each term of the operator.")

      // properties

      .def_prop_ro("parameters",
                             &fermion_op::get_parameter_descriptions,
                             "Returns a dictionary that maps each parameter "
                             "name to its description.")
      .def_prop_ro("degrees", &fermion_op::degrees,
                             "Returns a vector that lists all degrees of "
                             "freedom that the operator targets. "
                             "The order of degrees is from smallest to largest "
                             "and reflects the ordering of "
                             "the matrix returned by `to_matrix`. "
                             "Specifically, the indices of a statevector "
                             "with two qubits are {00, 01, 10, 11}. An "
                             "ordering of degrees {0, 1} then indicates "
                             "that a state where the qubit with index 0 equals "
                             "1 with probability 1 is given by "
                             "the vector {0., 1., 0., 0.}.")
      .def_prop_ro("min_degree", &fermion_op::min_degree,
                             "Returns the smallest index of the degrees of "
                             "freedom that the operator targets.")
      .def_prop_ro("max_degree", &fermion_op::max_degree,
                             "Returns the smallest index of the degrees of "
                             "freedom that the operator targets.")
      .def_prop_ro("term_count", &fermion_op::num_terms,
                             "Returns the number of terms in the operator.")

      // constructors

      .def(nb::init<>(),
           "Creates a default instantiated sum. A default instantiated "
           "sum has no value; it will take a value the first time an "
           "arithmetic operation "
           "is applied to it. In that sense, it acts as both the additive and "
           "multiplicative "
           "identity. To construct a `0` value in the mathematical sense "
           "(neutral element "
           "for addition), use `empty()` instead.")
      .def(nb::init<std::size_t>(),
           "Creates a sum operator with no terms, reserving "
           "space for the given number of terms.")
      .def(nb::init<const fermion_op_term &>(),
           "Creates a sum operator with the given term.")
      .def(nb::init<const fermion_op &>(), "Copy constructor.")
      .def(
          "copy", [](const fermion_op &self) { return fermion_op(self); },
          "Creates a copy of the operator.")

      // evaluations

      .def(
          "to_matrix",
          [](const fermion_op &self, dimension_map &dimensions,
             const parameter_map &params, bool invert_order) {
            auto cmat = self.to_matrix(dimensions, params, invert_order);
            return details::cmat_to_numpy(cmat);
          },
          nb::arg("dimensions") = dimension_map(),
          nb::arg("parameters") = parameter_map(),
          nb::arg("invert_order") = false,
          "Returns the matrix representation of the operator."
          "The matrix is ordered according to the convention (endianness) "
          "used in CUDA-Q, and the ordering returned by `degrees`. This order "
          "can be inverted by setting the optional `invert_order` argument to "
          "`True`. "
          "See also the documentation for `degrees` for more detail.")
      .def(
          "to_matrix",
          [](const fermion_op &self, dimension_map &dimensions,
             bool invert_order, const nb::kwargs &kwargs) {
            auto cmat = self.to_matrix(
                dimensions, details::kwargs_to_param_map(kwargs), invert_order);
            return details::cmat_to_numpy(cmat);
          },
          nb::arg("dimensions") = dimension_map(),
          nb::arg("invert_order") = false, nb::arg("kwargs"),
          "Returns the matrix representation of the operator."
          "The matrix is ordered according to the convention (endianness) "
          "used in CUDA-Q, and the ordering returned by `degrees`. This order "
          "can be inverted by setting the optional `invert_order` argument to "
          "`True`. "
          "See also the documentation for `degrees` for more detail.")
      .def(
          "to_sparse_matrix",
          [](const fermion_op &self, dimension_map &dimensions,
             const parameter_map &params, bool invert_order) {
            return self.to_sparse_matrix(dimensions, params, invert_order);
          },
          nb::arg("dimensions") = dimension_map(),
          nb::arg("parameters") = parameter_map(),
          nb::arg("invert_order") = false,
          "Return the sparse matrix representation of the operator. This "
          "representation is a "
          "`Tuple[list[complex], list[int], list[int]]`, encoding the "
          "non-zero values, rows, and columns of the matrix. "
          "This format is supported by `scipy.sparse.csr_array`."
          "The matrix is ordered according to the convention (endianness) "
          "used in CUDA-Q, and the ordering returned by `degrees`. This order "
          "can be inverted by setting the optional `invert_order` argument to "
          "`True`. "
          "See also the documentation for `degrees` for more detail.")
      .def(
          "to_sparse_matrix",
          [](const fermion_op &self, dimension_map &dimensions,
             bool invert_order, const nb::kwargs &kwargs) {
            return self.to_sparse_matrix(
                dimensions, details::kwargs_to_param_map(kwargs), invert_order);
          },
          nb::arg("dimensions") = dimension_map(),
          nb::arg("invert_order") = false, nb::arg("kwargs"),
          "Return the sparse matrix representation of the operator. This "
          "representation is a "
          "`Tuple[list[complex], list[int], list[int]]`, encoding the "
          "non-zero values, rows, and columns of the matrix. "
          "This format is supported by `scipy.sparse.csr_array`."
          "The matrix is ordered according to the convention (endianness) "
          "used in CUDA-Q, and the ordering returned by `degrees`. This order "
          "can be inverted by setting the optional `invert_order` argument to "
          "`True`. "
          "See also the documentation for `degrees` for more detail.")

      // comparisons

      .def("__eq__", &fermion_op::operator==,
           "Return true if the two operators are equivalent. The equivalence "
           "check takes "
           "commutation relations into account. Operators acting on different "
           "degrees of "
           "freedom are never equivalent, even if they only differ by an "
           "identity operator.")
      .def(
          "__eq__",
          [](const fermion_op &self, const fermion_op_term &other) {
            return self.num_terms() == 1 && *self.begin() == other;
          },
          "Return true if the two operators are equivalent.")

      // unary operators

      .def(-nb::self)
      .def(+nb::self)

      // in-place arithmetics

      .def(nb::self /= int())
      .def(nb::self *= int())
      .def(nb::self += int())
      .def(nb::self -= int())
      .def(nb::self /= double())
      .def(nb::self *= double())
      .def(nb::self += double())
      .def(nb::self -= double())
      .def(nb::self /= std::complex<double>())
      .def(nb::self *= std::complex<double>())
      .def(nb::self += std::complex<double>())
      .def(nb::self -= std::complex<double>())
      .def(nb::self /= scalar_operator())
      .def(nb::self *= scalar_operator())
      .def(nb::self += scalar_operator())
      .def(nb::self -= scalar_operator())
      .def(nb::self *= fermion_op_term())
      .def(nb::self += fermion_op_term())
      .def(nb::self -= fermion_op_term())
      .def(nb::self *= nb::self)
      .def(nb::self += nb::self)
// see issue https://github.com/pybind/pybind11/issues/1893
#ifdef __clang__
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wself-assign-overloaded"
#endif
      .def(nb::self -= nb::self)
#ifdef __clang__
#pragma clang diagnostic pop
#endif

      // right-hand arithmetics

      .def(nb::self / int())
      .def(nb::self * int())
      .def(nb::self + int())
      .def(nb::self - int())
      .def(nb::self / double())
      .def(nb::self * double())
      .def(nb::self + double())
      .def(nb::self - double())
      .def(nb::self / std::complex<double>())
      .def(nb::self * std::complex<double>())
      .def(nb::self + std::complex<double>())
      .def(nb::self - std::complex<double>())
      .def(nb::self / scalar_operator())
      .def(nb::self * scalar_operator())
      .def(nb::self + scalar_operator())
      .def(nb::self - scalar_operator())
      .def(nb::self * fermion_op_term())
      .def(nb::self + fermion_op_term())
      .def(nb::self - fermion_op_term())
      .def(nb::self * nb::self)
      .def(nb::self + nb::self)
      .def(nb::self - nb::self)
      .def(nb::self * matrix_op_term())
      .def(nb::self + matrix_op_term())
      .def(nb::self - matrix_op_term())
      .def(nb::self * matrix_op())
      .def(nb::self + matrix_op())
      .def(nb::self - matrix_op())

      // left-hand arithmetics

      .def(int() * nb::self)
      .def(int() + nb::self)
      .def(int() - nb::self)
      .def(double() * nb::self)
      .def(double() + nb::self)
      .def(double() - nb::self)
      .def(std::complex<double>() * nb::self)
      .def(std::complex<double>() + nb::self)
      .def(std::complex<double>() - nb::self)
      .def(scalar_operator() * nb::self)
      .def(scalar_operator() + nb::self)
      .def(scalar_operator() - nb::self)

      // common operators

      .def_static("empty", &fermion_op::empty,
                  "Creates a sum operator with no terms. And empty sum is the "
                  "neutral element for addition; "
                  "multiplying an empty sum with anything will still result in "
                  "an empty sum.")
      .def_static(
          "identity", []() { return fermion_op::identity(); },
          "Creates a product operator with constant value 1. The identity "
          "operator is the neutral "
          "element for multiplication.")
      .def_static(
          "identity",
          [](std::size_t target) { return fermion_op::identity(target); },
          "Creates a product operator that applies the identity to the given "
          "target index.")

      // general utility functions

      .def(
          "__str__", [](const fermion_op &self) { return self.to_string(); },
          "Returns the string representation of the operator.")
      .def("dump", &fermion_op::dump,
           "Prints the string representation of the operator to the standard "
           "output.")
      .def("trim", &fermion_op::trim, nb::arg("tol") = 0.0,
           nb::arg("parameters") = parameter_map(),
           "Removes all terms from the sum for which the absolute value of the "
           "coefficient is below "
           "the given tolerance.")
      .def(
          "trim",
          [](fermion_op &self, double tol, const nb::kwargs &kwargs) {
            return self.trim(tol, details::kwargs_to_param_map(kwargs));
          },
          nb::arg("tol") = 0.0, nb::arg("kwargs"),
          "Removes all terms from the sum for which the absolute value of the "
          "coefficient is below "
          "the given tolerance.")
      .def(
          "canonicalize", [](fermion_op &self) { return self.canonicalize(); },
          "Removes all identity operators from the operator.")
      .def(
          "canonicalize",
          [](fermion_op &self, const std::set<std::size_t> &degrees) {
            return self.canonicalize(degrees);
          },
          "Expands the operator to act on all given degrees, applying "
          "identities as needed. "
          "If an empty set is passed, canonicalizes all terms in the sum to "
          "act on the same "
          "degrees of freedom.")
      .def("distribute_terms", &fermion_op::distribute_terms,
           "Partitions the terms of the sums into the given number of separate "
           "sums.");

  fermion_op_term_class
      .def(
          "__iter__",
          [](fermion_op_term &self) {
            nb::list items;
            for (auto it = self.begin(); it != self.end(); ++it) {
              items.append(*it);
            }
            return items;
          },
          "Loop through each term of the operator.")

      // properties

      .def_prop_ro("parameters",
                             &fermion_op_term::get_parameter_descriptions,
                             "Returns a dictionary that maps each parameter "
                             "name to its description.")
      .def_prop_ro("degrees", &fermion_op_term::degrees,
                             "Returns a vector that lists all degrees of "
                             "freedom that the operator targets. "
                             "The order of degrees is from smallest to largest "
                             "and reflects the ordering of "
                             "the matrix returned by `to_matrix`. "
                             "Specifically, the indices of a statevector "
                             "with two qubits are {00, 01, 10, 11}. An "
                             "ordering of degrees {0, 1} then indicates "
                             "that a state where the qubit with index 0 equals "
                             "1 with probability 1 is given by "
                             "the vector {0., 1., 0., 0.}.")
      .def_prop_ro("min_degree", &fermion_op_term::min_degree,
                             "Returns the smallest index of the degrees of "
                             "freedom that the operator targets.")
      .def_prop_ro("max_degree", &fermion_op_term::max_degree,
                             "Returns the smallest index of the degrees of "
                             "freedom that the operator targets.")
      .def_prop_ro("ops_count", &fermion_op_term::num_ops,
                             "Returns the number of operators in the product.")
      .def_prop_ro(
          "term_id", &fermion_op_term::get_term_id,
          "The term id uniquely identifies the operators and targets (degrees) "
          "that they act on, "
          "but does not include information about the coefficient.")
      .def_prop_ro(
          "coefficient", &fermion_op_term::get_coefficient,
          "Returns the unevaluated coefficient of the operator. The "
          "coefficient is a "
          "callback function that can be invoked with the `evaluate` method.")

      // constructors

      .def(nb::init<>(),
           "Creates a product operator with constant value 1. The returned "
           "operator does not target any degrees of freedom but merely "
           "represents a constant.")
      .def(nb::init<std::size_t, std::size_t>(), nb::arg("first_degree"),
           nb::arg("last_degree"),
           "Creates a product operator that applies an identity operation to "
           "all degrees of "
           "freedom in the range [first_degree, last_degree).")
      .def(nb::init<double>(),
           "Creates a product operator with the given constant value. "
           "The returned operator does not target any degrees of freedom.")
      .def(nb::init<std::complex<double>>(),
           "Creates a product operator with the given "
           "constant value. The returned operator does not target any degrees "
           "of freedom.")
      .def("__init__", [](fermion_op_term *self, const scalar_operator &scalar) {
        new (self) fermion_op_term(fermion_op_term() * scalar)
        ;
      }, "Creates a product operator with non-constant scalar value.")
      .def(nb::init<fermion_handler>(),
           "Creates a product operator with the given elementary operator.")
      .def(nb::init<const fermion_op_term &, std::size_t>(),
           nb::arg("operator"), nb::arg("size") = 0,
           "Creates a copy of the given operator and reserves space for "
           "storing the given "
           "number of product terms (if a size is provided).")
      .def(
          "copy",
          [](const fermion_op_term &self) { return fermion_op_term(self); },
          "Creates a copy of the operator.")

      // evaluations

      .def("evaluate_coefficient", &fermion_op_term::evaluate_coefficient,
           nb::arg("parameters") = parameter_map(),
           "Returns the evaluated coefficient of the product operator. The "
           "parameters is a map of parameter names to their concrete, complex "
           "values.")
      .def(
          "to_matrix",
          [](const fermion_op_term &self, dimension_map &dimensions,
             const parameter_map &params, bool invert_order) {
            auto cmat = self.to_matrix(dimensions, params, invert_order);
            return details::cmat_to_numpy(cmat);
          },
          nb::arg("dimensions") = dimension_map(),
          nb::arg("parameters") = parameter_map(),
          nb::arg("invert_order") = false,
          "Returns the matrix representation of the operator."
          "The matrix is ordered according to the convention (endianness) "
          "used in CUDA-Q, and the ordering returned by `degrees`. This order "
          "can be inverted by setting the optional `invert_order` argument to "
          "`True`. "
          "See also the documentation for `degrees` for more detail.")
      .def(
          "to_matrix",
          [](const fermion_op_term &self, dimension_map &dimensions,
             bool invert_order, const nb::kwargs &kwargs) {
            auto cmat = self.to_matrix(
                dimensions, details::kwargs_to_param_map(kwargs), invert_order);
            return details::cmat_to_numpy(cmat);
          },
          nb::arg("dimensions") = dimension_map(),
          nb::arg("invert_order") = false, nb::arg("kwargs"),
          "Returns the matrix representation of the operator."
          "The matrix is ordered according to the convention (endianness) "
          "used in CUDA-Q, and the ordering returned by `degrees`. This order "
          "can be inverted by setting the optional `invert_order` argument to "
          "`True`. "
          "See also the documentation for `degrees` for more detail.")
      .def(
          "to_sparse_matrix",
          [](const fermion_op_term &self, dimension_map &dimensions,
             const parameter_map &params, bool invert_order) {
            return self.to_sparse_matrix(dimensions, params, invert_order);
          },
          nb::arg("dimensions") = dimension_map(),
          nb::arg("parameters") = parameter_map(),
          nb::arg("invert_order") = false,
          "Return the sparse matrix representation of the operator. This "
          "representation is a "
          "`Tuple[list[complex], list[int], list[int]]`, encoding the "
          "non-zero values, rows, and columns of the matrix. "
          "This format is supported by `scipy.sparse.csr_array`."
          "The matrix is ordered according to the convention (endianness) "
          "used in CUDA-Q, and the ordering returned by `degrees`. This order "
          "can be inverted by setting the optional `invert_order` argument to "
          "`True`. "
          "See also the documentation for `degrees` for more detail.")
      .def(
          "to_sparse_matrix",
          [](const fermion_op_term &self, dimension_map &dimensions,
             bool invert_order, const nb::kwargs &kwargs) {
            return self.to_sparse_matrix(
                dimensions, details::kwargs_to_param_map(kwargs), invert_order);
          },
          nb::arg("dimensions") = dimension_map(),
          nb::arg("invert_order") = false, nb::arg("kwargs"),
          "Return the sparse matrix representation of the operator. This "
          "representation is a "
          "`Tuple[list[complex], list[int], list[int]]`, encoding the "
          "non-zero values, rows, and columns of the matrix. "
          "This format is supported by `scipy.sparse.csr_array`."
          "The matrix is ordered according to the convention (endianness) "
          "used in CUDA-Q, and the ordering returned by `degrees`. This order "
          "can be inverted by setting the optional `invert_order` argument to "
          "`True`. "
          "See also the documentation for `degrees` for more detail.")

      // comparisons

      .def("__eq__", &fermion_op_term::operator==,
           "Return true if the two operators are equivalent. The equivalence "
           "check takes "
           "commutation relations into account. Operators acting on different "
           "degrees of "
           "freedom are never equivalent, even if they only differ by an "
           "identity operator.")
      .def(
          "__eq__",
          [](const fermion_op_term &self, const fermion_op &other) {
            return other.num_terms() == 1 && *other.begin() == self;
          },
          "Return true if the two operators are equivalent.")

      // unary operators

      .def(-nb::self)
      .def(+nb::self)

      // in-place arithmetics

      .def(nb::self /= int())
      .def(nb::self *= int())
      .def(nb::self /= double())
      .def(nb::self *= double())
      .def(nb::self /= std::complex<double>())
      .def(nb::self *= std::complex<double>())
      .def(nb::self /= scalar_operator())
      .def(nb::self *= scalar_operator())
      .def(nb::self *= nb::self)

      // right-hand arithmetics

      .def(nb::self / int())
      .def(nb::self * int())
      .def(nb::self + int())
      .def(nb::self - int())
      .def(nb::self / double())
      .def(nb::self * double())
      .def(nb::self + double())
      .def(nb::self - double())
      .def(nb::self / std::complex<double>())
      .def(nb::self * std::complex<double>())
      .def(nb::self + std::complex<double>())
      .def(nb::self - std::complex<double>())
      .def(nb::self / scalar_operator())
      .def(nb::self * scalar_operator())
      .def(nb::self + scalar_operator())
      .def(nb::self - scalar_operator())
      .def(nb::self * nb::self)
      .def(nb::self + nb::self)
      .def(nb::self - nb::self)
      .def(nb::self * fermion_op())
      .def(nb::self + fermion_op())
      .def(nb::self - fermion_op())
      .def(nb::self * matrix_op_term())
      .def(nb::self + matrix_op_term())
      .def(nb::self - matrix_op_term())
      .def(nb::self * matrix_op())
      .def(nb::self + matrix_op())
      .def(nb::self - matrix_op())

      // left-hand arithmetics

      .def(int() * nb::self)
      .def(int() + nb::self)
      .def(int() - nb::self)
      .def(double() * nb::self)
      .def(double() + nb::self)
      .def(double() - nb::self)
      .def(std::complex<double>() * nb::self)
      .def(std::complex<double>() + nb::self)
      .def(std::complex<double>() - nb::self)
      .def(scalar_operator() * nb::self)
      .def(scalar_operator() + nb::self)
      .def(scalar_operator() - nb::self)

      // general utility functions

      .def("is_identity", &fermion_op_term::is_identity,
           "Checks if all operators in the product are the identity. "
           "Note: this function returns true regardless of the value of the "
           "coefficient.")
      .def(
          "__str__",
          [](const fermion_op_term &self) { return self.to_string(); },
          "Returns the string representation of the operator.")
      .def("dump", &fermion_op_term::dump,
           "Prints the string representation of the operator to the standard "
           "output.")
      .def(
          "canonicalize",
          [](fermion_op_term &self) { return self.canonicalize(); },
          "Removes all identity operators from the operator.")
      .def(
          "canonicalize",
          [](fermion_op_term &self, const std::set<std::size_t> &degrees) {
            return self.canonicalize(degrees);
          },
          "Expands the operator to act on all given degrees, applying "
          "identities as needed. "
          "The canonicalization will throw a runtime exception if the operator "
          "acts on any degrees "
          "of freedom that are not included in the given set.");
}

void bindFermionWrapper(nb::module_ &mod) {
  bindFermionOperator(mod);
  nb::implicitly_convertible<double, fermion_op_term>();
  nb::implicitly_convertible<std::complex<double>, fermion_op_term>();
  nb::implicitly_convertible<scalar_operator, fermion_op_term>();
  nb::implicitly_convertible<fermion_op_term, fermion_op>();
  bindFermionModule(mod);
}

} // namespace cudaq
