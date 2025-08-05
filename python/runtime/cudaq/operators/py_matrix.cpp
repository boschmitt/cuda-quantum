/*******************************************************************************
 * Copyright (c) 2022 - 2025 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include <nanobind/stl/complex.h>
#include <nanobind/ndarray.h>
#include <nanobind/operators.h>
#include <nanobind/stl/vector.h>
#include <nanobind/stl/tuple.h>

#include "cudaq/utils/matrix.h"
#include "py_helpers.h"
#include "py_matrix.h"

#include <complex>

namespace cudaq {

/// @brief Extract the array data from a numpy array into our
/// own allocated data pointer.
void extractMatrixData(const nb::ndarray<nb::numpy, const std::complex<double>> &array, std::complex<double> *data) {
  if (array.ndim() != 2)
    throw std::runtime_error("Incompatible buffer shape.");

  auto rows = array.shape(0);
  auto cols = array.shape(1);
  
  // Copy data from numpy array
  memcpy(data, array.data(), sizeof(std::complex<double>) * (rows * cols));
}

void bindComplexMatrix(nb::module_ &mod) {
  nb::class_<complex_matrix>(
      mod, "ComplexMatrix",
      "The :class:`ComplexMatrix` is a thin wrapper around a "
      "matrix of complex<double> elements.")
      /// The following makes this compatible with NumPy via array interface
      .def("__array_interface__", [](complex_matrix &op) -> nb::dict {
        nb::dict array_interface;
        auto rows = op.rows();
        auto cols = op.cols();
        
        // Create shape tuple
        nb::tuple shape = nb::make_tuple(rows, cols);
        array_interface["shape"] = shape;
        
        // Set data type
        array_interface["typestr"] = "D";  // Complex double
        
        // Set data pointer as integer
        array_interface["data"] = nb::make_tuple(
            reinterpret_cast<std::uintptr_t>(op.get_data(complex_matrix::order::row_major)), 
            false  // not read-only
        );
        
        // Set version
        array_interface["version"] = 3;
        
        return array_interface;
      })
      .def("__init__", [](complex_matrix *self, const nb::ndarray<nb::numpy, const std::complex<double>> &array) {
             if (array.ndim() != 2)
               throw std::runtime_error("Array must be 2D");
             
             auto rows = array.shape(0);
             auto cols = array.shape(1);
             new (self) complex_matrix(rows, cols);
             extractMatrixData(array, self->get_data(complex_matrix::order::row_major));
           }, nb::arg("array"),
           "Create a :class:`ComplexMatrix` from a buffer of data, such as a "
           "numpy.ndarray.")
      .def(
          "num_rows", [](complex_matrix &m) { return m.rows(); },
          "Returns the number of rows in the matrix.")
      .def(
          "num_columns", [](complex_matrix &m) { return m.cols(); },
          "Returns the number of columns in the matrix.")
      .def(
          "__getitem__",
          [](complex_matrix &m, std::size_t i, std::size_t j) {
            return m(i, j);
          },
          "Return the matrix element at i, j.")
      .def(
          "__getitem__",
          [](complex_matrix &m, std::tuple<std::size_t, std::size_t> rowCol) {
            return m(std::get<0>(rowCol), std::get<1>(rowCol));
          },
          "Return the matrix element at i, j.")
      .def("minimal_eigenvalue", &complex_matrix::minimal_eigenvalue,
           "Return the lowest eigenvalue for this :class:`ComplexMatrix`.")
      .def(
          "dump", [](const complex_matrix &self) { self.dump(); },
          "Prints the matrix to the standard output.")
      .def(
          "__eq__",
          [](const complex_matrix &lhs, const complex_matrix &rhs) {
            return lhs == rhs;
          })
      .def("__str__", &complex_matrix::to_string,
           "Returns the string representation of the matrix.")
      .def(
          "to_numpy",
          [](complex_matrix &m) { return details::cmat_to_numpy(m); },
          "Convert :class:`ComplexMatrix` to numpy.ndarray.");
}

} // namespace cudaq
