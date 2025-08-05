/*******************************************************************************
 * Copyright (c) 2022 - 2025 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "py_helpers.h"
#include "cudaq/operators.h"
#include <complex>
#include <nanobind/stl/complex.h>
#include <nanobind/ndarray.h>
#include <nanobind/nanobind.h>
#include <nanobind/stl/string.h>

namespace cudaq::details {

cudaq::parameter_map kwargs_to_param_map(const nb::kwargs &kwargs) {
  cudaq::parameter_map params;
  for (auto [keyPy, valuePy] : kwargs) {
    std::string key = nb::cast<std::string>(keyPy);
    std::complex<double> value = nb::cast<std::complex<double>>(valuePy);
    params.insert(params.end(),
                  std::pair<std::string, std::complex<double>>(key, value));
  }
  return params;
}

std::unordered_map<std::string, std::string>
kwargs_to_param_description(const nb::kwargs &kwargs) {
  std::unordered_map<std::string, std::string> param_desc;
  for (auto [keyPy, valuePy] : kwargs) {
    std::string key = nb::cast<std::string>(keyPy);
    std::string value = nb::cast<std::string>(valuePy);
    param_desc.insert(param_desc.end(),
                      std::pair<std::string, std::string>(key, value));
  }
  return param_desc;
}

nb::ndarray<nb::numpy, std::complex<double>, nb::ndim<2>> cmat_to_numpy(complex_matrix &cmat) {
    std::size_t rows = cmat.rows();
    std::size_t cols = cmat.cols();
    std::complex<double> *data = cmat.get_data(complex_matrix::order::row_major);

    // Memory ownership:
    // If cmat is guaranteed to outlive the ndarray, use nb::handle() as owner.
    // If you want Python to own the data, create a capsule as owner.
    return nb::ndarray<nb::numpy, std::complex<double>, nb::ndim<2>>(data, {rows, cols}, nb::handle());
}

} // namespace cudaq::details
