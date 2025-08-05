/*******************************************************************************
 * Copyright (c) 2022 - 2025 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/
#include "py_evolve.h"
#include "LinkedLibraryHolder.h"
#include "common/ArgumentWrapper.h"
#include "common/Logger.h"
#include "cudaq/algorithms/evolve_internal.h"
#include "runtime/cudaq/platform/py_alt_launch_kernel.h"
#include "utils/OpaqueArguments.h"
#include "mlir/CAPI/IR.h"
#include <nanobind/stl/complex.h>
#include <nanobind/stl/function.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/vector.h>
#include <nanobind/stl/map.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/optional.h>

namespace cudaq {

template <typename numeric_type>
using spin_op_creator =
    std::function<spin_op(std::map<std::string, numeric_type>)>;

template <typename numeric_type>
evolve_result
pyEvolve(state initial_state, nb::object kernel,
         std::map<std::string, numeric_type> params,
         std::vector<spin_op_creator<numeric_type>> observables = {},
         int shots_count = -1) {
  if (nb::hasattr(kernel, "compile"))
    kernel.attr("compile")();

  auto kernelName = nb::cast<std::string>(kernel.attr("name"));
  auto kernelMod = nb::cast<MlirModule>(kernel.attr("module"));

  std::vector<spin_op> spin_ops = {};
  for (auto &observable : observables) {
    spin_ops.push_back(observable(params));
  }

  auto res = __internal__::evolve(
      initial_state,
      [kernelMod, kernelName](state state) mutable {
        auto *argData = new cudaq::OpaqueArguments();
        valueArgument(*argData, &state);
        pyAltLaunchKernel(kernelName, kernelMod, *argData, {});
        delete argData;
      },
      spin_ops, shots_count);
  return res;
}

template <typename numeric_type>
evolve_result
pyEvolve(state initial_state, std::vector<nb::object> kernels,
         std::vector<std::map<std::string, numeric_type>> params,
         std::vector<spin_op_creator<numeric_type>> observables = {},
         int shots_count = -1, bool save_intermediate_states = true) {
  std::vector<std::function<void(state)>> launchFcts = {};
  for (nb::object kernel : kernels) {
    if (nb::hasattr(kernel, "compile"))
      kernel.attr("compile")();

    auto kernelName = nb::cast<std::string>(kernel.attr("name"));
    auto kernelMod = nb::cast<MlirModule>(kernel.attr("module"));

    launchFcts.push_back([kernelMod, kernelName](state state) mutable {
      auto *argData = new cudaq::OpaqueArguments();
      valueArgument(*argData, &state);
      pyAltLaunchKernel(kernelName, kernelMod, *argData, {});
      delete argData;
    });
  }

  std::vector<std::vector<spin_op>> spin_ops = {};
  for (auto parameters : params) {
    std::vector<spin_op> ops = {};
    for (auto &observable : observables) {
      ops.push_back(observable(parameters));
    }
    spin_ops.push_back(std::move(ops));
  }

  return __internal__::evolve(initial_state, launchFcts, spin_ops, shots_count,
                              save_intermediate_states);
}

template <typename numeric_type>
async_evolve_result
pyEvolveAsync(state initial_state, nb::object kernel,
              std::map<std::string, numeric_type> params,
              std::vector<spin_op_creator<numeric_type>> observables = {},
              std::size_t qpu_id = 0,
              std::optional<cudaq::noise_model> noise_model = std::nullopt,
              int shots_count = -1) {
  if (nb::hasattr(kernel, "compile"))
    kernel.attr("compile")();

  auto kernelMod =
      wrap(unwrap(nb::cast<MlirModule>(kernel.attr("module"))).clone());
  auto kernelName = nb::cast<std::string>(kernel.attr("name"));

  std::vector<spin_op> spin_ops = {};
  for (auto observable : observables) {
    spin_ops.push_back(observable(params));
  }

  nb::gil_scoped_release release;
  return __internal__::evolve_async(
      initial_state,
      [kernelMod, kernelName](state state) mutable {
        auto *argData = new cudaq::OpaqueArguments();
        valueArgument(*argData, &state);
        pyAltLaunchKernel(kernelName, kernelMod, *argData, {});
        delete argData;
      },
      spin_ops, qpu_id, noise_model, shots_count);
}

template <typename numeric_type>
async_evolve_result
pyEvolveAsync(state initial_state, std::vector<nb::object> kernels,
              std::vector<std::map<std::string, numeric_type>> params,
              std::vector<spin_op_creator<numeric_type>> observables = {},
              std::size_t qpu_id = 0,
              std::optional<cudaq::noise_model> noise_model = std::nullopt,
              int shots_count = -1, bool save_intermediate_states = true) {
  std::vector<std::function<void(state)>> launchFcts = {};
  for (nb::object kernel : kernels) {
    if (nb::hasattr(kernel, "compile"))
      kernel.attr("compile")();

    // IMPORTANT: we need to make sure no Python data is accessed in the async.
    // functor.
    auto kernelMod =
        wrap(unwrap(nb::cast<MlirModule>(kernel.attr("module"))).clone());
    auto kernelName = nb::cast<std::string>(kernel.attr("name"));
    launchFcts.push_back(
        [kernelMod = std::move(kernelMod), kernelName](state state) mutable {
          cudaq::OpaqueArguments argData;
          valueArgument(argData, &state);
          pyAltLaunchKernel(kernelName, kernelMod, argData, {});
        });
  }

  std::vector<std::vector<spin_op>> spin_ops = {};
  for (auto parameters : params) {
    std::vector<spin_op> ops = {};
    for (auto observable : observables) {
      ops.push_back(observable(parameters));
    }
    spin_ops.push_back(std::move(ops));
  }

  nb::gil_scoped_release release;
  return __internal__::evolve_async(initial_state, launchFcts, spin_ops, qpu_id,
                                    noise_model, shots_count,
                                    save_intermediate_states);
}

/// @brief Bind the get_state cudaq function
void bindPyEvolve(nb::module_ &mod) {

  // Note: vector versions need to be first, otherwise the incorrect
  // overload is used.
  mod.def(
      "evolve",
      [](state initial_state, std::vector<nb::object> kernels,
         bool save_intermediate_states = true) {
        return pyEvolve<long>(initial_state, kernels, {}, {}, -1,
                              save_intermediate_states);
      },
      "");
  mod.def(
      "evolve",
      [](state initial_state, std::vector<nb::object> kernels,
         std::vector<std::map<std::string, long>> params,
         std::vector<spin_op_creator<long>> observables, int shots_count = -1,
         bool save_intermediate_states = true) {
        return pyEvolve(initial_state, kernels, params, observables,
                        shots_count, save_intermediate_states);
      },
      "");
  mod.def(
      "evolve",
      [](state initial_state, std::vector<nb::object> kernels,
         std::vector<std::map<std::string, double>> params,
         std::vector<spin_op_creator<double>> observables, int shots_count = -1,
         bool save_intermediate_states = true) {
        return pyEvolve(initial_state, kernels, params, observables,
                        shots_count, save_intermediate_states);
      },
      "");
  mod.def(
      "evolve",
      [](state initial_state, std::vector<nb::object> kernels,
         std::vector<std::map<std::string, std::complex<double>>> params,
         std::vector<spin_op_creator<std::complex<double>>> observables,
         int shots_count = -1, bool save_intermediate_states = true) {
        return pyEvolve(initial_state, kernels, params, observables,
                        shots_count, save_intermediate_states);
      },
      "");
  mod.def(
      "evolve",
      [](state initial_state, nb::object kernel) {
        return pyEvolve(initial_state, kernel, std::map<std::string, long>{});
      },
      "");
  mod.def(
      "evolve",
      [](state initial_state, nb::object kernel,
         std::map<std::string, long> params,
         std::vector<spin_op_creator<long>> observables, int shots_count = -1) {
        return pyEvolve(initial_state, kernel, params, observables,
                        shots_count);
      },
      "");
  mod.def(
      "evolve",
      [](state initial_state, nb::object kernel,
         std::map<std::string, double> params,
         std::vector<spin_op_creator<double>> observables,
         int shots_count = -1) {
        return pyEvolve(initial_state, kernel, params, observables,
                        shots_count);
      },
      "");
  mod.def(
      "evolve",
      [](state initial_state, nb::object kernel,
         std::map<std::string, std::complex<double>> params,
         std::vector<spin_op_creator<std::complex<double>>> observables,
         int shots_count = -1) {
        return pyEvolve(initial_state, kernel, params, observables,
                        shots_count);
      },
      "");

  // Note: vector versions need to be first, otherwise the incorrect
  // overload is used.
  mod.def(
      "evolve_async",
      [](state initial_state, std::vector<nb::object> kernels,
         std::size_t qpu_id,
         std::optional<cudaq::noise_model> noise_model = std::nullopt,
         bool save_intermediate_states = true) {
        return pyEvolveAsync<long>(initial_state, kernels, {}, {}, qpu_id,
                                   noise_model, -1, save_intermediate_states);
      },
      nb::arg("initial_state"), nb::arg("kernels"), nb::arg("qpu_id") = 0,
      nb::arg("noise_model") = std::nullopt, nb::kw_only(),
      nb::arg("save_intermediate_states") = true, "");
  mod.def(
      "evolve_async",
      [](state initial_state, std::vector<nb::object> kernels,
         std::vector<std::map<std::string, long>> params,
         std::vector<spin_op_creator<long>> observables, std::size_t qpu_id,
         std::optional<cudaq::noise_model> noise_model = std::nullopt,
         int shots_count = -1, bool save_intermediate_states = true) {
        return pyEvolveAsync(initial_state, kernels, params, observables,
                             qpu_id, noise_model, shots_count,
                             save_intermediate_states);
      },
      nb::arg("initial_state"), nb::arg("kernels"), nb::arg("params"),
      nb::arg("observables"), nb::arg("qpu_id") = 0,
      nb::arg("noise_model") = std::nullopt, nb::arg("shots_count") = -1,
      nb::kw_only(), nb::arg("save_intermediate_states") = true, "");
  mod.def(
      "evolve_async",
      [](state initial_state, std::vector<nb::object> kernels,
         std::vector<std::map<std::string, double>> params,
         std::vector<spin_op_creator<double>> observables, std::size_t qpu_id,
         std::optional<cudaq::noise_model> noise_model = std::nullopt,
         int shots_count = -1, bool save_intermediate_states = true) {
        return pyEvolveAsync(initial_state, kernels, params, observables,
                             qpu_id, noise_model, shots_count,
                             save_intermediate_states);
      },
      nb::arg("initial_state"), nb::arg("kernels"), nb::arg("params"),
      nb::arg("observables"), nb::arg("qpu_id") = 0,
      nb::arg("noise_model") = std::nullopt, nb::arg("shots_count") = -1,
      nb::kw_only(), nb::arg("save_intermediate_states") = true, "");
  mod.def(
      "evolve_async",
      [](state initial_state, std::vector<nb::object> kernels,
         std::vector<std::map<std::string, std::complex<double>>> params,
         std::vector<spin_op_creator<std::complex<double>>> observables,
         std::size_t qpu_id,
         std::optional<cudaq::noise_model> noise_model = std::nullopt,
         int shots_count = -1, bool save_intermediate_states = true) {
        return pyEvolveAsync(initial_state, kernels, params, observables,
                             qpu_id, noise_model, shots_count,
                             save_intermediate_states);
      },
      nb::arg("initial_state"), nb::arg("kernels"), nb::arg("params"),
      nb::arg("observables"), nb::arg("qpu_id") = 0,
      nb::arg("noise_model") = std::nullopt, nb::arg("shots_count") = -1,
      nb::kw_only(), nb::arg("save_intermediate_states") = true, "");
  mod.def(
      "evolve_async",
      [](state initial_state, nb::object kernel, std::size_t qpu_id,
         std::optional<cudaq::noise_model> noise_model = std::nullopt) {
        return pyEvolveAsync(initial_state, kernel,
                             std::map<std::string, long>{}, {}, qpu_id,
                             noise_model);
      },
      nb::arg("initial_state"), nb::arg("kernel"), nb::arg("qpu_id") = 0,
      nb::arg("noise_model") = std::nullopt, "");
  mod.def(
      "evolve_async",
      [](state initial_state, nb::object kernel,
         std::map<std::string, long> params,
         std::vector<spin_op_creator<long>> observables, std::size_t qpu_id,
         std::optional<cudaq::noise_model> noise_model = std::nullopt,
         int shots_count = -1) {
        return pyEvolveAsync(initial_state, kernel, params, observables, qpu_id,
                             noise_model, shots_count);
      },
      nb::arg("initial_state"), nb::arg("kernel"), nb::arg("params"),
      nb::arg("observables"), nb::arg("qpu_id") = 0,
      nb::arg("noise_model") = std::nullopt, nb::arg("shots_count") = -1, "");
  mod.def(
      "evolve_async",
      [](state initial_state, nb::object kernel,
         std::map<std::string, double> params,
         std::vector<spin_op_creator<double>> observables, std::size_t qpu_id,
         std::optional<cudaq::noise_model> noise_model = std::nullopt,
         int shots_count = -1) {
        return pyEvolveAsync(initial_state, kernel, params, observables, qpu_id,
                             noise_model, shots_count);
      },
      nb::arg("initial_state"), nb::arg("kernel"), nb::arg("params"),
      nb::arg("observables"), nb::arg("qpu_id") = 0,
      nb::arg("noise_model") = std::nullopt, nb::arg("shots_count") = -1, "");
  mod.def(
      "evolve_async",
      [](state initial_state, nb::object kernel,
         std::map<std::string, std::complex<double>> params,
         std::vector<spin_op_creator<std::complex<double>>> observables,
         std::size_t qpu_id,
         std::optional<cudaq::noise_model> noise_model = std::nullopt,
         int shots_count = -1) {
        return pyEvolveAsync(initial_state, kernel, params, observables, qpu_id,
                             noise_model, shots_count);
      },
      nb::arg("initial_state"), nb::arg("kernel"), nb::arg("params"),
      nb::arg("observables"), nb::arg("qpu_id") = 0,
      nb::arg("noise_model") = std::nullopt, nb::arg("shots_count") = -1, "");
  mod.def(
      "evolve_async",
      [](std::function<evolve_result()> evolveFunctor, std::size_t qpu_id = 0) {
        nb::gil_scoped_release release;
        return __internal__::evolve_async(evolveFunctor, qpu_id);
      },
      nb::arg("evolve_function"), nb::arg("qpu_id") = 0);
}

} // namespace cudaq
