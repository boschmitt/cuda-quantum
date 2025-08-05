/*******************************************************************************
 * Copyright (c) 2022 - 2025 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "cudaq/algorithms/run.h"
#include "runtime/cudaq/platform/py_alt_launch_kernel.h"
#include "utils/OpaqueArguments.h"
#include <future>
#include <nanobind/stl/complex.h>
#include <nanobind/stl/function.h>
#include <nanobind/ndarray.h>
#include <nanobind/nanobind.h>
#include <nanobind/stl/vector.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/optional.h>
#include <string>
#include <tuple>
#include <vector>

namespace cudaq {
namespace details {

std::vector<nb::object> readRunResults(mlir::ModuleOp module,
                                       mlir::func::FuncOp kernelFuncOp,
                                       mlir::Type ty, RunResultSpan &results,
                                       std::size_t count) {
  std::vector<nb::object> ret;
  std::size_t byteSize = results.lengthInBytes / count;
  for (std::size_t i = 0; i < results.lengthInBytes; i += byteSize) {
    nb::object obj = convertResult(module, kernelFuncOp, ty, results.data + i);
    ret.push_back(obj);
  }
  return ret;
}

static std::tuple<std::string, MlirModule, OpaqueArguments *,
                  mlir::func::FuncOp>
getKernelLaunchParameters(nb::object &kernel, nb::args args) {
  nb::object kernel_args = kernel.attr("arguments");
  if (nb::len(kernel_args) != args.size())
    throw std::runtime_error("Invalid number of arguments passed to run:" +
                             std::to_string(args.size()) + " expected " +
                             std::to_string(nb::len(kernel_args)));

  if (nb::hasattr(kernel, "compile"))
    kernel.attr("compile")();

  auto kernelName = nb::cast<std::string>(kernel.attr("name"));
  if (!nb::hasattr(kernel, "module") || kernel.attr("module").is_none())
    throw std::runtime_error(
        "Unsupported target / Invalid kernel for `run`: missing module");
  auto kernelMod = nb::cast<MlirModule>(kernel.attr("module"));
  args = simplifiedValidateInputArguments(args);
  auto *argData = toOpaqueArgs(args, kernelMod, kernelName);

  auto funcOp = getKernelFuncOp(kernelMod, kernelName);
  return {kernelName, kernelMod, argData, funcOp};
}

RunResultSpan pyRunTheKernel(const std::string &name, MlirModule module,
                             func::FuncOp funcOp,
                             cudaq::OpaqueArguments &runtimeArgs,
                             cudaq::quantum_platform &platform,
                             std::size_t shots_count, std::size_t qpu_id = 0) {

  auto returnTypes = funcOp.getResultTypes();
  if (returnTypes.empty() || returnTypes.size() > 1)
    throw std::runtime_error(
        "`cudaq.run` only supports kernels that return a value.");

  auto returnTy = returnTypes[0];
  // Disallow returning list / vectors from entry-point kernels.
  if (isa<cudaq::cc::StdvecType>(returnTy)) {
    throw std::runtime_error("`cudaq.run` does not yet support returning "
                             "`list` from entry-point kernels.");
  }

  auto mod = unwrap(module);

  auto [rawArgs, size, returnOffset, thunk] =
      pyAltLaunchKernelBase(name, module, returnTy, runtimeArgs, {}, 0, false);

  auto results = details::runTheKernel(
      [&]() mutable {
        pyLaunchKernel(name, thunk, mod, runtimeArgs, rawArgs, size,
                       returnOffset, {});
      },
      platform, name, shots_count, qpu_id);

  std::free(rawArgs);
  return results;
}

std::vector<nb::object> pyReadResults(RunResultSpan results, MlirModule module,
                                      func::FuncOp funcOp,
                                      std::size_t shots_count) {
  auto mod = unwrap(module);
  auto returnTypes = funcOp.getResultTypes();
  auto returnTy = returnTypes[0];

  return readRunResults(mod, funcOp, returnTy, results, shots_count);
}

} // namespace details

/// @brief Run `cudaq::run` on the provided kernel.
std::vector<nb::object> pyRun(nb::object &kernel, nb::args args,
                              std::size_t shots_count,
                              std::optional<noise_model> noise_model) {
  if (shots_count == 0)
    return {};

  auto [name, module, argData, func] =
      details::getKernelLaunchParameters(kernel, args);

  auto mod = unwrap(module);
  mod->setAttr(runtime::enableCudaqRun, mlir::UnitAttr::get(mod->getContext()));

  auto &platform = get_platform();
  if (noise_model.has_value()) {
    if (platform.is_remote())
      throw std::runtime_error(
          "Noise model is not supported on remote platforms.");
    // Launch the kernel in the appropriate context.
    platform.set_noise(&noise_model.value());
  }

  auto span = details::pyRunTheKernel(name, module, func, *argData, platform,
                                      shots_count);
  delete argData;
  auto results = details::pyReadResults(span, module, func, shots_count);

  if (noise_model.has_value())
    platform.reset_noise();

  return results;
}

// Internal struct representing buffer to be filled asynchronously.
// When the `ready` future is set, the content of the buffer is filled.
struct async_run_result {
  std::future<void> ready;
  std::vector<nb::object> *results;
  std::string *error;
};

/// @brief Run `cudaq::run_async` on the provided kernel.
async_run_result pyRunAsync(nb::object &kernel, nb::args args,
                            std::size_t shots_count,
                            std::optional<noise_model> noise_model,
                            std::size_t qpu_id) {
  kernel.inc_ref();
  auto &platform = get_platform();
  auto numQPUs = platform.num_qpus();
  if (qpu_id >= numQPUs)
    throw std::runtime_error("qpu_id (" + std::to_string(qpu_id) +
                             ") exceeds the number of available QPUs (" +
                             std::to_string(numQPUs) + ")");

  auto [name, module, argData, func] =
      details::getKernelLaunchParameters(kernel, args);

  auto mod = unwrap(module);
  mod->setAttr(runtime::enableCudaqRun, mlir::UnitAttr::get(mod->getContext()));

  if (noise_model.has_value() && platform.is_remote())
    throw std::runtime_error(
        "Noise model is not supported on remote platforms.");

  async_run_result result;
  result.results = new std::vector<nb::object>();
  result.error = new std::string();

  if (shots_count == 0) {
    std::promise<void> promise;
    result.ready = promise.get_future();
    promise.set_value();
    return result;
  }

  std::promise<details::RunResultSpan> spanPromise;
  auto spanFuture = spanPromise.get_future();

  std::promise<std::string> errorPromise;
  auto errorFuture = errorPromise.get_future();

  // Run the kernel and compute results span.
  {
    // Release GIL to allow c++ threads, all code inside the scope is c++, so
    // there is no need to re-acquire the GIL inside the thread.
    nb::gil_scoped_release gil_release{};
    QuantumTask wrapped = detail::make_copyable_function(
        [sp = std::move(spanPromise), ep = std::move(errorPromise), shots_count,
         qpu_id, argData, name, module, func,
         noise_model = std::move(noise_model)]() mutable {
          auto &platform = get_platform();

          // Launch the kernel in the appropriate context.
          if (noise_model.has_value())
            platform.set_noise(&noise_model.value());

          try {
            auto span = details::pyRunTheKernel(name, module, func, *argData,
                                                platform, shots_count, qpu_id);
            delete argData;
            sp.set_value(span);
            ep.set_value("");
          } catch (std::runtime_error &e) {
            auto message = std::string(e.what());
            sp.set_value({});
            ep.set_value(message);
          }
          platform.reset_noise();
        });
    platform.enqueueAsyncTask(qpu_id, wrapped);
  }

  // Convert results after the span is computed.
  {
    // Release GIL to allow c++ threads, re-acquire for conversion of the
    // results to python objects.
    nb::gil_scoped_release gil_release{};
    auto resultFuture =
        std::async(std::launch::deferred,
                   [sf = std::move(spanFuture), ef = std::move(errorFuture),
                    errorPtr = result.error, resultsPtr = result.results,
                    module, func, shots_count]() mutable {
                     auto error = ef.get();
                     std::swap(*errorPtr, error);
                     if (error.empty()) {
                       auto span = sf.get();
                       nb::gil_scoped_acquire gil{};
                       auto results = details::pyReadResults(span, module, func,
                                                             shots_count);
                       std::swap(*resultsPtr, results);
                     }
                   });
    result.ready = std::move(resultFuture);
  }

  return result;
}

/// @brief Bind the run cudaq function.
void bindPyRun(nb::module_ &mod) {
  mod.def("run", &pyRun, nb::arg("kernel"), nb::arg("args"), nb::kw_only(),
          nb::arg("shots_count") = 100, nb::arg("noise_model") = nb::none(),
          R"#(Run the provided `kernel` with the given kernel arguments over 
the specified number of circuit executions (`shots_count`).

Args:
  kernel: The kernel to execute `shots_count` times on the QPU.
  *arguments: The concrete values to evaluate the kernel function at.
  shots_count: The number of kernel executions on the QPU. Defaults to 100.
  noise_model: The optional noise model to add noise to the kernel execution.

Returns:
  A list of kernel return values from each execution. The length equals `shots_count`.
)#");
}

/// @brief Bind the run_async cudaq function.
void bindPyRunAsync(nb::module_ &mod) {
  nb::class_<async_run_result>(mod, "AsyncRunResult", "")
      .def(
          "get",
          [](async_run_result &self) {
            self.ready.get();
            auto err = *self.error;
            if (!err.empty()) {
              delete self.error;
              throw std::runtime_error(err);
            }
            auto ret = *self.results;
            delete self.results;
            return ret;
          },
          "");
  mod.def("run_async_internal", &pyRunAsync, nb::arg("kernel"), nb::arg("args"), nb::kw_only(),
          nb::arg("shots_count") = 100, nb::arg("noise_model") = nb::none(),
          nb::arg("qpu_id") = 0,
          R"#(Run the provided `kernel` with the given kernel arguments over 
the specified number of circuit executions (`shots_count`) asynchronously on the 
specified `qpu_id`.

Args:
  kernel: The kernel to execute `shots_count` times on the QPU.
  *arguments: The concrete values to evaluate the kernel function at.
  shots_count: The number of kernel executions on the QPU. Defaults to 100.
  noise_model: The optional noise model to add noise to the kernel execution.
  qpu_id: The id of the QPU. Defaults to 0.

Returns:
  AsyncRunResult: A handle which can be waited on via a `get()` method.
)#");
}
} // namespace cudaq
