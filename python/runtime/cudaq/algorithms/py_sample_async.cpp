/*******************************************************************************
 * Copyright (c) 2022 - 2025 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "cudaq/algorithms/sample.h"
#include "cudaq/utils/registry.h"
#include "runtime/cudaq/platform/py_alt_launch_kernel.h"
#include "utils/OpaqueArguments.h"
#include "mlir/CAPI/IR.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include <fmt/core.h>
#include <nanobind/stl/vector.h>
#include <nanobind/stl/string.h>

namespace nb = nanobind;

namespace cudaq {
std::string get_quake_by_name(const std::string &, bool,
                              std::optional<std::string>);

void pyAltLaunchKernel(const std::string &, MlirModule, OpaqueArguments &,
                       const std::vector<std::string> &);

void bindSampleAsync(nb::module_ &mod) {
  // Async. result wrapper for Python kernels, which also holds the Python MLIR
  // context.
  //
  // As a kernel is passed to an async. call, its lifetime on the main
  // Python thread decouples from the C++ functor in the execution queue. While
  // we can clone the MLIR module of the kernel in the functor, the context
  // needs to be alive. Hence, we hold the context here to keep it alive on the
  // main Python thread. For example,
  //  `async_handle = sample_async(kernel_factory())`,
  // where `kernel_factory()` returns a kernel object. The `async_handle` would
  // then track a reference (ref count) to the context of the temporary (rval)
  // kernel.
  class py_async_sample_result : public async_sample_result {
  public:
    // Ctors
    py_async_sample_result(async_sample_result &&res, nb::object &&mlirCtx)
        : async_sample_result(std::move(res)), ctx(std::move(mlirCtx)){};

  private:
    nb::object ctx;
  };

  nb::class_<async_sample_result, py_async_sample_result>(
      mod, "AsyncSampleResult",
      R"#(A data-type containing the results of a call to :func:`sample_async`. 
The `AsyncSampleResult` models a future-like type, whose 
:class:`SampleResult` may be returned via an invocation of the `get` method. This 
kicks off a wait on the current thread until the results are available.
See `future <https://en.cppreference.com/w/cpp/thread/future>`_ 
for more information on this programming pattern.)#")
      .def("__init__", [](async_sample_result *self, std::string inJson) {
        async_sample_result f;
        std::istringstream is(inJson);
        is >> f;
        new (self) async_sample_result(std::move(f));
      })
      .def("get", &async_sample_result::get,
           nb::call_guard<nb::gil_scoped_release>(),
           "Return the :class:`SampleResult` from the asynchronous sample "
           "execution.\n")
      .def("__str__", [](async_sample_result &res) {
        std::stringstream ss;
        ss << res;
        return ss.str();
      });

  mod.def(
      "sample_async",
      [&](nb::object kernel, nb::args args, std::size_t shots,
          bool explicitMeasurements, std::size_t qpu_id) {
        auto &platform = cudaq::get_platform();
        if (nb::hasattr(kernel, "compile"))
          kernel.attr("compile")();
        auto kernelName = nb::cast<std::string>(kernel.attr("name"));
        // Clone the kernel module
        auto kernelMod = mlirModuleFromOperation(
            wrap(unwrap(nb::cast<MlirModule>(kernel.attr("module")))->clone()));
        // Get the MLIR context associated with the kernel
        nb::object mlirCtx = kernel.attr("module").attr("context");
        args = simplifiedValidateInputArguments(args);

        // This kernel may not have been registered to the quake registry
        // (usually, the first invocation would register the kernel)
        // i.e., `cudaq::kernelHasConditionalFeedback` won't be able to tell if
        // this kernel has qubit measurement feedback on the first invocation.
        // Thus, add kernel's MLIR code to the registry.
        {
          auto moduleOp = unwrap(kernelMod);
          std::string mlirCode;
          llvm::raw_string_ostream outStr(mlirCode);
          mlir::OpPrintingFlags opf;
          moduleOp.print(outStr, opf);
          cudaq::registry::__cudaq_deviceCodeHolderAdd(kernelName.c_str(),
                                                       mlirCode.c_str());
        }

        // The function below will be executed multiple times
        // if the kernel has conditional feedback. In that case,
        // we have to be careful about deleting the `argData` and
        // only do so after the last invocation of that function.
        // Hence, pass it as a unique_ptr for the functor to manage its
        // lifetime.
        std::unique_ptr<OpaqueArguments> argData(
            toOpaqueArgs(args, kernelMod, kernelName));

        // Should only have C++ going on here, safe to release the GIL
        nb::gil_scoped_release release;
        return py_async_sample_result(
            cudaq::details::runSamplingAsync(
                // Notes:
                // (1) no Python data access is allowed in this lambda body.
                // (2) This lambda might be executed multiple times, e.g, when
                // the kernel contains measurement feedback.
                cudaq::detail::make_copyable_function(
                    [argData = std::move(argData), kernelName,
                     kernelMod]() mutable {
                      pyAltLaunchKernel(kernelName, kernelMod, *argData, {});
                    }),
                platform, kernelName, shots, explicitMeasurements, qpu_id),
            std::move(mlirCtx));
      },
      nb::arg("kernel"), nb::arg("args"), nb::kw_only(), nb::arg("shots_count") = 1000,
      nb::arg("explicit_measurements") = false, nb::arg("qpu_id") = 0,
      R"#(Asynchronously sample the state of the provided `kernel` at the 
specified number of circuit executions (`shots_count`).
When targeting a quantum platform with more than one QPU, the optional
`qpu_id` allows for control over which QPU to enable. Will return a
future whose results can be retrieved via `future.get()`.

Args:
  kernel (:class:`Kernel`): The :class:`Kernel` to execute `shots_count` 
    times on the QPU.
  *arguments (Optional[Any]): The concrete values to evaluate the kernel 
    function at. Leave empty if the kernel doesn't accept any arguments.
  shots_count (Optional[int]): The number of kernel executions on the 
    QPU. Defaults to 1000. Key-word only.
  explicit_measurements (Optional[bool]): A flag to indicate whether or not to 
    concatenate measurements in execution order for the returned sample result.
  qpu_id (Optional[int]): The optional identification for which QPU 
    on the platform to target. Defaults to zero. Key-word only.

Returns:
  :class:`AsyncSampleResult`: 
  A dictionary containing the measurement count results for the :class:`Kernel`.)#");
}
} // namespace cudaq
