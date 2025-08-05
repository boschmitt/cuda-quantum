/*******************************************************************************
 * Copyright (c) 2022 - 2025 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "cudaq.h"
#include "cudaq/Optimizer/Dialect/Quake/QuakeOps.h"
#include "cudaq/Todo.h"
#include "cudaq/algorithms/observe.h"
#include "runtime/cudaq/platform/py_alt_launch_kernel.h"
#include "utils/OpaqueArguments.h"
#include "mlir/CAPI/IR.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include <fmt/core.h>
#include <nanobind/stl/vector.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/function.h>

namespace nb = nanobind;

namespace cudaq {
inline constexpr int defaultShotsValue = -1;
inline constexpr int defaultQpuIdValue = 0;
enum class PyParType { thread, mpi };

/// @brief Analyze the MLIR Module for the kernel and check for
/// CUDA-Q specification adherence. Check that the kernel
/// returns void and does not contain measurements.
std::tuple<bool, std::string> isValidObserveKernel(nb::object &kernel) {
  if (nb::hasattr(kernel, "compile"))
    kernel.attr("compile")();

  auto kernelName = nb::cast<std::string>(kernel.attr("name"));
  auto kernelMod = nb::cast<MlirModule>(kernel.attr("module"));

  using namespace mlir;

  ModuleOp mod = unwrap(kernelMod);
  func::FuncOp kernelFunc;
  mod.walk([&](func::FuncOp function) {
    if (function.getName() == ("__nvqpp__mlirgen__" + kernelName)) {
      kernelFunc = function;
      return WalkResult::interrupt();
    }
    return WalkResult::advance();
  });

  // Do we have a return type?
  if (kernelFunc.getNumResults())
    return std::make_tuple(
        false, "kernels passed to observe must have void return type.");

  // Are measurements specified?
  bool hasMeasures = false;
  kernelFunc.walk([&](quake::MeasurementInterface measure) {
    hasMeasures = true;
    return WalkResult::interrupt();
  });
  if (hasMeasures)
    return std::make_tuple(
        false, "kernels passed to observe cannot have measurements specified.");

  // Valid kernel...
  return std::make_tuple(true, "");
}

async_observe_result pyObserveAsync(nb::object &kernel,
                                    const spin_op &spin_operator,
                                    nb::args &args, std::size_t qpu_id,
                                    int shots) {
  if (nb::hasattr(kernel, "compile"))
    kernel.attr("compile")();

  nb::object kernelBlockArgs = kernel.attr("arguments");
  if (nb::len(kernelBlockArgs) != args.size())
    throw std::runtime_error(
        "Invalid number of arguments passed to observe_async.");

  auto &platform = cudaq::get_platform();
  auto kernelName = nb::cast<std::string>(kernel.attr("name"));
  auto kernelMod = nb::cast<MlirModule>(kernel.attr("module"));
  args = simplifiedValidateInputArguments(args);
  auto *argData = toOpaqueArgs(args, kernelMod, kernelName);

  // Launch the asynchronous execution.
  nb::gil_scoped_release release;
  return details::runObservationAsync(
      [argData, kernelName, kernelMod]() mutable {
        pyAltLaunchKernel(kernelName, kernelMod, *argData, {});
        delete argData;
      },
      spin_operator, platform, shots, kernelName, qpu_id);
}

async_observe_result pyObserveAsyncWrapper(nb::object &kernel,
                                           nb::object &spin_operator_obj,
                                           nb::args &args, std::size_t qpu_id,
                                           int shots) {

  // FIXME(OperatorCpp): Remove this when the operator class is implemented in
  // C++
  cudaq::spin_op spin_operator = [](nb::object &obj) -> cudaq::spin_op {
    if (nb::hasattr(obj, "_to_spinop"))
      return nb::cast<cudaq::spin_op>(obj.attr("_to_spinop")());
    return nb::cast<cudaq::spin_op>(obj);
  }(spin_operator_obj);

  return pyObserveAsync(kernel, spin_operator, args, qpu_id, shots);
}

/// @brief Run `cudaq::observe` on the provided kernel and spin operator.
observe_result pyObservePar(const PyParType &type, nb::object &kernel,
                            spin_op &spin_operator, nb::args args = {},
                            int shots = defaultShotsValue,
                            std::optional<noise_model> noise = std::nullopt) {
  if (nb::hasattr(kernel, "compile"))
    kernel.attr("compile")();

  // Ensure the user input is correct.
  auto &platform = cudaq::get_platform();
  if (!platform.supports_task_distribution())
    throw std::runtime_error(
        "The current quantum_platform does not support parallel distribution "
        "of observe() expectation value computations.");

  // FIXME Handle noise modeling with parallel distribution.
  if (noise)
    TODO("Handle Noise Models with python parallel distribution.");

  auto name = nb::cast<std::string>(kernel.attr("name"));
  auto nQpus = platform.num_qpus();
  if (type == PyParType::thread) {
    // Does this platform expose more than 1 QPU
    // If so, let's distribute the work amongst the QPUs
    if (nQpus == 1)
      printf(
          "[cudaq::observe warning] distributed observe requested but only 1 "
          "QPU available. no speedup expected.\n");
    return details::distributeComputations(
        [&](std::size_t i, const spin_op &op) {
          return pyObserveAsync(kernel, op, args, i, shots);
        },
        spin_operator, nQpus);
  }

  if (!mpi::is_initialized())
    throw std::runtime_error("Cannot use mpi multi-node observe() without "
                             "MPI (did you initialize MPI?).");

  // Necessarily has to be MPI
  // Get the rank and the number of ranks
  auto rank = mpi::rank();
  auto nRanks = mpi::num_ranks();

  // Each rank gets a subset of the spin terms
  auto spins = spin_operator.distribute_terms(nRanks);

  // Get this rank's set of spins to compute
  auto localH = spins[rank];

  // Distribute locally, i.e. to the local nodes QPUs
  auto localRankResult = details::distributeComputations(
      [&](std::size_t i, const spin_op &op) {
        return pyObserveAsync(kernel, op, args, i, shots);
      },
      localH, nQpus);

  // combine all the data via an all_reduce
  auto exp_val = localRankResult.expectation();
  auto globalExpVal = mpi::all_reduce(exp_val, std::plus<double>());
  return observe_result(globalExpVal, spin_operator);
}

void bindObserveAsync(nb::module_ &mod) {
  auto parallelSubmodule = mod.def_submodule("parallel");
  nb::class_<cudaq::parallel::mpi>(
      parallelSubmodule, "mpi",
      "Type indicating that the :func:`observe` function should distribute its "
      "expectation value computations accross available MPI ranks and GPUs for "
      "each term.");
  nb::class_<cudaq::parallel::thread>(
      parallelSubmodule, "thread",
      "Type indicating that the :func:`observe` function should distribute its "
      "term "
      "expectation value computations across available GPUs via standard C++ "
      "threads.");

  mod.def("observe_async", &pyObserveAsyncWrapper, nb::arg("kernel"),
          nb::arg("spin_operator"), nb::arg("args"), nb::kw_only(),
          nb::arg("qpu_id") = defaultQpuIdValue,
          nb::arg("shots_count") = defaultShotsValue,
          R"#(Compute the expected value of the `spin_operator` with respect to 
the `kernel` asynchronously. If the kernel accepts arguments, it will 
be evaluated with respect to `kernel(*arguments)`. When targeting a
quantum platform with more than one QPU, the optional `qpu_id` allows
for control over which QPU to enable. Will return a future whose results
can be retrieved via `future.get()`.

Args:
  kernel (:class:`Kernel`): The :class:`Kernel` to evaluate the 
    expectation value with respect to.
  spin_operator (`SpinOperator`): The Hermitian spin operator to 
    calculate the expectation of.
  *arguments (Optional[Any]): The concrete values to evaluate the 
    kernel function at. Leave empty if the kernel doesn't accept any arguments.
  qpu_id (Optional[int]): The optional identification for which QPU on 
    the platform to target. Defaults to zero. Key-word only.
  shots_count (Optional[int]): The number of shots to use for QPU 
    execution. Defaults to -1 implying no shots-based sampling. Key-word only.

Returns:
  :class:`AsyncObserveResult`: 
  A future containing the result of the call to observe.)#");

  mod.def("isValidObserveKernel", &isValidObserveKernel);

  mod.def(
      "observe_parallel",
      [&](nb::object kernel, spin_op &spin_operator, nb::args arguments,
          int shots, nb::type_object execution,
          std::optional<noise_model> noise) -> observe_result {
        // Observe can be a single observe call, a parallel observe call,
        // or a observe broadcast. We'll handle them all here.

        using ObserveApplicator = std::function<std::vector<observe_result>(
            nb::object &, spin_op &, nb::args &, int,
            std::optional<noise_model>)>;

        std::unordered_map<std::string, ObserveApplicator> applicator{
            {"thread",
             [](nb::object &kernel, spin_op &spin_operator, nb::args arguments,
                int shots, std::optional<noise_model> noise) {
               return std::vector<observe_result>{
                   pyObservePar(PyParType::thread, kernel, spin_operator,
                                arguments, shots, noise)};
             }},
            {"mpi",
             [](nb::object &kernel, spin_op &spin_operator, nb::args arguments,
                int shots, std::optional<noise_model> noise) {
               return std::vector<observe_result>{
                   pyObservePar(PyParType::mpi, kernel, spin_operator,
                                arguments, shots, noise)};
             }}};

        std::string applicatorKey = nb::cast<std::string>(execution.attr("__name__"));

        // Run the observation task
        return applicator[applicatorKey](kernel, spin_operator, arguments,
                                         shots, noise)[0];
      },
      nb::arg("kernel"), nb::arg("spin_operator"), nb::arg("args"), nb::kw_only(),
      nb::arg("shots_count") = defaultShotsValue, nb::arg("execution"),
      nb::arg("noise_model") = nb::none(),
      R"#(Compute the expected value of the `spin_operator` with respect to 
the `kernel`. If the input `spin_operator` is a list of `SpinOperator` then compute 
the expected value of every operator in the list and return a list of results.
If the kernel accepts arguments, it will be evaluated 
with respect to `kernel(*arguments)`. Each argument in `arguments` provided
can be a list or ndarray of arguments of the specified kernel argument
type, and in this case, the `observe` functionality will be broadcasted over
all argument sets and a list of `observe_result` instances will be returned.
If both the input `spin_operator` and `arguments` are broadcast lists, 
a nested list of results over `arguments` then `spin_operator` will be returned.

Args:
  kernel (:class:`Kernel`): The :class:`Kernel` to evaluate the 
    expectation value with respect to.
  spin_operator (:class:`SpinOperator` or `list[SpinOperator]`): The Hermitian spin operator to 
    calculate the expectation of, or a list of such operators.
  *arguments (Optional[Any]): The concrete values to evaluate the 
    kernel function at. Leave empty if the kernel doesn't accept any arguments.
  shots_count (Optional[int]): The number of shots to use for QPU 
    execution. Defaults to -1 implying no shots-based sampling. Key-word only.
  noise_model (Optional[`NoiseModel`]): The optional :class:`NoiseModel` to add 
    noise to the kernel execution on the simulator. Defaults to an empty 
    noise model.

Returns:
  :class:`ObserveResult`: 
    A data-type containing the expectation value of the `spin_operator` with 
    respect to the `kernel(*arguments)`, or a list of such results in the case 
    of `observe` function broadcasting. If `shots_count` was provided, the 
    :class:`ObserveResult` will also contain a :class:`SampleResult` dictionary.)#");
}
} // namespace cudaq
