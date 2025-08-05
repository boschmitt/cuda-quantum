/*******************************************************************************
 * Copyright (c) 2022 - 2025 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/
#include "cudaq/algorithms/draw.h"
#include "runtime/cudaq/platform/py_alt_launch_kernel.h"
#include "utils/OpaqueArguments.h"
#include <nanobind/stl/complex.h>
#include <nanobind/stl/string.h>
#include <string>
#include <tuple>
#include <vector>

namespace cudaq {

namespace details {
std::tuple<std::string, MlirModule, OpaqueArguments *>
getKernelLaunchParameters(nb::object &kernel, nb::args args) {
  nb::object kernel_args = kernel.attr("arguments");
  if (nb::len(kernel_args) != args.size())
    throw std::runtime_error("Invalid number of arguments passed to draw:" +
                             std::to_string(args.size()) + " expected " +
                             std::to_string(nb::len(kernel_args)));

  if (nb::hasattr(kernel, "compile"))
    kernel.attr("compile")();

  auto kernelName = nb::cast<std::string>(kernel.attr("name"));
  auto kernelMod = nb::cast<MlirModule>(kernel.attr("module"));
  args = simplifiedValidateInputArguments(args);
  auto *argData = toOpaqueArgs(args, kernelMod, kernelName);

  return {kernelName, kernelMod, argData};
}

} // namespace details

/// @brief Run `cudaq::contrib::draw` on the provided kernel.
std::string pyDraw(nb::object &kernel, nb::args args) {
  auto [kernelName, kernelMod, argData] =
      details::getKernelLaunchParameters(kernel, args);

  return contrib::extractTrace([&]() mutable {
    pyAltLaunchKernel(kernelName, kernelMod, *argData, {});
    delete argData;
  });
}

/// @brief Run `cudaq::contrib::draw`'s string overload on the provided kernel.
std::string pyDraw(std::string format, nb::object &kernel, nb::args args) {
  if (format == "ascii") {
    return pyDraw(kernel, args);
  } else if (format == "latex") {
    auto [kernelName, kernelMod, argData] =
        details::getKernelLaunchParameters(kernel, args);

    return contrib::extractTraceLatex([&]() mutable {
      pyAltLaunchKernel(kernelName, kernelMod, *argData, {});
      delete argData;
    });
  } else {
    throw std::runtime_error("Invalid format passed to draw.");
  }
}

/// @brief Bind the draw cudaq function
void bindPyDraw(nb::module_ &mod) {
  mod.def("draw",
          static_cast<std::string (*)(std::string, nb::object &, nb::args)>(&pyDraw),
          R"#(Return a string representing the drawing of the execution path, 
in the format specified as the first argument. If the format is 
'ascii', the output will be a UTF-8 encoded string. If the format 
is 'latex', the output will be a LaTeX string.

Args:
  format (str): The format of the output. Can be 'ascii' or 'latex'.
  kernel (:class:`Kernel`): The :class:`Kernel` to draw.
  *arguments (Optional[Any]): The concrete values to evaluate the kernel 
    function at. Leave empty if the kernel doesn't accept any arguments.)#")
      .def(
          "draw", static_cast<std::string (*)(nb::object &, nb::args)>(&pyDraw),
          R"#(Return a UTF-8 encoded string representing drawing of the execution 
path, i.e., the trace, of the provided `kernel`.
      
Args:
  kernel (:class:`Kernel`): The :class:`Kernel` to draw.
  *arguments (Optional[Any]): The concrete values to evaluate the kernel 
    function at. Leave empty if the kernel doesn't accept any arguments.

Returns:
  The UTF-8 encoded string of the circuit, without measurement operations.

.. code-block:: python

  # Example
  import cudaq
  @cudaq.kernel
  def bell_pair():
      q = cudaq.qvector(2)
      h(q[0])
      cx(q[0], q[1])
      mz(q)
  print(cudaq.draw(bell_pair))
  # Output
  #      ╭───╮     
  # q0 : ┤ h ├──●──
  #      ╰───╯╭─┴─╮
  # q1 : ─────┤ x ├
  #           ╰───╯
  
  # Example with arguments
  import cudaq
  @cudaq.kernel
  def kernel(angle:float):
      q = cudaq.qubit()
      h(q)
      ry(angle, q)
  print(cudaq.draw(kernel, 0.59))
  # Output
  #      ╭───╮╭──────────╮
  # q0 : ┤ h ├┤ ry(0.59) ├
  #      ╰───╯╰──────────╯)#");
}

} // namespace cudaq
