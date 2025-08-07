/*******************************************************************************
 * Copyright (c) 2022 - 2025 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "py_runtime_target.h"
#include "LinkedLibraryHolder.h"
#include "common/FmtCore.h"
#include "common/Logger.h"
#include <nanobind/nanobind.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/map.h>

namespace nb = nanobind;

namespace cudaq {

std::map<std::string, std::string>
parseTargetKwArgs(const nb::kwargs &extraConfig) {
  if (extraConfig.contains("options"))
    throw std::runtime_error("The keyword `options` argument is not supported "
                             "in cudaq.set_target(). Please use the keyword "
                             "`option` in order to set the target options.");
  std::map<std::string, std::string> config;
  for (auto [key, value] : extraConfig) {
    std::string strValue = "";
    if (nb::isinstance<nb::bool_>(value))
      strValue = nb::cast<nb::bool_>(value) ? "true" : "false";
    else if (nb::isinstance<nb::str>(value))
      strValue = nb::cast<std::string>(value);
    else if (nb::isinstance<nb::int_>(value))
      strValue = std::to_string(nb::cast<int>(value));
    else
      throw std::runtime_error(
          "QPU kwargs config value must be cast-able to a string.");

    // Ignore empty parameter values
    if (!strValue.empty())
      config.emplace(nb::cast<std::string>(key), strValue);
  }
  return config;
}

void bindRuntimeTarget(nb::module_ &mod, LinkedLibraryHolder &holder) {

  nb::enum_<simulation_precision>(
      mod, "SimulationPrecision",
      "Enumeration describing the precision of the underyling simulation.")
      .value("fp32", simulation_precision::fp32)
      .value("fp64", simulation_precision::fp64);

  nb::class_<cudaq::RuntimeTarget>(
      mod, "Target",
      "The `cudaq.Target` represents the underlying infrastructure that "
      "CUDA-Q kernels will execute on. Instances of `cudaq.Target` describe "
      "what simulator they may leverage, the quantum_platform required for "
      "execution, and a description for the target.")
      .def_ro("name", &cudaq::RuntimeTarget::name,
                    "The name of the `cudaq.Target`.")
      .def_ro("simulator", &cudaq::RuntimeTarget::simulatorName,
                    "The name of the simulator this `cudaq.Target` leverages. "
                    "This will be empty for physical QPUs.")
      .def_ro("platform", &cudaq::RuntimeTarget::platformName,
                    "The name of the quantum_platform implementation this "
                    "`cudaq.Target` leverages.")
      .def_ro("description", &cudaq::RuntimeTarget::description,
                    "A string describing the features for this `cudaq.Target`.")
      .def("num_qpus", &cudaq::RuntimeTarget::num_qpus,
           "Return the number of QPUs available in this `cudaq.Target`.")
      .def("is_remote", &cudaq::RuntimeTarget::is_remote,
           "Returns true if the target consists of a remote REST QPU.")
      .def("is_emulated", &cudaq::RuntimeTarget::is_emulated,
           "Returns true if the emulation mode for the target has been "
           "activated.")
      .def(
          "is_remote_simulator", &cudaq::RuntimeTarget::is_remote_simulator,
          "Returns true if the target consists of a remote REST Simulator QPU.")
      .def("get_precision", &cudaq::RuntimeTarget::get_precision,
           "Return the simulation precision for the current target.")
      .def(
          "__str__",
          [](cudaq::RuntimeTarget &self) {
            std::string targetInfo = fmt::format(
                "Target {}\n\tsimulator={}\n\tplatform={}"
                "\n\tdescription={}\n\tprecision={}\n",
                self.name, self.simulatorName, self.platformName,
                self.description,
                self.get_precision() == simulation_precision::fp32 ? "fp32"
                                                                   : "fp64");
            const std::string argHelperStr = self.get_target_args_help_string();
            if (!argHelperStr.empty()) {
              targetInfo += ("Supported Arguments:\n" + argHelperStr);
            }
            return targetInfo;
          },
          "Persist the information in this `cudaq.Target` to a string.");

  mod.def(
      "has_target",
      [&](const std::string &name) { return holder.hasTarget(name); },
      "Return true if the `cudaq.Target` with the given name exists.");
  mod.def(
      "reset_target", [&]() { return holder.resetTarget(); },
      "Reset the current `cudaq.Target` to the default.");
  mod.def(
      "get_target",
      [&](const std::string &name) { return holder.getTarget(name); },
      "Return the `cudaq.Target` with the given name. Will raise an exception "
      "if the name is not valid.");
  mod.def(
      "get_target", [&]() { return holder.getTarget(); },
      "Return the `cudaq.Target` with the given name. Will raise an exception "
      "if the name is not valid.");
  mod.def(
      "get_targets", [&]() { return holder.getTargets(); },
      "Return all available `cudaq.Target` instances on the current system.");
  mod.def(
      "set_target",
      [&](const cudaq::RuntimeTarget &target, nb::kwargs extraConfig) {
        auto config = parseTargetKwArgs(extraConfig);
        holder.setTarget(target.name, config);
      },
      "Set the `cudaq.Target` to be used for CUDA-Q kernel execution. "
      "Can provide optional, target-specific configuration data via Python "
      "kwargs.");
  mod.def(
      "set_target",
      [&](const std::string &name, nb::kwargs extraConfig) {
        auto config = parseTargetKwArgs(extraConfig);
        holder.setTarget(name, config);
      },
      "Set the `cudaq.Target` with given name to be used for CUDA-Q "
      "kernel execution. Can provide optional, target-specific configuration "
      "data via Python kwargs.");
}

} // namespace cudaq
