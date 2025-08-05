/*******************************************************************************
 * Copyright (c) 2022 - 2025 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "common/ExecutionContext.h"
#include "common/RecordLogParser.h"
#include "cudaq/platform.h"
#include <fmt/core.h>
#include <nanobind/stl/complex.h>
#include <nanobind/stl/vector.h>
#include <nanobind/stl/string.h>
#include <nanobind/ndarray.h>

namespace nb = nanobind;

namespace nvqir {
std::string_view getQirOutputLog();
void clearQirOutputLog();
} // namespace nvqir

namespace cudaq {

void bindExecutionContext(nb::module_ &mod) {
  nb::class_<cudaq::ExecutionContext>(mod, "ExecutionContext")
      .def(nb::init<std::string>())
      .def(nb::init<std::string, int>())
      .def_ro("result", &cudaq::ExecutionContext::result)
      .def_rw("asyncExec", &cudaq::ExecutionContext::asyncExec)
      .def_ro("asyncResult", &cudaq::ExecutionContext::asyncResult)
      .def_rw("hasConditionalsOnMeasureResults",
                     &cudaq::ExecutionContext::hasConditionalsOnMeasureResults)
      .def_rw("totalIterations",
                     &cudaq::ExecutionContext::totalIterations)
      .def_rw("batchIteration", &cudaq::ExecutionContext::batchIteration)
      .def_rw("numberTrajectories",
                     &cudaq::ExecutionContext::numberTrajectories)
      .def_rw("explicitMeasurements",
                     &cudaq::ExecutionContext::explicitMeasurements)
      .def_ro("invocationResultBuffer",
                    &cudaq::ExecutionContext::invocationResultBuffer)
      .def("setSpinOperator",
           [](cudaq::ExecutionContext &ctx, cudaq::spin_op &spin) {
             ctx.spin = spin;
             assert(cudaq::spin_op::canonicalize(spin) == spin);
           })
      .def("getExpectationValue",
           [](cudaq::ExecutionContext &ctx) { return ctx.expectationValue; });
  mod.def(
      "setExecutionContext",
      [](cudaq::ExecutionContext &ctx) {
        auto &self = cudaq::get_platform();
        self.set_exec_ctx(&ctx);
      },
      "");
  mod.def(
      "resetExecutionContext",
      []() {
        auto &self = cudaq::get_platform();
        self.reset_exec_ctx();
      },
      "");
  mod.def("supportsConditionalFeedback", []() {
    auto &platform = cudaq::get_platform();
    return platform.supports_conditional_feedback();
  });
  mod.def("supportsExplicitMeasurements", []() {
    auto &platform = cudaq::get_platform();
    return platform.supports_explicit_measurements();
  });
  mod.def("getExecutionContextName", []() {
    auto &self = cudaq::get_platform();
    return self.get_exec_ctx()->name;
  });
  mod.def("getQirOutputLog", []() { return nvqir::getQirOutputLog(); });
  mod.def("clearQirOutputLog", []() { nvqir::clearQirOutputLog(); });
  mod.def("decodeQirOutputLog",
          [](const std::string &outputLog, nb::ndarray<uint8_t> decodedResults) {
            cudaq::RecordLogParser parser;
            parser.parse(outputLog);
            // Get the buffer and length of buffer (in bytes) from the parser.
            auto *origBuffer = parser.getBufferPtr();
            const std::size_t bufferSize = parser.getBufferSize();
            std::memcpy(decodedResults.data(), origBuffer, bufferSize);
          });
}
} // namespace cudaq
