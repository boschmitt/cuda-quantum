/*******************************************************************************
 * Copyright (c) 2022 - 2025 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include <nanobind/stl/vector.h>
#include <nanobind/stl/string.h>
#include <nanobind/operators.h>

#include "py_SampleResult.h"

#include "common/SampleResult.h"

#include <sstream>

namespace cudaq {

void bindMeasureCounts(nb::module_ &mod) {
  using namespace cudaq;

  // TODO Bind the variants of this functions that take the register name
  // as input.
  nb::class_<sample_result>(
      mod, "SampleResult",
      R"#(A data-type containing the results of a call to :func:`sample`. 
This includes all measurement counts data from both mid-circuit and 
terminal measurements.

Note:
	At this time, mid-circuit measurements are not directly supported. 
	Mid-circuit measurements may only be used if they are passed through 
	to `c_if`.

Attributes:
	register_names (List[str]): A list of the names of each measurement 
		register that are stored in `self`.)#")
      .def_prop_ro("register_names", &sample_result::register_names)
      .def(nb::init<>())
      .def(
          "dump", [](sample_result &self) { self.dump(); },
          "Print a string of the raw measurement counts data to the "
          "terminal.\n")
      .def("serialize", &sample_result::serialize,
           "Serialize this SampleResult to a vector of integer encoding.")
      .def("deserialize", &sample_result::deserialize,
           "Deserialize this SampleResult from an existing vector of integers "
           "adhering to the implicit encoding.")
      .def("get_total_shots", &sample_result::get_total_shots,
           "Get the total number of shots in the sample result")
      .def(
          "__str__",
          [](sample_result &self) {
            std::stringstream ss;
            self.dump(ss);
            return ss.str();
          },
          "Return a string of the raw measurement counts that are stored in "
          "`self`.\n")
      .def(
          "__getitem__",
          [](sample_result &self, const std::string &bitstring) {
            auto map = self.to_map();
            auto iter = map.find(bitstring);
            if (iter == map.end())
              throw nb::key_error(("bitstring '" + bitstring +
                                  "' does not exist").c_str());

            return iter->second;
          },
          nb::arg("bitstring"),
          R"#(Return the measurement counts for the given `bitstring`.

Args:
	bitstring (str): The binary string to return the measurement data of.

Returns:
	float: The number of times the given `bitstring` was measured 
	during the `shots_count` number of executions on the QPU.)#")
      .def(
          "__len__", [](sample_result &self) { return self.to_map().size(); },
          "Return the number of elements in `self`. Equivalent to "
          "the number of uniquely measured bitstrings.\n")
      .def(
          "__iter__",
          [](sample_result &self) {
            auto map = self.to_map();
            std::vector<std::string> keys;
            for (const auto &pair : map) {
              keys.push_back(pair.first);
            }
            return nb::cast(keys);
          },
          "Iterate through the :class:`SampleResult` dictionary.\n")
      .def("expectation", &sample_result::expectation,
           nb::arg("register_name") = GlobalRegisterName,
           "Return the expectation value in the Z-basis of the :class:`Kernel` "
           "that was sampled.\n")
      .def(
          "expectation_z",
          [](sample_result &self, const std::string_view register_name) {
            PyErr_WarnEx(PyExc_DeprecationWarning,
                         "expectation_z() is deprecated, use expectation() "
                         "with the same "
                         "argument structure.",
                         1);
            return self.expectation();
          },
          nb::arg("register_name") = GlobalRegisterName,
          "Return the expectation value in the Z-basis of the :class:`Kernel` "
          "that was sampled.\n")
      .def("probability", &sample_result::probability,
           "Return the probability of observing the given bit string.\n",
           nb::arg("bitstring"), nb::arg("register_name") = GlobalRegisterName,
           R"#(Return the probability of measuring the given `bitstring`.

Args:
  bitstring (str): The binary string to return the measurement 
		probability of.
  register_name (Optional[str]): The optional measurement register 
		name to extract the probability from. Defaults to the '__global__' 
		register.

Returns:
  float: 
	The probability of measuring the given `bitstring`. Equivalent 
	to the proportion of the total times the bitstring was measured 
	vs. the number of experiments (`shots_count`).)#")
      .def("most_probable", &sample_result::most_probable,
           nb::arg("register_name") = GlobalRegisterName,
           R"#(Return the bitstring that was measured most frequently in the 
experiment.

Args:
  register_name (Optional[str]): The optional measurement register 
		name to extract the most probable bitstring from. Defaults to the 
		'__global__' register.

Returns:
  str: The most frequently measured binary string during the experiment.)#")
      .def("count", &sample_result::count, nb::arg("bitstring"),
           nb::arg("register_name") = GlobalRegisterName,
           R"#(Return the number of times the given bitstring was observed.

Args:
  bitstring (str): The binary string to return the measurement counts for.
  register_name (Optional[str]): The optional measurement register name to 
		extract the probability from. Defaults to the '__global__' register.

Returns:
  int : The number of times the given bitstring was measured during the experiment.)#")
      .def("get_marginal_counts",
           static_cast<sample_result (sample_result::*)(
               const std::vector<std::size_t> &, const std::string_view) const>(
               &sample_result::get_marginal),
           nb::arg("marginal_indices"), nb::kw_only(),
           nb::arg("register_name") = GlobalRegisterName,
           R"#(Extract the measurement counts data for the provided subset of 
qubits (`marginal_indices`).

Args:
  marginal_indices (list[int]): A list of the qubit indices to extract the 
		measurement data from.
  register_name (Optional[str]): The optional measurement register name to extract 
		the counts data from. Defaults to the '__global__' register.
Returns:
  :class:`SampleResult`: 
	A new `SampleResult` dictionary containing the extracted measurement data.)#")
      .def("get_sequential_data", &sample_result::sequential_data,
           nb::arg("register_name") = GlobalRegisterName,
           "Return the data from the given register (`register_name`) as it "
           "was collected sequentially. A list of measurement results, not "
           "collated into a map.\n")
      .def(
          "get_register_counts",
          [&](sample_result &self, const std::string &registerName) {
            auto cd = self.to_map(registerName);
            ExecutionResult res(cd);
            return sample_result(res);
          },
          nb::arg("register_name"),
          "Extract the provided sub-register (`register_name`) as a new "
          ":class:`SampleResult`.\n")
      .def(
          "items",
          [](sample_result &self) {
            auto map = self.to_map();
            std::vector<std::pair<std::string, std::size_t>> items;
            for (const auto &pair : map) {
              items.push_back(pair);
            }
            return nb::cast(items);
          },
          "Return the key/value pairs in this :class:`SampleResult` "
          "dictionary.\n")
      .def(
          "values",
          [](sample_result &self) {
            auto map = self.to_map();
            std::vector<std::size_t> values;
            for (const auto &pair : map) {
              values.push_back(pair.second);
            }
            return nb::cast(values);
          },
          "Return all values (the counts) in this :class:`SampleResult` "
          "dictionary.\n")
      .def(nb::self += nb::self)
      .def("clear", &sample_result::clear,
           "Clear out all metadata from `self`.\n");
}

} // namespace cudaq
