/*******************************************************************************
 * Copyright (c) 2022 - 2025 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "py_qubit_qis.h"
#include "cudaq/qis/qubit_qis.h"
#include <fmt/core.h>
#include <nanobind/nanobind.h>
#include <nanobind/stl/string.h>
#include <nanobind/operators.h>
#include <functional>

namespace nb = nanobind;

namespace cudaq {

void bindQIS(nb::module_ &mod) {

  nb::class_<qubit>(
      mod, "qubit",
      "The qubit is the primary unit of information in a quantum computer. "
      "Qubits can be created individually or as part of larger registers.")
      .def(nb::init<>())
      .def(
          "__invert__", [](qubit &self) -> qubit & { return !self; },
          "Negate the control qubit.")
      .def("is_negated", &qubit::is_negative,
           "Returns true if this is a negated control qubit.")
      .def("reset_negation", &qubit::negate,
           "Removes the negated state of a control qubit.")
      .def(
          "id", [](qubit &self) { return self.id(); },
          "Return a unique integer identifier for this qubit.");

  nb::class_<qview<>>(mod, "qview",
                      "A non-owning view on a register of qubits.")
      .def(
          "size", [](qview<> &self) { return self.size(); },
          "Return the number of qubits in this view.")
      .def(
          "front", [](qview<> &self) -> qubit & { return self.front(); },
          "Return first qubit in this view.")
      .def(
          "front",
          [](qview<> &self, std::size_t count) { return self.front(count); },
          "Return first `count` qubits in this view.")
      .def(
          "back", [](qview<> &self) -> qubit & { return self.back(); },
          "Return the last qubit in this view.")
      .def(
          "back",
          [](qview<> &self, std::size_t count) { return self.back(count); },
          "Return the last `count` qubits in this view.")
      .def(
          "__iter__",
          [](qview<> &self) {
            std::vector<std::reference_wrapper<qubit>> qubits;
            for (auto it = self.begin(); it != self.end(); ++it) {
              qubits.push_back(std::ref(*it));
            }
            return nb::cast(qubits);
          })
      .def(
          "slice",
          [](qview<> &self, std::size_t start, std::size_t count) {
            return self.slice(start, count);
          },
          "Return the `[start, start+count]` qudits as a non-owning qview.")
      .def("__getitem__", &qview<>::operator[],
           "Return the qubit at the given index.");

  nb::class_<qvector<>>(
      mod, "qvector",
      "An owning, dynamically sized container for qubits. The semantics of the "
      "`qvector` follows that of a `std::vector` or `list` for qubits.")
      .def("__init__", [](qvector<> &self, std::size_t size) {
        new (&self) qvector<>(size);
      })
      .def(
          "size", [](qvector<> &self) { return self.size(); },
          "Return the number of qubits in this `qvector`.")
      .def(
          "front",
          [](qvector<> &self, std::size_t count) { return self.front(count); },
          "Return first `count` qubits in this `qvector` as a non-owning view.")
      .def(
          "front", [](qvector<> &self) -> qubit & { return self.front(); },
          "Return first qubit in this `qvector`.")
      .def(
          "back", [](qvector<> &self) -> qubit & { return self.back(); },
          "Return the last qubit in this `qvector`.")
      .def(
          "back",
          [](qvector<> &self, std::size_t count) { return self.back(count); },
          "Return the last `count` qubits in this `qvector` as a non-owning "
          "view.")
      .def(
          "__iter__",
          [](qvector<> &self) {
            std::vector<std::reference_wrapper<qubit>> qubits;
            for (auto it = self.begin(); it != self.end(); ++it) {
              qubits.push_back(std::ref(*it));
            }
            return nb::cast(qubits);
          })
      .def(
          "slice",
          [](qvector<> &self, std::size_t start, std::size_t count) {
            return self.slice(start, count);
          },
          "Return the `[start, start+count]` qudits as a non-owning qview.")
      .def("__getitem__", &qvector<2>::operator[],
           "Return the qubit at the given index.");

  nb::class_<pauli_word>(mod, "pauli_word",
                         "The `pauli_word` is a thin wrapper on a Pauli tensor "
                         "product string, e.g. `XXYZ` on 4 qubits.")
      .def(nb::init<>())
      .def(nb::init<const std::string>());
}
} // namespace cudaq
