# ============================================================================ #
# Copyright (c) 2022 - 2025 NVIDIA Corporation & Affiliates.                   #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

from __future__ import annotations
from dataclasses import dataclass

from cudaq.mlir.dialects import arith, math, quake
from cudaq.mlir.ir import Type, Value, IntegerAttr
from .._base import Iterable
from .._core import types as T
from .._core.scalars import constant
from ..int64 import Int64
from ..qref import QRef


def _fix_negative_index(index: Value, length: Value) -> Int64:
    zero = constant(0)
    index_lt_zero = arith.CmpIOp(IntegerAttr.get(T.i64(), 2), index, zero)
    reverse_index = arith.AddIOp(index, length).result
    index = arith.SelectOp(index_lt_zero, reverse_index, index).result
    return Int64(index)


@dataclass(frozen=True, slots=True)
class Veq(Iterable):
    value: Value

    @staticmethod
    def _allocate_memory(length: Value) -> Value:
        return quake.AllocaOp(T.veq(), size=length).result

    @staticmethod
    def create(length: Value) -> Veq:
        return Veq(Veq._allocate_memory(length))

    @staticmethod
    def from_list(values) -> Veq:
        length = values.get_length()
        data = values.data()
        num_qubits = math.CountTrailingZerosOp(length).result

        # TODO: Dynamically check if number of qubits is power of 2
        # and if the state is normalized

        qubits = Veq._allocate_memory(num_qubits)
        # TODO: Check why this operation returns a new veq
        init = quake.InitializeStateOp(qubits.type, qubits, data).result
        return Veq(init)

    @staticmethod
    def from_state(state: Value) -> Veq:
        num_qubits = quake.GetNumberOfQubitsOp(T.i64(), state).result
        qubits = Veq._allocate_memory(num_qubits)
        init = quake.InitializeStateOp(qubits.type, qubits, state).result
        return Veq(init)

    @staticmethod
    def from_qubits(qubits: list[Value]) -> Veq:
        return Veq(quake.ConcatOp(T.veq(), qubits).result)

    def get_length(self) -> Value:
        return quake.VeqSizeOp(T.i64(), self.value).result

    def get_item(self, index: Value) -> QRef:
        index = _fix_negative_index(index, self.get_length())
        return QRef(
            quake.ExtractRefOp(self.elements_type, self.value, -1,
                               index=index).result)

    def get_slice(self, begin: Value | None, end: Value | None) -> Veq:
        # TODO: Bound check
        dynamic_index = IntegerAttr.get(T.i64(), -1)
        if begin is None:
            begin = constant(0)
        else:
            begin = _fix_negative_index(begin, self.get_length())
        if end is None:
            end = self.get_length()
        else:
            end = _fix_negative_index(end, self.get_length())
        end = arith.SubIOp(end, constant(1)).result
        return Veq(
            quake.SubVeqOp(self.value.type,
                           self.value,
                           dynamic_index,
                           dynamic_index,
                           lower=begin,
                           upper=end).result)

    def back(self, num_elements: Value | None = None) -> Value:
        # TODO: Bound check
        size = self.get_length()
        if num_elements is None:
            last = arith.SubIOp(size, constant(1)).result
            return self.get_item(last)
        begin = arith.SubIOp(size, num_elements).result
        return self.get_slice(begin, size)

    def front(self, num_elements: Value | None = None) -> Value:
        # TODO: Bound check
        zero = constant(0)
        if num_elements is None:
            return self.get_item(zero)
        return self.get_slice(zero, num_elements)

    @property
    def elements_type(self) -> Type:
        return quake.RefType.get()

    @property
    def type(self) -> Type:
        return self.value.type
