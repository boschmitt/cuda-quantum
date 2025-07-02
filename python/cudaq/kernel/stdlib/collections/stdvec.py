# ============================================================================ #
# Copyright (c) 2022 - 2025 NVIDIA Corporation & Affiliates.                   #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

from __future__ import annotations
from dataclasses import dataclass

from cudaq.mlir.dialects import arith, cc
from cudaq.mlir.ir import Type, Value, IntegerAttr
from .._base import Iterable, try_downcast
from .._core import types as T
from .._core.control_flow import for_loop
from .._core.memory import alloca, load, store
from .._core.scalars import constant
from ..literals.int_literal import IntLiteral
from ..boolean import Bool
from ..int64 import Int64
from ..pointer import Pointer


def _fix_negative_index(index: Value, length: Value) -> Int64:
    zero = constant(0)
    index_lt_zero = arith.CmpIOp(IntegerAttr.get(T.i64(), 2), index, zero)
    reverse_index = arith.AddIOp(index, length).result
    index = arith.SelectOp(index_lt_zero, reverse_index, index).result
    return Int64(index)


@dataclass(frozen=True, slots=True)
class Stdvec(Iterable):
    value: Value

    @staticmethod
    def _allocate_memory(elements_type: Type, length: Value) -> Value:
        return Pointer(alloca(elements_type, size=length))

    @staticmethod
    def create(elements_type: Type, length: Value) -> Stdvec:
        data = Stdvec._allocate_memory(elements_type, length)
        vec = cc.StdvecInitOp(T.stdvec(elements_type), data,
                              length=length).result
        return Stdvec(vec)

    @staticmethod
    def from_list(values: list[Value]) -> Stdvec:
        # TODO: Check if values are homogeneous
        elements_type = values[0].type
        length = constant(len(values))
        data = Stdvec._allocate_memory(elements_type, length)
        for i, value in enumerate(values):
            data[IntLiteral(i)].store(value)
        vec = cc.StdvecInitOp(T.stdvec(elements_type), data,
                              length=length).result
        return Stdvec(vec)

    def get_length(self) -> Value:
        return cc.StdvecSizeOp(T.i64(), self.value).result

    def data(self) -> Value:
        ptr_type = T.ptr(T.array(self.elements_type))
        return Pointer(cc.StdvecDataOp(ptr_type, self.value).result)

    def get_item_address(self, index: Value) -> Value:
        index = _fix_negative_index(index, self.get_length())
        data = self.data()
        item_addr = data[index]
        return item_addr

    def get_item(self, index: Value):
        item_addr = self.get_item_address(index)
        return try_downcast(load(item_addr))

    def set_item(self, index, value):
        item_addr = self.get_item_address(index)
        store(value, item_addr)

    def get_slice(self, begin: Value | None, end: Value | None):
        if begin is None:
            begin = constant(0)
        if end is None:
            end = self.get_length()
        else:
            end = _fix_negative_index(end, self.get_length())
        begin_addr = self.get_item_address(begin)
        length = arith.SubIOp(end, begin).result
        return Stdvec(
            cc.StdvecInitOp(self.value.type, begin_addr, length=length).result)

    def back(self, num_elements: Value | None = None) -> Value:
        size = self.get_length()
        if num_elements is None:
            last = arith.SubIOp(size, constant(1)).result
            return self.get_item(last)
        begin = arith.SubIOp(size, num_elements).result
        return self.get_slice(begin, size)

    def front(self, num_elements: Value | None = None) -> Value:
        zero = constant(0)
        if num_elements is None:
            return self.get_item(zero)
        return self.get_slice(zero, num_elements)

    def __contains__(self, value):
        size = self.get_length()

        result = alloca(T.i1())
        store(constant(False), result)

        def body_builder(index):
            element = self.get_item(index)
            cmp = element.__eq__(value)
            current = load(result)
            store(arith.OrIOp(current, cmp), result)

        for_loop(size, body_builder)
        return Bool(load(result))

    @property
    def elements_type(self) -> Type:
        return cc.StdvecType.getElementType(self.value.type)

    @property
    def type(self) -> Type:
        return self.value.type
