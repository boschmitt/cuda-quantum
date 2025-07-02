# ============================================================================ #
# Copyright (c) 2022 - 2025 NVIDIA Corporation & Affiliates.                   #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

from __future__ import annotations
from dataclasses import dataclass

from cudaq.mlir.dialects import arith, math
from cudaq.mlir.ir import Type, Value, UnitAttr
from .._base import AbstractValue, Iterable
from .._core import types as T
from .._core.control_flow import for_loop
from ..collections.stdvec import Stdvec
from ..int64 import Int64, Int64able
from ..literals.int_literal import IntLiteral


class Range(AbstractValue, Iterable):
    __slots__ = ('start', 'step', 'stop')

    def __init__(self, *args):
        self.start = None
        self.step = None
        self.stop = None
        if len(args) == 1:
            self.stop = args[0]
        elif len(args) == 2:
            self.start = args[0]
            self.stop = args[1]
        elif len(args) == 3:
            self.start = args[0]
            self.stop = args[1]
            self.step = args[2]
        else:
            raise ValueError("Range requires 1 to 3 integer arguments")

    def get_length(self) -> Value:
        # Useful helpers
        zero = IntLiteral(0)
        one = IntLiteral(1)

        start = Int64(self.start or zero)
        step = Int64(self.step or one)
        stop = Int64(self.stop)

        # We need to check for a empty range
        diff = stop - start

        # First condition: step > 0 and diff <= 0
        step_gt_zero = step > zero
        diff_le_zero = diff <= zero
        first_condition = arith.AndIOp(step_gt_zero, diff_le_zero)

        # Second condition: step < 0 and diff >= 0
        step_lt_zero = step < zero
        diff_ge_zero = diff >= zero
        second_condition = arith.AndIOp(step_lt_zero, diff_ge_zero)

        is_empty = arith.OrIOp(first_condition, second_condition)

        step = math.AbsIOp(step)
        length = arith.CeilDivSIOp(math.AbsIOp(diff).result, step).result
        return Int64(
            arith.SelectOp(is_empty, zero.materialize(), length).result)

    def get_item(self, index: Int64able) -> Int64:
        start = Int64(self.start if self.start is not None else IntLiteral(0))
        step = Int64(self.step if self.step is not None else IntLiteral(1))
        index = index.__as_int64__()
        return start.__add__(step.__mul__(index))

    @property
    def elements_type(self) -> Type:
        return T.i64()

    def materialize(self, *, with_type: Type | None = None) -> Stdvec:
        size = self.get_length()
        vec = Stdvec.create(T.i64(), size)

        def body_builder(index):
            index = Int64(index)
            value = self.get_item(index)
            vec.set_item(index, value)

        loop = for_loop(size, body_builder)
        loop.attributes['invariant'] = UnitAttr.get()
        return vec

    def __repr__(self):
        return f"Range(start={self.start}, stop={self.stop}, step={self.step})"

    @property
    def type(self):
        return Range
