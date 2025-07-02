# ============================================================================ #
# Copyright (c) 2022 - 2025 NVIDIA Corporation & Affiliates.                   #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

from __future__ import annotations
from dataclasses import dataclass

from cudaq.mlir.ir import Type, Value
from .._core import types as T
from .._base import AbstractValue, Iterable
from ..int64 import Int64able
from ..literals.int_literal import IntLiteral


@dataclass(frozen=True, slots=True, init=False)
class Enumerate(AbstractValue, Iterable):
    iterable: Iterable
    start: Value | Int64able

    def __init__(self, iterable, start: Value | Int64able | None = None):
        assert isinstance(iterable, Iterable)
        object.__setattr__(self, 'iterable', iterable)
        object.__setattr__(self, 'start', start or IntLiteral(0))

    def get_length(self) -> Value:
        return self.iterable.get_length()

    def get_item(self, index: Int64able) -> list[Value]:
        index = index.__as_int64__()
        start = self.start.__as_int64__()
        base_values = self.iterable.get_item(index)
        index = index.__add__(start)
        return [index, base_values]

    @property
    def elements_type(self) -> Type:
        element_type = self.iterable.elements_type
        return T.struct([T.i64(), element_type])

    def __repr__(self):
        return f"Enumerate(iterable={self.iterable}, start={self.start})"

    @property
    def type(self):
        return Enumerate
