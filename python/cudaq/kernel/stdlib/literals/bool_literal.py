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
from .._base import Literal
from .._core import types as T
from .._core.scalars import constant


@dataclass(frozen=True, slots=True)
class BoolLiteral(Literal):
    value: bool

    def __as_bool__(self):
        from ..boolean import Bool
        return Bool(self.materialize(with_type=T.i1()))

    def materialize(self, *, with_type: Type | None = None) -> Value:
        if with_type is not None:
            return constant(self.value, with_type)
        return constant(self.value, T.i1())

    @property
    def type(self):
        return BoolLiteral
