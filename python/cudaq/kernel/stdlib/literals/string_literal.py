# ============================================================================ #
# Copyright (c) 2022 - 2025 NVIDIA Corporation & Affiliates.                   #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

from __future__ import annotations
from dataclasses import dataclass

from cudaq.mlir.dialects import cc
from cudaq.mlir.ir import StringAttr, Type, Value
from .._core import types as T
from .._base import Literal


@dataclass(frozen=True, slots=True)
class StringLiteral(Literal):
    value: str

    def materialize(self, *, with_type: Type | None = None) -> Value:
        type = T.ptr(T.array(T.i8(), len(self.value) + 1))
        return cc.CreateStringLiteralOp(type, StringAttr.get(self.value)).result

    @property
    def type(self):
        return StringLiteral
