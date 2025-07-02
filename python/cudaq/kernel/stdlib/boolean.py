# ============================================================================ #
# Copyright (c) 2022 - 2025 NVIDIA Corporation & Affiliates.                   #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

from __future__ import annotations
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

from cudaq.mlir.ir import Type, Value
from cudaq.mlir.dialects import arith
from ._core import types as T
from ._core.scalars import constant, compare


@runtime_checkable
class Boolable(Protocol):
    "This protocol describes a type that can be converted to an `bool`."

    def __bool__(self):
        pass


@runtime_checkable
class ImplicitBoolable(Protocol):
    "This protocol describes a type that can be implicitly converted to an `bool`."

    def __as_bool__(self):
        pass


@dataclass(frozen=True, slots=True, init=False)
class Bool:
    value: Value

    def __init__(self, value: Value):
        assert isinstance(value, Value), f"Expected Value, got {type(value)}"
        assert value.type == T.i1(), f"Expected i1, got {value.type}"
        object.__setattr__(self, "value", value)

    # ------------------------------------------------------------------------ #
    # Implicit type conversions
    # ------------------------------------------------------------------------ #

    def __as_bool__(self):
        return self

    # ------------------------------------------------------------------------ #
    # Comparison
    # ------------------------------------------------------------------------ #

    def _compare(self, other, predicate):
        from .literals.bool_literal import BoolLiteral
        from .literals.int_literal import IntLiteral
        if isinstance(other, (BoolLiteral, IntLiteral)):
            other = other.__as_bool__()
        if not isinstance(other, Bool):
            return None
        return Bool(compare(self.value, other.value, predicate))

    def __eq__(self, other):
        return self._compare(other, "eq")

    def __ne__(self, other):
        return self._compare(other, "ne")

    # ------------------------------------------------------------------------ #
    # Logical
    # ------------------------------------------------------------------------ #

    def __not__(self):
        return Bool(
            arith.XOrIOp(constant(1, self.value.type), self.value).result)

    @property
    def type(self) -> Type:
        return self.value.type
