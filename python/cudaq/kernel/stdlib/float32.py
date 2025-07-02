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
from cudaq.mlir.dialects import arith, cc
from ._core import types as T
from ._core.scalars import constant, compare


@runtime_checkable
class Float32able(Protocol):
    "This protocol describes a type that can be converted to a `float32`."

    def __float32__(self) -> Float32:
        pass


@runtime_checkable
class ImplicitFloat32able(Protocol):
    "This protocol describes a type that can be implicitly converted to a `float32`."

    def __float32__(self) -> Float32:
        pass


@dataclass(frozen=True, slots=True, init=False)
class Float32:
    value: Value

    def __init__(self, value: Value | Float32able):
        assert isinstance(
            value,
            (Value,
             Float32able)), f"Expected Value or Float32able, got {type(value)}"
        if isinstance(value, Float32):
            value = value.value
        if isinstance(value, Float32able):
            value = value.__float32__().value
        assert value.type == T.f32(), f"Expected f32, got {value.type}"
        object.__setattr__(self, "value", value)

    # ------------------------------------------------------------------------ #
    # Explicit type conversions
    # ------------------------------------------------------------------------ #

    def __float32__(self) -> Float32:
        return self

    def __float64__(self):
        from .float64 import Float64
        return Float64(cc.CastOp(T.f64(), self.value).result)

    def __int32__(self):
        from .int32 import Int32
        return Int32(
            cc.CastOp(T.i32(), self.value, sint=True, zint=False).result)

    def __int64__(self):
        from .int64 import Int64
        return Int64(
            cc.CastOp(T.i64(), self.value, sint=True, zint=False).result)

    # ------------------------------------------------------------------------ #
    # Implicit type conversions
    # ------------------------------------------------------------------------ #

    def __bool__(self):
        zero = constant(0, self.value.type)
        return self.__eq__(zero)

    def __as_float32__(self) -> Float32:
        return self

    def __as_float64__(self):
        return self.__float64__()

    # Equality
    def __eq__(self, other: Float32):
        from .boolean import Bool
        return Bool(compare(self.value, other.value, "eq"))

    def __ne__(self, other: Float32):
        from .boolean import Bool
        return Bool(compare(self.value, other.value, "ne"))

    # Arithmetic
    def __neg__(self):
        return arith.NegFOp(self.value).result

    @property
    def type(self) -> Type:
        return self.value.type
