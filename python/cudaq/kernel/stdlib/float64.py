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
class Float64able(Protocol):
    "This protocol describes a type that can be converted to a `float64`."

    def __float64__(self) -> Float64:
        pass


@runtime_checkable
class ImplicitFloat64able(Protocol):
    "This protocol describes a type that can be implicitly converted to a `float64`."

    def __as_float64__(self) -> Float64:
        pass


@dataclass(frozen=True, slots=True, init=False)
class Float64:
    value: Value

    def __init__(self, value: Value | Float64able):
        assert isinstance(
            value,
            (Value,
             Float64able)), f"Expected Value or Float64able, got {type(value)}"
        if isinstance(value, Float64):
            value = value.value
        if isinstance(value, Float64able):
            value = value.__float64__().value
        assert value.type == T.f64(), f"Expected f64, got {value.type}"
        object.__setattr__(self, "value", value)

    # ------------------------------------------------------------------------ #
    # Explicit type conversions
    # ------------------------------------------------------------------------ #

    def __complex64__(self):
        from .complex64 import Complex64
        return Complex64(self)

    def __complex128__(self):
        from .complex128 import Complex128
        return Complex128(self)

    def __float64__(self):
        return self

    def __float32__(self):
        from .float32 import Float32
        return Float32(cc.CastOp(T.f32(), self.value).result)

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

    def __as_bool__(self):
        zero = constant(0, self.value.type)
        return self.__ne__(zero)

    def __as_complex64__(self):
        # TODO: Must give an warning.
        return self.__complex64__()

    def __as_complex128__(self):
        return self.__complex128__()

    def __as_float32__(self):
        return self.__float32__()

    def __as_float64__(self):
        return self

    # ------------------------------------------------------------------------ #
    # Comparison
    # ------------------------------------------------------------------------ #

    def __eq__(self, other: Float64):
        from .boolean import Bool
        return Bool(compare(self.value, other.value, "eq"))

    def __ne__(self, other: Float64):
        from .boolean import Bool
        return Bool(compare(self.value, other.value, "ne"))

    # ------------------------------------------------------------------------ #
    # Unary arithmetic
    # ------------------------------------------------------------------------ #

    def __neg__(self):
        return Float64(arith.NegFOp(self.value).result)

    # ------------------------------------------------------------------------ #
    # Left-hand side arithmetic  (self.__op__(other))
    # ------------------------------------------------------------------------ #

    def _binary_op(self, other, optor, *, rhs=False):
        if not hasattr(other, '__as_float64__'):
            return None
        other = other.__as_float64__()
        if rhs:
            return Float64(optor(other.value, self.value).result)
        return Float64(optor(self.value, other.value).result)

    def __add__(self, other):
        return self._binary_op(other, arith.AddFOp)

    def __sub__(self, other):
        return self._binary_op(other, arith.SubFOp)

    def __mul__(self, other):
        return self._binary_op(other, arith.MulFOp)

    def __truediv__(self, other):
        return self._binary_op(other, arith.DivFOp)

    # ------------------------------------------------------------------------ #
    # Right-hand side arithmetic  (other.__rop__(self))
    # ------------------------------------------------------------------------ #

    def __radd__(self, other):
        return self._binary_op(other, arith.AddFOp, rhs=True)

    def __rsub__(self, other):
        return self._binary_op(other, arith.SubFOp, rhs=True)

    def __rmul__(self, other):
        return self._binary_op(other, arith.MulFOp, rhs=True)

    def __rtruediv__(self, other):
        return self._binary_op(other, arith.DivFOp, rhs=True)

    # ------------------------------------------------------------------------ #
    # In-place arithmetic
    # ------------------------------------------------------------------------ #

    def __iadd__(self, other):
        result = self.__add__(other) or other.__radd__(self)
        if hasattr(result, '__as_float64__'):
            return result.__as_float64__()
        return None

    def __isub__(self, other):
        result = self.__sub__(other) or other.__rsub__(self)
        if hasattr(result, '__as_float64__'):
            return result.__as_float64__()
        return None

    def __imul__(self, other):
        result = self.__mul__(other) or other.__rmul__(self)
        if hasattr(result, '__as_float64__'):
            return result.__as_float64__()
        return None

    def __itruediv__(self, other):
        result = self.__truediv__(other) or other.__rtruediv__(self)
        if hasattr(result, '__as_float64__'):
            return result.__as_float64__()
        return None

    @property
    def type(self) -> Type:
        return self.value.type
