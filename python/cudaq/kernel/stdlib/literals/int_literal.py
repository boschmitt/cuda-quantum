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
from .._core.scalars import constant, implicit_conversion


@dataclass(frozen=True, slots=True)
class IntLiteral(Literal):
    value: int

    # ------------------------------------------------------------------------ #
    # Explicit type conversions
    # ------------------------------------------------------------------------ #

    def __int64__(self):
        from ..int64 import Int64
        return Int64(self.materialize(with_type=T.i64()))

    def __int32__(self):
        from ..int32 import Int32
        return Int32(self.materialize(with_type=T.i32()))

    def __float32__(self):
        from ..float32 import Float32
        return Float32(self.materialize(with_type=T.f32()))

    def __float64__(self):
        from ..float64 import Float64
        return Float64(self.materialize(with_type=T.f64()))

    # ------------------------------------------------------------------------ #
    # Implicit type conversions
    # ------------------------------------------------------------------------ #

    def __as_bool__(self):
        from ..boolean import Bool
        return Bool(self.materialize(with_type=T.i1()))

    def __as_int64__(self):
        return self.__int64__()

    def __as_int32__(self):
        return self.__int32__()

    def __as_float32__(self):
        return self.__float32__()

    def __as_float64__(self):
        return self.__float64__()

    # ------------------------------------------------------------------------ #
    # Unary arithmetic
    # ------------------------------------------------------------------------ #

    def __neg__(self):
        return IntLiteral(-self.value)

    # ------------------------------------------------------------------------ #
    # Left-hand side arithmetic  (self.__op__(other))
    # ------------------------------------------------------------------------ #

    def __add__(self, other):
        if isinstance(other, IntLiteral):
            return IntLiteral(self.value + other.value)
        return None

    def __sub__(self, other):
        if isinstance(other, IntLiteral):
            return IntLiteral(self.value - other.value)
        return None

    def __mul__(self, other):
        if isinstance(other, IntLiteral):
            return IntLiteral(self.value * other.value)
        return None

    def __truediv__(self, other):
        if isinstance(other, IntLiteral):
            return IntLiteral(self.value / other.value)
        return None

    def __mod__(self, other):
        if isinstance(other, IntLiteral):
            return IntLiteral(self.value % other.value)
        return None

    def __floordiv__(self, other):
        if isinstance(other, IntLiteral):
            return IntLiteral(self.value // other.value)
        return None

    def __pow__(self, other):
        if isinstance(other, IntLiteral):
            return IntLiteral(self.value**other.value)
        return None

    # Bitwise operations

    def __lshift__(self, other):
        if isinstance(other, IntLiteral):
            return IntLiteral(self.value << other.value)
        return None

    def __rshift__(self, other):
        if isinstance(other, IntLiteral):
            return IntLiteral(self.value >> other.value)
        return None

    def __and__(self, other):
        if isinstance(other, IntLiteral):
            return IntLiteral(self.value & other.value)
        return None

    def __or__(self, other):
        if isinstance(other, IntLiteral):
            return IntLiteral(self.value | other.value)
        return None

    def __xor__(self, other):
        if isinstance(other, IntLiteral):
            return IntLiteral(self.value ^ other.value)
        return None

    # ------------------------------------------------------------------------ #
    # Right-hand side arithmetic  (other.__rop__(self))
    # ------------------------------------------------------------------------ #

    def __radd__(self, other):
        return None

    def __rsub__(self, other):
        return None

    def __rmul__(self, other):
        return None

    def __rtruediv__(self, other):
        return None

    def __rmod__(self, other):
        return None

    def __rfloordiv__(self, other):
        return None

    def __rpow__(self, other):
        return None

    def __rlshift__(self, other):
        return None

    def __rrshift__(self, other):
        return None

    def __rand__(self, other):
        return None

    def __ror__(self, other):
        return None

    def __rxor__(self, other):
        return None

    # ------------------------------------------------------------------------ #
    # Materialization
    # ------------------------------------------------------------------------ #

    def materialize(self, *, with_type: Type | None = None) -> Value:
        if with_type is None:
            return constant(self.value, T.i64())
        if with_type == T.i1():
            return constant(self.value != 0, with_type)
        if T.is_integer_like_type(with_type):
            return constant(self.value, with_type)
        if T.is_float_type(with_type):
            return constant(float(self.value), with_type)
        if T.is_complex_type(with_type):
            return constant(complex(self.value), with_type)
        raise ValueError(f"Cannot convert {self.value} to {with_type}")

    @property
    def type(self):
        return IntLiteral
