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
from cudaq.mlir.dialects import arith, cc, math
from ._core import types as T
from ._core.scalars import compare


@runtime_checkable
class Int64able(Protocol):
    "This protocol describes a type that can be converted to an `int`."

    def __int64__(self):
        pass


@runtime_checkable
class ImplicitInt64able(Protocol):
    "This protocol describes a type that can be implicitly converted to an `int`."

    def __as_int64__(self):
        pass


@dataclass(frozen=True, slots=True, init=False)
class Int64:
    value: Value

    def __init__(self, value: Value | Int64able):
        assert isinstance(
            value,
            (Value,
             Int64able)), f"Expected Value or Int64able, got {type(value)}"
        if isinstance(value, Int64):
            value = value.value
        if isinstance(value, Int64able):
            value = value.__int64__().value
        assert value.type == T.i64(), f"Expected i64, got {value.type}"
        object.__setattr__(self, "value", value)

    # ------------------------------------------------------------------------ #
    # Explicit type conversions
    # ------------------------------------------------------------------------ #

    def __int64__(self) -> Int64:
        return self

    def __int32__(self):
        # TODO: Must give an warning.
        from .int32 import Int32
        return Int32(cc.CastOp(T.i32(), self.value).result)

    def __float32__(self):
        from .float32 import Float32
        return Float32(
            cc.CastOp(T.f32(), self.value, sint=True, zint=False).result)

    def __float64__(self):
        from .float64 import Float64
        return Float64(
            cc.CastOp(T.f64(), self.value, sint=True, zint=False).result)

    def __complex64__(self):
        from .complex64 import Complex64
        return Complex64(self.__float32__())

    def __complex128__(self):
        from .complex128 import Complex128
        return Complex128(self.__float64__())

    # ------------------------------------------------------------------------ #
    # Implicit type conversions
    # ------------------------------------------------------------------------ #

    def __as_bool__(self):
        from .literals import IntLiteral
        zero = IntLiteral(0)
        return self.__ne__(zero)

    def __as_int64__(self):
        return self.__int64__()

    def __as_int32__(self):
        # TODO: Must give an warning.
        return self.__int32__()

    def __as_float32__(self):
        return self.__float32__()

    def __as_float64__(self):
        return self.__float64__()

    def __as_complex64__(self):
        return self.__complex64__()

    def __as_complex128__(self):
        return self.__complex128__()

    # ------------------------------------------------------------------------ #
    # Comparison
    # ------------------------------------------------------------------------ #

    def _compare(self, other, predicate: str):
        from .literals import IntLiteral
        if isinstance(other, IntLiteral):
            other = other.__int64__()
        if not isinstance(other, Int64):
            return None

        from .boolean import Bool
        return Bool(compare(self.value, other.value, predicate))

    def __eq__(self, other):
        return self._compare(other, "eq")

    def __ne__(self, other):
        return self._compare(other, "ne")

    def __lt__(self, other):
        return self._compare(other, "lt")

    def __le__(self, other):
        return self._compare(other, "le")

    def __gt__(self, other):
        return self._compare(other, "gt")

    def __ge__(self, other):
        return self._compare(other, "ge")

    # ------------------------------------------------------------------------ #
    # Unary arithmetic
    # ------------------------------------------------------------------------ #

    def __neg__(self):
        from .literals import IntLiteral
        zero = IntLiteral(0).__int64__()
        return zero.__sub__(self)

    # ------------------------------------------------------------------------ #
    # Left-hand side arithmetic  (self.__op__(other))
    # ------------------------------------------------------------------------ #

    def _binary_op(self, other, optor, *, rhs=False):
        if not isinstance(other, ImplicitInt64able):
            return None
        other = other.__as_int64__()
        if rhs:
            return Int64(optor(other.value, self.value).result)
        return Int64(optor(self.value, other.value).result)

    def __add__(self, other):
        return self._binary_op(other, arith.AddIOp)

    def __sub__(self, other):
        return self._binary_op(other, arith.SubIOp)

    def __mul__(self, other):
        return self._binary_op(other, arith.MulIOp)

    def __truediv__(self, other):
        return self._binary_op(other, arith.DivSIOp)

    def __mod__(self, other):
        return self._binary_op(other, arith.RemSIOp)

    def __floordiv__(self, other):
        return self._binary_op(other, arith.FloorDivSIOp)

    def __pow__(self, other):
        return self._binary_op(other, math.IPowIOp)

    # Binary bitwise

    def __lshift__(self, other):
        return self._binary_op(other, arith.ShLIOp)

    def __rshift__(self, other):
        return self._binary_op(other, arith.ShRSIOp)

    def __and__(self, other):
        return self._binary_op(other, arith.AndIOp)

    def __or__(self, other):
        return self._binary_op(other, arith.OrIOp)

    def __xor__(self, other):
        return self._binary_op(other, arith.XOrIOp)

    # ------------------------------------------------------------------------ #
    # Right-hand side arithmetic  (other.__rop__(self))
    # ------------------------------------------------------------------------ #

    def __radd__(self, other):
        return self._binary_op(other, arith.AddIOp, rhs=True)

    def __rsub__(self, other):
        return self._binary_op(other, arith.SubIOp, rhs=True)

    def __rmul__(self, other):
        return self._binary_op(other, arith.MulIOp, rhs=True)

    def __rtruediv__(self, other):
        return self._binary_op(other, arith.DivSIOp, rhs=True)

    def __rmod__(self, other):
        return self._binary_op(other, arith.RemSIOp, rhs=True)

    def __rfloordiv__(self, other):
        return self._binary_op(other, arith.FloorDivSIOp, rhs=True)

    def __rpow__(self, other):
        return self._binary_op(other, math.IPowIOp, rhs=True)

    # ------------------------------------------------------------------------ #
    # In-place arithmetic
    # ------------------------------------------------------------------------ #

    def __iadd__(self, other):
        result = self.__add__(other) or other.__radd__(self)
        if isinstance(result, ImplicitInt64able):
            return result.__as_int64__()
        return None

    def __isub__(self, other):
        result = self.__sub__(other) or other.__rsub__(self)
        if isinstance(result, ImplicitInt64able):
            return result.__as_int64__()
        return None

    def __imul__(self, other):
        result = self.__mul__(other) or other.__rmul__(self)
        if isinstance(result, ImplicitInt64able):
            return result.__as_int64__()
        return None

    def __itruediv__(self, other):
        result = self.__truediv__(other) or other.__rtruediv__(self)
        if isinstance(result, ImplicitInt64able):
            return result.__as_int64__()
        return None

    def __imod__(self, other):
        result = self.__mod__(other) or other.__rmod__(self)
        if isinstance(result, ImplicitInt64able):
            return result.__as_int64__()
        return None

    def __ifloordiv__(self, other):
        result = self.__floordiv__(other) or other.__rfloordiv__(self)
        if isinstance(result, ImplicitInt64able):
            return result.__as_int64__()
        return None

    def __ilshift__(self, other):
        result = self.__lshift__(other) or other.__ilshift__(self)
        if isinstance(result, ImplicitInt64able):
            return result.__as_int64__()
        return None

    def __irshift__(self, other):
        result = self.__rshift__(other) or other.__irshift__(self)
        if isinstance(result, ImplicitInt64able):
            return result.__as_int64__()
        return None

    def __iand__(self, other):
        result = self.__and__(other) or other.__iand__(self)
        if isinstance(result, ImplicitInt64able):
            return result.__as_int64__()
        return None

    def __ior__(self, other):
        result = self.__or__(other) or other.__ior__(self)
        if isinstance(result, ImplicitInt64able):
            return result.__as_int64__()
        return None

    def __ixor__(self, other):
        result = self.__xor__(other) or other.__ixor__(self)
        if isinstance(result, ImplicitInt64able):
            return result.__as_int64__()
        return None

    @property
    def type(self) -> Type:
        return self.value.type
