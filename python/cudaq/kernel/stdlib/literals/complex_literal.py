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
from ..complex128 import ImplicitComplex128able


@dataclass(frozen=True, slots=True)
class ComplexLiteral(Literal):
    value: complex

    def real(self):
        from .float_literal import FloatLiteral
        return FloatLiteral(self.value.real)

    def imag(self):
        from .float_literal import FloatLiteral
        return FloatLiteral(self.value.imag)

    # ------------------------------------------------------------------------ #
    # Explicit type conversions
    # ------------------------------------------------------------------------ #

    def __complex64__(self):
        from ..complex64 import Complex64
        return Complex64(self.materialize(with_type=T.complex64()))

    def __complex128__(self):
        from ..complex128 import Complex128
        return Complex128(self.materialize(with_type=T.complex128()))

    # ------------------------------------------------------------------------ #
    # Implicit type conversions
    # ------------------------------------------------------------------------ #

    def __as_bool__(self):
        from ..boolean import Bool
        return Bool(self.materialize(with_type=T.i1()))

    def __as_complex64__(self):
        return self.__complex64__()

    def __as_complex128__(self):
        return self.__complex128__()

    # ------------------------------------------------------------------------ #
    # Unary arithmetic
    # ------------------------------------------------------------------------ #

    def __neg__(self):
        return ComplexLiteral(-self.value)

    # ------------------------------------------------------------------------ #
    # Left-hand side arithmetic  (self.__op__(other))
    # ------------------------------------------------------------------------ #

    def _is_promotable_literal(self, other):
        from .int_literal import IntLiteral
        from .float_literal import FloatLiteral
        if isinstance(other, (FloatLiteral, IntLiteral, ComplexLiteral)):
            return True
        return False

    def _try_complex128_binary_op(self, other, op: str):
        if not isinstance(other, ImplicitComplex128able):
            return None
        value = self.__as_complex128__()
        other = other.__as_complex128__()
        return getattr(value, f"__{op}__")(other)

    def __add__(self, other):
        if self._is_promotable_literal(other):
            return ComplexLiteral(self.value + complex(other.value))
        return self._try_complex128_binary_op(other, "add")

    def __sub__(self, other):
        if self._is_promotable_literal(other):
            return ComplexLiteral(self.value - complex(other.value))
        return self._try_complex128_binary_op(other, "sub")

    def __mul__(self, other):
        if self._is_promotable_literal(other):
            return ComplexLiteral(self.value * complex(other.value))
        return self._try_complex128_binary_op(other, "mul")

    def __truediv__(self, other):
        if self._is_promotable_literal(other):
            return ComplexLiteral(self.value / complex(other.value))
        return self._try_complex128_binary_op(other, "truediv")

    def __mod__(self, other):
        if self._is_promotable_literal(other):
            return ComplexLiteral(self.value % complex(other.value))
        return self._try_complex128_binary_op(other, "mod")

    def __floordiv__(self, other):
        if self._is_promotable_literal(other):
            return ComplexLiteral(self.value // complex(other.value))
        return self._try_complex128_binary_op(other, "floordiv")

    def __pow__(self, other):
        if self._is_promotable_literal(other):
            return ComplexLiteral(self.value**complex(other.value))
        return self._try_complex128_binary_op(other, "pow")

    # Bitwise operations

    def __lshift__(self, other):
        return None

    def __rshift__(self, other):
        return None

    def __and__(self, other):
        return None

    def __or__(self, other):
        return None

    def __xor__(self, other):
        return None

    # ------------------------------------------------------------------------ #
    # Right-hand side arithmetic  (other.__rop__(self))
    # ------------------------------------------------------------------------ #

    def __radd__(self, other):
        if self._is_promotable_literal(other):
            return ComplexLiteral(complex(other.value) + self.value)
        return self._try_complex128_binary_op(other, "radd")

    def __rsub__(self, other):
        if self._is_promotable_literal(other):
            return ComplexLiteral(complex(other.value) - self.value)
        return self._try_complex128_binary_op(other, "rsub")

    def __rmul__(self, other):
        if self._is_promotable_literal(other):
            return ComplexLiteral(complex(other.value) * self.value)
        return self._try_complex128_binary_op(other, "rmul")

    def __rtruediv__(self, other):
        if self._is_promotable_literal(other):
            return ComplexLiteral(complex(other.value) / self.value)
        return self._try_complex128_binary_op(other, "rtruediv")

    def __rmod__(self, other):
        if self._is_promotable_literal(other):
            return ComplexLiteral(complex(other.value) % self.value)
        return self._try_complex128_binary_op(other, "rmod")

    def __rfloordiv__(self, other):
        if self._is_promotable_literal(other):
            return ComplexLiteral(complex(other.value) // self.value)
        return self._try_complex128_binary_op(other, "rfloordiv")

    def __rpow__(self, other):
        if self._is_promotable_literal(other):
            return ComplexLiteral(complex(other.value)**self.value)
        return self._try_complex128_binary_op(other, "rpow")

    # ------------------------------------------------------------------------ #
    # Materialization
    # ------------------------------------------------------------------------ #

    def materialize(self, *, with_type: Type | None = None) -> Value:
        if with_type is None:
            return constant(self.value, T.complex128())
        if with_type == T.i1():
            return constant(self.value != complex(0, 0), with_type)
        if T.is_complex_type(with_type):
            return constant(complex(self.value), with_type)
        raise ValueError(f"Cannot convert {self.value} to {with_type}")

    @property
    def type(self):
        return ComplexLiteral
