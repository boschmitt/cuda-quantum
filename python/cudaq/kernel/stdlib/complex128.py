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
from cudaq.mlir.dialects import cc, complex as complex_dialect
from ._core import types as T
from ._core.scalars import constant


@runtime_checkable
class Complex128able(Protocol):
    "This protocol describes a type that can be converted to a `complex128`."

    def __complex128__(self) -> Complex128:
        pass


@runtime_checkable
class ImplicitComplex128able(Protocol):
    "This protocol describes a type that can be implicitly converted to a `complex128`."

    def __as_complex128__(self) -> Complex128:
        pass


@dataclass(frozen=True, slots=True, init=False)
class Complex128:
    value: Value

    def __init__(self, real, imag=None):
        if isinstance(real, Value):
            assert real.type == T.complex128(
            ), f"Expected complex128, got {real.type}"
            object.__setattr__(self, "value", real)
            return
        if isinstance(real, Complex128):
            object.__setattr__(self, "value", real.value)
            return

        if hasattr(real, '__as_float64__'):
            real = real.__as_float64__().value
        if imag is None:
            imag = constant(0, T.f64())
        elif hasattr(imag, '__as_float64__'):
            imag = imag.__as_float64__().value
        value = complex_dialect.CreateOp(T.complex128(), real, imag).result
        object.__setattr__(self, "value", value)

    # ------------------------------------------------------------------------ #
    # Explicit type conversions
    # ------------------------------------------------------------------------ #

    def __complex64__(self):
        from .complex64 import Complex64
        real = complex_dialect.ReOp(self.value).result
        imag = complex_dialect.ImOp(self.value).result
        real = cc.CastOp(T.f32(), real).result
        imag = cc.CastOp(T.f32(), imag).result
        return Complex64(
            complex_dialect.CreateOp(T.complex64(), real, imag).result)

    def __complex128__(self) -> Complex128:
        return self

    # ------------------------------------------------------------------------ #
    # Implicit type conversions
    # ------------------------------------------------------------------------ #

    def __as_bool__(self):
        zero = constant(0, self.value.type)
        return self.__ne__(zero)

    def __as_complex64__(self):
        return self.__complex64__()

    def __as_complex128__(self) -> Complex128:
        return self

    # ------------------------------------------------------------------------ #
    # Comparison
    # ------------------------------------------------------------------------ #

    def __eq__(self, other):
        from .boolean import Bool
        return Bool(complex_dialect.EqualOp(self.value, other.value).result)

    def __ne__(self, other):
        from .boolean import Bool
        return Bool(complex_dialect.NotEqualOp(self.value, other.value).result)

    # ------------------------------------------------------------------------ #
    # Arithmetic
    # ------------------------------------------------------------------------ #

    def _binary_op(self, other, optor, *, rhs=False):
        if not hasattr(other, '__as_complex128__'):
            return None
        other = other.__as_complex128__()
        if rhs:
            return Complex128(optor(other.value, self.value).result)
        return Complex128(optor(self.value, other.value).result)

    def __add__(self, other):
        return self._binary_op(other, complex_dialect.AddOp)

    def __radd__(self, other):
        return self._binary_op(other, complex_dialect.AddOp, rhs=True)

    def __sub__(self, other):
        return self._binary_op(other, complex_dialect.SubOp)

    def __rsub__(self, other):
        return self._binary_op(other, complex_dialect.SubOp, rhs=True)

    def __mul__(self, other):
        return self._binary_op(other, complex_dialect.MulOp)

    def __rmul__(self, other):
        return self._binary_op(other, complex_dialect.MulOp, rhs=True)

    def __truediv__(self, other):
        return self._binary_op(other, complex_dialect.DivOp)

    def __rtruediv__(self, other):
        return self._binary_op(other, complex_dialect.DivOp, rhs=True)

    def __neg__(self):
        return complex_dialect.NegOp(self.value).result

    @property
    def real(self):
        from .float64 import Float64
        return Float64(complex_dialect.ReOp(self.value).result)

    @property
    def imag(self):
        from .float64 import Float64
        return Float64(complex_dialect.ImOp(self.value).result)

    @property
    def type(self) -> Type:
        return self.value.type
