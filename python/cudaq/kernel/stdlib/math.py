# ============================================================================ #
# Copyright (c) 2022 - 2025 NVIDIA Corporation & Affiliates.                   #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

from cudaq.mlir.dialects import math, complex as complex_dialect
from .complex64 import Complex64
from .complex128 import Complex128
from .float64 import Float64
from .int64 import Int64


def cos(value):
    """
    Compute the cosine of a value. If the input is a complex number, returns the complex cosine.
    Otherwise returns the real cosine.

    NOTE: This function follows the NumPy convetion of requiring 64-bit floating point numbers.
    Thus the input value is cast to f64 (or complex128) before computing the cosine.

    Args:
        value: The input value to compute cosine of. Can be real or complex.

    Returns:
        The cosine of the input value. For complex inputs, returns complex128 type.
        For real inputs, returns float64 type.
    """
    if isinstance(value, (Complex64, Complex128)):
        return Complex128(
            complex_dialect.CosOp(value.__as_complex128__().value).result)
    return Float64(math.CosOp(value.__as_float64__().value).result)


def sin(value):
    """
    Compute the sine of a value. If the input is a complex number, returns the complex sine.
    Otherwise returns the real sine.

    NOTE: This function follows the NumPy convetion of requiring 64-bit floating point numbers.
    Thus the input value is cast to f64 (or complex128) before computing the sine.

    Args:
        value: The input value to compute cosine of. Can be real or complex.

    Returns:
        The cosine of the input value. For complex inputs, returns complex128 type.
        For real inputs, returns float64 type.
    """
    if isinstance(value, (Complex64, Complex128)):
        return Complex128(
            complex_dialect.SinOp(value.__as_complex128__().value).result)
    return Float64(math.SinOp(value.__as_float64__().value).result)


def sqrt(value):
    """
    Compute the square root of a value. If the input is a complex number, returns the complex square root.
    Otherwise returns the real square root.

    NOTE: This function follows the NumPy convetion of requiring 64-bit floating point numbers.
    Thus the input value is cast to f64 (or complex128) before computing the square root.

    Args:
        value: The input value to compute cosine of. Can be real or complex.

    Returns:
        The square root of the input value. For complex inputs, returns complex128 type.
        For real inputs, returns float64 type.
    """
    if isinstance(value, (Complex64, Complex128)):
        return complex_dialect.SqrtOp(value.__as_complex128__().value).result
    return math.SqrtOp(value.__as_float64__().value).result


def exp(value):
    """
    Compute the exponential of a value. If the input is a complex number, returns the complex exponential.
    Otherwise returns the real exponential.

    NOTE: This function follows the NumPy convetion of requiring 64-bit floating point numbers.
    Thus the input value is cast to f64 (or complex128) before computing the exponential.

    Args:
        value: The input value to compute cosine of. Can be real or complex.

    Returns:
        The exponential of the input value. For complex inputs, returns complex128 type.
        For real inputs, returns float64 type.
    """
    if isinstance(value, (Complex64, Complex128)):
        value = value.__as_complex128__().value
        #return Complex128(complex_dialect.ExpOp(value).result)
        # Note: using `complex.ExpOp` results in a  "can't legalize `complex.exp`"
        # error. Using Euler's' formula instead:
        #
        # "e^(x+i*y) = (e^x) * (cos(y)+i*sin(y))"

        real = complex_dialect.ReOp(value).result
        # Need to convert here, otherwise the result will be f64.
        left = Float64(math.ExpOp(real).result).__as_complex128__().value

        im = complex_dialect.ImOp(value).result
        re = math.CosOp(im).result
        im = math.SinOp(im).result
        right = complex_dialect.CreateOp(value.type, re, im).result
        return Complex128(complex_dialect.MulOp(left, right).result)
    return Float64(math.ExpOp(value.__as_float64__().value).result)


def ceil(value):
    if isinstance(value, (Complex64, Complex128)):
        raise ValueError(
            "numpy call (ceil) is not supported for complex numbers")
    result = Float64(math.CeilOp(value.__as_float64__().value).result)
    return Int64(result)
