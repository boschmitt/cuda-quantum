# ============================================================================ #
# Copyright (c) 2022 - 2025 NVIDIA Corporation & Affiliates.                   #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

from enum import IntEnum
from cudaq.mlir.dialects import arith, cc, complex as complex_dialect
from cudaq.mlir.ir import Type, Value, IntegerAttr, FloatAttr, StringAttr, IntegerType, F32Type, F64Type, ComplexType
from .._core import types as T


def constant(value: T.InferableType, ir_type: Type | None = None) -> Value:
    if ir_type is None:
        ir_type = T.infer_type(value)

    if T.is_complex_type(ir_type):
        element_type = ComplexType(ir_type).element_type
        re = constant(value.real, element_type)
        im = constant(value.imag, element_type)
        return complex_dialect.CreateOp(ir_type, re, im).result

    if isinstance(value, str):
        return cc.CreateStringLiteralOp(
            ir_type,
            StringAttr.get(value),
        ).result

    if IntegerType.isinstance(ir_type):
        value = IntegerAttr.get(ir_type, value)
    if F32Type.isinstance(ir_type):
        value = FloatAttr.get(ir_type, value)
    if F64Type.isinstance(ir_type):
        value = FloatAttr.get(ir_type, value)

    return arith.ConstantOp(ir_type, value).result


# ---------------------------------------------------------------------------- #
#  Comparison
# ---------------------------------------------------------------------------- #


class _CmpIPredicate(IntEnum):
    eq = 0  # equal
    ne = 1  # not equal
    slt = 2  # signed less than
    sle = 3  # signed less than or equal
    sgt = 4  # signed greater than
    sge = 5  # signed greater than or equal
    ult = 6  # unsigned less than
    ule = 7  # unsigned less than or equal
    ugt = 8  # unsigned greater than
    uge = 9  # unsigned greater than or equal


def _cmpi_predicate_attr(predicate):
    match predicate:
        case "eq":
            return IntegerAttr.get(T.i64(), _CmpIPredicate.eq)
        case "ne":
            return IntegerAttr.get(T.i64(), _CmpIPredicate.ne)
        case "lt":
            return IntegerAttr.get(T.i64(), _CmpIPredicate.slt)
        case "le":
            return IntegerAttr.get(T.i64(), _CmpIPredicate.sle)
        case "gt":
            return IntegerAttr.get(T.i64(), _CmpIPredicate.sgt)
        case "ge":
            return IntegerAttr.get(T.i64(), _CmpIPredicate.sge)
        case _:
            return None


class _CmpFPredicate(IntEnum):
    alwaysfalse = 0
    # An ordered comparison checks if neither operand is NaN.
    oeq = 1
    ogt = 2
    oge = 3
    olt = 4
    ole = 5
    one = 6
    ord = 7
    # An unordered comparison checks if either operand is a NaN.
    ueq = 8
    ugt = 9
    uge = 10
    ult = 11
    ule = 12
    une = 13
    uno = 14
    alwaystrue = 15


def _cmpf_predicate_attr(predicate):
    match predicate:
        case "eq":
            return IntegerAttr.get(T.i64(), _CmpFPredicate.oeq)
        case "ne":
            return IntegerAttr.get(T.i64(), _CmpFPredicate.one)
        case "lt":
            return IntegerAttr.get(T.i64(), _CmpFPredicate.olt)
        case "le":
            return IntegerAttr.get(T.i64(), _CmpFPredicate.ole)
        case "gt":
            return IntegerAttr.get(T.i64(), _CmpFPredicate.ogt)
        case "ge":
            return IntegerAttr.get(T.i64(), _CmpFPredicate.oge)
        case _:
            return None


def compare(right: Value, left: Value, predicate: str) -> Value:
    assert right.type == left.type, f"Expected {right.type} == {left.type}"
    if T.is_integer_like_type(right.type):
        predicate = _cmpi_predicate_attr(predicate)
        assert predicate is not None, f"Invalid predicate: {predicate}"
        return arith.CmpIOp(predicate, right, left).result
    if T.is_float_type(right.type):
        predicate = _cmpf_predicate_attr(predicate)
        assert predicate is not None, f"Invalid predicate: {predicate}"
        return arith.CmpFOp(predicate, right, left).result
    raise ValueError(f"Unsupported type: {right.type}")


# ============================================================================ #
# Implicit Conversions
# ============================================================================ #


def implicit_conversion(value, target_type):
    if value.type == target_type:
        return value
    if IntegerType.isinstance(target_type):
        target_width = IntegerType(target_type).width
        if target_width == 1:
            zero = constant(0, value.type)
            if IntegerType.isinstance(value.type):
                return arith.CmpIOp(IntegerAttr.get(T.i64(), 1), zero,
                                    value).result
            if T.is_float_type(value.type):
                return arith.CmpFOp(IntegerAttr.get(T.i64(), 6), zero,
                                    value).result
            if ComplexType.isinstance(value.type):
                return complex_dialect.NotEqualOp(zero, value).result
            return None
        if not IntegerType.isinstance(value.type):
            return None
        return _i_coerce(value, target_width)
    if F32Type.isinstance(target_type):
        if ComplexType.isinstance(value.type):
            return None
        return f32_coerce(value)
    if F64Type.isinstance(target_type):
        if ComplexType.isinstance(value.type):
            return None
        return f64_coerce(value)
    if ComplexType.isinstance(target_type):
        if ComplexType(target_type).element_type == T.f32():
            return complex64_coerce(value)
        if ComplexType(target_type).element_type == T.f64():
            return complex128_coerce(value)
        raise ValueError(f"Unsupported complex type: {target_type}")
    return None


def arithmetic_promotion(value, target_type):
    #assert T.is_arithmetic_type(value) and T.is_arithmetic_type(target_type)
    if value.type == target_type:
        return value
    if IntegerType.isinstance(target_type):
        if IntegerType.isinstance(value.type):
            width = IntegerType(value.type).width
            target_width = IntegerType(target_type).width
            if width < target_width:
                return _i_coerce(value, target_width)
        return value
    if F32Type.isinstance(target_type):
        if IntegerType.isinstance(value.type):
            return f32_coerce(value)
        return value
    if F64Type.isinstance(target_type):
        if IntegerType.isinstance(value.type):
            return f64_coerce(value)
        if F32Type.isinstance(value.type):
            return f64_coerce(value)
        return value
    if ComplexType.isinstance(target_type):
        if ComplexType(target_type).element_type == T.f32():
            if IntegerType.isinstance(value.type):
                return complex64_coerce(value)
            if F32Type.isinstance(value.type):
                return complex64_coerce(value)
            if F64Type.isinstance(value.type):
                return complex64_coerce(value)
            return value
        if ComplexType(target_type).element_type == T.f64():
            return complex128_coerce(value)
        raise ValueError(f"Unsupported complex type: {target_type}")
    return None


def _i_coerce(value, target_width):
    if IntegerType.isinstance(value.type):
        width = IntegerType(value.type).width
        if width == target_width:
            return value
        if target_width < width:
            return cc.CastOp(T.i(target_width), value).result
        zeroext = width == 1
        return cc.CastOp(T.i(target_width),
                         value,
                         sint=not zeroext,
                         zint=zeroext).result
    if T.is_float_type(value.type):
        return cc.CastOp(T.i(target_width), value, sint=True, zint=False).result
    return None


def i64_coerce(value):
    return _i_coerce(value, 64)


def i32_coerce(value):
    return _i_coerce(value, 32)


def i8_coerce(value):
    return _i_coerce(value, 8)


def f64_coerce(value):
    if F64Type.isinstance(value.type):
        return value
    if F32Type.isinstance(value.type):
        return cc.CastOp(T.f64(), value).result
    if IntegerType.isinstance(value.type):
        zeroext = IntegerType(value.type).width == 1
        return cc.CastOp(T.f64(), value, sint=not zeroext, zint=zeroext).result


def f32_coerce(value):
    if F32Type.isinstance(value.type):
        return value
    if F64Type.isinstance(value.type):
        return cc.CastOp(T.f32(), value).result
    if IntegerType.isinstance(value.type):
        zeroext = IntegerType(value.type).width == 1
        return cc.CastOp(T.f32(), value, sint=not zeroext, zint=zeroext).result
    return None


def complex64_coerce(value):
    if ComplexType.isinstance(value.type):
        if ComplexType(value.type).element_type == T.f32():
            return value
        re = f32_coerce(complex_dialect.ReOp(value).result)
        im = f32_coerce(complex_dialect.ImOp(value).result)
        return complex_dialect.CreateOp(T.complex64(), re, im).result

    re = f32_coerce(value)
    im = constant(0.0, T.f32())
    return complex_dialect.CreateOp(T.complex64(), re, im).result


def complex128_coerce(value):
    if ComplexType.isinstance(value.type):
        if ComplexType(value.type).element_type == T.f64():
            return value
        re = f64_coerce(complex_dialect.ReOp(value).result)
        im = f64_coerce(complex_dialect.ImOp(value).result)
        return complex_dialect.CreateOp(T.complex128(), re, im).result

    re = f64_coerce(value)
    im = constant(0.0, T.f64())
    return complex_dialect.CreateOp(T.complex128(), re, im).result
