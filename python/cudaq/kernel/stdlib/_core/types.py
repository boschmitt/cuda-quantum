# ============================================================================ #
# Copyright (c) 2022 - 2025 NVIDIA Corporation & Affiliates.                   #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

import numpy as np

from cudaq.mlir.ir import (
    ComplexType,
    Context,
    F32Type,
    F64Type,
    IndexType,
    IntegerType,
    Type,
)
from cudaq.mlir.dialects import cc, quake

# ============================================================================ #
# Core types
# ============================================================================ #


def i(width: int) -> Type:
    return IntegerType.get_signless(width)


def cmplx(type: Type) -> Type:
    return ComplexType.get(type)


# Integer types
i1 = lambda: i(1)
i8 = lambda: i(8)
i32 = lambda: i(32)
i64 = lambda: i(64)

# Float types
f32 = lambda: F32Type.get()
f64 = lambda: F64Type.get()

# Complex types
complex64 = lambda: cmplx(f32())
complex128 = lambda: cmplx(f64())

# ============================================================================ #
# CC types
# ============================================================================ #

ptr = lambda element_type: cc.PointerType.get(element_type)


def struct(types: list[Type], name: str | None = None) -> Type:
    if name is None:
        return cc.StructType.get(types)
    return cc.StructType.getNamed(name, types)


def array(element_type: Type, size: int | None = None) -> Type:
    if size is None:
        return cc.ArrayType.get(element_type)
    return cc.ArrayType.get(element_type, size)


def stdvec(element_type: Type) -> Type:
    return cc.StdvecType.get(element_type)


# ============================================================================ #
# Quake types
# ============================================================================ #

qref = lambda: quake.RefType.get()
measurement = lambda: quake.MeasureType.get()


def veq(size: int | None = None) -> Type:
    if size is None:
        return quake.VeqType.get()
    return quake.VeqType.get(size)


def struq(types: list[Type], name: str | None = None) -> Type:
    if name is None:
        return quake.StruqType.get(types)
    return quake.StruqType.getNamed(name, types)


# ============================================================================ #
# Helper types
# ============================================================================ #

CapturableType = bool | complex | float | int | np.float32 | np.float64 | np.complex64 | np.complex128
InferableType = CapturableType | str

# ============================================================================ #
# Utility functions
# ============================================================================ #


def _isa(obj, cls):
    try:
        cls(obj)
    except ValueError:
        return False
    return True


def _is_any_of(obj, classes):
    return any(_isa(obj, cls) for cls in classes)


def is_quantum_type(type):
    return _is_any_of(type, [quake.RefType, quake.VeqType, quake.StruqType])


def is_struq_type(type):
    return _isa(type, quake.StruqType)


def is_pointer_type(type):
    return _isa(type, cc.PointerType)


def is_integer_like_type(type):
    return _is_any_of(type, [IntegerType, IndexType])


def is_float_type(type):
    return _is_any_of(type, [F32Type, F64Type])


def is_complex_type(type_: Type) -> bool:
    return _isa(type_, ComplexType)


def is_arithmetic_type(type_: Type) -> bool:
    return _is_any_of(type_, [IntegerType, F64Type, F32Type, ComplexType])


def infer_type(value: InferableType,) -> Type:
    if isinstance(value, bool):
        return i1()
    if isinstance(value, int):
        return i64()
    if isinstance(value, np.float32):
        return f32()
    if isinstance(value, (float, np.float64)):
        return f64()
    if isinstance(value, np.complex64):
        return complex64()
    if isinstance(value, (complex, np.complex128)):
        return complex128()
    if isinstance(value, str):
        return ptr(i8())
    raise ValueError(f"Unsupported type: {type(value)}")
