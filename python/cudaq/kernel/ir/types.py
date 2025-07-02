# ============================================================================ #
# Copyright (c) 2022 - 2025 NVIDIA Corporation & Affiliates.                   #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

import numpy as np
from typing import List, Optional

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


def i(width: int, context: Optional[Context] = None) -> Type:
    return IntegerType.get_signless(width, context=context)


def cmplx(type: Type) -> Type:
    return ComplexType.get(type)


# Integer types
i1 = lambda context=None: i(1, context)
i8 = lambda context=None: i(8, context)
i32 = lambda context=None: i(32, context)
i64 = lambda context=None: i(64, context)

# Float types
f32 = lambda context=None: F32Type.get(context)
f64 = lambda context=None: F64Type.get(context)

# Complex types
complex64 = lambda context=None: cmplx(f32(context))
complex128 = lambda context=None: cmplx(f64(context))

# ============================================================================ #
# CC types
# ============================================================================ #

ptr = lambda element_type: cc.PointerType.get(element_type, element_type.context
                                             )


def struct(types: List[Type], name: Optional[str] = None) -> Type:
    context = types[0].context
    if name is None:
        return cc.StructType.get(types, context=context)
    return cc.StructType.getNamed(name, types, context=context)


def array(element_type: Type, size: Optional[int] = None) -> Type:
    context = element_type.context
    if size is None:
        return cc.ArrayType.get(element_type, context=context)
    return cc.ArrayType.get(element_type, size, context=context)


def stdvec(element_type: Type) -> Type:
    return cc.StdvecType.get(element_type, context=element_type.context)


# ============================================================================ #
# Quake types
# ============================================================================ #

qref = lambda context=None: quake.RefType.get(context=context)
measurement = lambda context=None: quake.MeasureType.get(context=context)


def veq(size: Optional[int] = None,
        *,
        context: Optional[Context] = None) -> Type:
    if size is None:
        return quake.VeqType.get(context=context)
    return quake.VeqType.get(size, context=context)


def struq(types: List[Type], name: Optional[str] = None) -> Type:
    context = types[0].context
    if name is None:
        return quake.StruqType.get(types, context=context)
    return quake.StruqType.getNamed(name, types, context=context)


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


def infer_type(value: InferableType,
               *,
               context: Optional[Context] = None) -> Type:
    if isinstance(value, bool):
        return i1(context)
    if isinstance(value, int):
        return i64(context)
    if isinstance(value, np.float32):
        return f32(context)
    if isinstance(value, (float, np.float64)):
        return f64(context)
    if isinstance(value, np.complex64):
        return complex64(context)
    if isinstance(value, (complex, np.complex128)):
        return complex128(context)
    if isinstance(value, str):
        return ptr(i8(context))
    raise ValueError(f"Unsupported type: {type(value)}")
