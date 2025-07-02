# ============================================================================ #
# Copyright (c) 2022 - 2025 NVIDIA Corporation & Affiliates.                   #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

from __future__ import annotations

from cudaq.mlir.ir import (
    ComplexType,
    F32Type,
    F64Type,
    IntegerType,
    Value,
)
from cudaq.mlir.dialects import cc, quake
from ._core import types as T


from ._base import AbstractValue, Iterable
from .abstract import Tuple, AList, Range, Enumerate
from .collections import Stdvec, Veq
from .literals import BoolLiteral, ComplexLiteral, FloatLiteral, IntLiteral, StringLiteral

from .boolean import Bool
from .complex64 import Complex64
from .complex128 import Complex128
from .int32 import Int32
from .int64 import Int64
from .float32 import Float32
from .float64 import Float64
from .pointer import Pointer
from .struct import Struct
from .struq import Struq
from .qref import QRef


def try_downcast(value):
    if not isinstance(value, Value):
        return value
    if IntegerType.isinstance(value.type):
        width = IntegerType(value.type).width
        if width == 1:
            return Bool(value)
        if width == 32:
            return Int32(value)
        if width == 64:
            return Int64(value)
        return value
    if ComplexType.isinstance(value.type):
        if ComplexType(value.type).element_type == T.f32():
            return Complex64(value)
        if ComplexType(value.type).element_type == T.f64():
            return Complex128(value)
        return value
    if F32Type.isinstance(value.type):
        return Float32(value)
    if F64Type.isinstance(value.type):
        return Float64(value)
    if cc.PointerType.isinstance(value.type):
        return Pointer(value)
    if cc.StdvecType.isinstance(value.type):
        return Stdvec(value)
    if cc.StructType.isinstance(value.type):
        return Struct(value)
    if quake.RefType.isinstance(value.type):
        return QRef(value)
    if quake.StruqType.isinstance(value.type):
        return Struq(value)
    if quake.VeqType.isinstance(value.type):
        return Veq(value)
    return value


# ============================================================================ #
# Quantum operations
# ============================================================================ #


def qalloca() -> Value:
    return QRef(quake.AllocaOp(T.qref()).result)


from .qops.adjoint import adjoint
from .qops.control import control
from .qops.compute_action import compute_action
from .qops.exp_pauli import exp_pauli
from .qops.gates import *
from .qops.measurement import mx, my, mz
from .qops.reset import reset

# ============================================================================ #
# Math functions
# ============================================================================ #

from .math import cos, sin, sqrt, exp, ceil
