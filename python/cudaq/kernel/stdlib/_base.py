# ============================================================================ #
# Copyright (c) 2022 - 2025 NVIDIA Corporation & Affiliates.                   #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

from abc import ABC, abstractmethod

from cudaq.mlir.ir import (
    ComplexType,
    F32Type,
    F64Type,
    IntegerType,
    Value,
)
from cudaq.mlir.dialects import cc, quake
from ._core import types as T


class Iterable(ABC):
    """Base class for all iterable types"""

    @abstractmethod
    def get_length(self):
        """
        Returns the length of the iterable
        """
        pass

    @abstractmethod
    def get_item(self, index):
        """
        Get item(s) at the given index
        Returns a list of values (for unpacking/enumerate support)
        """
        pass


class AbstractValue:
    pass


class Literal(AbstractValue):
    pass


def has_arithmetic_type(value: Value | list[Value]):
    if hasattr(value, "type"):
        return T.is_arithmetic_type(value.type)
    if isinstance(value, list):
        return all(T.is_arithmetic_type(v.type) for v in value)
    return False


def has_integer_like_type(value: Value | list[Value]):
    if hasattr(value, "type"):
        return T.is_integer_like_type(value.type)
    if isinstance(value, list):
        return all(T.is_integer_like_type(v.type) for v in value)
    return False


def has_float_type(value: Value | list[Value]):
    if hasattr(value, "type"):
        return T.is_float_type(value.type)
    if isinstance(value, list):
        return all(T.is_float_type(v.type) for v in value)
    return False


def has_complex_type(value: Value | list[Value]):
    if hasattr(value, "type"):
        return T.is_complex_type(value.type)
    if isinstance(value, list):
        return all(T.is_complex_type(v.type) for v in value)
    return False


def try_downcast(value):
    if not isinstance(value, Value):
        return value
    from .boolean import Bool
    from .int32 import Int32
    from .int64 import Int64
    from .complex64 import Complex64
    from .complex128 import Complex128
    from .float32 import Float32
    from .float64 import Float64
    from .pointer import Pointer
    from .collections.stdvec import Stdvec
    from .collections.veq import Veq
    from .struct import Struct
    from .struq import Struq
    from .qref import QRef

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


class Error:
    message: str
