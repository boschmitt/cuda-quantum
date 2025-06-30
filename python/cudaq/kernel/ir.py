# ============================================================================ #
# Copyright (c) 2022 - 2025 NVIDIA Corporation & Affiliates.                   #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

import numpy as np
import random
import re
import string
from functools import partialmethod
from typing import List, Optional, get_origin

from cudaq.mlir.ir import (
    BoolAttr,
    Block,
    ComplexType,
    Context,
    DenseI32ArrayAttr,
    DictAttr,
    F32Type,
    F64Type,
    FlatSymbolRefAttr,
    FloatAttr,
    InsertionPoint,
    IntegerAttr,
    IntegerType,
    Location,
    Module,
    StringAttr,
    SymbolTable,
    Type,
    TypeAttr,
    Value,
    UnitAttr,
)
from cudaq.mlir.dialects import arith, cc, complex as complex_dialect

# ==============================================================================
# High-level helpers
# ==============================================================================


def _try_downcast(value: Value):
    if cc.PointerType.isinstance(value.type):
        return _Pointer(value)
    return value


class _WrappedValue:
    pass


class _Pointer(_WrappedValue):

    def __init__(self, value: Value):
        self.value = value

    def load(self) -> Value:
        return _try_downcast(cc.load(self.value))

    def store(self, value: Value):
        cc.store(value, self.value)

    def __repr__(self):
        return f"Pointer({self.value})"


class _Stdvec(_WrappedValue):

    @staticmethod
    def create(elements_type: Type, length: Value, loc: Location):
        data = cc.alloca(elements_type, size=length, loc=loc)
        vec = cc.StdvecInitOp(cc.StdvecType.get(loc.context, elements_type),
                              data,
                              length=length)
        return _Stdvec(vec)

    @staticmethod
    def from_list(values: List[Value], loc: Location):
        i64 = IntegerType.get_signless(64, loc.context)
        # TODO: Check if values are homogeneous
        elements_type = values[0].type
        length = arith.ConstantOp(i64,
                                  IntegerAttr.get(i64, len(values)),
                                  loc=loc).result
        data = cc.alloca(elements_type, size=length, loc=loc)
        for i, value in enumerate(values):
            cc.StoreOp(value, cc.compute_ptr(data, i, loc=loc))
        vec = cc.StdvecInitOp(cc.StdvecType.get(loc.context, elements_type),
                              data,
                              length=length,
                              loc=loc).result
        return _Stdvec(vec)

    def __init__(self, value: Value):
        self.value = value

    def get_length(self, loc: Location):
        return cc.StdvecSizeOp(IntegerType.get_signless(64, loc.context),
                               self.value,
                               loc=loc).result

    def get_elements_type(self, context: Context):
        return cc.StdvecType.getElementType(self.value.type)

    def get_data(self, loc: Location):
        array_type = cc.ArrayType.get(loc.context,
                                      self.get_elements_type(loc.context))
        ptr_array_type = cc.PointerType.get(loc.context, array_type)
        return cc.StdvecDataOp(ptr_array_type, self.value, loc=loc).result

    def item_address(self, index: Value, loc: Location):
        i64 = IntegerType.get_signless(64, loc.context)
        zero = arith.ConstantOp(i64, IntegerAttr.get(i64, 0), loc=loc).result
        index_lt_zero = arith.CmpIOp(IntegerAttr.get(i64, 2),
                                     index,
                                     zero,
                                     loc=loc)
        reverse_index = arith.AddIOp(index, self.get_length(loc),
                                     loc=loc).result
        index = arith.SelectOp(index_lt_zero, reverse_index, index,
                               loc=loc).result

        data = self.get_data(loc)
        return cc.compute_ptr(data, index, loc=loc)

    def get_item(self, index: Value, loc: Location):
        item_addr = self.item_address(index, loc)
        return [cc.LoadOp(item_addr, loc=loc).result]

    def set_item(self, index: Value, value: Value, loc: Location):
        item_addr = self.item_address(index, loc)
        cc.StoreOp(value, item_addr, loc=loc)

    def get_slice(self, begin: Value, end: Value, loc: Location):
        begin_addr = self.item_address(begin, loc)
        length = arith.SubIOp(end, begin).result
        return _Stdvec(
            cc.StdvecInitOp(self.value.type, begin_addr, length=length).result)

    def materialize(self, loc: Location):
        return self.value

    def __repr__(self):
        return f"Stdvec({self.value})"


# ==============================================================================
# Low-level helpers
# ==============================================================================

# CC Dialect `ComputePtrOp` in C++ sets the dynamic index as
# `std::numeric_limits<int32_t>::min()` (see CCOps.tc line 898). We'll duplicate
# that here by just setting it manually
kDynamicPtrIndex: int = -2147483648


def _i(width: int, context: Context = None) -> Type:
    return IntegerType.get_signless(width, context=context)


i1 = lambda context=None: _i(1, context)
i8 = lambda context=None: _i(8, context)
i32 = lambda context=None: _i(32, context)
i64 = lambda context=None: _i(64, context)

f32 = lambda context=None: F32Type.get(context)
f64 = lambda context=None: F64Type.get(context)


def _complex(type: Type) -> Type:
    return ComplexType.get(type)


complex64 = lambda context: _complex(f32(context))
complex128 = lambda context: _complex(f64(context))


def _is_complex_type(type_: Type) -> bool:
    return ComplexType.isinstance(type_)


array = lambda element_type, length: cc.ArrayType.get(element_type, length,
                                                      element_type.context)
dyn_array = lambda element_type: cc.ArrayType.get(element_type,
                                                  context=element_type.context)
ptr = lambda element_type: cc.PointerType.get(element_type, element_type.context
                                             )
struct = lambda types, context=None: cc.StructType.get(types, context)


def _infer_mlir_type(value: bool | complex | float | int | str,
                     *,
                     context: Context = None) -> Type:
    if isinstance(value, bool):
        return i1(context)
    if isinstance(value, int):
        return i64(context)
    if isinstance(value, float):
        return f64(context)
    if isinstance(value, complex):
        return complex128(context)
    raise ValueError(f"Unsupported type: {type(value)}")


def constant(
    value: bool | complex | float | int | str,
    ir_type: Optional[Type] = None,
    *,
    loc: Optional[Location] = None,
    ip: Optional[InsertionPoint] = None,
    context: Optional[Context] = None,
) -> Value:
    if ir_type is None:
        ir_type = _infer_mlir_type(value, context=context)

    if _is_complex_type(ir_type):
        re = constant(value.real, f64(context), loc=loc, ip=ip)
        im = constant(value.imag, f64(context), loc=loc, ip=ip)
        return complex_dialect.CreateOp(ir_type, re, im, loc=loc, ip=ip).result

    if isinstance(value, str):
        ir_type = ptr(ir_type, context)
        return cc.CreateStringLiteralOp(ir_type,
                                        StringAttr.get(value),
                                        loc=loc,
                                        ip=ip).result

    return arith.ConstantOp(ir_type, value, loc=loc, ip=ip).result


def alloca(element_type: Type,
           *,
           size: Optional[Value] = None,
           loc: Optional[Location] = None,
           ip: Optional[InsertionPoint] = None):
    type_attr = TypeAttr.get(element_type)
    if size is not None:
        element_type = dyn_array(element_type)
    return_type = ptr(element_type)
    return cc.AllocaOp(return_type, type_attr, seqSize=size, loc=loc,
                       ip=ip).result


def store(value: Value,
          memory: Value,
          *,
          loc: Optional[Location] = None,
          ip: Optional[InsertionPoint] = None):
    cc.StoreOp(value, memory, loc=loc, ip=ip)


def compute_ptr(base: Value,
                indices: Value | int | List[Value] | List[int],
                *,
                loc: Optional[Location] = None,
                ip: Optional[InsertionPoint] = None):
    if not isinstance(indices, list):
        indices = [indices]

    array_or_struc_type = cc.PointerType.getElementType(base.type)
    if cc.ArrayType.isinstance(array_or_struc_type):
        element_type = cc.ArrayType.getElementType(array_or_struc_type)
    elif cc.StructType.isinstance(array_or_struc_type):
        assert isinstance(indices[0], int)
        fields_types = cc.StructType.getTypes(array_or_struc_type)
        element_type = fields_types[indices[0]]
    else:
        raise ValueError(f"Unsupported type: {array_or_struc_type}")

    return_type = ptr(element_type)

    if all(isinstance(x, int) for x in indices):
        indices_attr = DenseI32ArrayAttr.get(indices, context=base.type.context)
        return cc.ComputePtrOp(return_type,
                               base, [],
                               indices_attr,
                               loc=loc,
                               ip=ip).result
    if all(isinstance(x, Value) for x in indices):
        indices_attr = DenseI32ArrayAttr.get([kDynamicPtrIndex],
                                             context=base.type.context)
        return cc.ComputePtrOp(return_type,
                               base,
                               indices,
                               indices_attr,
                               loc=loc,
                               ip=ip).result
    raise ValueError(f"Unsupported indices: {indices}")
