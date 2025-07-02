# ============================================================================ #
# Copyright (c) 2022 - 2025 NVIDIA Corporation & Affiliates.                   #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

from cudaq.mlir.dialects import cc, quake
from cudaq.mlir.ir import Type, TypeAttr, Value, DenseI32ArrayAttr
from .._core import types as T
from .._base import globalRegisteredTypes


def alloca(element_type: Type, *, size: Value | None = None) -> Value:
    type_attr = TypeAttr.get(element_type)
    if size is not None:
        element_type = T.array(element_type)
    return_type = T.ptr(element_type)
    return cc.AllocaOp(return_type, type_attr, seqSize=size).result


def load(memory: Value) -> Value:
    return cc.LoadOp(memory).result


def store(value: Value, memory: Value) -> None:
    cc.StoreOp(value, memory)


# CC Dialect `ComputePtrOp` in C++ sets the dynamic index as
# `std::numeric_limits<int32_t>::min()` (see CCOps.tc line 898). We'll
# duplicate that here by just setting it manually.
kDynamicIndex: int = -2147483648


def _get_field_index(struct: Type, field_name: str) -> int:
    if cc.StructType.isinstance(struct):
        name = cc.StructType.getName(struct)
    elif quake.StruqType.isinstance(struct):
        name = quake.StruqType.getName(struct)
    else:
        raise ValueError(f"Unsupported type: {struct}")
    assert name and name != 'tuple'
    if not globalRegisteredTypes.isRegisteredClass(name):
        raise RuntimeError(f'Dataclass is not registered: {name}')

    _, userType = globalRegisteredTypes.getClassAttributes(name)
    for i, (k, _) in enumerate(userType.items()):
        if k == field_name:
            return i
    raise RuntimeError(f'Field {field_name} not found in {name}')


def compute_ptr(base: Value, index: Value | int | str) -> Value:
    array_or_struc_type = cc.PointerType.getElementType(base.type)
    if cc.ArrayType.isinstance(array_or_struc_type):
        element_type = cc.ArrayType.getElementType(array_or_struc_type)
    elif cc.StructType.isinstance(array_or_struc_type):
        if isinstance(index, str):
            index = _get_field_index(array_or_struc_type, index)
        fields_types = cc.StructType.getTypes(array_or_struc_type)
        element_type = fields_types[index]
    else:
        raise ValueError(f"Unsupported type: {array_or_struc_type}")

    ptr_type = T.ptr(element_type)

    if isinstance(index, int):
        index = DenseI32ArrayAttr.get([index])
        return cc.ComputePtrOp(ptr_type, base, [], index).result
    if isinstance(index, Value):
        dyn_index = DenseI32ArrayAttr.get([kDynamicIndex])
        return cc.ComputePtrOp(ptr_type, base, [index], dyn_index).result
    raise ValueError(f"Unsupported index: {index}")
