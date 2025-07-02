# ============================================================================ #
# Copyright (c) 2022 - 2025 NVIDIA Corporation & Affiliates.                   #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

from __future__ import annotations
from dataclasses import dataclass

from cudaq.mlir.ir import (
    DenseI32ArrayAttr,
    DenseI64ArrayAttr,
    Type,
    Value,
)
from cudaq.mlir.dialects import cc
from ._core import types as T
from ._core.memory import _get_field_index
from ._core.scalars import implicit_conversion
from .literals.int_literal import IntLiteral


@dataclass(frozen=True, slots=True)
class Struct:
    value: Value

    @staticmethod
    def create(fields: list[Type], *, name: str | None = None) -> Struct:
        return Struct(cc.UndefOp(T.struct(fields, name)).result)

    @property
    def name(self) -> str:
        return cc.StructType.getName(self.value.type)

    @property
    def num_fields(self) -> int:
        return len(cc.StructType.getTypes(self.value.type))

    def get_field(self, name_or_index: str | int) -> Value:
        field_index = name_or_index
        if isinstance(name_or_index, str):
            field_index = _get_field_index(self.value.type, name_or_index)
        if isinstance(name_or_index, IntLiteral):
            field_index = name_or_index.value
        field_type = cc.StructType.getTypes(self.value.type)
        assert field_index < len(field_type)
        field_type = field_type[field_index]
        return cc.ExtractValueOp(field_type, self.value, [],
                                 DenseI32ArrayAttr.get([field_index])).result

    def set_field(self, name_or_index: str | int, value: Value) -> Struct:
        field_index = name_or_index
        if isinstance(name_or_index, str):
            field_index = self._get_field_index(name_or_index)
        if isinstance(name_or_index, IntLiteral):
            field_index = name_or_index.value
        field_type = cc.StructType.getTypes(self.value.type)
        assert field_index < len(field_type)
        field_type = field_type[field_index]
        value = implicit_conversion(value, field_type)
        return Struct(
            cc.InsertValueOp(self.type, self.value, value,
                             DenseI64ArrayAttr.get([field_index])).result)

    @property
    def type(self) -> Type:
        return self.value.type
