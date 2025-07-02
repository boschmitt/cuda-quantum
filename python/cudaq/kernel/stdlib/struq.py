# ============================================================================ #
# Copyright (c) 2022 - 2025 NVIDIA Corporation & Affiliates.                   #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

from __future__ import annotations
from dataclasses import dataclass

from cudaq.mlir.ir import Type, Value
from cudaq.mlir.dialects import quake
from ._core import types as T
from ._core.memory import _get_field_index


@dataclass(frozen=True, slots=True)
class Struq:
    value: Value

    @staticmethod
    def create(args: list[Value], *, name: str | None = None) -> Struq:
        types = [arg.type for arg in args]
        struq_type = T.struq(types, name)
        return Struq(quake.MakeStruqOp(struq_type, args).result)

    @property
    def name(self) -> str:
        return quake.StruqType.getName(self.value.type)

    @property
    def num_fields(self) -> int:
        return len(quake.StruqType.getTypes(self.value.type))

    def get_field(self, name_or_index: str | int) -> Value:
        field_index = name_or_index
        if isinstance(name_or_index, str):
            field_index = _get_field_index(self.value.type, name_or_index)
        field_type = quake.StruqType.getTypes(self.value.type)
        assert field_index < len(field_type)
        field_type = field_type[field_index]
        return quake.GetMemberOp(field_type, self.value, field_index).result

    @property
    def type(self) -> Type:
        return self.value.type
