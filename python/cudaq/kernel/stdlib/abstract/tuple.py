# ============================================================================ #
# Copyright (c) 2022 - 2025 NVIDIA Corporation & Affiliates.                   #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

from __future__ import annotations
from dataclasses import dataclass

from cudaq.mlir.ir import Type
from .._base import AbstractValue
from ..struct import Struct


@dataclass(frozen=True, slots=True, init=False)
class Tuple(AbstractValue):
    values: tuple

    def __init__(self, *values):
        object.__setattr__(self, "values", tuple(values))

    def __getitem__(self, index: int) -> AbstractValue:
        return self.values[index]

    def __len__(self) -> int:
        return len(self.values)

    def __iter__(self):
        return iter(self.values)

    def materialize(self, *, with_type: Type | None = None) -> Struct:
        values = [
            value.materialize() if isinstance(value, AbstractValue) else value
            for value in self.values
        ]
        types = [value.type for value in values]
        struct = Struct.create(types, name="tuple")
        for i, value in enumerate(values):
            struct = struct.set_field(i, value)
        return struct

    @property
    def type(self):
        return Tuple
