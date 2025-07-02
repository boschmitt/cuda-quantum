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
from cudaq.mlir.dialects import cc
from ._base import try_downcast
from ._core.memory import compute_ptr


@dataclass(frozen=True, slots=True)
class Pointer:
    value: Value

    @property
    def element_type(self) -> Type:
        return cc.PointerType.getElementType(self.value.type)

    def load(self) -> Value:
        return try_downcast(cc.LoadOp(self.value).result)

    def store(self, value: Value):
        if not isinstance(value, Value):
            value = value.value
        cc.StoreOp(value, self.value)

    def __getitem__(self, index: Value | int | str) -> Pointer:
        if isinstance(index, (int, str)):
            return Pointer(compute_ptr(self.value, index))
        return Pointer(compute_ptr(self.value, index.value))

    @property
    def type(self) -> Type:
        return self.value.type
