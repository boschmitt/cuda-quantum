# ============================================================================ #
# Copyright (c) 2022 - 2025 NVIDIA Corporation & Affiliates.                   #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

from __future__ import annotations
from dataclasses import dataclass

from cudaq.mlir.dialects import cc
from cudaq.mlir.ir import Type
from .._base import AbstractValue, Literal
from .._core.scalars import implicit_conversion
from ..collections.stdvec import Stdvec
from ..collections.veq import Veq
from ..literals.bool_literal import BoolLiteral
from ..qref import QRef


@dataclass(frozen=True, slots=True)
class AList(AbstractValue):
    values: list

    def __contains__(self, item):
        if isinstance(item, Literal):
            return BoolLiteral(item.value in self.values)
        value = self.materialize()
        return value.__contains__(item)

    def materialize(self, *, with_type: Type | None = None) -> Stdvec:
        elements = self.values
        if isinstance(elements[0], QRef):
            return Veq.from_qubits(elements)
        if isinstance(elements[0], AbstractValue):
            elements = [element.materialize() for element in elements]
        if with_type is not None:
            if cc.StdvecType.isinstance(with_type):
                with_type = cc.StdvecType.getElementType(with_type)
            else:
                raise ValueError(f"Unsupported type: {with_type}")
            elements = [
                implicit_conversion(element, with_type) for element in elements
            ]
        return Stdvec.from_list(elements)

    @property
    def type(self):
        return AList
