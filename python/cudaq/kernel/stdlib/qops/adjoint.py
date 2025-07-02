# ============================================================================ #
# Copyright (c) 2022 - 2025 NVIDIA Corporation & Affiliates.                   #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

from cudaq.mlir.dialects import quake
from cudaq.mlir.ir import FlatSymbolRefAttr, Value
from .._base import Error
from ...utils import globalKernelRegistry, nvqppPrefix


def adjoint(symbol_or_value: FlatSymbolRefAttr | Value,
            args: list[Value] | Value):
    if isinstance(symbol_or_value, Value):
        quake.ApplyOp([], [symbol_or_value], [], args, is_adj=True)
        return
    if FlatSymbolRefAttr.isinstance(symbol_or_value):
        # TODO: This is a hack to get the name from the symbol.
        name = FlatSymbolRefAttr(symbol_or_value).value[len(nvqppPrefix):]
        if len(args) != len(globalKernelRegistry[name].arguments):
            return Error(
                f"Adjoint of {name} requires {len(globalKernelRegistry[name].arguments)} arguments, got {len(args)}"
            )
        quake.ApplyOp([], [], [], args, callee=symbol_or_value, is_adj=True)
        return
