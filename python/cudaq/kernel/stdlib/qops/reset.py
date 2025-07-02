# ============================================================================ #
# Copyright (c) 2022 - 2025 NVIDIA Corporation & Affiliates.                   #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

from cudaq.mlir.dialects import quake
from cudaq.mlir.ir import Value
from .._base import Error
from .._core.control_flow import for_loop
from ..collections.veq import Veq


def reset(target: Value) -> Error | None:
    # TODO: Support quantum structs ?
    if quake.RefType.isinstance(target.type):
        quake.ResetOp([], target)
        return None
    if isinstance(target, Veq):
        length = target.get_length()
        for_loop(length, lambda i: quake.ResetOp([], target.get_item(i)))
        return None
    return Error(f"Reset quantum operation on incorrect type {target.type}")
