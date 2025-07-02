# ============================================================================ #
# Copyright (c) 2022 - 2025 NVIDIA Corporation & Affiliates.                   #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

from typing import Callable

from cudaq.mlir.dialects import quake
from cudaq.mlir.ir import Value
from .._base import Error, try_downcast
from ..collections.veq import Veq
from .._core import types as T


def _measure(op: Callable,
             targets: list[Value] | Value,
             *,
             label: str | None = None) -> Error | None:

    if isinstance(targets[0], Veq) or len(targets) > 1:
        measure_type = T.stdvec(T.measurement())
        result_type = T.stdvec(T.i1())
    elif quake.RefType.isinstance(targets[0].type):
        measure_type = T.measurement()
        result_type = T.i1()
    else:
        return Error(
            f"Measure quantum operation on incorrect type {targets[0].type}")

    measure_result = op(measure_type, [], targets, registerName=label)
    result = quake.DiscriminateOp(result_type, measure_result).result
    return try_downcast(result)


def mx(args: list[Value] | Value, *, label: str | None = None) -> Error | None:
    return _measure(quake.MxOp, args, label=label)


def my(args: list[Value] | Value, *, label: str | None = None) -> Error | None:
    return _measure(quake.MyOp, args, label=label)


def mz(args: list[Value] | Value, *, label: str | None = None) -> Error | None:
    return _measure(quake.MzOp, args, label=label)
