# ============================================================================ #
# Copyright (c) 2022 - 2025 NVIDIA Corporation & Affiliates.                   #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

from typing import Callable

from cudaq.mlir._mlir_libs._quakeDialects import cudaq_runtime
from cudaq.mlir.dialects import arith, cc
from cudaq.mlir.ir import Value, Block, InsertionPoint, IntegerAttr, BoolAttr
from .._core import types as T
from .scalars import constant


def _has_terminator(block: Block):
    if len(block.operations) > 0:
        return cudaq_runtime.isTerminator(
            block.operations[len(block.operations) - 1])
    return False


def for_loop(num_iterations: Value,
             body: Callable[[Value], None],
             else_body: Callable[[], None] | None = None) -> Value:
    i64 = T.i64()
    start = constant(0, i64)
    step = constant(1, i64)
    loop = cc.LoopOp([start.type], [start], BoolAttr.get(False))

    whileBlock = Block.create_at_start(loop.whileRegion, [start.type])
    with InsertionPoint(whileBlock):
        pred = IntegerAttr.get(i64, 2)
        test = arith.CmpIOp(pred, whileBlock.arguments[0],
                            num_iterations).result
        cc.ConditionOp(test, whileBlock.arguments)

    stepBlock = Block.create_at_start(loop.stepRegion, [start.type])
    with InsertionPoint(stepBlock):
        incr = arith.AddIOp(stepBlock.arguments[0], step).result
        cc.ContinueOp([incr])

    bodyBlock = Block.create_at_start(loop.bodyRegion, [start.type])
    with InsertionPoint(bodyBlock):
        body(bodyBlock.arguments[0])
        if not _has_terminator(bodyBlock):
            cc.ContinueOp(bodyBlock.arguments)

    if else_body is not None:
        elseBlock = Block.create_at_start(loop.elseRegion, [start.type])
        with InsertionPoint(elseBlock):
            else_body()
            if not _has_terminator(elseBlock):
                cc.ContinueOp(elseBlock.arguments)

    return loop


def while_loop(test: Callable[[], Value],
               body: Callable[[], None],
               else_body: Callable[[], None] | None = None) -> Value:
    loop = cc.LoopOp([], [], BoolAttr.get(False))

    whileBlock = Block.create_at_start(loop.whileRegion, [])
    with InsertionPoint(whileBlock):
        condition = test()
        cc.ConditionOp(condition, [])

    bodyBlock = Block.create_at_start(loop.bodyRegion, [])
    with InsertionPoint(bodyBlock):
        body()
        if not _has_terminator(bodyBlock):
            cc.ContinueOp([])

    if else_body is not None:
        elseBlock = Block.create_at_start(loop.elseRegion, [])
        with InsertionPoint(elseBlock):
            else_body()
            if not _has_terminator(elseBlock):
                cc.ContinueOp([])

    return loop


def if_else(condition: Value, then_body: Callable[[], None],
            else_body: Callable[[], None] | None):

    if_op = cc.IfOp([], condition, [])
    thenBlock = Block.create_at_start(if_op.thenRegion, [])
    with InsertionPoint(thenBlock):
        then_body()
        if not _has_terminator(thenBlock):
            cc.ContinueOp([])

    if else_body is not None:
        elseBlock = Block.create_at_start(if_op.elseRegion, [])
        with InsertionPoint(elseBlock):
            else_body()
            if not _has_terminator(elseBlock):
                cc.ContinueOp([])


def continue_op() -> None:
    ip = InsertionPoint.current

    # We need to find two things:
    # 1. The innermost loop
    # 2. The innermost `if`, if any (it might be that we are not inside an `if` at all)
    loop_op = ip.block.owner.operation
    is_inside_if = False
    while loop_op is not None:
        opview = loop_op.opview
        if isinstance(opview, cc.IfOp):
            is_inside_if = True
        if isinstance(opview, cc.LoopOp):
            break
        loop_op = loop_op.parent

    if loop_op is None:
        raise ValueError("Continue statement outside of loop")

    body = loop_op.opview.bodyRegion.blocks[0]
    if is_inside_if:
        cc.UnwindContinueOp(body.arguments, ip=ip)
        return

    cc.ContinueOp(body.arguments, ip=ip)


def break_op() -> None:
    ip = InsertionPoint.current
    loop_op = ip.block.owner.operation
    is_inside_if = False
    while loop_op is not None:
        opview = loop_op.opview
        if isinstance(opview, cc.IfOp):
            is_inside_if = True
        if isinstance(opview, cc.LoopOp):
            break
        loop_op = loop_op.parent

    if loop_op is None:
        raise ValueError("Break statement outside of loop")

    body = loop_op.opview.bodyRegion.blocks[0]
    if is_inside_if:
        cc.UnwindBreakOp(body.arguments, ip=ip)
        return

    cc.BreakOp(body.arguments, ip=ip)
