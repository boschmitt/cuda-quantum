# ============================================================================ #
# Copyright (c) 2022 - 2025 NVIDIA Corporation & Affiliates.                   #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

from __future__ import annotations
from abc import ABC, abstractmethod
import ast
from dataclasses import dataclass
from enum import IntEnum
from typing import Callable

from cudaq.mlir.ir import (
    Block,
    BoolAttr,
    ComplexType,
    DenseI32ArrayAttr,
    DenseI64ArrayAttr,
    F32Type,
    F64Type,
    FlatSymbolRefAttr,
    InsertionPoint,
    IntegerAttr,
    FloatAttr,
    IntegerType,
    StringAttr,
    Type,
    TypeAttr,
    UnitAttr,
    Value,
)
from cudaq.mlir._mlir_libs._quakeDialects import cudaq_runtime
from cudaq.mlir.dialects import arith, cc, complex as complex_dialect, math, quake
from . import types as T
from ..utils import globalRegisteredTypes, globalKernelRegistry, nvqppPrefix

# CC Dialect `ComputePtrOp` in C++ sets the dynamic index as
# `std::numeric_limits<int32_t>::min()` (see CCOps.tc line 898). We'll
# duplicate that here by just setting it manually.
kDynamicIndex: int = -2147483648


def alloca(element_type: Type, *, size: Value | None = None) -> PointerValue:
    type_attr = TypeAttr.get(element_type)
    if size is not None:
        element_type = T.array(element_type)
    return_type = T.ptr(element_type)
    return PointerValue(
        cc.AllocaOp(return_type, type_attr, seqSize=size).result)


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


def _compute_ptr(base: Value, index: Value | int | str) -> Value:
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


def constant(value: T.InferableType, ir_type: Type | None = None) -> Value:
    if ir_type is None:
        ir_type = T.infer_type(value)

    if T.is_complex_type(ir_type):
        element_type = ComplexType(ir_type).element_type
        re = constant(value.real, element_type)
        im = constant(value.imag, element_type)
        return complex_dialect.CreateOp(ir_type, re, im).result

    if isinstance(value, str):
        return cc.CreateStringLiteralOp(
            ir_type,
            StringAttr.get(value),
        ).result

    if IntegerType.isinstance(ir_type):
        value = IntegerAttr.get(ir_type, value)
    if F32Type.isinstance(ir_type):
        value = FloatAttr.get(ir_type, value)
    if F64Type.isinstance(ir_type):
        value = FloatAttr.get(ir_type, value)

    return arith.ConstantOp(ir_type, value).result


def literal(value: str) -> StringLiteral:
    return StringLiteral(value)


def load(memory: Value) -> Value:
    return cc.LoadOp(memory).result


def store(value: Value, memory: Value):
    cc.StoreOp(value, memory)


def break_op() -> Error | None:
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


def continue_op() -> Error | None:
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


def if_op(condition: Value, then_body: Callable[[], None],
          else_body: Callable[[], None] | None):

    condition = implicit_conversion(condition, T.i1())
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


def try_downcast(value):
    if not isinstance(value, Value):
        return value
    if ComplexType.isinstance(value.type):
        return ComplexValue(value)
    if cc.PointerType.isinstance(value.type):
        return PointerValue(value)
    if cc.StdvecType.isinstance(value.type):
        return StdvecValue(value)
    if cc.StructType.isinstance(value.type):
        return StructValue(value)
    if quake.RefType.isinstance(value.type):
        return QRefValue(value)
    if quake.StruqType.isinstance(value.type):
        return StruqValue(value)
    if quake.VeqType.isinstance(value.type):
        return VeqValue(value)
    return value


class _CmpIPredicate(IntEnum):
    eq = 0  # equal
    ne = 1  # not equal
    slt = 2  # signed less than
    sle = 3  # signed less than or equal
    sgt = 4  # signed greater than
    sge = 5  # signed greater than or equal
    ult = 6  # unsigned less than
    ule = 7  # unsigned less than or equal
    ugt = 8  # unsigned greater than
    uge = 9  # unsigned greater than or equal


def _arith_cmpi_predicate_attr(predicate):
    match predicate:
        case ast.Eq:
            return IntegerAttr.get(T.i64(), _CmpIPredicate.eq)
        case ast.NotEq:
            return IntegerAttr.get(T.i64(), _CmpIPredicate.ne)
        case ast.Lt:
            return IntegerAttr.get(T.i64(), _CmpIPredicate.slt)
        case ast.LtE:
            return IntegerAttr.get(T.i64(), _CmpIPredicate.sle)
        case ast.Gt:
            return IntegerAttr.get(T.i64(), _CmpIPredicate.sgt)
        case ast.GtE:
            return IntegerAttr.get(T.i64(), _CmpIPredicate.sge)
        case _:
            return None


class _CmpFPredicate(IntEnum):
    alwaysfalse = 0
    # An ordered comparison checks if neither operand is NaN.
    oeq = 1
    ogt = 2
    oge = 3
    olt = 4
    ole = 5
    one = 6
    ord = 7
    # An unordered comparison checks if either operand is a NaN.
    ueq = 8
    ugt = 9
    uge = 10
    ult = 11
    ule = 12
    une = 13
    uno = 14
    alwaystrue = 15


def _arith_cmpf_predicate_attr(predicate):
    match predicate:
        case ast.Eq:
            return IntegerAttr.get(T.i64(), _CmpFPredicate.oeq)
        case ast.NotEq:
            return IntegerAttr.get(T.i64(), _CmpFPredicate.one)
        case ast.Lt:
            return IntegerAttr.get(T.i64(), _CmpFPredicate.olt)
        case ast.LtE:
            return IntegerAttr.get(T.i64(), _CmpFPredicate.ole)
        case ast.Gt:
            return IntegerAttr.get(T.i64(), _CmpFPredicate.ogt)
        case ast.GtE:
            return IntegerAttr.get(T.i64(), _CmpFPredicate.oge)
        case _:
            return None


def compare(left: Value, right: Value, predicate: ast.cmpop):
    if not has_arithmetic_type([left, right]):
        return None

    left = arithmetic_promotion(left, right.type)
    right = arithmetic_promotion(right, left.type)

    if has_float_type(left):
        predicate_attr = _arith_cmpf_predicate_attr(type(predicate))
        if predicate_attr is None:
            return None
        return arith.CmpFOp(predicate_attr, left, right).result

    if has_integer_like_type(left):
        predicate_attr = _arith_cmpi_predicate_attr(type(predicate))
        if predicate_attr is None:
            return None
        return arith.CmpIOp(predicate_attr, left, right).result

    if has_complex_type(left):
        match type(predicate):
            case ast.Eq:
                return complex_dialect.EqualOp(left, right).result
            case ast.NotEq:
                return complex_dialect.NotEqualOp(left, right).result
            case _:
                return None

    return None


def has_arithmetic_type(value: Value | list[Value]):
    if isinstance(value, Value):
        return T.is_arithmetic_type(value.type)
    return all(T.is_arithmetic_type(v.type) for v in value)


def has_integer_like_type(value: Value | list[Value]):
    if isinstance(value, Value):
        return T.is_integer_like_type(value.type)
    if isinstance(value, list):
        return all(T.is_integer_like_type(v.type) for v in value)
    return False


def has_float_type(value: Value | list[Value]):
    if isinstance(value, Value):
        return T.is_float_type(value.type)
    if isinstance(value, list):
        return all(T.is_float_type(v.type) for v in value)
    return False


def has_complex_type(value: Value | list[Value]):
    if isinstance(value, Value):
        return T.is_complex_type(value.type)
    return all(T.is_complex_type(v.type) for v in value)


def binary_op(left: Value, right: Value, op: ast.cmpop):
    if isinstance(op, ast.Pow):
        if has_integer_like_type([left, right]):
            return math.IPowIOp(left, right).result
        if has_float_type([left, right]):
            return math.FPowIOp(left, right).result
        if has_float_type([left, right]):
            return math.PowFOp(left, right).result
        if has_complex_type([left, right]):
            return complex_dialect.PowOp(left, right).result
        return None

    left = arithmetic_promotion(left, right.type)
    right = arithmetic_promotion(right, left.type)

    match type(op):
        case ast.Add:
            if has_integer_like_type(left):
                return arith.AddIOp(left, right).result
            if has_float_type(left):
                return arith.AddFOp(left, right).result
            if has_complex_type(left):
                return complex_dialect.AddOp(left, right).result
        case ast.Sub:
            if has_integer_like_type(left):
                return arith.SubIOp(left, right).result
            if has_float_type(left):
                return arith.SubFOp(left, right).result
            if has_complex_type(left):
                return complex_dialect.SubOp(left, right).result
        case ast.Mult:
            if has_integer_like_type(left):
                return arith.MulIOp(left, right).result
            if has_float_type(left):
                return arith.MulFOp(left, right).result
            if has_complex_type(left):
                return complex_dialect.MulOp(left, right).result
        case ast.Div:
            if has_integer_like_type(left):
                return arith.DivSIOp(left, right).result
            if has_float_type(left):
                return arith.DivFOp(left, right).result
            if has_complex_type(left):
                return complex_dialect.DivOp(left, right).result
        case ast.Mod:
            if has_integer_like_type(left):
                return arith.RemSIOp(left, right).result
            if has_float_type(left):
                return arith.RemFOp(left, right).result
        case ast.FloorDiv:
            if has_integer_like_type([left, right]):
                return arith.FloorDivSIOp(left, right).result
        # Shitf operations
        case ast.LShift:
            if has_integer_like_type([left, right]):
                return arith.ShLIOp(left, right).result
        case ast.RShift:
            if has_integer_like_type([left, right]):
                return arith.ShRSIOp(left, right).result
        case ast.BitAnd:
            if has_integer_like_type([left, right]):
                return arith.AndIOp(left, right).result
        case ast.BitOr:
            if has_integer_like_type([left, right]):
                return arith.OrIOp(left, right).result
        case ast.BitXor:
            if has_integer_like_type([left, right]):
                return arith.XOrIOp(left, right).result
        case _:
            return None
    return None


def unary_op(operand: Value, op: ast.unaryop):
    match type(op):
        case ast.USub:
            if has_integer_like_type(operand):
                zero = constant(0, operand.type)
                return arith.SubIOp(zero, operand).result
            if has_float_type(operand):
                return arith.NegFOp(operand).result
            if has_complex_type(operand):
                return complex_dialect.NegOp(operand).result
        case ast.Not:
            operand = implicit_conversion(operand, T.i1())
            one = constant(0, operand.type)
            return arith.XOrIOp(one, operand).result
        case _:
            return None


# ============================================================================ #
# Implicit Conversions
# ============================================================================ #


def implicit_conversion(value, target_type):
    if value.type == target_type:
        return value
    if IntegerType.isinstance(target_type):
        target_width = IntegerType(target_type).width
        if target_width == 1:
            zero = constant(0, value.type)
            if IntegerType.isinstance(value.type):
                return arith.CmpIOp(IntegerAttr.get(T.i64(), 1), zero,
                                    value).result
            if T.is_float_type(value.type):
                return arith.CmpFOp(IntegerAttr.get(T.i64(), 6), zero,
                                    value).result
            if ComplexType.isinstance(value.type):
                return complex_dialect.NotEqualOp(zero, value).result
            return None
        if not IntegerType.isinstance(value.type):
            return None
        return _i_coerce(value, target_width)
    if F32Type.isinstance(target_type):
        if ComplexType.isinstance(value.type):
            return None
        return f32_coerce(value)
    if F64Type.isinstance(target_type):
        if ComplexType.isinstance(value.type):
            return None
        return f64_coerce(value)
    if ComplexType.isinstance(target_type):
        if ComplexType(target_type).element_type == T.f32():
            return complex64_coerce(value)
        if ComplexType(target_type).element_type == T.f64():
            return complex128_coerce(value)
        raise ValueError(f"Unsupported complex type: {target_type}")
    return None


def arithmetic_promotion(value, target_type):
    #assert T.is_arithmetic_type(value) and T.is_arithmetic_type(target_type)
    if value.type == target_type:
        return value
    if IntegerType.isinstance(target_type):
        if IntegerType.isinstance(value.type):
            width = IntegerType(value.type).width
            target_width = IntegerType(target_type).width
            if width < target_width:
                return _i_coerce(value, target_width)
        return value
    if F32Type.isinstance(target_type):
        if IntegerType.isinstance(value.type):
            return f32_coerce(value)
        return value
    if F64Type.isinstance(target_type):
        if IntegerType.isinstance(value.type):
            return f64_coerce(value)
        if F32Type.isinstance(value.type):
            return f64_coerce(value)
        return value
    if ComplexType.isinstance(target_type):
        if ComplexType(target_type).element_type == T.f32():
            if IntegerType.isinstance(value.type):
                return complex64_coerce(value)
            if F32Type.isinstance(value.type):
                return complex64_coerce(value)
            if F64Type.isinstance(value.type):
                return complex64_coerce(value)
            return value
        if ComplexType(target_type).element_type == T.f64():
            return complex128_coerce(value)
        raise ValueError(f"Unsupported complex type: {target_type}")
    return None


def _i_coerce(value, target_width):
    if IntegerType.isinstance(value.type):
        width = IntegerType(value.type).width
        if width == target_width:
            return value
        if target_width < width:
            return cc.CastOp(T.i(target_width), value).result
        zeroext = width == 1
        return cc.CastOp(T.i(target_width),
                         value,
                         sint=not zeroext,
                         zint=zeroext).result
    if T.is_float_type(value.type):
        return cc.CastOp(T.i(target_width), value, sint=True, zint=False).result
    return None


def i64_coerce(value):
    return _i_coerce(value, 64)


def i32_coerce(value):
    return _i_coerce(value, 32)


def i8_coerce(value):
    return _i_coerce(value, 8)


def f64_coerce(value):
    if F64Type.isinstance(value.type):
        return value
    if F32Type.isinstance(value.type):
        return cc.CastOp(T.f64(), value).result
    if IntegerType.isinstance(value.type):
        zeroext = IntegerType(value.type).width == 1
        return cc.CastOp(T.f64(), value, sint=not zeroext, zint=zeroext).result


def f32_coerce(value):
    if F32Type.isinstance(value.type):
        return value
    if F64Type.isinstance(value.type):
        return cc.CastOp(T.f32(), value).result
    if IntegerType.isinstance(value.type):
        zeroext = IntegerType(value.type).width == 1
        return cc.CastOp(T.f32(), value, sint=not zeroext, zint=zeroext).result
    return None


def complex64_coerce(value):
    if ComplexType.isinstance(value.type):
        if ComplexType(value.type).element_type == T.f32():
            return value
        re = f32_coerce(complex_dialect.ReOp(value).result)
        im = f32_coerce(complex_dialect.ImOp(value).result)
        return complex_dialect.CreateOp(T.complex64(), re, im).result

    re = f32_coerce(value)
    im = constant(0.0, T.f32())
    return complex_dialect.CreateOp(T.complex64(), re, im).result


def complex128_coerce(value):
    if ComplexType.isinstance(value.type):
        if ComplexType(value.type).element_type == T.f64():
            return value
        re = f64_coerce(complex_dialect.ReOp(value).result)
        im = f64_coerce(complex_dialect.ImOp(value).result)
        return complex_dialect.CreateOp(T.complex128(), re, im).result

    re = f64_coerce(value)
    im = constant(0.0, T.f64())
    return complex_dialect.CreateOp(T.complex128(), re, im).result


# ============================================================================ #
# Quantum operations
# ============================================================================ #


@dataclass(frozen=True, slots=True)
class Error:
    message: str


def qalloca() -> Value:
    return QRefValue(quake.AllocaOp(T.qref()).result)


def _check_controls_and_targets(controls: list[Value],
                                targets: list[Value]) -> Error | None:
    for i, control in enumerate(controls):
        if not T.is_quantum_type(control.type):
            return Error(f'control operand {i} is not of quantum type.')
    for i, target in enumerate(targets):
        if not T.is_quantum_type(target.type):
            return Error(f'target operand {i} is not of quantum type.')
    return None


class Gate:
    __slots__ = ('op', 'num_parameters', 'num_targets', 'is_adjoint')

    def __init__(self,
                 name,
                 *,
                 num_parameters=0,
                 num_targets=1,
                 is_adjoint=False):
        self.op = getattr(quake, '{}Op'.format(name.title()))
        self.num_parameters = num_parameters
        self.num_targets = num_targets
        self.is_adjoint = is_adjoint

    def __call__(self, args, *, is_adjoint: bool = False):
        targets = args[self.num_parameters:]
        parameters = args[:self.num_parameters]
        parameters = [f64_coerce(param) for param in parameters]
        is_adjoint = is_adjoint ^ self.is_adjoint
        if self.num_targets == 1:
            for target in targets:
                if isinstance(target, VeqValue):
                    length = target.get_length()
                    for_loop(
                        length,
                        lambda i: self.op([],
                                          parameters, [], [target.get_item(i)],
                                          is_adj=is_adjoint))
                elif isinstance(target, QRefValue):
                    self.op([], parameters, [], [target], is_adj=is_adjoint)
                else:
                    return Error(
                        f'quantum operation {self.op.__name__} on incorrect quantum type {target.type}.'
                    )
        else:
            assert len(targets) == self.num_targets
            self.op([], parameters, [], targets, is_adj=is_adjoint)

    def adj(self, args):
        return self(args, is_adjoint=True)

    def ctrl(self, args):
        targets = args[-self.num_targets:]
        parameters = args[:self.num_parameters]
        parameters = [f64_coerce(param) for param in parameters]
        controls = args[self.num_parameters:-self.num_targets]
        self.op([], parameters, controls, targets, is_adj=self.is_adjoint)


# One-target gates
h = Gate('h', num_targets=1)
s = Gate('s', num_targets=1)
sdg = Gate('s', num_targets=1, is_adjoint=True)
t = Gate('t', num_targets=1)
tdg = Gate('t', num_targets=1, is_adjoint=True)
x = Gate('x', num_targets=1)
y = Gate('y', num_targets=1)
z = Gate('z', num_targets=1)

# One-target gates with parameters
r1 = Gate('r1', num_parameters=1, num_targets=1)
rx = Gate('rx', num_parameters=1, num_targets=1)
ry = Gate('ry', num_parameters=1, num_targets=1)
rz = Gate('rz', num_parameters=1, num_targets=1)
u3 = Gate('u3', num_parameters=3, num_targets=1)

# Two-target gates
swap = Gate('swap', num_targets=2)


def reset(target: Value) -> Error | None:
    result = _check_controls_and_targets([], [target])
    if isinstance(result, Error):
        return result

    # TODO: Support quantum structs ?
    if quake.RefType.isinstance(target.type):
        quake.ResetOp([], target)
        return None
    if isinstance(target, VeqValue):
        length = target.get_length()
        for_loop(length, lambda i: quake.ResetOp([], target.get_item(i)))
        return None
    return Error(f"Reset quantum operation on incorrect type {target.type}")


def _measure(op: Callable,
             targets: list[Value] | Value,
             *,
             label: str | None = None) -> Error | None:

    result = _check_controls_and_targets([], targets)
    if isinstance(result, Error):
        return result

    if isinstance(targets[0], VeqValue) or len(targets) > 1:
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


def exp_pauli(rotation: Value, qubits: list[Value],
              pauli_word: str | StringLiteral):
    if isinstance(pauli_word, StringLiteral):
        pauli_word = pauli_word.materialize()
    _check_controls_and_targets([], [qubits])
    rotation = f64_coerce(rotation)
    quake.ExpPauliOp([], [rotation], [], [qubits], pauli=pauli_word)


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


def control(symbol_or_value: FlatSymbolRefAttr | Value,
            controls: list[Value] | Value, args: list[Value] | Value):
    if isinstance(symbol_or_value, Value):
        quake.ApplyOp([], [symbol_or_value], [controls], args)
        return
    if FlatSymbolRefAttr.isinstance(symbol_or_value):
        # TODO: This is a hack to get the name from the symbol.
        name = FlatSymbolRefAttr(symbol_or_value).value[len(nvqppPrefix):]
        if len(args) != len(globalKernelRegistry[name].arguments):
            return Error(
                f"Control of {name} requires {len(globalKernelRegistry[name].arguments)} arguments, got {len(args)}"
            )
        quake.ApplyOp([], [], [controls], args, callee=symbol_or_value)
        return


def compute_action(compute, action):
    quake.ComputeActionOp(compute, action)


# ============================================================================ #
# Math functions
# ============================================================================ #


def cos(value):
    """
    Compute the cosine of a value. If the input is a complex number, returns the complex cosine.
    Otherwise returns the real cosine.

    NOTE: This function follows the NumPy convetion of requiring 64-bit floating point numbers.
    Thus the input value is cast to f64 (or complex128) before computing the cosine.

    Args:
        value: The input value to compute cosine of. Can be real or complex.

    Returns:
        The cosine of the input value. For complex inputs, returns complex128 type.
        For real inputs, returns float64 type.
    """
    if ComplexType.isinstance(value.type):
        return complex_dialect.CosOp(complex128_coerce(value)).result
    return math.CosOp(f64_coerce(value)).result


def sin(value):
    """
    Compute the sine of a value. If the input is a complex number, returns the complex sine.
    Otherwise returns the real sine.

    NOTE: This function follows the NumPy convetion of requiring 64-bit floating point numbers.
    Thus the input value is cast to f64 (or complex128) before computing the sine.

    Args:
        value: The input value to compute cosine of. Can be real or complex.

    Returns:
        The cosine of the input value. For complex inputs, returns complex128 type.
        For real inputs, returns float64 type.
    """
    if ComplexType.isinstance(value.type):
        return complex_dialect.SinOp(complex128_coerce(value)).result
    return math.SinOp(f64_coerce(value)).result


def sqrt(value):
    """
    Compute the square root of a value. If the input is a complex number, returns the complex square root.
    Otherwise returns the real square root.

    NOTE: This function follows the NumPy convetion of requiring 64-bit floating point numbers.
    Thus the input value is cast to f64 (or complex128) before computing the square root.

    Args:
        value: The input value to compute cosine of. Can be real or complex.

    Returns:
        The square root of the input value. For complex inputs, returns complex128 type.
        For real inputs, returns float64 type.
    """
    if ComplexType.isinstance(value.type):
        return complex_dialect.SqrtOp(complex128_coerce(value)).result
    return math.SqrtOp(f64_coerce(value)).result


def exp(value):
    """
    Compute the exponential of a value. If the input is a complex number, returns the complex exponential.
    Otherwise returns the real exponential.

    NOTE: This function follows the NumPy convetion of requiring 64-bit floating point numbers.
    Thus the input value is cast to f64 (or complex128) before computing the exponential.

    Args:
        value: The input value to compute cosine of. Can be real or complex.

    Returns:
        The exponential of the input value. For complex inputs, returns complex128 type.
        For real inputs, returns float64 type.
    """
    if ComplexType.isinstance(value.type):
        # Note: using `complex.ExpOp` results in a
        # "can't legalize `complex.exp`" error.
        # Using Euler's' formula instead:
        #
        # "e^(x+i*y) = (e^x) * (cos(y)+i*sin(y))"
        value = complex128_coerce(value)

        real = complex_dialect.ReOp(value).result
        left = complex128_coerce(math.ExpOp(real).result)

        im = complex_dialect.ImOp(value).result
        re = math.CosOp(im).result
        im = math.SinOp(im).result
        right = complex_dialect.CreateOp(value.type, re, im).result
        return complex_dialect.MulOp(left, right).result
    return math.ExpOp(f64_coerce(value)).result


def ceil(value):
    if ComplexType.isinstance(value.type):
        self.emitFatalError(
            f"numpy call ({node.func.attr}) is not supported for complex numbers",
            node)
        return
    return math.CeilOp(value).result


# ============================================================================ #
# Value Adapters
# ============================================================================ #


@dataclass(frozen=True, slots=True)
class PointerValue:
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

    def __getitem__(self, index: Value | int) -> PointerValue:
        return PointerValue(_compute_ptr(self.value, index))

    @property
    def type(self) -> Type:
        return self.value.type


@dataclass(frozen=True, slots=True)
class QRefValue:
    value: Value

    @property
    def type(self) -> Type:
        return self.value.type


@dataclass(frozen=True, slots=True)
class ComplexValue:
    value: Value

    @property
    def real(self) -> Value:
        return complex_dialect.ReOp(self.value).result

    @property
    def imag(self) -> Value:
        return complex_dialect.ImOp(self.value).result

    @property
    def type(self) -> Type:
        return self.value.type


# TODO: Should we differentiate between structs and tuples?
@dataclass(frozen=True, slots=True)
class StructValue:
    value: Value

    @staticmethod
    def create(fields: list[Type], *, name: str | None = None) -> StructValue:
        return StructValue(cc.UndefOp(T.struct(fields, name)).result)

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
        field_type = cc.StructType.getTypes(self.value.type)
        assert field_index < len(field_type)
        field_type = field_type[field_index]
        return cc.ExtractValueOp(field_type, self.value, [],
                                 DenseI32ArrayAttr.get([field_index])).result

    def set_field(self, name_or_index: str | int, value: Value) -> StructValue:
        field_index = name_or_index
        if isinstance(name_or_index, str):
            field_index = self._get_field_index(name_or_index)
        field_type = cc.StructType.getTypes(self.value.type)
        assert field_index < len(field_type)
        field_type = field_type[field_index]
        value = implicit_conversion(value, field_type)
        return StructValue(
            cc.InsertValueOp(self.type, self.value, value,
                             DenseI64ArrayAttr.get([field_index])).result)

    @property
    def type(self) -> Type:
        return self.value.type


@dataclass(frozen=True, slots=True)
class StruqValue:
    value: Value

    @staticmethod
    def create(args: list[Value], *, name: str | None = None) -> StruqValue:
        types = [arg.type for arg in args]
        struq_type = T.struq(types, name)
        return StruqValue(quake.MakeStruqOp(struq_type, args).result)

    @property
    def name(self) -> str:
        return quake.StruqType.getName(self.value.type)

    @property
    def num_fields(self) -> int:
        return len(cc.StructType.getTypes(self.value.type))

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


class Iterable(ABC):
    """Base class for all iterable types"""

    @abstractmethod
    def get_length(self):
        """
        Returns the length of the iterable
        """
        pass

    @abstractmethod
    def get_item(self, index):
        """
        Get item(s) at the given index
        Returns a list of values (for unpacking/enumerate support)
        """
        pass


def _fix_negative_index(index: Value, length: Value) -> Value:
    i64 = T.i64(index.type.context)
    zero = constant(0, i64)
    index_lt_zero = arith.CmpIOp(IntegerAttr.get(i64, 2), index, zero)
    reverse_index = arith.AddIOp(index, length).result
    index = arith.SelectOp(index_lt_zero, reverse_index, index).result
    return index


@dataclass(frozen=True, slots=True)
class VeqValue(Iterable):
    value: Value

    @staticmethod
    def _allocate_memory(length: Value) -> Value:
        return quake.AllocaOp(T.veq(), size=length).result

    @staticmethod
    def create(length: Value) -> VeqValue:
        return VeqValue(VeqValue._allocate_memory(length))

    @staticmethod
    def from_list(values) -> VeqValue:
        length = values.get_length()
        data = values.data()
        num_qubits = math.CountTrailingZerosOp(length).result

        # TODO: Dynamically check if number of qubits is power of 2
        # and if the state is normalized

        qubits = VeqValue._allocate_memory(num_qubits)
        # TODO: Check why this operation returns a new veq
        init = quake.InitializeStateOp(qubits.type, qubits, data).result
        return VeqValue(init)

    @staticmethod
    def from_state(state: Value) -> VeqValue:
        num_qubits = quake.GetNumberOfQubitsOp(T.i64(), state).result
        qubits = VeqValue._allocate_memory(num_qubits)
        init = quake.InitializeStateOp(qubits.type, qubits, state).result
        return VeqValue(init)

    @staticmethod
    def from_qubits(qubits: list[Value]) -> VeqValue:
        return VeqValue(quake.ConcatOp(T.veq(), qubits).result)

    def get_length(self) -> Value:
        return quake.VeqSizeOp(T.i64(), self.value).result

    def get_item(self, index: Value) -> QRefValue:
        index = _fix_negative_index(index, self.get_length())
        return QRefValue(
            quake.ExtractRefOp(self.elements_type, self.value, -1,
                               index=index).result)

    def get_slice(self, begin: Value | None, end: Value | None) -> VeqValue:
        # TODO: Bound check
        dynamic_index = IntegerAttr.get(T.i64(), -1)
        if begin is None:
            begin = constant(0)
        else:
            begin = _fix_negative_index(begin, self.get_length())
        if end is None:
            end = self.get_length()
        else:
            end = _fix_negative_index(end, self.get_length())
        end = arith.SubIOp(end, constant(1)).result
        return VeqValue(
            quake.SubVeqOp(self.value.type,
                           self.value,
                           dynamic_index,
                           dynamic_index,
                           lower=begin,
                           upper=end).result)

    def back(self, num_elements: Value | None = None) -> Value:
        # TODO: Bound check
        size = self.get_length()
        if num_elements is None:
            last = arith.SubIOp(size, constant(1)).result
            return self.get_item(last)
        begin = arith.SubIOp(size, num_elements).result
        return self.get_slice(begin, size)

    def front(self, num_elements: Value | None = None) -> Value:
        # TODO: Bound check
        zero = constant(0)
        if num_elements is None:
            return self.get_item(zero)
        return self.get_slice(zero, num_elements)

    @property
    def elements_type(self) -> Type:
        return quake.RefType.get()

    @property
    def type(self) -> Type:
        return self.value.type


@dataclass(frozen=True, slots=True)
class StdvecValue(Iterable):
    value: Value

    @staticmethod
    def _allocate_memory(elements_type: Type, length: Value) -> Value:
        return alloca(elements_type, size=length)

    @staticmethod
    def create(elements_type: Type, length: Value) -> StdvecValue:
        data = StdvecValue._allocate_memory(elements_type, length)
        vec = cc.StdvecInitOp(T.stdvec(elements_type), data,
                              length=length).result
        return StdvecValue(vec)

    @staticmethod
    def from_list(values: list[Value]) -> StdvecValue:
        # TODO: Check if values are homogeneous
        elements_type = values[0].type
        length = constant(len(values))
        data = StdvecValue._allocate_memory(elements_type, length)
        for i, value in enumerate(values):
            data[i].store(value)
        vec = cc.StdvecInitOp(T.stdvec(elements_type), data,
                              length=length).result
        return StdvecValue(vec)

    def get_length(self) -> Value:
        return cc.StdvecSizeOp(T.i64(), self.value).result

    def data(self) -> Value:
        ptr_type = T.ptr(T.array(self.elements_type))
        return PointerValue(cc.StdvecDataOp(ptr_type, self.value).result)

    def get_item_address(self, index: Value) -> Value:
        index = _fix_negative_index(index, self.get_length())
        data = self.data()
        item_addr = data[index]
        return item_addr

    def get_item(self, index: Value):
        item_addr = self.get_item_address(index)
        return try_downcast(load(item_addr))

    def set_item(self, index, value):
        item_addr = self.get_item_address(index)
        store(value, item_addr)

    def get_slice(self, begin: Value | None, end: Value | None):
        if begin is None:
            begin = constant(0)
        if end is None:
            end = self.get_length()
        else:
            end = _fix_negative_index(end, self.get_length())
        begin_addr = self.get_item_address(begin)
        length = arith.SubIOp(end, begin).result
        return StdvecValue(
            cc.StdvecInitOp(self.value.type, begin_addr, length=length).result)

    def back(self, num_elements: Value | None = None) -> Value:
        size = self.get_length()
        if num_elements is None:
            last = arith.SubIOp(size, constant(1)).result
            return self.get_item(last)
        begin = arith.SubIOp(size, num_elements).result
        return self.get_slice(begin, size)

    def front(self, num_elements: Value | None = None) -> Value:
        zero = constant(0)
        if num_elements is None:
            return self.get_item(zero)
        return self.get_slice(zero, num_elements)

    def check_contains(self, value: Value):
        size = self.get_length()

        result = alloca(T.i1())
        store(constant(False), result)

        def body_builder(index):
            element = self.get_item(index)
            # TODO: passing a instance of ast predicate seems wrong
            cmp = compare(element, value, ast.Eq())
            current = load(result)
            store(arith.OrIOp(current, cmp), result)

        for_loop(size, body_builder)
        return load(result)

    @property
    def elements_type(self) -> Type:
        return cc.StdvecType.getElementType(self.value.type)

    @property
    def type(self) -> Type:
        return self.value.type


class AbstractValue:
    pass


@dataclass(frozen=True, slots=True)
class StringLiteral(AbstractValue):
    value: str

    def materialize(self) -> Value:
        return cc.CreateStringLiteralOp(T.ptr(T.i8()),
                                        StringAttr.get(self.value)).result


class Range(AbstractValue, Iterable):
    __slots__ = ('start', 'step', 'stop')

    def __init__(self, *args):
        self.start = None
        self.step = None
        self.stop = None
        if len(args) == 1:
            self.stop = args[0]
        elif len(args) == 2:
            self.start = args[0]
            self.stop = args[1]
        elif len(args) == 3:
            self.start = args[0]
            self.stop = args[1]
            self.step = args[2]
        else:
            raise ValueError("Range requires 1 to 3 integer arguments")

    def get_length(self) -> Value:
        # Useful helpers
        i64 = T.i64()
        zero = constant(0)
        one = constant(1)

        start = i64_coerce(self.start or zero)
        step = i64_coerce(self.step or one)
        stop = i64_coerce(self.stop)

        # We need to check for a empty range
        diff = arith.SubIOp(stop, start).result

        # First condition: step > 0 and diff <= 0
        step_gt_zero = arith.CmpIOp(IntegerAttr.get(i64, 4), step, zero)
        diff_le_zero = arith.CmpIOp(IntegerAttr.get(i64, 3), diff, zero)
        first_condition = arith.AndIOp(step_gt_zero, diff_le_zero)

        # Second condition: step < 0 and diff >= 0
        step_lt_zero = arith.CmpIOp(IntegerAttr.get(i64, 2), step, zero)
        diff_ge_zero = arith.CmpIOp(IntegerAttr.get(i64, 5), diff, zero)
        second_condition = arith.AndIOp(step_lt_zero, diff_ge_zero)

        is_empty = arith.OrIOp(first_condition, second_condition)

        step = math.AbsIOp(step)
        length = arith.CeilDivSIOp(math.AbsIOp(diff).result, step).result
        return arith.SelectOp(is_empty, zero, length).result

    def get_item(self, index: Value) -> Value:
        start = i64_coerce(self.start or constant(0))
        step = i64_coerce(self.step or constant(1))
        return arith.AddIOp(start, arith.MulIOp(index, step)).result

    @property
    def elements_type(self) -> Type:
        return T.i64()

    def materialize(self) -> StdvecValue:
        size = self.get_length()
        vec = StdvecValue.create(T.i64(), size)

        def body_builder(index):
            value = self.get_item(index)
            vec.set_item(index, value)

        loop = for_loop(size, body_builder)
        loop.attributes['invariant'] = UnitAttr.get()
        return vec

    def __repr__(self):
        return f"Range(start={self.start}, stop={self.stop}, step={self.step})"


class Enumerate(AbstractValue, Iterable):
    __slots__ = ('iterable', 'start')

    def __init__(self, iterable, start=None):
        self.iterable = iterable
        assert isinstance(self.iterable, Iterable)
        if self.iterable is None:
            raise CompilerError()
        self.start = start

    def get_length(self) -> Value:
        return self.iterable.get_length()

    def get_item(self, index: Value) -> list[Value]:
        base_values = self.iterable.get_item(index)
        if self.start is not None:
            start = i64_coerce(self.start)
            index = arith.AddIOp(index, start).result

        return [index, base_values]

    @property
    def elements_type(self) -> Type:
        element_type = self.iterable.elements_type
        return T.struct([T.i64(), element_type])

    def __repr__(self):
        return f"Enumerate(iterable={self.iterable}, start={self.start})"


class Tuple(AbstractValue):
    __slots__ = ('values',)

    def __init__(self, *values):
        self.values = tuple(values)

    def __getitem__(self, index: int) -> AbstractValue:
        return self.values[index]

    def __len__(self) -> int:
        return len(self.values)

    def __iter__(self):
        return iter(self.values)

    def materialize(self) -> StructValue:
        types = [value.type for value in self.values]
        struct = StructValue.create(types, name="tuple")
        for i, value in enumerate(self.values):
            struct = struct.set_field(i, value)
        return struct


@dataclass(frozen=True, slots=True)
class AList(AbstractValue):
    values: list

    def materialize(self, with_type: Type | None = None) -> StdvecValue:
        elements = self.values
        if isinstance(elements[0], QRefValue):
            return VeqValue.from_qubits(elements)
        if with_type is not None:
            if cc.StdvecType.isinstance(with_type):
                with_type = cc.StdvecType.getElementType(with_type)
            else:
                raise ValueError(f"Unsupported type: {with_type}")
            elements = [
                implicit_conversion(element, with_type) for element in elements
            ]
        return StdvecValue.from_list(elements)
