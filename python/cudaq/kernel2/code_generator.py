# ============================================================================ #
# Copyright (c) 2022 - 2025 NVIDIA Corporation & Affiliates.                   #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

import ast
from collections.abc import Iterable
from enum import IntEnum
import inspect
from textwrap import dedent

from cudaq.mlir.ir import (
    Context,
    DictAttr,
    Block,
    BoolAttr,
    F32Type,
    F64Type,
    FunctionType,
    InsertionPoint,
    IndexType,
    IntegerAttr,
    IntegerType,
    Location,
    Module,
    StringAttr,
    TypeAttr,
    UnitAttr,
    Value as IRValue,
)
from cudaq.kernel.captured_data import CapturedDataStorage
from cudaq.kernel.utils import globalKernelRegistry
from cudaq.mlir.dialects import arith, cc, func, quake
from cudaq.mlir._mlir_libs._quakeDialects import cudaq_runtime, register_all_dialects
from .diagnostic_emitter import DiagnosticEmitter
from .type_converter import TypeConverter
from .utils import ScopedTable

#-------------------------------------------------------------------------------
# Misc helpers
#-------------------------------------------------------------------------------

# TOOD: Perhaps find a better place to put these.

def _isa(obj, cls):
  try:
    cls(obj)
  except ValueError:
    return False
  return True

def _is_any_of(obj, classes):
  return any(_isa(obj, cls) for cls in classes)

def _is_integer_like_type(type):
  return _is_any_of(type, [IntegerType, IndexType])

def _is_float_type(type):
  return _is_any_of(type, [F32Type, F64Type])

class _CmpIPredicate(IntEnum):
    eq  = 0 # equal
    ne  = 1 # not equal
    slt = 2 # signed less than
    sle = 3 # signed less than or equal
    sgt = 4 # signed greater than
    sge = 5 # signed greater than or equal
    ult = 6 # unsigned less than
    ule = 7 # unsigned less than or equal
    ugt = 8 # unsigned greater than
    uge = 9 # unsigned greater than or equal

def _arith_cmpi_predicate_attr(predicate):
    ty = IntegerType.get_signless(64)
    match predicate:
        case ast.Eq:
            return IntegerAttr.get(ty, _CmpIPredicate.eq)
        case ast.NotEq:
            return IntegerAttr.get(ty, _CmpIPredicate.ne)
        case ast.Lt:
            return IntegerAttr.get(ty, _CmpIPredicate.slt)
        case ast.LtE:
            return IntegerAttr.get(ty, _CmpIPredicate.sle)
        case ast.Gt:
            return IntegerAttr.get(ty, _CmpIPredicate.sgt)
        case ast.GtE:
            return IntegerAttr.get(ty, _CmpIPredicate.sge)
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
    ty = IntegerType.get_signless(64)
    match predicate:
        case ast.Eq:
            return IntegerAttr.get(ty, _CmpFPredicate.oeq)
        case ast.NotEq:
            return IntegerAttr.get(ty, _CmpFPredicate.one)
        case ast.Lt:
            return IntegerAttr.get(ty, _CmpFPredicate.olt)
        case ast.LtE:
            return IntegerAttr.get(ty, _CmpFPredicate.ole)
        case ast.Gt:
            return IntegerAttr.get(ty, _CmpFPredicate.ogt)
        case ast.GtE:
            return IntegerAttr.get(ty, _CmpFPredicate.oge)
        case _:
            return None


class Value:
    __slots__ = ('py_type', 'ir_value')

    def __init__(self, py_type, ir_value):
        self.py_type = py_type
        self.ir_value = ir_value

class Range:
    __slots__ = ('start', 'step', 'stop')

    def __init__(self, *args):
        self.start = 0
        self.step = 1
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

    # Add this method this class is a Iterable
    def __iter__(self):
        pass

    def __repr__(self):
        return f"Range(start={self.start}, stop={self.stop}, step={self.step})"

#-------------------------------------------------------------------------------
# Builtin namespace
#-------------------------------------------------------------------------------

class qubit:
    __slots__ = ()

    @classmethod
    def _mlir_alloca(cls, loc):
        return quake.AllocaOp(quake.RefType.get(loc.context), loc=loc).result

class qvector:
    __slots__ = ()

    @classmethod
    def _mlir_alloca(cls, size, loc):
        # TODO?
        #if isinstance(size.owner.opview, arith.ConstantOp):
        #    size = size.owner.opview.value
        #    return quake.AllocaOp(quake.VeqType.get(loc.context, size), loc=loc).result
        return quake.AllocaOp(quake.VeqType.get(loc.context), size=size, loc=loc).result

class Gate:
    __slots__ = ('op', 'num_parameters', 'num_targets', 'is_adjoint')

    def __init__(self, name, *, num_parameters=0, num_targets=1, is_adjoint=False):
        self.op = getattr(quake, '{}Op'.format(name.title()))
        self.num_parameters = num_parameters
        self.num_targets = num_targets
        self.is_adjoint = is_adjoint

    def _apply(self, parameters, controls, targets, *, is_adjoint=False, loc):
        # Ouch, need to account for the possibility of returning wires
        self.op([], parameters, controls, targets, is_adj=is_adjoint ^ self.is_adjoint, loc=loc)


builtin_namespace = {
    'cudaq.qvector': qvector,
    'cudaq.qubit': qubit,
    # One-target gates
    'x': Gate('x'),
    'y': Gate('y'),
    'z': Gate('z'),
    'h': Gate('h'),
    's': Gate('s'),
    'sdg': Gate('s', is_adjoint=True),
    't': Gate('t'),
    'tdg': Gate('t', is_adjoint=True),
    # One-target, parameterized gates
    'r1': Gate('r1', num_parameters=1),
    'rx': Gate('rx', num_parameters=1),
    'ry': Gate('ry', num_parameters=1),
    'rz': Gate('rz', num_parameters=1),
}


#-------------------------------------------------------------------------------
# Visitor
#-------------------------------------------------------------------------------

class CodeGenerator(ast.NodeVisitor):

    unsupported_cmpop = {ast.Is, ast.IsNot, ast.In, ast.NotIn}

    # TODO: There are some outside code that depends on the MLIR context attribute
    #       being named `ctx`. We should not have that. 
    def __init__(self, capturedDataStorage: CapturedDataStorage, **kwargs):

        filename, line_number = kwargs.get('locationOffset', ('', 0))

        if 'existingModule' in kwargs:
            self.module = kwargs['existingModule']
            self.ctx = self.module.context
            location = Location.file(filename, line_number, 0, context=self.ctx)
        else:
            self.ctx = Context()
            register_all_dialects(self.ctx)
            quake.register_dialect(self.ctx)
            cc.register_dialect(self.ctx)
            cudaq_runtime.registerLLVMDialectTranslation(self.ctx)
            location = Location.file(filename, line_number, 0, context=self.ctx)
            self.module = Module.create(loc=location)

        # Create a new captured data storage or use the existing one
        # passed from the current kernel decorator.
        self.capturedDataStorage = capturedDataStorage
        if (self.capturedDataStorage == None):
            self.capturedDataStorage = CapturedDataStorage(ctx=self.ctx,
                                                           loc=location,
                                                           name=None,
                                                           module=self.module)
        else:
            self.capturedDataStorage.setKernelContext(ctx=self.ctx,
                                                      loc=location,
                                                      name=None,
                                                      module=self.module)

        self.capturedVars = kwargs.get('capturedVariables', {})
        self.disableEntryPointTag = kwargs.get('disableEntryPointTag', False)
        self.disableNvqppPrefix = kwargs.get('disableNvqppPrefix', False)
        self.knownResultType = kwargs.get('knownResultType', None)
        self.verbose = kwargs.get('verbose', False)

        self.diagnostic = DiagnosticEmitter(filename, line_number)
        self.type_converter = TypeConverter(self.diagnostic, self.ctx)

        self.dependentCaptureVars = {}
        self.local_symbols = ScopedTable()
        self.return_type = None

    #---------------------------------------------------------------------------
    # TODO: Remove these
    #---------------------------------------------------------------------------

    def validateArgumentAnnotations(self, astModule):
        """
        Utility function for quickly validating that we have
        all arguments annotated.
        """

        class ValidateArgumentAnnotations(ast.NodeVisitor):
            """
            Utility visitor for finding argument annotations
            """

            def __init__(self, bridge):
                self.bridge = bridge

            def visit_FunctionDef(self, node):
                for arg in node.args.args:
                    if arg.annotation == None:
                        self.bridge.emitFatalError(
                            'cudaq.kernel functions must have argument type annotations.',
                            arg)

        ValidateArgumentAnnotations(self).visit(astModule)

    #---------------------------------------------------------------------------
    # CC helpers
    #---------------------------------------------------------------------------
    # These helpers won't be necessary with newer LLVM

    def _cc_alloca(self, type_, loc):
        return cc.AllocaOp(cc.PointerType.get(self.ctx, type_), TypeAttr.get(type_), loc=loc).result
    def _cc_load(self, memory, loc):
        return cc.LoadOp(memory, loc=loc).result

    #---------------------------------------------------------------------------
    # Helpers
    #---------------------------------------------------------------------------
    
    def _create_location(self, node):
        return Location.file(
            self.diagnostic.filename,
            getattr(node, "lineno", 0) + self.diagnostic.first_line,
            getattr(node, "col_offset", 0),
            self.ctx
        )

    def _binary_op(self, lhs, rhs, op):
        op = op.capitalize()
        if _is_float_type(lhs.type) and _is_float_type(rhs.type):
            op += "F"
        elif _is_integer_like_type(lhs.type) and _is_integer_like_type(rhs.type):
            op += "I"
        else:
            return None

        op = getattr(arith, f"{op}Op")
        return op(lhs, rhs).result

    def _cmp_op(self, lhs, rhs, predicate):
        if _is_float_type(lhs.type) and _is_float_type(rhs.type):
            predicate_attr = _arith_cmpf_predicate_attr(predicate)
            return arith.CmpFOp(predicate_attr, lhs, rhs).result

        if _is_integer_like_type(lhs.type) and _is_integer_like_type(rhs.type):
            predicate_attr = _arith_cmpi_predicate_attr(predicate)
            return arith.CmpIOp(predicate_attr, lhs, rhs).result

        return None

    #---------------------------------------------------------------------------
    # Generic visiting
    #---------------------------------------------------------------------------

    def visit(self, node):
        return super().visit(node)

    def generic_visit(self, node):
        self.diagnostic.emit_error(
            getattr(node, "lineno", 0),
            getattr(node, "col_offset", 0),
            f"Unsupported AST node {str(node)}",
        )

    #---------------------------------------------------------------------------

    def visit_AnnAssign(self, node):
        target = self.visit(node.target)
        type_ = self.visit(node.annotation)
        value = self.visit(node.value)
        
        print(target)
        print(type_)
        print(value)

    def visit_Assign(self, node):
        # TODO: Make sure we are not assigning a nonlocal or global.
        loc = self._create_location(node)

        # TODO: Support fancier assignments. For now, we limit this to simple
        # <name> = <value>
        if len(node.targets) > 1 or isinstance(node.value, ast.Tuple | ast.List):
            self.diagnostic.emit_error(
                node.lineno, node.col_offset,
                f"Unsupported Assign node {str(node)}",
            )

        if not isinstance(node.targets[0], ast.Name):
            self.diagnostic.emit_error(
                node.lineno, node.col_offset,
                f"Unsupported Assign node {str(node)}",
            )

        # Symbol resolution on assigment is a bit different
        name = node.targets[0].id
        target = self.visit(node.targets[0])
        value = self.visit(node.value)

        # If `value` is not a MLIR value, then we are dealing with some python
        # object that has not been represented in the IR, thus we just store it
        # as-is in the local symbols table.
        if not isinstance(value, IRValue):
            self.local_symbols.insert(name, value)
            return

        if not target:
            # We are seeing the identifier for the first time. Thus we need to 
            # create a name, which is an identifier that bind to an object.

            # When dealing with quantum types, we just update the local symbols.
            if quake.RefType.isinstance(value.type) or quake.VeqType.isinstance(value.type):
                self.local_symbols.insert(name, value)
                return

            # We are creating a new name
            memory = self._cc_alloca(value.type, loc)
            self.local_symbols.insert(name, memory)
            target = memory
        elif quake.RefType.isinstance(target.type):
            # Quantum types are not SSA values in the IR, they are memory and 
            # we don't have an operation capable of adding an extra level of 
            # indirection here.
            self.diagnostic.emit_error(
                node.lineno, node.col_offset,
                f"Variables of a quantum type are immutable",
            )
        elif cc.PointerType.isinstance(target.type):
            # Check whether types of the existing target and the value are the
            # same. If not, then we need to allocate a new memory and re-assign
            # the name.
            type_ = cc.PointerType.getElementType(target.type)
            if type_ != value.type:
                self.diagnostic.emit_error(
                    node.lineno, node.col_offset,
                    f"Cannot assing a value of type {value.type} to a name that previously contained a value of type {type_}",
                )

        cc.StoreOp(value, target, loc=loc)

    def visit_Attribute(self, node):
        # Avoid recursing in the cases where we have "foo.bar.zaz"
        new_node = node.value
        chain = [node.attr]
        while isinstance(new_node, ast.Attribute):
            chain.append(new_node.attr)
            new_node = new_node.value

        if not isinstance(new_node, ast.Name):
            self.diagnostic.emit_error(
                node.lineno, node.col_offset,
                f"Unsupported attribute node {str(node)}",
            )

        # FIXME: Eventually we should properly handle these by resolving objects
        #        appropriately. For now, we just build a name node with the full
        #        attribute chain as a string and visit it.
        chain.append(new_node.id)
        full_name = '.'.join(reversed(chain))
        return self.visit(ast.Name(full_name, ast.Load(), lineno=node.lineno, col_offset=node.col_offset))

        # TODO: In the future, we should do something likethis:
        # obj = self.visit(new_node)
        # for attr in reversed(chain):
        #     # Using subscript is faster than `getattr` for dict.
        #     if isinstance(obj, dict):
        #         obj = obj[attr]
        #     else:
        #         obj = getattr(obj, attr)

        # return obj

    def visit_BinOp(self, node):
        left = self.visit(node.left)
        right = self.visit(node.right)

        if isinstance(node.op, (ast.Add, ast.Sub)):
            right = self.type_converter.coerce_value_type(left.type, right)
            left = self.type_converter.coerce_value_type(right.type, left)

        result = None
        match node.op:
            case ast.Add():
                result = self._binary_op(left, right, op="add")
            case ast.BitAnd():
                result = self._binary_op(left, right, op="and")
            case ast.Sub():
                result = self._binary_op(left, right, op="sub")
            case _:
                self.diagnostic.emit_error(
                    node.lineno, node.col_offset,
                    f"Unsupported operator {ast.unparse(node)} {node.op}"
                )

        if result:
            return result

        self.diagnostic.emit_error(
            node.lineno, node.col_offset,
            f"Error when handling operator {ast.unparse(node)} {node.op}"
        )

    def visit_Break(self, node):
        # TODO: Handle edge cases? This statement only make sense with an 
        # ancestor scope is a loop, `for` or `while`
        cc.UnwindBreakOp([])

    def visit_Continue(self, node):
        # TODO: Handle edge cases? This statement only make sense with an 
        # ancestor scope is a loop, `for` or `while`
        cc.UnwindContinueOp([])

    def visit_Call(self, node):
        # FIXME: This a big hack, we need it because attributes are not handled properly.
        func = node.func
        is_adjoint = False
        is_control = False
        if isinstance(node.func, ast.Attribute):
            if node.func.attr == 'adj':
                is_adjoint = True
                func = node.func.value
            elif node.func.attr == 'ctrl':
                is_control = True
                func = node.func.value
                
        fn = self.visit(func)
        args = [self.visit(arg) for arg in node.args]
        loc = self._create_location(node)

        # Check if we are dealing with a type. In such cases we are dealing with
        # their initializer, probably...
        if isinstance(fn, type):
            if hasattr(fn, '_mlir_alloca'):
                return fn._mlir_alloca(*args, loc)
            return fn(*args)

        if isinstance(fn, Gate):
            targets = args[fn.num_parameters:]
            parameters = args[0:fn.num_parameters]
            fn._apply(parameters, [], targets, is_adjoint=is_adjoint, loc=loc)
            return None

        self.diagnostic.emit_error(
            getattr(node, "lineno", 0),
            getattr(node, "col_offset", 0),
            f"Unsupported call {ast.unparse(node)}",
        )

    def visit_Compare(self, node):
        # TODO: For now, allow a single comparison only.
        if len(node.comparators) != 1 or len(node.ops) != 1:
            self.diagnostic.emit_error(
                node.lineno, node.col_offset,
                f"Expected a single comparator, but found {len(node.comparators)}.",
            )

        cmp_op = type(node.ops[0])
        if cmp_op in self.unsupported_cmpop:
            self.diagnostic.emit_error(
                node.lineno, node.col_offset,
                f"Unsupported comparison operation '{cmp_op}'.",
            )
        left = self.visit(node.left)
        right = self.visit(node.comparators[0])
        if left.type != right.type:
            self.diagnostic.emit_error(
                node.lineno, node.col_offset,
                f"Expected the same types for comparison operator '{cmp_op}', but got {left.type} and {right.type}."
            )

        return self._cmp_op(left, right, cmp_op)

    def visit_Constant(self, node):
        loc = self._create_location(node)
        ty = self.type_converter.convert_type(type(node.value))
        return arith.ConstantOp(ty, node.value, loc=loc).result

    def visit_Expr(self, node):
        if not isinstance(node.value, ast.Call):
            self.diagnostic.emit_error(
                getattr(node, "lineno", 0),
                getattr(node, "col_offset", 0),
                f"Unsupported expression node {str(node)}",
            )

        return self.visit(node.value)

    def visit_For(self, node):
        # We don't support all sort of iterables, for now we only support
        # iterables that are vector-like, i.e. have a specified size or can
        # be queried for its size.

        iter = self.visit(node.iter)
        if quake.VeqType.isinstance(iter.type):
            size = quake.VeqType.getSize(iter.type)
            if quake.VeqType.hasSpecifiedSize(iter.type):
                size = arith.ConstantOp(IntegerType.get_signless(64, self.ctx), size).result
            else:
                size = quake.VeqSizeOp(IntegerType.get_signless(64, self.ctx), iter).result

            extractFunctor = lambda idx: [quake.ExtractRefOp(quake.RefType.get(self.ctx), iter, -1, index=idx).result]

        
        int64_ty = IntegerType.get_signless(64, self.ctx)
        start = arith.ConstantOp(int64_ty, 0).result
        step = arith.ConstantOp(int64_ty, 1).result

        loop = cc.LoopOp([int64_ty], [start], BoolAttr.get(False))

        whileBlock = Block.create_at_start(loop.whileRegion, [int64_ty])
        with InsertionPoint(whileBlock):
            test = self._cmp_op(whileBlock.arguments[0], size, ast.Lt)
            cc.ConditionOp(test, whileBlock.arguments)

        bodyBlock = Block.create_at_start(loop.bodyRegion, [int64_ty])
        with InsertionPoint(bodyBlock):
            self.local_symbols.push_scope()
            targets = [node.target.id] if isinstance(node.target, ast.Name) else [elt.id for elt in node.target.elts]
            values = extractFunctor(bodyBlock.arguments[0])
            for target, value in zip(targets, values):
                self.local_symbols.insert(target, value)
            for stmt in node.body:
                self.visit(stmt)
            cc.ContinueOp(bodyBlock.arguments)
            self.local_symbols.pop_scope()

        stepBlock = Block.create_at_start(loop.stepRegion, [int64_ty])
        with InsertionPoint(stepBlock):
            incr = arith.AddIOp(stepBlock.arguments[0], step).result
            cc.ContinueOp([incr])

    def visit_FunctionDef(self, node):
        print(ast.dump(node, indent=4))

        # Process arguments
        args = []
        # TODO: This attribute is required by users of the code generator.
        #       We not have it in this way.
        self.argTypes = []
        for arg in node.args.args:
            args.append(arg.arg)
            if arg.annotation is None:
                self.diagnostic.emit_error(
                    arg.lineno, arg.col_offset,
                    f"Unsupported typeless argument {str(arg.arg)}",
                )
            self.argTypes.append(self.type_converter.convert_type_hint(arg.annotation))

        # the full function name in MLIR is `__nvqpp__mlirgen__` + the function name
        name = node.name if self.disableNvqppPrefix else f"__nvqpp__mlirgen__{node.name}"

        # It is a bit weird to have to set this attribute in the bridge instead of having
        # the compiler to figure it out
        attr = DictAttr.get(
            {
                name:
                    StringAttr.get(
                        f'{name}_PyKernelEntryPointRewrite',
                        context=self.ctx)
            },
            context=self.ctx)
        self.module.operation.attributes['quake.mangled_name_map'] = attr

        # Create `func` operation within the MLIR module
        loc = self._create_location(node)
        with InsertionPoint(self.module.body), loc:

            func_op = func.FuncOp(name, (self.argTypes, []))
            func_op.attributes["cudaq-kernel"] = UnitAttr.get()

            entry_block = func_op.add_entry_block()
            for arg_name, arg in zip(args, entry_block.arguments):
                self.local_symbols.insert(arg_name, arg)
            with InsertionPoint(entry_block):
                for stmt in node.body:
                    if isinstance(stmt, ast.FunctionDef):
                        self.diagnostic.emit_error(
                            getattr(node, "lineno", 0),
                            getattr(node, "col_offset", 0),
                            f"Defining functions inside kernels is unsupported",
                        )
                    self.visit(stmt)

                if self.return_type:
                    func_type = FunctionType.get(self.argTypes, [self.return_type])
                    func_op.attributes['function_type'] = TypeAttr.get(func_type)
                else:
                    func.ReturnOp([], loc=loc)

            globalKernelRegistry[node.name] = func_op

    def visit_If(self, node):
        test = self.visit(node.test)

        # FIXME: Handle the case where test is not a bool, coercion!
        # (Think about the None value?)

        if_op = cc.IfOp([], test, [])
        block = Block.create_at_start(if_op.thenRegion, [])
        with InsertionPoint(block):
            self.local_symbols.push_scope()
            for stmt in node.body:
                self.visit(stmt)
            # FIXME: Check for terminator ?
            cc.ContinueOp([])
            self.local_symbols.pop_scope()

        if len(node.orelse) > 0:
            block = Block.create_at_start(if_op.elseRegion, [])
            with InsertionPoint(block):
                self.local_symbols.push_scope()
                for stmt in node.body:
                    self.visit(stmt)
                # FIXME: Check for terminator ?
                cc.ContinueOp([])
                self.local_symbols.pop_scope()

    def visit_Module(self, node):
        assert len(node.body) == 1
        assert isinstance(node.body[0], ast.FunctionDef)
        self.visit(node.body[0])

    def visit_Name(self, node):
        # The ast.Name node has a context that is either `load` or `store`.
        if type(node.ctx) is ast.Store:
            # In the case of a store, we are either (1) creating a new name or
            # (2) re-assigning a name to a new value.
            return self.local_symbols.lookup(node.id)

        if node.id in builtin_namespace:
            return builtin_namespace[node.id]

        value = self.local_symbols.lookup(node.id)
        if value:
            if isinstance(value, IRValue):
                type_ = value.type
                if cc.PointerType.isinstance(type_):
                    type_ = cc.PointerType.getElementType(type_)
                    loc = self._create_location(node)
                    return self._cc_load(value, loc=loc)
            return value


        # When in the `load` context, not finding the symbol is a fatal error.
        self.diagnostic.emit_error(
            node.lineno, node.col_offset,
            f"Unknown variable {node.id}"
        )

    def visit_Return(self, node):
        # Check whether we need to return from the kernel scope or some inner.
        # In the later case, we need to use `cc`'s unwind return operation.
        # For example:
        #
        # def kernel(...):
        #   if <cond>:
        #       ...
        #       return <exp> # kernel return from an inner scope (unwind return)
        #   ...
        #   return <exp> # kernel return from kernel scope (normal return)
        return_op = func.ReturnOp
        if self.local_symbols.depth() > 1:
            return_op = cc.UnwindReturnOp

        loc = self._create_location(node)
        if node.value is None:
            return_op([], loc=loc)
            return

        return_expr = self.visit(node.value)
        self.return_type = return_expr.type
        return_op([return_expr], loc=loc)

    def visit_Tuple(self, node):
        return tuple(self.visit(elt) for elt in node.elts)

    def visit_While(self, node):
        loop = cc.LoopOp([], [], BoolAttr.get(False))
        block = Block.create_at_start(loop.whileRegion, [])
        with InsertionPoint(block):
            test = self.visit(node.test)
            # FIXME: Handle the case where test is not a bool, coercion!
            # (Think about the None value?)

            cc.ConditionOp(test, [])

        block = Block.create_at_start(loop.bodyRegion, [])
        with InsertionPoint(block):
            self.local_symbols.push_scope()
            for stmt in node.body:
                self.visit(stmt)
            # FIXME: Check for terminator ?
            cc.ContinueOp([])
            self.local_symbols.pop_scope()

