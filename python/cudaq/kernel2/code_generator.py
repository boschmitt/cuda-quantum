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
    DenseI32ArrayAttr,
    DenseI64ArrayAttr,
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
from cudaq.mlir.dialects import arith, cc, func, quake, math
from cudaq.mlir._mlir_libs._quakeDialects import cudaq_runtime, register_all_dialects
from .diagnostic import Diagnostic, DiagnosticSeverity, DiagnosticSpan
from .diagnostic_emitter import DiagnosticEmitter
from .type_converter import TypeConverter
from .utils import ScopedTable

#-------------------------------------------------------------------------------
# Misc helpers
#-------------------------------------------------------------------------------

# TOOD: Perhaps find a better place to put these.

# CC Dialect `ComputePtrOp` in C++ sets the dynamic index as `std::numeric_limits<int32_t>::min()`
# (see CCOps.tc line 898). We'll duplicate that here by just setting it manually
kDynamicPtrIndex: int = -2147483648

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


class Symbol:
    __slots__ = ('name', 'def_node', 'ir_type', 'ir_slot')

    def __init__(self, name):
        self.name = name
        self.def_node = None
        self.ir_type = None
        self.ir_slot = None

    def __repr__(self):
        return f"Symbol(name={self.name}, def_node={self.def_node}, ir_type={self.ir_type}, ir_slot={self.ir_slot})"

#-------------------------------------------------------------------------------
# Iterable helpers
#-------------------------------------------------------------------------------

class Iterable(ABC):
    """Base class for all iterable types"""

    @abstractmethod
    def get_loop_params(self, loc):
        """
        Returns loop parameters (start, size, step)
        loc: Source location
        """
        pass

    @abstractmethod
    def extract_element(self, index, loc):
        """
        Extract element(s) at the given index
        Returns a list of values (for unpacking/enumerate support)
        """
        pass

class Range(Iterable):
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

    def get_loop_params(self, loc):
        int64_ty = IntegerType.get_signless(64, loc.context)
        start = self.start if self.start else arith.ConstantOp(int64_ty, 0, loc=loc).result
        size = self.stop
        step = self.step if self.step else arith.ConstantOp(int64_ty, 1, loc=loc).result
        return start, size, step

    def extract_element(self, index, loc):
        return [index]

    def __iter__(self):
        pass  # Keep existing iterator method

    def __repr__(self):
        return f"Range(start={self.start}, stop={self.stop}, step={self.step})"

class Enumerate(Iterable):
    __slots__ = ('iterable', 'start')

    def __init__(self, iterable, start=None):
        self.iterable = iterable
        self.start = start

    def get_loop_params(self, loc):
        # Delegate to the wrapped iterable
        return self.iterable.get_loop_params(loc)

    def extract_element(self, index, loc):
        # Get the base element and add the index
        base_values = self.iterable.extract_element(index, loc)
        return [index] + base_values

    def __iter__(self):
        pass  # Keep existing iterator method

    def __repr__(self):
        return f"Enumerate(iterable={self.iterable}, start={self.start})"

class VeqIterable(Iterable):
    """Iterable adapter for quantum vector values"""

    __slots__ = ('value',)

    def __init__(self, value):
        self.value = value

    def get_loop_params(self, loc):
        int64_ty = IntegerType.get_signless(64, loc.context)
        start = arith.ConstantOp(int64_ty, 0, loc=loc).result
        step = arith.ConstantOp(int64_ty, 1, loc=loc).result

        size = quake.VeqType.getSize(self.value.type)
        if quake.VeqType.hasSpecifiedSize(self.value.type):
            size = arith.ConstantOp(int64_ty, size, loc=loc).result
        else:
            size = quake.VeqSizeOp(int64_ty, self.value, loc=loc).result

        return start, size, step

    def extract_element(self, index, loc):
        return [quake.ExtractRefOp(
            quake.RefType.get(loc.context),
            self.value,
            -1,
            index=index,
            loc=loc
        ).result]

class ArrayIterable(Iterable):
    """Iterable adapter for array values"""

    __slots__ = ('value',)

    def __init__(self, value):
        self.value = value

    def get_loop_params(self, loc):
        int64_ty = IntegerType.get_signless(64, loc.context)
        start = arith.ConstantOp(int64_ty, 0, loc=loc).result
        step = arith.ConstantOp(int64_ty, 1, loc=loc).result

        array_type = cc.PointerType.getElementType(self.value.type)
        size = arith.ConstantOp(int64_ty, cc.ArrayType.getSize(array_type), loc=loc).result

        return start, size, step

    def extract_element(self, index, loc):
        array_type = cc.PointerType.getElementType(self.value.type)
        element_type = cc.ArrayType.getElementType(array_type)
        address = cc.ComputePtrOp(
            cc.PointerType.get(loc.context, element_type),
            self.value,
            [index],
            DenseI32ArrayAttr.get([kDynamicPtrIndex], context=loc.context),
            loc=loc
        ).result
        return [cc.LoadOp(address, loc=loc).result]

class StdvecIterable(Iterable):
    """Iterable adapter for standard vector values"""

    __slots__ = ('value',)

    def __init__(self, value):
        self.value = value

    def get_loop_params(self, loc):
        int64_ty = IntegerType.get_signless(64, loc.context)
        start = arith.ConstantOp(int64_ty, 0, loc=loc).result
        step = arith.ConstantOp(int64_ty, 1, loc=loc).result
        size = cc.StdvecSizeOp(int64_ty, self.value, loc=loc).result
        return start, size, step

    def extract_element(self, index, loc):
        element_type = cc.StdvecType.getElementType(self.value.type)
        ptr_type = cc.PointerType.get(loc.context, element_type)
        array_type = cc.ArrayType.get(loc.context, element_type)
        ptr_array_type = cc.PointerType.get(loc.context, array_type)
        vec_ptr = cc.StdvecDataOp(ptr_array_type, self.value, loc=loc).result
        elem_addr = cc.ComputePtrOp(
            ptr_type,
            vec_ptr,
            [index],
            DenseI32ArrayAttr.get([kDynamicPtrIndex], context=loc.context),
            loc=loc
        ).result
        return [cc.LoadOp(elem_addr, loc=loc).result]

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

class Measurement:
    __slots__ = ('op', 'register_name')

    def __init__(self, name, *, register_name=None):
        self.op = getattr(quake, '{}Op'.format(name.title()))
        self.register_name = register_name

    def _mlir_materialize(self, targets, loc):
        measure_type = quake.MeasureType.get(loc.context)
        result_type = IntegerType.get_signless(1)
        if len(targets) > 1 or quake.VeqType.isinstance(targets[0].type):
            measure_type = cc.StdvecType.get(loc.context, measure_type)
            result_type = cc.StdvecType.get(loc.context, result_type)
        measure_result = self.op(measure_type, [], targets, loc=loc).result
        return quake.DiscriminateOp(result_type, measure_result, loc=loc).result

builtin_namespace = {
    # Python builtins
    'range': Range,
    'enumerate': Enumerate,
    # CUDA-Q builtins    
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
    # Measurements
    'mz': Measurement('mz'),
    'mx': Measurement('mx'),
    'my': Measurement('my'),
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
        self.source = kwargs.get('source', None)

        self.diagnostic = DiagnosticEmitter(filename, line_number, self.source)
        self.type_converter = TypeConverter(self.diagnostic, self.ctx)

        self.dependentCaptureVars = {}
        self.local_symbols = ScopedTable()
        self.value_to_node = ScopedTable()
        self.return_type = None

    #---------------------------------------------------------------------------
    # TODO: Remove these
    #---------------------------------------------------------------------------

    def validateArgumentAnnotations(self, astModule):
        # TODO: Checking for type hints happen in various places python runtime.
        #       We should probably do it in only one place. For now, I leaving it here
        #       because this seems to be the first place.
        class ValidateArgumentAnnotations(ast.NodeVisitor):
            def __init__(self, code_generator):
                self.code_generator = code_generator

            def visit_FunctionDef(self, node):
                for arg in node.args.args:
                    if arg.annotation is None:
                        diagnostic = Diagnostic(
                            DiagnosticSeverity.ERROR,
                            'Missing type annotation for kernel argument.',
                        ).add_span(DiagnosticSpan.from_ast_node(
                            arg,
                            file=self.code_generator.diagnostic.filename,
                            is_primary=True,
                            label=f"Argument has no type annotation",
                        ))
                        self.code_generator.diagnostic.emit(diagnostic)

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
        dialect = arith
        if op == "Pow":
            dialect = math
            if _is_float_type(lhs.type):
                op = f"F{op}"
            elif _is_integer_like_type(lhs.type):
                op = f"I{op}"
            else:
                return None

        if _is_float_type(rhs.type):
            op += "F"
        elif _is_integer_like_type(rhs.type):
            op += "I"
        else:
            return None

        op = getattr(dialect, f"{op}Op")
        return op(lhs, rhs).result

    def _cmp_op(self, lhs, rhs, predicate):
        if _is_float_type(lhs.type) and _is_float_type(rhs.type):
            predicate_attr = _arith_cmpf_predicate_attr(predicate)
            return arith.CmpFOp(predicate_attr, lhs, rhs).result

        if _is_integer_like_type(lhs.type) and _is_integer_like_type(rhs.type):
            predicate_attr = _arith_cmpi_predicate_attr(predicate)
            return arith.CmpIOp(predicate_attr, lhs, rhs).result

        return None

    def _load_if_pointer(self, value):
        if cc.PointerType.isinstance(value.type):
            return cc.LoadOp(value).result
        return value

    def _is_quantum_type(self, _type):
        return quake.RefType.isinstance(_type) or quake.VeqType.isinstance(_type)

    def _bufferize_array(self, value, loc):
        if cc.ArrayType.isinstance(value.type):
            buffer = cc.AllocaOp(cc.PointerType.get(self.ctx, value.type), TypeAttr.get(value.type), loc=loc).result
            cc.StoreOp(value, buffer, loc=loc)
            return buffer
        return value

    def _make_iterable(self, value, node):
        """
        Convert a value to an Iterable if possible
        Returns None and emits diagnostics if value cannot be made iterable

        Args:
            value: The value to convert to an Iterable
            node: The AST node for diagnostics

        Returns:
            An Iterable object or None if conversion failed
        """
        if isinstance(value, Iterable):
            # Check if Enumerate wraps a non-iterable
            if isinstance(value, Enumerate):
                value.iterable = self._make_iterable(value.iterable, node)
                if not value.iterable:
                    self.diagnostic.emit(Diagnostic(
                        DiagnosticSeverity.ERROR,
                        "enumerate() argument must be an iterable"
                    ).add_span(DiagnosticSpan.from_ast_node(
                        node,
                        file=self.diagnostic.filename,
                        is_primary=True
                    )))
            return value

        if isinstance(value, IRValue):
            value = self._bufferize_array(value, self._create_location(node))
            # Create appropriate adapter based on IR type
            if quake.VeqType.isinstance(value.type):
                return VeqIterable(value)
            elif cc.PointerType.isinstance(value.type):
                array_type = cc.PointerType.getElementType(value.type)
                if cc.ArrayType.isinstance(array_type):
                    return ArrayIterable(value)
            elif cc.StdvecType.isinstance(value.type):
                return StdvecIterable(value)

        # Not an iterable
        self.diagnostic.emit(Diagnostic(
            DiagnosticSeverity.ERROR,
            f"'{type(value).__name__}' object is not iterable"
        ).add_span(DiagnosticSpan.from_ast_node(
            node,
            file=self.diagnostic.filename,
            is_primary=True
        )))
        return None

    #---------------------------------------------------------------------------
    # Generic visiting
    #---------------------------------------------------------------------------

    def visit(self, node):
        return super().visit(node)

    def generic_visit(self, node):
        diagnostic = Diagnostic(
            DiagnosticSeverity.ERROR,
            f"Unsupported AST node {str(node)}"
        ).add_span(DiagnosticSpan.from_ast_node(
            node,
            file=self.diagnostic.filename,
            is_primary=True
        ))
        self.diagnostic.emit(diagnostic)

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

        # TODO: Support chained assignment, e.g. `a = b = 1`
        if len(node.targets) > 1:
            diagnostic = Diagnostic(
                DiagnosticSeverity.ERROR,
                f"Unsupported chained assignment.",
            ).add_span(DiagnosticSpan.from_ast_node(
                node,
                file=self.diagnostic.filename,
                is_primary=True,
            ))
            self.diagnostic.emit(diagnostic)
            return

        # TODO: Support unpacking assignment, e.g. `a, b = 1, 2`
        if isinstance(node.targets[0], ast.Tuple):
            diagnostic = Diagnostic(
                DiagnosticSeverity.ERROR,
                f"Unsupported unpacking assignment.",
            ).add_span(DiagnosticSpan.from_ast_node(
                node.targets[0],
                file=self.diagnostic.filename,
                is_primary=True,
                label=f"The target of the assignment is a tuple"
            ))
            self.diagnostic.emit(diagnostic)

        # Symbol resolution on assigment is a bit different
        target = self.visit(node.targets[0])
        value = self.visit(node.value)

        # Make sure we are dealing with an IRValue
        if not isinstance(value, IRValue):
            diagnostic = Diagnostic(
                DiagnosticSeverity.ERROR,
                f"Cannot assign a value of type {type(value)} to a name",
            ).add_span(DiagnosticSpan.from_ast_node(
                node.value,
                file=self.diagnostic.filename,
                is_primary=True,
            ))
            self.diagnostic.emit(diagnostic)

        if not target.def_node:
            if cc.PointerType.isinstance(value.type):
                value = cc.LoadOp(value).result

            # We are seeing the identifier for the first time. 
            target.def_node = node
            target.ir_type = value.type

            # When dealing with quantum types, we just update the local symbols.
            if quake.RefType.isinstance(value.type) or quake.VeqType.isinstance(value.type):
                target.ir_slot = value
                return

            # We are creating a new name
            memory = self._cc_alloca(value.type, loc)
            target.ir_slot = memory
            self.value_to_node.insert(memory,  target)

        elif quake.RefType.isinstance(target.ir_type):
            # Quantum types are not SSA values in the IR, they are memory and 
            # we don't have an operation capable of adding an extra level of 
            # indirection here.
            self.diagnostic.emit_error(
                node.lineno, node.col_offset,
                f"Variables of a quantum type are immutable",
            )
        elif target.ir_type != value.type:
            # Check whether types of the existing target and the value are the
            # same. If not, then we need to allocate a new memory and re-assign
            # the name.
            diagnostic = Diagnostic(
                DiagnosticSeverity.ERROR,
                f"Cannot assign a value of type {value.type} to a name that previously contained a value of type {target.ir_type}",
            ).add_span(DiagnosticSpan.from_ast_node(
                node,
                file=self.diagnostic.filename,
                is_primary=True,
                label=f"Trying to assign a value of type {value.type} to a variable of type {target.ir_type}"
            )).add_span(DiagnosticSpan.from_ast_node(
                target.def_node,
                file=self.diagnostic.filename,
                is_primary=True,
                label=f"'{target.name}' is defined here"
            ))
            self.diagnostic.emit(diagnostic)

        cc.StoreOp(value, target.ir_slot, loc=loc)

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

        # TODO: In the future, we should do something like this:
        # obj = self.visit(new_node)
        # for attr in reversed(chain):
        #     # Using subscript is faster than `getattr` for dict.
        #     if isinstance(obj, dict):
        #         obj = obj[attr]
        #     else:
        #         obj = getattr(obj, attr)

        # return obj

    def visit_BinOp(self, node) -> IRValue:
        left = self._load_if_pointer(self.visit(node.left))
        right = self._load_if_pointer(self.visit(node.right))

        if isinstance(node.op, (ast.Add, ast.Sub, ast.Mult)):
            right = self.type_converter.coerce_value_type(left.type, right)
            left = self.type_converter.coerce_value_type(right.type, left)

        result = None
        match node.op:
            case ast.Add():
                result = self._binary_op(left, right, op="add")
            case ast.BitAnd():
                result = self._binary_op(left, right, op="and")
            case ast.Mult():
                result = self._binary_op(left, right, op="mul")
            case ast.Pow():
                result = self._binary_op(left, right, op="pow") 
            case ast.Sub():
                result = self._binary_op(left, right, op="sub")
            case _:
                diagnostic = Diagnostic(
                    DiagnosticSeverity.ERROR,
                    f"Unsupported binary operator."
                ).add_span(DiagnosticSpan.from_ast_node(
                    node,
                    file=self.diagnostic.filename,
                    is_primary=True,
                ))
                self.diagnostic.emit(diagnostic)

        if result:
            return result

        self.diagnostic.emit_error(
            node.lineno, node.col_offset,
            f"Error when handling operator {ast.unparse(node)} {node.op}"
        )

    def visit_Break(self, node):
        # TODO: Handle edge cases? This statement only make sense with an 
        # ancestor scope is a loop, `for` or `while`
        cc.UnwindBreakOp([], loc=self._create_location(node))

    def visit_Continue(self, node):
        # TODO: Handle edge cases? This statement only make sense with an 
        # ancestor scope is a loop, `for` or `while`
        cc.UnwindContinueOp([], loc=self._create_location(node))

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

        if isinstance(fn, Measurement):
            measure_result = fn._mlir_materialize(args, loc)
            return measure_result

        self.diagnostic.emit_error(
            getattr(node, "lineno", 0),
            getattr(node, "col_offset", 0),
            f"Unsupported call {ast.unparse(node)}",
        )

    def visit_Compare(self, node) -> IRValue:
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

    def visit_Constant(self, node) -> IRValue:
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
        iter_value = self.visit(node.iter)
        self.local_symbols.push_scope()
        targets = self.visit(node.target)
        if isinstance(targets, Symbol):
            targets = [targets]

        loc = self._create_location(node)
        iterable = self._make_iterable(iter_value, node.iter)
        start, size, step = iterable.get_loop_params(loc)

        # Materialize the loop
        loop = cc.LoopOp([start.type], [start], BoolAttr.get(False))

        # Create while condition block
        whileBlock = Block.create_at_start(loop.whileRegion, [start.type])
        with InsertionPoint(whileBlock):
            test = self._cmp_op(whileBlock.arguments[0], size, ast.Lt)
            cc.ConditionOp(test, whileBlock.arguments)

        # Create step block
        stepBlock = Block.create_at_start(loop.stepRegion, [start.type])
        with InsertionPoint(stepBlock):
            incr = arith.AddIOp(stepBlock.arguments[0], step).result
            cc.ContinueOp([incr])

        # Create and populate the body block
        bodyBlock = Block.create_at_start(loop.bodyRegion, [start.type])
        with InsertionPoint(bodyBlock):
            # Extract elements for this iteration
            values = iterable.extract_element(bodyBlock.arguments[0], loc)

            # Validate target/value count
            if len(targets) != len(values):
                self.diagnostic.emit(Diagnostic(
                    DiagnosticSeverity.ERROR,
                    f"Cannot unpack {len(values)} values into {len(targets)} targets"
                ).add_span(DiagnosticSpan.from_ast_node(
                    node.target,
                    file=self.diagnostic.filename,
                    is_primary=True,
                    label=f"Expected {len(values)} targets, got {len(targets)}"
                )))

            # Assign values to targets
            for target, value in zip(targets, values):
                if target.ir_slot is None:
                    # Initialize new target
                    target.ir_type = value.type
                    target.ir_slot = self._cc_alloca(value.type, loc)
                    cc.StoreOp(value, target.ir_slot, loc=loc)
                    self.local_symbols.insert(target.name, target)
                else:
                    # Check type compatibility
                    if target.ir_type != value.type:
                        self.diagnostic.emit(Diagnostic(
                            DiagnosticSeverity.ERROR,
                            f"Cannot assign a value of type {value.type} to a name that previously contained a value of type {target.ir_type}"
                        ).add_span(DiagnosticSpan.from_ast_node(
                            node.target,
                            file=self.diagnostic.filename,
                            is_primary=True
                        )))
                    cc.StoreOp(value, target.ir_slot, loc=loc)

            # Process the loop body
            for stmt in node.body:
                self.visit(stmt)

            cc.ContinueOp(bodyBlock.arguments)

        stepBlock = Block.create_at_start(loop.stepRegion, [int64_ty])
        with InsertionPoint(stepBlock):
            incr = arith.AddIOp(stepBlock.arguments[0], step).result
            cc.ContinueOp([incr])

        self.local_symbols.pop_scope()

    def visit_FunctionDef(self, node):
        #print(ast.dump(node, indent=4))

        # Process arguments
        # TODO: This attribute is required by users of the code generator.
        #       We not have it in this way.
        self.argTypes = []
        for arg in node.args.args:
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
        attr = DictAttr.get({
            name: StringAttr.get(f'{name}_PyKernelEntryPointRewrite', context=self.ctx)
        }, context=self.ctx)

        self.module.operation.attributes['quake.mangled_name_map'] = attr

        # Create `func` operation within the MLIR module
        loc = self._create_location(node)
        with InsertionPoint(self.module.body), loc:

            func_op = func.FuncOp(name, (self.argTypes, []))
            func_op.attributes["cudaq-kernel"] = UnitAttr.get()

            entry_block = func_op.add_entry_block()
            with InsertionPoint(entry_block):
                # Create slots for the arguments
                # TODO: There are arguments that don't need slots, e.g. quantum types.
                for arg, block_arg in zip(node.args.args, entry_block.arguments):
                    symbol = Symbol(arg.arg)
                    symbol.ir_type = block_arg.type
                    arg_loc = self._create_location(arg)
                    symbol.ir_slot = self._cc_alloca(block_arg.type, arg_loc)
                    cc.StoreOp(block_arg, symbol.ir_slot, loc=arg_loc)
                    self.local_symbols.insert(arg.arg, symbol)

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

    def visit_List(self, node):
        loc = self._create_location(node)
        values = [self.visit(elt) for elt in node.elts]

        # Check if empty list
        if not values:
            return []

        # Get first value's type
        first_type = values[0].type

        # Check if all values have same type
        for value in values[1:]:
            value_type = value.type
            if value_type != first_type:
                diagnostic = Diagnostic(
                    DiagnosticSeverity.ERROR,
                    f"List elements must all have the same type. Found {first_type} and {value_type}",
                ).add_span(DiagnosticSpan.from_ast_node(
                    node,
                    file=self.diagnostic.filename,
                    is_primary=True,
                ))
                self.diagnostic.emit(diagnostic)

        if self._is_quantum_type(values[0].type):
            veq_type = quake.VeqType.get(self.ctx, len(values))
            return quake.ConcatOp(veq_type, values, loc=loc).result
        array_type = cc.ArrayType.get(self.ctx, values[0].type, len(values))
        array_value = cc.UndefOp(array_type, loc=loc)
        for i, value in enumerate(values):
            array_value = cc.InsertValueOp(array_type, array_value, value, DenseI64ArrayAttr.get([i]), loc=loc)

        return array_value.result

    def visit_Module(self, node):
        assert len(node.body) == 1
        assert isinstance(node.body[0], ast.FunctionDef)
        self.visit(node.body[0])
        print(self.module)

    def visit_Name(self, node):
        # The ast.Name node has a context that is either `load` or `store`.
        if type(node.ctx) is ast.Store:
            # In the case of a store, we are either:
            #   (1) creating a new name or
            #   (2) re-assigning a name to a new value.
            return self.local_symbols.lookup_or_insert(node.id, Symbol(node.id))

        if node.id in builtin_namespace:
            return builtin_namespace[node.id]

        symbol = self.local_symbols.lookup(node.id)
        if symbol:
            loc = self._create_location(node)
            # TODO: Rethink this, it makes handling of arrays a bit awkward.
            if self._is_quantum_type(symbol.ir_type) or cc.ArrayType.isinstance(symbol.ir_type):
                return symbol.ir_slot
            return self._cc_load(symbol.ir_slot, loc=loc)

        # When in the `load` context, not finding the symbol is a fatal error.
        diagnostic = Diagnostic(
            DiagnosticSeverity.ERROR,
            f"Unknown name '{node.id}'",
        ).add_span(DiagnosticSpan.from_ast_node(
            node,
            file=self.diagnostic.filename,
            is_primary=True,
        ))
        self.diagnostic.emit(diagnostic)

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
        if not isinstance(return_expr, IRValue):
            diagnostic = Diagnostic(
                DiagnosticSeverity.ERROR,
                "Unsupported return type",
            ).add_span(DiagnosticSpan.from_ast_node(
                node.value,
                file=self.diagnostic.filename,
                is_primary=True,
                label=f"The value has type {type(return_expr)}"
            ))
            self.diagnostic.emit(diagnostic)
            return

        return_expr = self._load_if_pointer(return_expr)
        self.return_type = return_expr.type
        return_op([return_expr], loc=loc)

    def visit_Subscript(self, node):
        # The ast.Subscript node has a context that is either `load` or `store`.
        value = self.visit(node.value)
        slice = self.visit(node.slice)

        if quake.VeqType.isinstance(value.type):
            return quake.ExtractRefOp(quake.RefType.get(self.ctx), value, -1, index=slice).result
        if cc.PointerType.isinstance(value.type):
            array_type = cc.PointerType.getElementType(value.type)
            element_type = cc.ArrayType.getElementType(array_type)
            address = cc.ComputePtrOp(
                cc.PointerType.get(self.ctx, element_type), value,
                [slice], DenseI32ArrayAttr.get([kDynamicPtrIndex])
            ).result
            if type(node.ctx) is ast.Store:
                target = self.value_to_node.lookup(value)
                symbol = Symbol(f"{target.name}")
                symbol.def_node = target.def_node
                symbol.ir_type = element_type
                symbol.ir_slot = address
                return symbol
            return address
        if cc.StdvecType.isinstance(value.type):
            return cc.StdvecExtractOp(cc.StdvecType.getElementType(value.type), value, slice).result
        if cc.StructType.isinstance(value.type):
            # TODO: We cannot generally support tuples since CC has no way to extract dynamic elements.
            # Check if the slice is a constant that we can use for static extraction.
            if isinstance(slice.owner.opview, arith.ConstantOp) and IntegerAttr.isinstance(slice.owner.opview.value):
                slice = IntegerAttr(slice.owner.opview.value).value
            else:
                diagnostic = Diagnostic(
                    DiagnosticSeverity.ERROR,
                    f"Unsupported dynamic subscript operation on tuple"
                ).add_span(DiagnosticSpan.from_ast_node(
                    node.slice,
                    file=self.diagnostic.filename,
                    is_primary=True,
                    label=f"Could not statically infer value"
                ))
                self.diagnostic.emit(diagnostic)

            element_types = cc.StructType.getTypes(value.type)
            return cc.ExtractValueOp(element_types[slice], value, [], DenseI32ArrayAttr.get([slice])).result

        diagnostic = Diagnostic(
            DiagnosticSeverity.ERROR,
            f"Unsupported subscript operation on a value of type {value.type}"
        ).add_span(DiagnosticSpan.from_ast_node(
            node.value,
            file=self.diagnostic.filename,
            is_primary=True,
            label=f"The value has type {value.type}"
        ))
        self.diagnostic.emit(diagnostic)

    def visit_Tuple(self, node):
        values = [self.visit(elt) for elt in node.elts]
        if type(node.ctx) is ast.Store:
            return tuple(values)

        # Materialize the tuple
        loc = self._create_location(node)
        tuple_type = cc.StructType.get(self.ctx, [t.type for t in values])
        tuple_value = cc.UndefOp(tuple_type, loc=loc)
        for i, value in enumerate(values):
            tuple_value = cc.InsertValueOp(tuple_type, tuple_value, value, DenseI64ArrayAttr.get([i]), loc=loc)
        return tuple_value.result

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
