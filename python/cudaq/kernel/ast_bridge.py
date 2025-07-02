# ============================================================================ #
# Copyright (c) 2022 - 2025 NVIDIA Corporation & Affiliates.                   #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #
import ast
import importlib
import graphlib
import textwrap
import numpy as np
import os
import sys
from collections import deque
from types import FunctionType

from cudaq.mlir._mlir_libs._quakeDialects import (
    cudaq_runtime, load_intrinsic, gen_vector_of_complex_constant,
    register_all_dialects)
from cudaq.mlir.dialects import arith, cc, func, quake
from cudaq.mlir.ir import (BoolAttr, Block, Context, ComplexType,
                           DenseBoolArrayAttr, DictAttr, F32Type, F64Type,
                           FlatSymbolRefAttr, FunctionType, InsertionPoint,
                           IntegerAttr, IntegerType, Location, Module,
                           StringAttr, SymbolTable, UnitAttr, Value)
from cudaq.mlir.passmanager import PassManager
from . import analysis, stdlib
from .stdlib import _core
from .stdlib._core import types as T
from .stdlib.int64 import ImplicitInt64able
from .stdlib.float64 import ImplicitFloat64able
from .captured_data import CapturedDataStorage
from .utils import (
    Color,
    globalAstRegistry,
    globalKernelRegistry,
    globalRegisteredOperations,
    globalRegisteredTypes,
    nvqppPrefix,
    mlirTypeFromAnnotation,
    mlirTypeFromPyType,
    mlirTypeToPyType,
)

State = cudaq_runtime.State

# This file implements the CUDA-Q Python AST to MLIR conversion.
# It provides a `PyASTBridge` class that implements the `ast.NodeVisitor` type
# to walk the Python AST for a `cudaq.kernel` annotated function and generate
# valid MLIR code using `Quake`, `CC`, `Arith`, and `Math` dialects.

ALLOWED_TYPES_IN_A_DATACLASS = [int, float, bool, cudaq_runtime.qview]


class PyScopedSymbolTable(object):

    def __init__(self):
        self.symbolTable = deque()

    def pushScope(self):
        self.symbolTable.append({})

    def popScope(self):
        self.symbolTable.pop()

    def numLevels(self):
        return len(self.symbolTable)

    def add(self, symbol, value, level=-1):
        """
        Add a symbol to the scoped symbol table at any scope level.
        """
        self.symbolTable[level][symbol] = value

    def __contains__(self, symbol):
        for st in reversed(self.symbolTable):
            if symbol in st:
                return True

        return False

    def __setitem__(self, symbol, value):
        # default to nearest surrounding scope
        self.add(symbol, value)
        return

    def __getitem__(self, symbol):
        for st in reversed(self.symbolTable):
            if symbol in st:
                return st[symbol]

        raise RuntimeError(
            f"{symbol} is not a valid variable name in this scope.")

    def clear(self):
        while len(self.symbolTable):
            self.symbolTable.pop()
        return


class CompilerError(RuntimeError):
    """
    Custom exception class for improved error diagnostics.
    """

    def __init__(self, *args, **kwargs):
        RuntimeError.__init__(self, *args, **kwargs)


class PyASTBridge(ast.NodeVisitor):
    """
    The `PyASTBridge` class implements the `ast.NodeVisitor` type to convert a 
    python function definition (annotated with cudaq.kernel) to an MLIR `ModuleOp`
    containing a `func.FuncOp` representative of the original python function but leveraging 
    the Quake and CC dialects provided by CUDA-Q. This class keeps track of a 
    MLIR Value stack that is pushed to and popped from during visitation of the 
    function AST nodes. We leverage the auto-generated MLIR Python bindings for the internal 
    C++ CUDA-Q dialects to build up the MLIR code. 

    For kernels that call other kernels, we require that the `ModuleOp` contain the 
    kernel being called. This is enabled via the `FindDepKernelsVisitor` in the local 
    analysis module, and is handled by the below `compile_to_mlir` function. For 
    callable block arguments, we leverage runtime-known callable argument function names 
    and synthesize them away with an internal C++ MLIR pass. 
    """

    def __init__(self, capturedDataStorage: CapturedDataStorage, **kwargs):
        """
        The constructor. Initializes the `mlir.Value` stack, the `mlir.Context`, and the 
        `mlir.Module` that we will be building upon. This class keeps track of a 
        symbol table, which maps variable names to constructed `mlir.Values`. 
        """
        self.valueStack = deque()
        self.knownResultType = kwargs[
            'knownResultType'] if 'knownResultType' in kwargs else None
        if 'existingModule' in kwargs:
            self.module = kwargs['existingModule']
            self.ctx = self.module.context
            self.loc = Location.unknown(context=self.ctx)
        else:
            self.ctx = Context()
            register_all_dialects(self.ctx)
            quake.register_dialect(context=self.ctx)
            cc.register_dialect(context=self.ctx)
            cudaq_runtime.registerLLVMDialectTranslation(self.ctx)
            self.loc = Location.unknown(context=self.ctx)
            self.module = Module.create(loc=self.loc)

        # Create a new captured data storage or use the existing one
        # passed from the current kernel decorator.
        self.capturedDataStorage = capturedDataStorage
        if (self.capturedDataStorage == None):
            self.capturedDataStorage = CapturedDataStorage(ctx=self.ctx,
                                                           loc=self.loc,
                                                           name=None,
                                                           module=self.module)
        else:
            self.capturedDataStorage.setKernelContext(ctx=self.ctx,
                                                      loc=self.loc,
                                                      name=None,
                                                      module=self.module)

        # If the driver of this AST bridge instance has indicated
        # that there is a return type from analysis on the Python AST,
        # then we want to set the known result type so that the
        # FuncOp can have it.
        if 'returnTypeIsFromPython' in kwargs and kwargs[
                'returnTypeIsFromPython'] and self.knownResultType is not None:
            self.knownResultType = mlirTypeFromPyType(self.knownResultType,
                                                      self.ctx)

        self.capturedVars = kwargs[
            'capturedVariables'] if 'capturedVariables' in kwargs else {}
        self.dependentCaptureVars = {}
        self.locationOffset = kwargs[
            'locationOffset'] if 'locationOffset' in kwargs else ('', 0)
        self.disableEntryPointTag = kwargs[
            'disableEntryPointTag'] if 'disableEntryPointTag' in kwargs else False
        self.disableNvqppPrefix = kwargs[
            'disableNvqppPrefix'] if 'disableNvqppPrefix' in kwargs else False
        self.symbolTable = PyScopedSymbolTable()
        self.indent_level = 0
        self.indent = 4 * " "
        self.buildingEntryPoint = False
        self.currentAssignVariableName = None
        self.walkingReturnNode = False
        self.controlNegations = []
        self.verbose = 'verbose' in kwargs and kwargs['verbose']
        self.verbose = True

    def debug_msg(self, msg, node=None):
        if self.verbose:
            print(f'{self.indent * self.indent_level}{msg()}')
            if node is not None:
                print(
                    textwrap.indent(ast.unparse(node),
                                    (self.indent * (self.indent_level + 1))))

    def emitWarning(self, msg, astNode=None):
        """
        Emit a warning, providing the user with source file information and
        the offending code.
        """
        codeFile = os.path.basename(self.locationOffset[0])
        lineNumber = '' if astNode == None else astNode.lineno + self.locationOffset[
            1] - 1

        print(Color.BOLD, end='')
        msg = codeFile + ":" + str(
            lineNumber
        ) + ": " + Color.YELLOW + "warning: " + Color.END + Color.BOLD + msg + (
            "\n\t (offending source -> " + ast.unparse(astNode) + ")" if
            hasattr(ast, 'unparse') and astNode is not None else '') + Color.END
        print(msg)

    def emitFatalError(self, msg, astNode=None):
        """
        Emit a fatal error, providing the user with source file information and
        the offending code.
        """
        codeFile = os.path.basename(self.locationOffset[0])
        lineNumber = '' if astNode == None else astNode.lineno + self.locationOffset[
            1] - 1

        print(Color.BOLD, end='')
        msg = codeFile + ":" + str(
            lineNumber
        ) + ": " + Color.RED + "error: " + Color.END + Color.BOLD + msg + (
            "\n\t (offending source -> " + ast.unparse(astNode) + ")" if
            hasattr(ast, 'unparse') and astNode is not None else '') + Color.END
        raise CompilerError(msg)

    def simulationPrecision(self):
        target = cudaq_runtime.get_target()
        return target.get_precision()

    def simulationDType(self):
        if self.simulationPrecision() == cudaq_runtime.SimulationPrecision.fp64:
            return T.complex128()
        return T.complex64()

    def pushValue(self, value):
        self.debug_msg(lambda: f'push {value}')
        self.valueStack.append(stdlib.try_downcast(value))

    def popValue(self):
        val = self.valueStack.pop()
        self.debug_msg(lambda: f'pop {val}')
        return val

    def hasTerminator(self, block):
        if len(block.operations) > 0:
            return cudaq_runtime.isTerminator(
                block.operations[len(block.operations) - 1])
        return False

    def ifPointerThenLoad(self, value):
        if isinstance(value, stdlib.Pointer):
            return value.load()
        return value

    def getValue(self, value):
        if isinstance(value, stdlib.AbstractValue):
            return value.materialize()
        if isinstance(value, stdlib.Pointer):
            return value.load()
        return value

    def ifNotPointerThenStore(self, value):
        if not isinstance(value, stdlib.Pointer):
            slot = stdlib.Pointer(_core.alloca(value.type))
            slot.store(value)
            return slot
        return value

    def __copyVectorAndCastElements(self, source, target_element_type):
        if not isinstance(source, stdlib.Stdvec):
            raise RuntimeError(
                f"expected vector type in __copyVectorAndCastElements but received {source}"
            )

        # Exit early if no copy is needed to avoid an unneeded store.
        if (source.elements_type == target_element_type):
            return source

        source_size = source.get_length()
        new_stdvec = stdlib.Stdvec.create(target_element_type, source_size)

        def bodyBuilder(iterVar):
            source_item = source.get_item(iterVar)
            # TODO: Really dubious sort of thing to do here. Shouldn't it be arithmetic_promotion?
            casted_item = _core.implicit_conversion(source_item,
                                                    target_element_type)
            new_stdvec.set_item(iterVar, casted_item)

        _core.for_loop(source_size, bodyBuilder)
        return new_stdvec

    def _print_i64(self, value):
        assert isinstance(
            value, ImplicitInt64able
        ), f"print_i64 requires an integer value, got {type(value)}"
        symbol_table = SymbolTable(self.module.operation)
        if 'print_i64' not in symbol_table:
            with InsertionPoint(self.module.body):
                fn = func.FuncOp('print_i64', ([T.ptr(T.i8()), T.i64()], []))
                fn.sym_visibility = StringAttr.get("private")
            symbol_table = SymbolTable(self.module.operation)

        fn_symbol = symbol_table['print_i64']
        ir_value = value.__as_int64__().value
        assert ir_value.type == T.i64()
        string = stdlib.StringLiteral('[cudaq-ast-dbg] %ld\n').materialize()
        string = cc.CastOp(T.ptr(T.i8()), string).result
        func.CallOp(fn_symbol, [string, ir_value])

    def _print_f64(self, value):
        assert isinstance(
            value, ImplicitFloat64able
        ), f"print_f64 requires a float value, got {type(value)}"
        symbol_table = SymbolTable(self.module.operation)
        if 'print_f64' not in symbol_table:
            with InsertionPoint(self.module.body):
                fn = func.FuncOp('print_f64', ([T.ptr(T.i8()), T.f64()], []))
                fn.sym_visibility = StringAttr.get("private")
            symbol_table = SymbolTable(self.module.operation)

        fn_symbol = symbol_table['print_f64']
        ir_value = value.__as_float64__().value
        assert ir_value.type == T.f64()
        string = stdlib.StringLiteral('[cudaq-ast-dbg] %.12lf\n').materialize()
        string = cc.CastOp(T.ptr(T.i8()), string).result
        func.CallOp(fn_symbol, [string, ir_value])

    def mlirTypeFromAnnotation(self, annotation):
        msg = None
        try:
            return mlirTypeFromAnnotation(annotation, self.ctx, raiseError=True)
        except RuntimeError as e:
            msg = str(e)

        if msg is not None:
            self.emitFatalError(msg, annotation)

    def argumentsValidForFunction(self, values, functionTy):
        return False not in [
            ty == values[i].type
            for i, ty in enumerate(FunctionType(functionTy).inputs)
        ]

    def checkControlAndTargetTypes(self, controls, targets):
        [
            self.emitFatalError(f'control operand {i} is not of quantum type.')
            if not T.is_quantum_type(control.type) else None
            for i, control in enumerate(controls)
        ]
        [
            self.emitFatalError(f'target operand {i} is not of quantum type.')
            if not T.is_quantum_type(target.type) else None
            for i, target in enumerate(targets)
        ]

    def needsStackSlot(self, type):
        # FIXME add more as we need them
        return ComplexType.isinstance(type) or F64Type.isinstance(
            type) or F32Type.isinstance(type) or IntegerType.isinstance(
                type) or cc.StructType.isinstance(type)

    def visit(self, node):
        self.debug_msg(lambda: f'[Visit {type(node).__name__}]', node)
        self.indent_level += 1
        super().visit(node)
        self.indent_level -= 1

    def visit_FunctionDef(self, node):
        if self.buildingEntryPoint:
            # This is an inner function def, we will
            # treat it as a cc.callable (cc.create_lambda)
            self.debug_msg(lambda: f'Visiting inner FunctionDef {node.name}')

            arguments = node.args.args
            if len(arguments):
                self.emitFatalError(
                    "inner function definitions cannot have arguments.", node)

            ty = cc.CallableType.get([])
            createLambda = cc.CreateLambdaOp(ty)
            initRegion = createLambda.initRegion
            initBlock = Block.create_at_start(initRegion, [])
            # TODO: process all captured variables in the main function
            # definition first to avoid reusing code not defined in the
            # same or parent scope of the produced MLIR.
            with InsertionPoint(initBlock):
                [self.visit(n) for n in node.body]
                cc.ReturnOp([])
            self.symbolTable[node.name] = createLambda.result
            return

        with self.ctx, InsertionPoint(self.module.body), self.loc:

            # Get the potential documentation string
            self.docstring = ast.get_docstring(node)

            # Get the argument types and argument names
            # this will throw an error if the types aren't annotated
            self.argTypes = [
                self.mlirTypeFromAnnotation(arg.annotation)
                for arg in node.args.args
            ]
            parentResultType = self.knownResultType
            if node.returns is not None and not (isinstance(
                    node.returns, ast.Constant) and
                                                 (node.returns.value is None)):
                self.knownResultType = self.mlirTypeFromAnnotation(node.returns)

            # Get the argument names
            argNames = [arg.arg for arg in node.args.args]

            self.name = node.name
            self.capturedDataStorage.name = self.name

            # the full function name in MLIR is `__nvqpp__mlirgen__` + the function name
            if not self.disableNvqppPrefix:
                fullName = nvqppPrefix + node.name
            else:
                fullName = node.name

            # Create the FuncOp
            f = func.FuncOp(fullName, (self.argTypes, [] if self.knownResultType
                                       == None else [self.knownResultType]),
                            loc=self.loc)
            self.kernelFuncOp = f

            areQuantumTypes = [T.is_quantum_type(ty) for ty in self.argTypes]
            f.attributes.__setitem__('cudaq-kernel', UnitAttr.get())
            if True not in areQuantumTypes and not self.disableEntryPointTag:
                f.attributes.__setitem__('cudaq-entrypoint', UnitAttr.get())

            # Create the entry block
            self.entry = f.add_entry_block()

            # Set the insertion point to the start of the entry block
            with InsertionPoint(self.entry):
                self.buildingEntryPoint = True
                self.symbolTable.pushScope()
                # Add the block arguments to the symbol table,
                # create a stack slot for value arguments
                blockArgs = self.entry.arguments
                for i, b in enumerate(blockArgs):
                    if self.needsStackSlot(b.type):
                        stackSlot = _core.alloca(b.type)
                        _core.store(b, stackSlot)
                        self.symbolTable[argNames[i]] = stackSlot
                    else:
                        self.symbolTable[argNames[i]] = stdlib.try_downcast(b)

                for stmt in node.body:
                    self.visit(stmt)
                # Add the return operation
                if not self.hasTerminator(self.entry):
                    # If the function has a known (non-None) return type, emit an `undef` of that
                    # type and return it; else return void.
                    #
                    # NOTE: At this point, the kernel has already been validated by the
                    # `analysis.ValidateReturnStatements` visitor. Thus, it is safe to add the
                    # return of an undefined value just as a mean to satisfies the requirement
                    # that a block terminator is present.
                    if self.knownResultType is not None:
                        undef = cc.UndefOp(self.knownResultType).result
                        func.ReturnOp([undef])
                    else:
                        func.ReturnOp([])
                self.buildingEntryPoint = False
                self.symbolTable.popScope()

            if True not in areQuantumTypes:
                attr = DictAttr.get(
                    {
                        fullName:
                            StringAttr.get(
                                fullName + '_PyKernelEntryPointRewrite',
                                context=self.ctx)
                    },
                    context=self.ctx)
                self.module.operation.attributes.__setitem__(
                    'quake.mangled_name_map', attr)

            globalKernelRegistry[node.name] = f
            self.symbolTable.clear()
            self.valueStack.clear()

            self.knownResultType = parentResultType

    def visit_Expr(self, node):
        if hasattr(node, 'value') and isinstance(node.value, ast.Constant):
            self.debug_msg(lambda: f'[(Inline) Visit Constant]', node.value)
            constant = node.value
            if isinstance(constant.value, str):
                return

        self.visit(node.value)

    def visit_Lambda(self, node):
        """
        Map a lambda expression in a CUDA-Q kernel to a CC Lambda (a Value of `cc.callable` type 
        using the `cc.create_lambda` operation). Note that we extend Python with a novel 
        syntax to specify a list of independent statements (Python lambdas must have a single statement) by 
        allowing programmers to return a Tuple where each element is an independent statement. 

        ```python
            functor = lambda : (h(qubits), x(qubits), ry(np.pi, qubits))  # qubits captured from parent region
            # is equivalent to 
            def functor(qubits):
                h(qubits)
                x(qubits)
                ry(np.pi, qubits)
        ```
        """
        arguments = node.args.args
        if len(arguments):
            self.emitFatalError("CUDA-Q lambdas cannot have arguments.", node)

        ty = cc.CallableType.get([])
        createLambda = cc.CreateLambdaOp(ty)
        initBlock = Block.create_at_start(createLambda.initRegion, [])
        with InsertionPoint(initBlock):
            # Python lambdas can only have a single statement.
            # Here we will enhance our language by processing a single Tuple statement
            # as a set of statements for each element of the tuple
            if isinstance(node.body, ast.Tuple):
                self.debug_msg(lambda: f'[(Inline) Visit Tuple]', node.body)
                [self.visit(element) for element in node.body.elts]
            else:
                self.visit(
                    node.body)  # only one statement in a python lambda :(
            cc.ReturnOp([])
        self.pushValue(createLambda.result)
        return

    def _is_reassignable_value(self, value):
        if isinstance(value, stdlib.Veq):
            return False
        if T.is_quantum_type(value.type):
            return False
        if cc.CallableType.isinstance(value.type):
            return False
        return True

    def _needs_stack_slot(self, value):
        if isinstance(value, stdlib.Veq):
            return False
        if T.is_quantum_type(value.type):
            return False
        if cc.CallableType.isinstance(value.type):
            return False
        return True

    def visit_Assign(self, node):
        # TODO: Support chained assignments like `a = b = c = <expr>`
        if len(node.targets) > 1:
            self.emitFatalError("CUDA-Q does not support chained assignments.",
                                node)

        self.visit(node.targets[0])
        vars = self.popValue()
        if not isinstance(vars, stdlib.Tuple):
            vars = stdlib.Tuple(vars)

        values_nodes = [node.value]
        if len(vars) > 1:
            if not isinstance(node.value, ast.Tuple):
                self.emitFatalError(
                    f"Invalid assignment detected. Expected a tuple, got {type(node.value)}.",
                    node)
            values_nodes = node.value.elts
            if len(vars) != len(values_nodes):
                self.emitFatalError(
                    f"Invalid assignment detected. Expected {len(vars)} values, got {len(values_nodes)}.",
                    node)

        for var, value_node in zip(vars, values_nodes):
            if isinstance(var, str):
                # We are seen the name for the first time
                self.currentAssignVariableName = str(var)
                self.visit(value_node)
                self.currentAssignVariableName = None
                if len(self.valueStack) == 0:
                    self.emitFatalError("invalid assignment detected.", node)
                value = self.getValue(self.popValue())
                if self._needs_stack_slot(value):
                    slot = _core.alloca(value.type)
                    _core.store(value, slot)
                    value = slot
                self.symbolTable[var] = value
                continue

            self.visit(value_node)
            if len(self.valueStack) == 0:
                self.emitFatalError("invalid assignment detected.", node)
            value = self.getValue(self.popValue())
            if not self._is_reassignable_value(value):
                self.emitFatalError(
                    f"Cannot reassign to {var} of type {value.type}.", node)

            _core.store(value, var)

    def visit_Attribute(self, node):
        attr = node.attr
        if not isinstance(attr, str):
            self.emitFatalError(f"Attribute chain not supported.", node)

        if isinstance(node.ctx, ast.Store):
            self.visit(node.value)
            var = self.popValue()
            if isinstance(var, stdlib.Pointer):
                pointee_type = var.element_type
                if cc.StructType.isinstance(pointee_type):
                    name = cc.StructType.getName(pointee_type)
                    if name == 'tuple':
                        self.emitFatalError("Cannot address a tuple this way.",
                                            node)
                    address = var.__getitem__(attr)
                    self.pushValue(address)
                    return
            self.emitFatalError("Cannot attribute store context.", node)

        if isinstance(node.value, ast.Name):
            self.debug_msg(lambda: f'[(Inline) Visit Name]', node.value)
            if node.value.id in ['np', 'numpy', 'math']:
                if node.attr == 'complex64':
                    self.pushValue(T.complex64())
                    return
                if node.attr == 'complex128':
                    self.pushValue(T.complex128())
                    return
                if node.attr == 'pi':
                    self.pushValue(stdlib.FloatLiteral(np.pi))
                    return
                if node.attr == 'e':
                    self.pushValue(stdlib.FloatLiteral(np.e))
                    return
                if node.attr == 'euler_gamma':
                    self.pushValue(stdlib.FloatLiteral(np.euler_gamma))
                    return
                raise RuntimeError(
                    "math expression {}.{} was not understood".format(
                        node.value.id, node.attr))

            if node.value.id == 'cudaq':
                if node.attr in [
                        'DepolarizationChannel', 'AmplitudeDampingChannel',
                        'PhaseFlipChannel', 'BitFlipChannel', 'PhaseDamping',
                        'ZError', 'XError', 'YError', 'Pauli1', 'Pauli2',
                        'Depolarization1', 'Depolarization2'
                ]:
                    cudaq_module = importlib.import_module('cudaq')
                    channel_class = getattr(cudaq_module, node.attr)
                    self.pushValue(_core.constant(channel_class.num_parameters))
                    self.pushValue(_core.constant(hash(channel_class)))
                return

        self.visit(node.value)
        value = self.ifPointerThenLoad(self.popValue())

        if isinstance(value, (stdlib.Struct, stdlib.Struq)):
            self.pushValue(value.get_field(attr))
            return

        if attr == 'size' and isinstance(value, stdlib.Veq):
            self.pushValue(value.get_length())
            return

        if node.attr in ['imag', 'real']:
            if isinstance(value, (stdlib.Complex64, stdlib.Complex128)):
                if (node.attr == 'real'):
                    self.pushValue(value.real)
                    return

                if (node.attr == 'imag'):
                    self.pushValue(value.imag)
                    return

    def visit_Call(self, node):
        global globalRegisteredOperations

        for arg in node.args:
            self.visit(arg)
        args = [
            self.ifPointerThenLoad(self.popValue())
            for _ in range(len(node.args))
        ]
        args.reverse()

        namedArgs = {}
        for keyword in node.keywords:
            self.visit(keyword.value)
            namedArgs[keyword.arg] = self.popValue()

        # do not walk the FunctionDef decorator_list arguments
        if isinstance(node.func, ast.Attribute):
            self.debug_msg(lambda: f'[(Inline) Visit Attribute]', node.func)
            if hasattr(
                    node.func.value, 'id'
            ) and node.func.value.id == 'cudaq' and node.func.attr == 'kernel':
                return

            # If we have a `func = ast.Attribute``, then it could be that
            # we have a previously defined kernel function call with manually specified module names
            # e.g. `cudaq.lib.test.hello.fermionic_swap``. In this case, we assume
            # FindDepKernels has found something like this, loaded it, and now we just
            # want to get the function name and call it.

            # First let's check for registered C++ kernels
            cppDevModNames = []
            value = node.func.value
            if isinstance(value, ast.Name) and value.id != 'cudaq':
                self.debug_msg(lambda: f'[(Inline) Visit Name]', value)
                cppDevModNames = [node.func.attr, value.id]
            else:
                self.debug_msg(lambda: f'[(Inline) Visit Attribute]', value)
                while isinstance(value, ast.Attribute):
                    cppDevModNames.append(value.attr)
                    value = value.value
                    if isinstance(value, ast.Name):
                        self.debug_msg(lambda: f'[(Inline) Visit Name]', value)
                        cppDevModNames.append(value.id)
                        break

            devKey = '.'.join(cppDevModNames[::-1])

            def get_full_module_path(partial_path):
                parts = partial_path.split('.')
                for module_name, module in sys.modules.items():
                    if module_name.endswith(parts[0]):
                        try:
                            obj = module
                            for part in parts[1:]:
                                obj = getattr(obj, part)
                            return f"{module_name}.{'.'.join(parts[1:])}"
                        except AttributeError:
                            continue
                return partial_path

            devKey = get_full_module_path(devKey)
            if cudaq_runtime.isRegisteredDeviceModule(devKey):
                maybeKernelName = cudaq_runtime.checkRegisteredCppDeviceKernel(
                    self.module, devKey + '.' + node.func.attr)
                if maybeKernelName == None:
                    maybeKernelName = cudaq_runtime.checkRegisteredCppDeviceKernel(
                        self.module, devKey)
                if maybeKernelName != None:
                    otherKernel = SymbolTable(
                        self.module.operation)[maybeKernelName]
                    fType = otherKernel.type
                    if len(fType.inputs) != len(node.args):
                        funcName = node.func.id if hasattr(
                            node.func, 'id') else node.func.attr
                        self.emitFatalError(
                            f"invalid number of arguments passed to callable {funcName} ({len(node.args)} vs required {len(fType.inputs)})",
                            node)

                    func.CallOp(otherKernel, args)
                    return

            # Start by seeing if we have mod1.mod2.mod3...
            moduleNames = []
            value = node.func.value
            while isinstance(value, ast.Attribute):
                self.debug_msg(lambda: f'[(Inline) Visit Attribute]', value)
                moduleNames.append(value.attr)
                value = value.value
                if isinstance(value, ast.Name):
                    self.debug_msg(lambda: f'[(Inline) Visit Name]', value)
                    moduleNames.append(value.id)
                    break

            if all(x in moduleNames for x in ['cudaq', 'dbg', 'ast']):
                # Handle a debug print statement
                if len(args) != 1:
                    self.emitFatalError(
                        f"cudaq.dbg.ast.{node.func.attr} call invalid - too many arguments passed.",
                        node)

                if node.func.attr == 'print_i64':
                    self._print_i64(args[0])
                elif node.func.attr == 'print_f64':
                    self._print_f64(args[0])
                else:
                    self.emitFatalError(
                        f"cudaq.dbg.ast.{node.func.attr} call invalid - unknown function.",
                        node)
                return

            # If we did have module names, then this is what we are looking for
            if len(moduleNames):
                name = node.func.attr
                if not name in globalKernelRegistry:
                    moduleNames.reverse()
                    self.emitFatalError(
                        "{}.{} is not a valid quantum kernel to call.".format(
                            '.'.join(moduleNames), node.func.attr), node)

                # If it is in `globalKernelRegistry`, it has to be in this Module
                otherKernel = SymbolTable(self.module.operation)[nvqppPrefix +
                                                                 name]
                fType = otherKernel.type
                if len(fType.inputs) != len(node.args):
                    funcName = node.func.id if hasattr(node.func,
                                                       'id') else node.func.attr
                    self.emitFatalError(
                        f"invalid number of arguments passed to callable {funcName} ({len(node.args)} vs required {len(fType.inputs)})",
                        node)

                func.CallOp(otherKernel, args)
                return

        if isinstance(node.func, ast.Name):
            self.debug_msg(lambda: f'[(Inline) Visit Name]', node.func)
            if node.func.id == 'len':
                assert len(args) == 1
                if isinstance(args[0], (stdlib.Stdvec, stdlib.Veq)):
                    self.pushValue(args[0].get_length())
                    return
                self.emitFatalError(
                    "__len__ not supported on variables of this type.", node)

            if node.func.id == 'range':
                self.pushValue(stdlib.Range(*args))
                return

            if node.func.id == 'enumerate':
                self.pushValue(stdlib.Enumerate(*args))
                return

            if node.func.id == 'complex':
                if len(namedArgs) == 0:
                    assert len(args) == 2
                    self.pushValue(stdlib.Complex128(args[0], args[1]))
                    return
                self.pushValue(
                    stdlib.Complex128(namedArgs['real'], namedArgs['imag']))
                return

            if node.func.id in [
                    'h', 'x', 'y', 'z', 's', 'sdg', 't', 'tdg', 'rx', 'ry',
                    'rz', 'r1', 'u3', 'swap'
            ]:
                args = [
                    arg.materialize()
                    if isinstance(arg, stdlib.AbstractValue) else arg
                    for arg in args
                ]
                result = getattr(stdlib, node.func.id)(args)
                if isinstance(result, stdlib.Error):
                    self.emitFatalError(result.message, node)
                return

            if node.func.id in [
                    'ch', 'cx', 'cy', 'cz', 'cs', 'ct', 'crx', 'cry', 'crz',
                    'cr1'
            ]:
                args = [
                    arg.materialize()
                    if isinstance(arg, stdlib.AbstractValue) else arg
                    for arg in args
                ]
                result = getattr(stdlib, node.func.id[1:]).ctrl(args)
                if isinstance(result, stdlib.Error):
                    self.emitFatalError(result.message, node)
                return

            if node.func.id in ['mx', 'my', 'mz']:
                args = [
                    arg.materialize()
                    if isinstance(arg, stdlib.AbstractValue) else arg
                    for arg in args
                ]
                registerName = self.currentAssignVariableName
                # If `registerName` is None, then we know that we
                # are not assigning this measure result to anything
                # so we therefore should not push it on the stack
                pushResultToStack = registerName != None or self.walkingReturnNode

                # By default we set the `register_name` for the measurement
                # to the assigned variable name (if there is one). But
                # the use could have manually specified `register_name='something'`
                # check for that here and use it there
                if len(namedArgs) == 1 and 'register_name' in namedArgs:
                    registerName = namedArgs['register_name']
                    if not isinstance(registerName, stdlib.StringLiteral):
                        self.emitFatalError(
                            "measurement register_name keyword must be a string literal.",
                            node)
                    registerName = registerName.value

                registerName = registerName if registerName else None
                result = getattr(stdlib, node.func.id)(args, label=registerName)
                if isinstance(result, stdlib.Error):
                    self.emitFatalError(result.message, node)
                if pushResultToStack:
                    self.pushValue(result)
                return

            if node.func.id == 'reset':
                args = [
                    arg.materialize()
                    if isinstance(arg, stdlib.AbstractValue) else arg
                    for arg in args
                ]
                result = stdlib.reset(args[0])
                if isinstance(result, stdlib.Error):
                    self.emitFatalError(result.message, node)
                return

            if node.func.id in globalRegisteredOperations:
                unitary = globalRegisteredOperations[node.func.id]
                numTargets = int(np.log2(np.sqrt(unitary.size)))

                if len(args) != numTargets:
                    self.emitFatalError(
                        f'invalid number of arguments ({len(args)}) passed to {node.func.id} (requires {numTargets} arguments)',
                        node)

                for i, t in enumerate(args):
                    if not isinstance(t, stdlib.QRef):
                        self.emitFatalError(
                            f'invalid target operand {i}, broadcasting is not supported on custom operations.'
                        )

                globalName = f'{nvqppPrefix}{node.func.id}_generator_{numTargets}.rodata'

                currentST = SymbolTable(self.module.operation)
                if not globalName in currentST:
                    with InsertionPoint(self.module.body):
                        gen_vector_of_complex_constant(self.loc, self.module,
                                                       globalName,
                                                       unitary.tolist())
                quake.CustomUnitarySymbolOp(
                    [],
                    generator=FlatSymbolRefAttr.get(globalName),
                    parameters=[],
                    controls=[],
                    targets=args,
                    is_adj=False)
                return

            # Handle the case where we are capturing an opaque kernel
            # function. It has to be in the capture vars and it has to
            # be a PyKernelDecorator.
            if node.func.id in self.capturedVars and node.func.id not in globalKernelRegistry:
                from .kernel_decorator import PyKernelDecorator
                var = self.capturedVars[node.func.id]
                if isinstance(var, PyKernelDecorator):
                    # If we found it, then compile its ASTModule to MLIR so
                    # that it is in the proper registries, then give it
                    # the proper function alias
                    PyASTBridge(var.capturedDataStorage,
                                existingModule=self.module,
                                locationOffset=var.location).visit(
                                    var.astModule)
                    # If we have an alias, make sure we point back to the
                    # kernel registry correctly for the next conditional check
                    if var.name in globalKernelRegistry:
                        node.func.id = var.name

            if node.func.id in globalKernelRegistry:
                # If in `globalKernelRegistry`, it has to be in this Module
                otherKernel = SymbolTable(self.module.operation)[nvqppPrefix +
                                                                 node.func.id]
                fType = otherKernel.type
                if len(fType.inputs) != len(args):
                    self.emitFatalError(
                        "invalid number of arguments passed to callable {} ({} vs required {})"
                        .format(node.func.id, len(args),
                                len(fType.inputs)), node)

                args = [
                    arg.materialize(with_type=fType.inputs[i]) if isinstance(
                        arg, stdlib.AbstractValue) else arg
                    for i, arg in enumerate(args)
                ]
                if len(fType.results) == 0:
                    func.CallOp(otherKernel, args)
                else:
                    result = func.CallOp(otherKernel, args).result
                    self.pushValue(result)
                return

            elif node.func.id in self.symbolTable:
                val = self.symbolTable[node.func.id]
                if cc.CallableType.isinstance(val.type):
                    callableTy = cc.CallableType.getFunctionType(val.type)
                    if not self.argumentsValidForFunction(args, callableTy):
                        self.emitFatalError(
                            "invalid argument types for callable function ({} vs {})"
                            .format([v.type for v in args], callableTy), node)

                    callable = cc.CallableFuncOp(callableTy, val).result
                    func.CallIndirectOp([], callable, args)
                    return

            elif node.func.id == 'exp_pauli':
                stdlib.exp_pauli(rotation=args[0],
                                 qubits=args[1],
                                 pauli_word=args[2])
                return

            elif node.func.id == 'int':
                casted = args[0].__int64__()
                if not casted:
                    self.emitFatalError(
                        f'Invalid cast to integer: {args[0].type}', node)
                self.pushValue(casted)

            elif node.func.id == 'list':
                if len(args) == 1:
                    # FIXME: This is a hack to get things going
                    if isinstance(args[0], stdlib.AbstractValue):
                        self.pushValue(args[0])
                        return
                    if isinstance(args[0], stdlib.Stdvec):
                        self.pushValue(args[0])
                        return
                self.emitFatalError('Invalid list() cast requested.', node)

            elif node.func.id == 'print_i64':
                self._print_i64(args[0])
                return
            elif node.func.id == 'print_f64':
                self._print_f64(args[0])
                return

            elif node.func.id in globalRegisteredTypes.classes:
                # Handle User-Custom Struct Constructor
                cls, annotations = globalRegisteredTypes.getClassAttributes(
                    node.func.id)

                if '__slots__' not in cls.__dict__:
                    self.emitWarning(
                        f"Adding new fields in data classes is not yet supported. The dataclass must be declared with @dataclass(slots=True) or @dataclasses.dataclass(slots=True).",
                        node)

                # Disallow user specified methods on structs
                if len({
                        k: v
                        for k, v in cls.__dict__.items()
                        if not (k.startswith('__') and k.endswith('__')) and
                        isinstance(v, FunctionType)
                }) != 0:
                    self.emitFatalError(
                        'struct types with user specified methods are not allowed.',
                        node)

                structTys = [
                    mlirTypeFromPyType(v, self.ctx)
                    for _, v in annotations.items()
                ]
                # Ensure we don't use hybrid data types
                numQuantumMemberTys = sum(
                    [1 if T.is_quantum_type(ty) else 0 for ty in structTys])
                if numQuantumMemberTys != 0:  # we have quantum member types
                    if numQuantumMemberTys != len(structTys):
                        self.emitFatalError(
                            f'hybrid quantum-classical data types not allowed in kernel code',
                            node)

                isStruq = not (not structTys)
                for fieldTy in structTys:
                    if not T.is_quantum_type(fieldTy):
                        isStruq = False
                if isStruq:
                    structTy = T.struq(structTys, node.func.id)
                    # Disallow recursive quantum struct types.
                    for fieldTy in structTys:
                        if T.is_struq_type(fieldTy):
                            self.emitFatalError(
                                'recursive quantum struct types not allowed.',
                                node)
                else:
                    structTy = stdlib.Struct.create(structTys,
                                                    name=node.func.id)

                if isStruq:
                    # If we have a quantum struct. We cannot allocate classical
                    # memory and load / store quantum type values to that memory
                    # space, so use `quake.MakeStruqOp`.
                    self.pushValue(quake.MakeStruqOp(structTy, args).result)
                    return

                if len(args) != len(structTys):
                    self.emitFatalError(
                        f'constructor struct type {node.func.id} requires '
                        f'{len(structTys)} arguments, but was given {len(args)}.',
                        node)

                for i, arg in enumerate(args):
                    arg = arg.materialize(with_type=structTys[i]) if isinstance(
                        arg, stdlib.AbstractValue) else arg
                    structTy = structTy.set_field(i, arg)
                self.pushValue(structTy)
                return

            else:
                self.emitFatalError(
                    "unhandled function call - {}, known kernels are {}".format(
                        node.func.id, globalKernelRegistry.keys()), node)

        elif isinstance(node.func, ast.Attribute):
            self.debug_msg(lambda: f'[(Inline) Visit Attribute]', node.func)
            self.debug_msg(lambda: f'[(Inline) Visit Name]', node.func.value)

            if node.func.value.id in ['numpy', 'np']:
                if node.func.attr == 'array':
                    if isinstance(args[0], stdlib.AbstractValue):
                        args[0] = args[0].materialize()
                    if isinstance(args[0], stdlib.Stdvec):
                        # `np.array(vec, <dtype = ty>)`
                        eleTy = args[0].elements_type
                        dTy = eleTy
                        if len(namedArgs) > 0:
                            dTy = namedArgs['dtype']

                        # Convert the vector to the provided data type if needed.
                        self.pushValue(
                            self.__copyVectorAndCastElements(args[0], dTy))
                        return

                    raise self.emitFatalError(
                        f"unexpected numpy array initializer type: {args[0].type}",
                        node)

                result = None
                match node.func.attr:
                    case 'complex128':
                        result = args[0].__complex128__()
                    case 'complex64':
                        result = args[0].__complex64__()
                    case 'float64':
                        result = args[0].__float64__()
                    case 'float32':
                        result = args[0].__float32__()
                    case 'int32':
                        result = args[0].__int32__()
                    case 'int64':
                        result = args[0].__int64__()
                    case 'cos':
                        result = stdlib.cos(args[0])
                    case 'sin':
                        result = stdlib.sin(args[0])
                    case 'sqrt':
                        result = stdlib.sqrt(args[0])
                    case 'ceil':
                        result = stdlib.ceil(args[0])
                    case 'exp':
                        result = stdlib.exp(args[0])
                    case _:
                        self.emitFatalError(
                            f"unsupported NumPy call ({node.func.attr})", node)

                self.pushValue(result)
                return

            if node.func.value.id == 'cudaq':
                if node.func.attr == 'complex':
                    self.pushValue(self.simulationDType())
                    return

                if node.func.attr == 'amplitudes':
                    if isinstance(args[0], stdlib.AbstractValue):
                        args[0] = args[0].materialize()
                    if isinstance(args[0], stdlib.Stdvec):
                        self.pushValue(args[0])
                        return

                    self.emitFatalError(
                        f"unsupported amplitudes argument type: {args[0].type}",
                        node)

                if node.func.attr == 'qvector':
                    if len(args) == 0:
                        self.emitFatalError(
                            'qvector does not have default constructor. Init from size or existing state.',
                            node)

                    if isinstance(args[0], stdlib.AbstractValue):
                        args[0] = args[0].materialize()
                    if (IntegerType.isinstance(args[0].type)):
                        self.pushValue(stdlib.Veq.create(args[0]))
                        return
                    if isinstance(args[0], stdlib.Stdvec):
                        # FIXME: Remove this
                        listScalar = None
                        arrNode = node.args[0]
                        if isinstance(arrNode, ast.List):
                            listScalar = arrNode.elts

                        if isinstance(arrNode, ast.Call) and isinstance(
                                arrNode.func, ast.Attribute):
                            if arrNode.func.value.id in [
                                    'numpy', 'np'
                            ] and arrNode.func.attr == 'array':
                                lst = node.args[0].args[0]
                                if isinstance(lst, ast.List):
                                    listScalar = lst.elts

                        if listScalar != None:
                            size = len(listScalar)
                            numQubits = np.log2(size)
                            if not numQubits.is_integer():
                                self.emitFatalError(
                                    "Invalid input state size for qvector init (not a power of 2)",
                                    node)
                        self.pushValue(stdlib.Veq.from_list(args[0]))
                        return

                    if cc.StateType.isinstance(args[0].type):
                        # Ouch, this is not looking good. We just loaded the pointer
                        # but now we need to get it back to a memory location.
                        state_ptr = self.ifNotPointerThenStore(args[0])
                        self.pushValue(stdlib.Veq.from_state(state_ptr))
                        return

                    self.emitFatalError(
                        f"unsupported qvector argument type: {args[0].type}",
                        node)
                    return

                if node.func.attr == "qubit":
                    if len(args) >= 1:
                        self.emitFatalError(
                            'cudaq.qubit() constructor does not take any arguments. To construct a vector of qubits, use `cudaq.qvector(N)`.'
                        )
                    self.pushValue(stdlib.qalloca())
                    return

                if node.func.attr == 'adjoint':
                    args = [
                        arg.materialize()
                        if isinstance(arg, stdlib.AbstractValue) else arg
                        for arg in args
                    ]
                    result = stdlib.adjoint(args[0], args[1:])
                    if result:
                        self.emitFatalError(result.message)
                    return

                if node.func.attr == 'control':
                    args = [
                        arg.materialize()
                        if isinstance(arg, stdlib.AbstractValue) else arg
                        for arg in args
                    ]
                    result = stdlib.control(symbol_or_value=args[0],
                                            controls=args[1],
                                            args=args[2:])
                    if result:
                        self.emitFatalError(result.message)
                    return

                if node.func.attr == 'apply_noise':
                    # Pop off all the arguments we need
                    numParamsVal = args[0]
                    # Shrink the arguments down
                    values = args[1:]

                    # Need to get the number of parameters as an integer
                    concreteIntAttr = IntegerAttr(
                        numParamsVal.owner.attributes['value'])
                    numParams = concreteIntAttr.value

                    # Next Value is our generated key for the channel
                    # Get it and shrink the list
                    key = args[0]
                    values = args[1:]

                    # Now we know the next `numParams` arguments are
                    # our Kraus channel parameters
                    params = args[:numParams]
                    for i, p in enumerate(params):
                        # If we have a F64 value, we want to
                        # store it to a pointer
                        if F64Type.isinstance(p.type):
                            alloca = _core.alloca(p.type)
                            _core.store(p, alloca)
                            params[i] = alloca

                    # The remaining arguments are the qubits
                    veq = stdlib.Veq.from_qubits(values[numParams:])
                    quake.ApplyNoiseOp(params, [veq], key=key)
                    return

                if node.func.attr == 'compute_action':
                    if len(args) != 2:
                        self.emitFatalError(
                            "compute_action requires 2 arguments.", node)
                    stdlib.compute_action(compute=args[0], action=args[1])
                    return

                self.emitFatalError(
                    f'Invalid function or class type requested from the cudaq module ({node.func.attr})',
                    node)

            if node.func.value.id in self.symbolTable:
                # Method call on one of our variables
                var = self.symbolTable[node.func.value.id]
                if isinstance(var, stdlib.Veq):
                    if node.func.attr == 'size':
                        # Handled already in the Attribute visit
                        # FIXME: It is weird that this get handled in a different place
                        self.pushValue(var.get_length())
                        return

                    # `qreg` or `qview` method call
                    if node.func.attr == 'back':
                        self.pushValue(var.back(*args))
                        return
                    if node.func.attr == 'front':
                        self.pushValue(var.front(*args))
                        return

            # We have a `func_name.ctrl`
            if node.func.value.id in [
                    'h', 'x', 'y', 'z', 's', 't', 'rx', 'ry', 'rz', 'r1', 'u3',
                    'swap'
            ]:
                args = [
                    arg.materialize()
                    if isinstance(arg, stdlib.AbstractValue) else arg
                    for arg in args
                ]
                op = getattr(stdlib, node.func.value.id)
                if node.func.attr == 'ctrl':
                    result = op.ctrl(args)
                elif node.func.attr == 'adj':
                    result = op.adj(args)
                else:
                    self.emitFatalError(
                        f'Unknown attribute on quantum operation {node.func.value.id} ({node.func.attr}).'
                    )

                if isinstance(result, stdlib.Error):
                    self.emitFatalError(result.message, node)
                return

            # custom `ctrl` and `adj`
            if node.func.value.id in globalRegisteredOperations:
                if not node.func.attr == 'ctrl' and not node.func.attr == 'adj':
                    self.emitFatalError(
                        f'Unknown attribute on custom operation {node.func.value.id} ({node.func.attr}).'
                    )

                unitary = globalRegisteredOperations[node.func.value.id]
                numTargets = int(np.log2(np.sqrt(unitary.size)))
                targets = args[-numTargets:]

                for i, t in enumerate(targets):
                    if not isinstance(t, stdlib.QRef):
                        self.emitFatalError(
                            f'invalid target operand {i}, broadcasting is not supported on custom operations.'
                        )

                globalName = f'{nvqppPrefix}{node.func.value.id}_generator_{numTargets}.rodata'

                currentST = SymbolTable(self.module.operation)
                if not globalName in currentST:
                    with InsertionPoint(self.module.body):
                        gen_vector_of_complex_constant(self.loc, self.module,
                                                       globalName,
                                                       unitary.tolist())

                negatedControlQubits = None
                controls = []
                is_adj = False

                if node.func.attr == 'ctrl':
                    controls = args[:-numTargets]
                    if not controls:
                        self.emitFatalError(
                            'controlled operation requested without any control argument(s).',
                            node)
                    negatedControlQubits = None
                    if len(self.controlNegations):
                        negCtrlBools = [None] * len(controls)
                        for i, c in enumerate(controls):
                            negCtrlBools[i] = c in self.controlNegations
                        negatedControlQubits = DenseBoolArrayAttr.get(
                            negCtrlBools)
                        self.controlNegations.clear()
                if node.func.attr == 'adj':
                    is_adj = True

                self.checkControlAndTargetTypes(controls, targets)
                quake.CustomUnitarySymbolOp(
                    [],
                    generator=FlatSymbolRefAttr.get(globalName),
                    parameters=[],
                    controls=controls,
                    targets=targets,
                    is_adj=is_adj,
                    negated_qubit_controls=negatedControlQubits)
                return

            self.emitFatalError(
                f"Invalid function call - '{node.func.value.id}' is unknown.")

    def visit_ListComp(self, node):
        if len(node.generators) > 1:
            self.emitFatalError(
                "CUDA-Q only supports single generators for list comprehension.",
                node)

        if not isinstance(node.generators[0].target, ast.Name):
            self.emitFatalError(
                "only support named targets in list comprehension", node)

        # Handle the case of `[qOp(q) for q in veq]`
        # FIXME: This is assumption that is not always true. It might as well be that the we are calling kernels that return a value.
        if isinstance(
                node.generators[0].iter,
                ast.Name) and node.generators[0].iter.id in self.symbolTable:
            self.debug_msg(lambda: f'[(Inline) Visit Name]',
                           node.generators[0].iter)
            if isinstance(self.symbolTable[node.generators[0].iter.id],
                          stdlib.Veq):
                # now we know we have `[expr(r) for r in iterable]`
                # reuse what we do in `visit_For()`
                forNode = ast.For()
                forNode.iter = node.generators[0].iter
                forNode.target = node.generators[0].target
                forNode.body = [node.elt]
                forNode.orelse = []
                self.visit_For(forNode)
                return

        self.visit(node.generators[0].iter)
        assert len(self.valueStack) == 1
        iterable = self.popValue()
        assert isinstance(iterable, stdlib.Iterable)

        start = _core.constant(0)
        size = iterable.get_length()
        step = _core.constant(1)

        targets = []
        if isinstance(node.generators[0].target, ast.Name):
            self.debug_msg(lambda: f'[(Inline) Visit Name]',
                           node.generators[0].target)
            targets.append(node.generators[0].target.id)
        else:
            # has to be a `ast.Tuple`
            self.debug_msg(lambda: f'[(Inline) Visit Tuple]',
                           node.generators[0].target)
            for elt in node.generators[0].target.elts:
                targets.append(elt.id)

        # Materialize the loop
        loop = cc.LoopOp([start.type], [start], BoolAttr.get(False))

        whileBlock = Block.create_at_start(loop.whileRegion, [start.type])
        with InsertionPoint(whileBlock):
            pred = IntegerAttr.get(T.i64(), 2)
            test = arith.CmpIOp(pred, whileBlock.arguments[0], size).result
            cc.ConditionOp(test, whileBlock.arguments)

        stepBlock = Block.create_at_start(loop.stepRegion, [start.type])
        with InsertionPoint(stepBlock):
            incr = arith.AddIOp(stepBlock.arguments[0], step).result
            cc.ContinueOp([incr])

        bodyBlock = Block.create_at_start(loop.bodyRegion, [start.type])
        with InsertionPoint(bodyBlock):
            index = stdlib.Int64(bodyBlock.arguments[0])
            values = iterable.get_item(index)
            if not isinstance(values, list):
                values = [values]
            if len(targets) != len(values):
                raise CompilerError(
                    f"Cannot unpack {len(values)} values into {len(targets)} targets"
                )

            self.symbolTable.pushScope()

            # Assign values to targets
            for target, value in zip(targets, values):
                # FIXME: A target might be a variable that already exists
                slot = _core.alloca(value.type)
                _core.store(value, slot)
                self.symbolTable[target] = slot

            self.visit(node.elt)
            # FIXME: This is assumption that is not always true. It might as well be that the body doesn't return a value.
            assert len(self.valueStack) == 1
            result = self.popValue()
            if isinstance(result, stdlib.AbstractValue):
                result = result.materialize()

            # Now that we have the result, we can allocate the array we are creating
            with InsertionPoint(loop):
                array = stdlib.Stdvec.create(result.type, size)
            array.set_item(index, result)
            cc.ContinueOp(bodyBlock.arguments)

            self.symbolTable.popScope()

        self.pushValue(array)
        return

    def visit_List(self, node):
        for elt in node.elts:
            self.visit(elt)

        elements = [
            self.ifPointerThenLoad(self.popValue())
            for _ in range(len(node.elts))
        ]
        elements.reverse()

        # We do not store lists of pointers
        first_type = elements[0].type
        is_homogeneous = all(first_type == v.type for v in elements[1:])

        # If we are still not homogenous
        if not is_homogeneous:
            self.emitFatalError(
                "non-homogenous list not allowed - must all be same type: {}".
                format([v.type for v in elements]), node)

        # Turn this List into a StdVec<T>
        self.pushValue(stdlib.AList(elements))

    def visit_Constant(self, node):
        if isinstance(node.value, bool):
            self.pushValue(stdlib.BoolLiteral(node.value))
            return
        if isinstance(node.value, float):
            self.pushValue(stdlib.FloatLiteral(node.value))
            return
        if isinstance(node.value, int):
            self.pushValue(stdlib.IntLiteral(node.value))
            return
        if isinstance(node.value, str):
            self.pushValue(stdlib.StringLiteral(node.value))
            return
        if isinstance(node.value, complex):
            self.pushValue(stdlib.ComplexLiteral(node.value))
            return

        self.emitFatalError("unhandled constant value", node)

    def visit_Subscript(self, node):
        self.visit(node.value)
        var = self.popValue()

        if isinstance(node.ctx, ast.Store):
            if isinstance(node.slice, ast.Slice):
                self.emitFatalError("Cannot store to a slice.", node)
            self.visit(node.slice)
            index = self.getValue(self.popValue())
            if not IntegerType.isinstance(index.type):
                self.emitFatalError(
                    f'Invalid index variable type ({index.type})', node)
            if isinstance(var, stdlib.Veq):
                self.pushValue(var.get_item(index))
                return
            if isinstance(var, stdlib.Stdvec):
                item_addr = var.get_item_address(index)
                self.pushValue(item_addr)
                return
            if isinstance(var, stdlib.Pointer):
                pointee_type = var.element_type
                if cc.StdvecType.isinstance(pointee_type):
                    # TODO:Review this. Does it make sense to have a pointer to a stdvec?
                    var = var.load()
                    self.pushValue(var.get_item_address(index))
                    return
                if cc.StructType.isinstance(pointee_type):
                    name = cc.StructType.getName(pointee_type)
                    if name != 'tuple':
                        self.emitFatalError("Cannot address a struct this way.",
                                            node)
                    address = var[index]
                    self.pushValue(address)
                    return
            self.emitFatalError(
                f"unhandled subscript operation on store context {var.type}",
                node)

        # We are on the `load` context. Thus we need to return a value.

        # Handle slices, e.g. `var[lower:upper]`
        if isinstance(node.slice, ast.Slice):
            self.debug_msg(lambda: f'[(Inline) Visit Slice]', node.slice)
            if node.slice.step:
                self.emitFatalError("Step value in slice is not supported.",
                                    node)

            lower, upper = (None, None)
            if node.slice.lower:
                self.visit(node.slice.lower)
                lower = self.getValue(self.popValue())
            if node.slice.upper:
                self.visit(node.slice.upper)
                upper = self.getValue(self.popValue())

            if isinstance(var, (stdlib.Veq, stdlib.Stdvec)):
                self.pushValue(var.get_slice(lower, upper))
                return

            self.emitFatalError(
                f"unhandled slice operation, cannot handle type {var.type}",
                node)

        self.visit(node.slice)
        # TODO: Does a pointer to stdvec makes sense?
        var = self.getValue(var)

        if isinstance(var, (stdlib.Veq, stdlib.Stdvec)):
            idx = self.getValue(self.popValue())
            if not IntegerType.isinstance(idx.type):
                self.emitFatalError(f'invalid index variable type ({idx.type})',
                                    node)
            self.pushValue(var.get_item(idx))
            return

        if isinstance(var, stdlib.Struct):
            if var.name != 'tuple':
                self.emitFatalError("Cannot address a struct this way.", node)
            idx = self.popValue()
            if not isinstance(idx, stdlib.IntLiteral):
                self.emitFatalError(
                    "non-constant subscript value on a tuple is not supported",
                    node)

            if idx.value >= var.num_fields:
                self.emitFatalError(f'tuple index is out of range: {idx}', node)

            self.pushValue(var.get_field(idx))
            return

        self.emitFatalError(f"unhandled subscript: {var}", node)

    def visit_For(self, node):
        self.visit(node.iter)
        assert len(self.valueStack) == 1
        iterable = self.popValue()
        iterable = stdlib.try_downcast(self.ifPointerThenLoad(iterable))
        assert isinstance(iterable, stdlib.Iterable)

        targets = []
        if isinstance(node.target, ast.Name):
            self.debug_msg(lambda: f'[(Inline) Visit Name]', node.target)
            targets.append(node.target.id)
        else:
            # has to be a `ast.Tuple`
            self.debug_msg(lambda: f'[(Inline) Visit Tuple]', node.target)
            for elt in node.target.elts:
                targets.append(elt.id)

        def body_builder(index: Value):
            index = stdlib.Int64(index)
            values = iterable.get_item(index)
            if not isinstance(values, list):
                values = [values]
            if len(targets) != len(values):
                self.emitFatalError(
                    f"Cannot unpack {len(values)} values into {len(targets)} targets",
                    node)

            self.symbolTable.pushScope()

            # Assign values to targets
            for target, value in zip(targets, values):
                # FIXME: A target might be a variable that already exists
                if isinstance(value, stdlib.QRef):
                    self.symbolTable[target] = value
                else:
                    slot = _core.alloca(value.type)
                    _core.store(value, slot)
                    self.symbolTable[target] = slot

            # Process the loop body
            for stmt in node.body:
                self.visit(stmt)

            self.symbolTable.popScope()

        # For now this a completely new scope, it doesn't have access to the
        # loop variables! Shoud it have tho?
        def else_body():
            self.symbolTable.pushScope()
            for stmt in node.orelse:
                self.visit(stmt)
            self.symbolTable.popScope()

        size = iterable.get_length()
        _core.for_loop(size, body_builder,
                       else_body if len(node.orelse) > 0 else None)

    def visit_While(self, node):
        verbose = self.verbose

        def test() -> Value:
            # This is a hack to avoid the printing of MLIR values. The printing
            # triggers the verifier to be called, which is not what we want because
            # it will fail as the operation is still in the process of being built.
            # TODO: There might be a better way to do this. Operations can be printed
            # by using the `get_asm` method, which has a `assume_verified` argument.
            self.verbose = False
            self.visit(node.test)
            condition = self.ifPointerThenLoad(self.popValue())
            condition = condition.__as_bool__()
            return condition

        def body():
            self.symbolTable.pushScope()
            for stmt in node.body:
                self.visit(stmt)
            self.symbolTable.popScope()

        def else_body():
            self.symbolTable.pushScope()
            for stmt in node.orelse:
                self.visit(stmt)
            self.symbolTable.popScope()

        _core.while_loop(test, body,
                         else_body if len(node.orelse) > 0 else None)
        self.verbose = verbose

    def visit_BoolOp(self, node):
        # We can have the same semantics as Python because that would require variants
        # for the short-circuiting behavior. E.g. `a or b` would be `a` if `a` is truthy,
        # otherwise `b`. In Python, `a` and `b` can be of different types, so
        # we would need to promote them to a common type.

        # Visit the LHS and pop the value
        # Note we want any `mz(q)` calls to push their
        # result value to the stack, so we set a non-None
        # variable name here.
        self.currentAssignVariableName = ''
        self.visit(node.values[0])
        lhs = self.ifPointerThenLoad(self.popValue())
        lhs_condition = _core.implicit_conversion(lhs, T.i1())

        ifOp = cc.IfOp([lhs_condition.type], lhs_condition, [])
        thenBlock = Block.create_at_start(ifOp.thenRegion, [])
        with InsertionPoint(thenBlock):
            if isinstance(node.op, ast.Or):
                cc.ContinueOp([lhs_condition])
            else:
                self.visit(node.values[1])
                rhs = self.ifPointerThenLoad(self.popValue())
                rhs_condition = _core.implicit_conversion(rhs, T.i1())
                cc.ContinueOp([rhs_condition])

        elseBlock = Block.create_at_start(ifOp.elseRegion, [])
        with InsertionPoint(elseBlock):
            if isinstance(node.op, ast.Or):
                self.visit(node.values[1])
                rhs = self.ifPointerThenLoad(self.popValue())
                rhs_condition = _core.implicit_conversion(rhs, T.i1())
                cc.ContinueOp([rhs_condition])
            else:
                cc.ContinueOp([lhs_condition])

        # Reset the assign variable name
        self.currentAssignVariableName = None
        self.pushValue(ifOp.result)
        return

    def visit_Compare(self, node):
        if len(node.ops) > 1:
            self.emitFatalError("only single comparators are supported.", node)

        if isinstance(node.left, ast.Name):
            self.debug_msg(lambda: f'[(Inline) Visit Name]', node.left)
            if node.left.id not in self.symbolTable:
                self.emitFatalError(
                    f"{node.left.id} was not initialized before use in compare expression.",
                    node)

        self.visit(node.left)
        left = self.ifPointerThenLoad(self.popValue())
        self.visit(node.comparators[0])
        right = self.ifPointerThenLoad(self.popValue())
        op = node.ops[0]

        result = None
        match type(op):
            case ast.Eq:
                result = left.__eq__(right)
            case ast.NotEq:
                result = left.__ne__(right)
            case ast.Lt:
                result = left.__lt__(right)
            case ast.LtE:
                result = left.__le__(right)
            case ast.Gt:
                result = left.__gt__(right)
            case ast.GtE:
                result = left.__ge__(right)
            case ast.In:
                result = right.__contains__(left)
            case ast.NotIn:
                result = right.__contains__(left)
                result = result.__not__()
            case _:
                self.emitFatalError(
                    f"Unsupported compare operator - {type(op)}", node)

        if result is None:
            self.emitFatalError(
                f"Unsupported comparison - {type(op)}, {left} {right}", node)

        self.pushValue(result)

    def visit_AugAssign(self, node):
        self.visit(node.target)
        target = self.popValue()
        assert isinstance(target, stdlib.Pointer)

        self.visit(node.value)
        value = self.ifPointerThenLoad(self.popValue())

        loaded_target = target.load()

        result = None
        match type(node.op):
            case ast.Add:
                result = loaded_target.__iadd__(value)
            case ast.Sub:
                result = loaded_target.__isub__(value)
            case ast.Mult:
                result = loaded_target.__imul__(value)
            case ast.Div:
                result = loaded_target.__itruediv__(value)
            case ast.Mod:
                result = loaded_target.__imod__(value)
            case ast.FloorDiv:
                result = loaded_target.__ifloordiv__(value)
            case ast.LShift:
                result = loaded_target.__ilshift__(value)
            case ast.RShift:
                result = loaded_target.__irshift__(value)
            case ast.BitAnd:
                result = loaded_target.__iand__(value)
            case ast.BitOr:
                result = loaded_target.__ior__(value)
            case ast.BitXor:
                result = loaded_target.__ixor__(value)

        if not result:
            self.emitFatalError(
                f"Unsupported binary operator - {type(node.op)}", node)
        result = _core.implicit_conversion(result, loaded_target.type)
        if not result:
            self.emitFatalError(f"Invalid AugAssing operator - {type(node.op)}",
                                node)
        target.store(result)
        return

    def visit_If(self, node):
        # Visit the conditional node, retain
        # measurement results by assigning a dummy variable name
        self.currentAssignVariableName = ''
        self.visit(node.test)
        self.currentAssignVariableName = None

        condition = self.ifPointerThenLoad(self.popValue()).__as_bool__()

        def then_body():
            self.symbolTable.pushScope()
            for stmt in node.body:
                self.visit(stmt)
            self.symbolTable.popScope()

        def else_body():
            self.symbolTable.pushScope()
            for stmt in node.orelse:
                self.visit(stmt)
            self.symbolTable.popScope()

        _core.if_else(condition.value, then_body,
                      else_body if len(node.orelse) > 0 else None)

    def visit_Return(self, node):
        if node.value == None:
            return

        self.walkingReturnNode = True
        self.visit(node.value)
        self.walkingReturnNode = False

        if len(self.valueStack) == 0:
            return

        result = self.ifPointerThenLoad(self.popValue())
        if isinstance(result, stdlib.AbstractValue):
            result = result.materialize(with_type=self.knownResultType)

        if result.type != self.knownResultType:
            result = _core.implicit_conversion(result, self.knownResultType)

        if result.type != self.knownResultType:
            self.emitFatalError(
                f"Invalid return type, function was defined to return a {mlirTypeToPyType(self.knownResultType)} but the value being returned is of type {mlirTypeToPyType(result.type)}",
                node)

        if isinstance(result, stdlib.Stdvec):
            symName = '__nvqpp_vectorCopyCtor'
            load_intrinsic(self.module, symName)
            eleTy = result.elements_type
            ptrTy = T.ptr(T.i8())
            resBuf = result.data()
            # TODO Revisit this calculation
            byteWidth = 16 if ComplexType.isinstance(eleTy) else 8
            eleSize = _core.constant(byteWidth)
            dynSize = result.get_length()
            resBuf = cc.CastOp(ptrTy, resBuf)
            heapCopy = func.CallOp([ptrTy], symName,
                                   [resBuf, dynSize, eleSize]).result
            res = cc.StdvecInitOp(result.type, heapCopy, length=dynSize).result
            func.ReturnOp([res])
            return

        if self.symbolTable.numLevels() > 1:
            # We are in an inner scope, release all scopes before returning
            cc.UnwindReturnOp([result])
            return

        func.ReturnOp([result])

    def visit_Tuple(self, node):
        for elt in node.elts:
            self.visit(elt)

        if isinstance(node.ctx, ast.Store):
            vars = [self.popValue() for _ in range(len(node.elts))]
            vars.reverse()
            self.pushValue(stdlib.Tuple(*vars))
            return

        values = [
            self.ifPointerThenLoad(self.popValue())
            for _ in range(len(node.elts))
        ]
        values.reverse()
        self.pushValue(stdlib.Tuple(*values))

    def visit_UnaryOp(self, node):
        self.visit(node.operand)
        operand = self.ifPointerThenLoad(self.popValue())

        # Handle qubit negations
        if isinstance(node.op, ast.Invert):
            if isinstance(operand, stdlib.QRef):
                self.controlNegations.append(operand)
                self.pushValue(operand)
                return

        match type(node.op):
            case ast.USub:
                if hasattr(operand, "__neg__"):
                    self.pushValue(operand.__neg__())
                    return
            case ast.Not:
                if hasattr(operand, "__bool__"):
                    result = operand.__bool__()
                    self.pushValue(result.__not__())
                    return

        self.emitFatalError(f"Unsupported unary operator - {type(node.op)}",
                            node)

    def visit_Break(self, node):
        err = _core.break_op()
        if err:
            self.emitFatalError(err.message, node)
            return

    def visit_Continue(self, node):
        err = _core.continue_op()
        if err:
            self.emitFatalError(err.message, node)
            return

    def visit_BinOp(self, node):
        self.visit(node.left)
        left = self.ifPointerThenLoad(self.popValue())
        self.visit(node.right)
        right = self.ifPointerThenLoad(self.popValue())

        result = None
        match type(node.op):
            case ast.Add:
                result = left.__add__(right) or right.__radd__(left)
            case ast.Sub:
                result = left.__sub__(right) or right.__rsub__(left)
            case ast.Mult:
                result = left.__mul__(right) or right.__rmul__(left)
            case ast.Div:
                result = left.__truediv__(right) or right.__rtruediv__(left)
            case ast.Mod:
                result = left.__mod__(right) or right.__rmod__(left)
            case ast.FloorDiv:
                result = left.__floordiv__(right) or right.__rfloordiv__(left)
            case ast.Pow:
                result = left.__pow__(right) or right.__rpow__(left)
            case ast.LShift:
                result = left.__lshift__(right) or right.__rlshift__(left)
            case ast.RShift:
                result = left.__rshift__(right) or right.__rrshift__(left)
            case ast.BitAnd:
                result = left.__and__(right) or right.__rand__(left)
            case ast.BitOr:
                result = left.__or__(right) or right.__ror__(left)
            case ast.BitXor:
                result = left.__xor__(right) or right.__rxor__(left)
            case _:
                self.emitFatalError(
                    f"Unsupported binary operator - {type(node.op)}", node)

        if result is None:
            self.emitFatalError(
                f"Unsupported binary operator - {type(node.op)}, {left} {right}",
                node)

        self.pushValue(result)

    def visit_Name(self, node):
        if isinstance(node.ctx, ast.Store):
            if node.id in self.capturedVars:
                self.emitFatalError(
                    f"CUDA-Q does not allow assignment to variables captured from parent scope.",
                    node)

        if node.id in globalKernelRegistry:
            self.pushValue(FlatSymbolRefAttr.get(nvqppPrefix + node.id))
            return

        if node.id == 'complex':
            self.pushValue(T.complex128())
            return

        if node.id == 'float':
            self.pushValue(T.f64())
            return

        if node.id in self.symbolTable:
            self.pushValue(self.symbolTable[node.id])
            return

        if node.id in self.capturedVars:
            if isinstance(node.ctx, ast.Store):
                self.emitFatalError(
                    f"CUDA-Q does not allow assignment to variables captured from parent scope.",
                    node)

            # Only support a small subset of types here
            complexType = type(1j)
            value = self.capturedVars[node.id]

            if isinstance(value, State):
                self.pushValue(self.capturedDataStorage.storeCudaqState(value))
                return

            if isinstance(value, (list, np.ndarray)) and isinstance(
                    value[0], T.CapturableType):
                # FIXME: This is wrong, but necessary to keep current behavior.
                # For example, it accepts `[1j, 1.0]` as a vector of complex
                # numbers, but it should not the types are `[complex, float]`.
                with InsertionPoint.at_block_begin(self.entry):
                    elements_type = T.infer_type(value[0])
                    elementValues = [
                        _core.constant(el, elements_type) for el in value
                    ]
                    # Save the copy of the captured list so we can compare
                    # it to the scope to detect changes on recompilation.
                    self.dependentCaptureVars[node.id] = value.copy()
                    mlirVal = stdlib.Stdvec.from_list(elementValues)
                    self.symbolTable.add(node.id, mlirVal, 0)
                    self.pushValue(mlirVal)
                    return

            if isinstance(value, T.CapturableType):
                self.dependentCaptureVars[node.id] = value
                with InsertionPoint.at_block_begin(self.entry):
                    mlirVal = _core.constant(value)
                    stackSlot = _core.alloca(mlirVal.type)
                    _core.store(mlirVal, stackSlot)
                    # Store at the top-level
                    self.symbolTable.add(node.id, stackSlot, 0)
                    self.pushValue(stackSlot)
                    return

            errorType = type(value).__name__
            if (isinstance(value, list)):
                errorType = f"{errorType}[{type(value[0]).__name__}]"

            try:
                if issubclass(value, cudaq_runtime.KrausChannel):
                    # Here we have a KrausChannel as part of the AST
                    # We want to create a hash value from it, and
                    # we then want to push the number of parameters and
                    # that hash value. This can only be used with apply_noise
                    if not hasattr(value, 'num_parameters'):
                        self.emitFatalError(
                            'apply_noise kraus channels must have `num_parameters` constant class attribute specified.'
                        )

                    self.pushValue(_core.constant(value.num_parameters))
                    self.pushValue(_core.constant(hash(value)))
                    return
            except TypeError:
                pass

            self.emitFatalError(
                f"Invalid type for variable ({node.id}) captured from parent scope (only int, bool, float, complex, cudaq.State, and list/np.ndarray[int|bool|float|complex] accepted, type was {errorType}).",
                node)

        if isinstance(node.ctx, ast.Load):
            self.emitFatalError(
                f"Invalid variable name requested - '{node.id}' is not defined within the quantum kernel it is used in.",
                node)

        self.pushValue(node.id)


def compile_to_mlir(astModule, capturedDataStorage: CapturedDataStorage,
                    **kwargs):
    """
    Compile the given Python AST Module for the CUDA-Q 
    kernel FunctionDef to an MLIR `ModuleOp`. 
    Return both the `ModuleOp` and the list of function 
    argument types as MLIR Types. 

    This function will first check to see if there are any dependent 
    kernels that are required by this function. If so, those kernels 
    will also be compiled into the `ModuleOp`. The AST will be stored 
    later for future potential dependent kernel lookups. 
    """

    global globalAstRegistry
    verbose = 'verbose' in kwargs and kwargs['verbose']
    returnType = kwargs['returnType'] if 'returnType' in kwargs else None
    lineNumberOffset = kwargs['location'] if 'location' in kwargs else ('', 0)
    parentVariables = kwargs[
        'parentVariables'] if 'parentVariables' in kwargs else {}

    # Create the AST Bridge
    bridge = PyASTBridge(capturedDataStorage,
                         verbose=verbose,
                         knownResultType=returnType,
                         returnTypeIsFromPython=True,
                         locationOffset=lineNumberOffset,
                         capturedVariables=parentVariables)

    # Validate the arguments and return statements
    analysis.ValidateArgumentAnnotations(bridge).visit(astModule)
    analysis.ValidateReturnStatements(bridge).visit(astModule)

    # Find any dependent kernels, they have to be built as part of this ModuleOp.
    vis = analysis.FindDepKernelsVisitor(bridge.ctx)
    vis.visit(astModule)
    depKernels = vis.depKernels

    # Keep track of a kernel call graph, we will
    # sort this later after we build up the graph
    callGraph = {vis.kernelName: {k for k, v in depKernels.items()}}

    # Visit dependent kernels recursively to
    # ensure we have all necessary kernels added to the
    # module
    transitiveDeps = depKernels
    while len(transitiveDeps):
        # For each found dependency, see if that kernel
        # has further dependencies
        for depKernelName, depKernelAst in transitiveDeps.items():
            localVis = analysis.FindDepKernelsVisitor(bridge.ctx)
            localVis.visit(depKernelAst[0])
            # Append the found dependencies to our running tally
            depKernels = {**depKernels, **localVis.depKernels}
            # Reset for the next go around
            transitiveDeps = localVis.depKernels
            # Update the call graph
            callGraph[localVis.kernelName] = {
                k for k, v in localVis.depKernels.items()
            }

    # Sort the call graph topologically
    callGraphSorter = graphlib.TopologicalSorter(callGraph)
    sortedOrder = callGraphSorter.static_order()

    # Add all dependent kernels to the MLIR Module,
    # Do not check any 'dependent' kernels that
    # have the same name as the main kernel here, i.e.
    # ignore kernels that have the same name as this one.
    for funcName in sortedOrder:
        if funcName != vis.kernelName and funcName in depKernels:
            # Build an AST Bridge and visit the dependent kernel
            # function. Provide the dependent kernel source location as well.
            PyASTBridge(capturedDataStorage,
                        existingModule=bridge.module,
                        locationOffset=depKernels[funcName][1]).visit(
                            depKernels[funcName][0])

    # Build the MLIR Module for this kernel
    bridge.visit(astModule)

    if verbose:
        print(bridge.module)

    # Canonicalize the code, check for measurement(s) readout
    pm = PassManager.parse(
        "builtin.module(canonicalize,cse,func.func(quake-add-metadata))",
        context=bridge.ctx)

    try:
        pm.run(bridge.module)
    except:
        raise RuntimeError("could not compile code for '{}'.".format(
            bridge.name))

    extraMetaData = {}
    if len(bridge.dependentCaptureVars):
        extraMetaData['dependent_captures'] = bridge.dependentCaptureVars

    return bridge.module, bridge.argTypes, extraMetaData
