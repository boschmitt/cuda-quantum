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
from cudaq.mlir.dialects import arith, cc, complex, func, math, quake
from cudaq.mlir.ir import (BoolAttr, Block, BlockArgument, Context, ComplexType,
                           DenseBoolArrayAttr, DenseI32ArrayAttr,
                           DenseI64ArrayAttr, DictAttr, F32Type, F64Type,
                           FlatSymbolRefAttr, FloatAttr, FunctionType,
                           InsertionPoint, IntegerAttr, IntegerType, Location,
                           Module, StringAttr, SymbolTable, TypeAttr, UnitAttr)
from cudaq.mlir.passmanager import PassManager
from .analysis import FindDepKernelsVisitor
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

# CC Dialect `ComputePtrOp` in C++ sets the
# dynamic index as `std::numeric_limits<int32_t>::min()`
# (see CCOps.tc line 898). We'll duplicate that
# here by just setting it manually
kDynamicPtrIndex: int = -2147483648

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
        self.inForBodyStack = deque()
        self.inIfStmtBlockStack = deque()
        self.currentAssignVariableName = None
        self.walkingReturnNode = False
        self.controlNegations = []
        self.subscriptPushPointerValue = False
        self.attributePushPointerValue = False
        self.verbose = 'verbose' in kwargs and kwargs['verbose']
        self.currentNode = None

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
        if astNode == None:
            astNode = self.currentNode
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
        if astNode == None:
            astNode = self.currentNode
        lineNumber = '' if astNode == None else astNode.lineno + self.locationOffset[
            1] - 1

        print(Color.BOLD, end='')
        msg = codeFile + ":" + str(
            lineNumber
        ) + ": " + Color.RED + "error: " + Color.END + Color.BOLD + msg + (
            "\n\t (offending source -> " + ast.unparse(astNode) + ")" if
            hasattr(ast, 'unparse') and astNode is not None else '') + Color.END
        raise CompilerError(msg)

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

        # Ensure that functions with a return-type annotation actually has a valid return statement
        # in all paths, if not throw an error.
        class ValidateReturnStatements(ast.NodeVisitor):

            def __init__(self, bridge):
                self.bridge = bridge

            def visit_FunctionDef(self, node):
                # skip if un-annotated or explicitly marked as None
                is_none_ret = (isinstance(node.returns, ast.Constant) and
                               node.returns.value is None) or (
                                   isinstance(node.returns, ast.Name) and
                                   node.returns.id == 'None')

                if node.returns is None or is_none_ret:
                    return self.generic_visit(node)

                def all_paths_return(stmts):
                    for stmt in stmts:
                        if isinstance(stmt, ast.Return):
                            return True

                        if isinstance(stmt, ast.If):
                            if all_paths_return(stmt.body) and all_paths_return(
                                    stmt.orelse):
                                return True

                        if isinstance(stmt, (ast.For, ast.While)):
                            if all_paths_return(stmt.body) or all_paths_return(
                                    stmt.orelse):
                                return True

                    return False

                if not all_paths_return(node.body):
                    self.bridge.emitFatalError(
                        'cudaq.kernel functions with return type annotations must have a return statement.',
                        node)

                self.generic_visit(node)

        ValidateReturnStatements(self).visit(astModule)

    def getVeqType(self, size=None):
        """
        Return a `quake.VeqType`. Pass the size of the `quake.veq` if known. 
        """
        if size == None:
            return quake.VeqType.get()
        return quake.VeqType.get(size)

    def getRefType(self):
        """
        Return a `quake.RefType`.
        """
        return quake.RefType.get()

    def isQuantumType(self, ty):
        """
        Return True if the given type is quantum (is a `VeqType` or `RefType`). 
        Return False otherwise.
        """
        return quake.RefType.isinstance(ty) or quake.VeqType.isinstance(
            ty) or quake.StruqType.isinstance(ty)

    def isMeasureResultType(self, ty, value):
        """
        Return true if the given type is a qubit measurement result type (an i1 type).
        """
        if hasattr(value, 'owner') and hasattr(
                value.owner,
                'name') and not 'quake.discriminate' == value.owner.name:
            return False
        return IntegerType.isinstance(ty) and ty == IntegerType.get_signless(1)

    def getIntegerType(self, width=64):
        """
        Return an MLIR `IntegerType` of the given bit width (defaults to 64 bits).
        """
        return IntegerType.get_signless(width)

    def getIntegerAttr(self, type, value):
        """
        Return an MLIR Integer Attribute of the given `IntegerType`.
        """
        return IntegerAttr.get(type, value)

    def getFloatType(self, width=64):
        """
        Return an MLIR float type (single or double precision).
        """
        # Note:
        # `numpy.float64` is the same as `float` type, with width of 64 bit.
        # `numpy.float32` type has width of 32 bit.
        return F64Type.get() if width == 64 else F32Type.get()

    def getFloatAttr(self, type, value):
        """
        Return an MLIR float attribute (single or double precision).
        """
        return FloatAttr.get(type, value)

    def getConstantFloat(self, value, width=64):
        """
        Create a constant float operation and return its MLIR result Value.
        Takes as input the concrete float value.
        """
        ty = self.getFloatType(width=width)
        return self.getConstantFloatWithType(value, ty)

    def getConstantFloatWithType(self, value, ty):
        """
        Create a constant float operation and return its MLIR result Value.
        Takes as input the concrete float value.
        """
        return arith.ConstantOp(ty, self.getFloatAttr(ty, value)).result

    def getComplexType(self, width=64):
        """
        Return an MLIR complex type (single or double precision).
        """
        # Note:
        # `numpy.complex128` is the same as `complex` type,
        # with element width of 64bit (`np.complex64` and `float`)
        # `numpy.complex64` type has element type of `np.float32`.
        return self.getComplexTypeWithElementType(
            self.getFloatType(width=width))

    def getComplexTypeWithElementType(self, eTy):
        """
        Return an MLIR complex type (single or double precision).
        """
        return ComplexType.get(eTy)

    def getConstantComplex(self, value, width=64):
        """
        Create a constant complex operation and return its MLIR result Value.
        Takes as input the concrete complex value.
        """
        ty = self.getComplexType(width=width)
        return complex.CreateOp(ty,
                                self.getConstantFloat(value.real, width=width),
                                self.getConstantFloat(value.imag,
                                                      width=width)).result

    def getConstantComplexWithElementType(self, value, eTy):
        """
        Create a constant complex operation and return its MLIR result Value.
        Takes as input the concrete complex value.
        """
        ty = self.getComplexTypeWithElementType(eTy)
        return complex.CreateOp(ty,
                                self.getConstantFloatWithType(value.real, eTy),
                                self.getConstantFloatWithType(value.imag,
                                                              eTy)).result

    def getConstantInt(self, value, width=64):
        """
        Create a constant integer operation and return its MLIR result Value.
        Takes as input the concrete integer value. Can specify the integer bit width.
        """
        ty = self.getIntegerType(width)
        return arith.ConstantOp(ty, self.getIntegerAttr(ty, value)).result

    def promoteOperandType(self, ty, operand):
        if ty == operand.type:
            return operand

        if ComplexType.isinstance(ty):
            complexType = ComplexType(ty)
            floatType = complexType.element_type
            if ComplexType.isinstance(operand.type):
                otherComplexType = ComplexType(operand.type)
                otherFloatType = otherComplexType.element_type
                if (floatType != otherFloatType):
                    real = self.promoteOperandType(floatType,
                                                   complex.ReOp(operand).result)
                    imag = self.promoteOperandType(floatType,
                                                   complex.ImOp(operand).result)
                    operand = complex.CreateOp(complexType, real, imag).result
            else:
                real = self.promoteOperandType(floatType, operand)
                imag = self.getConstantFloatWithType(0.0, floatType)
                operand = complex.CreateOp(complexType, real, imag).result

        if (cc.StdvecType.isinstance(ty)):
            eleTy = cc.StdvecType.getElementType(ty)
            if cc.StdvecType.isinstance(operand.type):
                operand = self.__copyVectorAndCastElements(operand, eleTy)

        if F64Type.isinstance(ty):
            if F32Type.isinstance(operand.type):
                operand = cc.CastOp(ty, operand).result
            if IntegerType.isinstance(operand.type):
                zeroext = IntegerType(operand.type).width == 1
                operand = cc.CastOp(ty, operand, sint=not zeroext,
                                    zint=zeroext).result

        if F32Type.isinstance(ty):
            if F64Type.isinstance(operand.type):
                operand = cc.CastOp(ty, operand).result
            if IntegerType.isinstance(operand.type):
                zeroext = IntegerType(operand.type).width == 1
                operand = cc.CastOp(ty, operand, sint=not zeroext,
                                    zint=zeroext).result

        if IntegerType.isinstance(ty):
            if F64Type.isinstance(operand.type) or F32Type.isinstance(
                    operand.type):
                operand = cc.CastOp(ty, operand, sint=True, zint=False).result
            if IntegerType.isinstance(operand.type):
                if IntegerType(ty).width < IntegerType(operand.type).width:
                    operand = cc.CastOp(ty, operand).result
                else:
                    zeroext = IntegerType(operand.type).width == 1
                    operand = cc.CastOp(ty,
                                        operand,
                                        sint=not zeroext,
                                        zint=zeroext).result

        return operand

    def simulationPrecision(self):
        """
        Return precision for the current simulation backend,
        see `cudaq_runtime.SimulationPrecision`.
        """
        target = cudaq_runtime.get_target()
        return target.get_precision()

    def simulationDType(self):
        """
        Return the data type for the current simulation backend,
        either `numpy.complex128` or `numpy.complex64`.
        """
        if self.simulationPrecision() == cudaq_runtime.SimulationPrecision.fp64:
            return self.getComplexType(width=64)
        return self.getComplexType(width=32)

    def pushValue(self, value):
        """
        Push an MLIR Value onto the stack for usage in a subsequent AST node visit method.
        """
        self.debug_msg(lambda: f'push {value}')
        self.valueStack.append(value)

    def popValue(self):
        """
        Pop an MLIR Value from the stack. 
        """
        val = self.valueStack.pop()
        self.debug_msg(lambda: f'pop {val}')
        return val

    def pushForBodyStack(self, bodyBlockArgs):
        """
        Indicate that we are entering a for loop body block. 
        """
        self.inForBodyStack.append(bodyBlockArgs)

    def popForBodyStack(self):
        """
        Indicate that we have left a for loop body block.
        """
        self.inForBodyStack.pop()

    def pushIfStmtBlockStack(self):
        """
        Indicate that we are entering an if statement then or else block.
        """
        self.inIfStmtBlockStack.append(0)

    def popIfStmtBlockStack(self):
        """
        Indicate that we have just left an if statement then 
        or else block.
        """
        self.inIfStmtBlockStack.pop()

    def isInForBody(self):
        """
        Return True if the current insertion point is within 
        a for body block. 
        """
        return len(self.inForBodyStack) > 0

    def isInIfStmtBlock(self):
        """
        Return True if the current insertion point is within 
        an if statement then or else block.
        """
        return len(self.inIfStmtBlockStack) > 0

    def hasTerminator(self, block):
        """
        Return True if the given Block has a Terminator operation.
        """
        if len(block.operations) > 0:
            return cudaq_runtime.isTerminator(
                block.operations[len(block.operations) - 1])
        return False

    def isArithmeticType(self, type):
        """
        Return True if the given type is an integer, float, or complex type. 
        """
        return IntegerType.isinstance(type) or F64Type.isinstance(
            type) or F32Type.isinstance(type) or ComplexType.isinstance(type)

    def ifPointerThenLoad(self, value):
        """
        If the given value is of pointer type, load the pointer
        and return that new value.
        """
        if cc.PointerType.isinstance(value.type):
            return cc.LoadOp(value).result
        return value

    def ifNotPointerThenStore(self, value):
        """
        If the given value is not of a pointer type, allocate a
        slot on the stack, store the the value in the slot, and
        return the slot address.
        """
        if not cc.PointerType.isinstance(value.type):
            slot = cc.AllocaOp(cc.PointerType.get(value.type),
                               TypeAttr.get(value.type)).result
            cc.StoreOp(value, slot)
            return slot
        return value

    def __createStdvecWithKnownValues(self, size, listElementValues):
        # Turn this List into a StdVec<T>
        arrSize = self.getConstantInt(size)
        arrTy = cc.ArrayType.get(listElementValues[0].type)
        alloca = cc.AllocaOp(cc.PointerType.get(arrTy),
                             TypeAttr.get(listElementValues[0].type),
                             seqSize=arrSize).result

        for i, v in enumerate(listElementValues):
            eleAddr = cc.ComputePtrOp(
                cc.PointerType.get(listElementValues[0].type), alloca,
                [self.getConstantInt(i)],
                DenseI32ArrayAttr.get([kDynamicPtrIndex],
                                      context=self.ctx)).result
            cc.StoreOp(v, eleAddr)

        vecTy = listElementValues[0].type
        if cc.PointerType.isinstance(vecTy):
            vecTy = cc.PointerType.getElementType(vecTy)

        return cc.StdvecInitOp(cc.StdvecType.get(vecTy), alloca,
                               length=arrSize).result

    def getStructMemberIdx(self, memberName, structTy):
        """
        For the given struct type and member variable name, return 
        the index of the variable in the struct and the specific 
        MLIR type for the variable.
        """
        if cc.StructType.isinstance(structTy):
            structName = cc.StructType.getName(structTy)
        else:
            structName = quake.StruqType.getName(structTy)
        structIdx = None
        if not globalRegisteredTypes.isRegisteredClass(structName):
            self.emitFatalError(f'Dataclass is not registered: {structName})')

        _, userType = globalRegisteredTypes.getClassAttributes(structName)
        for i, (k, _) in enumerate(userType.items()):
            if k == memberName:
                structIdx = i
                break
        if structIdx == None:
            self.emitFatalError(
                f'Invalid struct member: {structName}.{memberName} (members={[k for k,_ in userType.items()]})'
            )
        return structIdx, mlirTypeFromPyType(userType[memberName], self.ctx)

    # Create a new vector with source elements converted to the target element type if needed.
    def __copyVectorAndCastElements(self, source, targetEleType):
        if not cc.PointerType.isinstance(source.type):
            if cc.StdvecType.isinstance(source.type):
                # Exit early if no copy is needed to avoid an unneeded store.
                sourceEleType = cc.StdvecType.getElementType(source.type)
                if (sourceEleType == targetEleType):
                    return source

        sourcePtr = source
        if not cc.PointerType.isinstance(sourcePtr.type):
            sourcePtr = self.ifNotPointerThenStore(sourcePtr)

        sourceType = cc.PointerType.getElementType(sourcePtr.type)
        if not cc.StdvecType.isinstance(sourceType):
            raise RuntimeError(
                f"expected vector type in __copyVectorAndCastElements but received {sourceType}"
            )

        sourceEleType = cc.StdvecType.getElementType(sourceType)
        if (sourceEleType == targetEleType):
            return sourcePtr

        sourceArrType = cc.ArrayType.get(sourceEleType)
        sourceElePtrTy = cc.PointerType.get(sourceEleType)
        sourceArrElePtrTy = cc.PointerType.get(sourceArrType)
        sourceValue = self.ifPointerThenLoad(sourcePtr)
        sourceDataPtr = cc.StdvecDataOp(sourceArrElePtrTy, sourceValue).result
        sourceSize = cc.StdvecSizeOp(self.getIntegerType(), sourceValue).result

        targetElePtrType = cc.PointerType.get(targetEleType)
        targetTy = cc.ArrayType.get(targetEleType)
        targetArrElePtrTy = cc.PointerType.get(targetTy)
        targetVecTy = cc.StdvecType.get(targetEleType)
        targetPtr = cc.AllocaOp(targetArrElePtrTy,
                                TypeAttr.get(targetEleType),
                                seqSize=sourceSize).result

        rawIndex = DenseI32ArrayAttr.get([kDynamicPtrIndex], context=self.ctx)

        def bodyBuilder(iterVar):
            eleAddr = cc.ComputePtrOp(sourceElePtrTy, sourceDataPtr, [iterVar],
                                      rawIndex).result
            loadedEle = cc.LoadOp(eleAddr).result
            castedEle = self.promoteOperandType(targetEleType, loadedEle)
            targetEleAddr = cc.ComputePtrOp(targetElePtrType, targetPtr,
                                            [iterVar], rawIndex).result
            cc.StoreOp(castedEle, targetEleAddr)

        self.createInvariantForLoop(sourceSize, bodyBuilder)
        return cc.StdvecInitOp(targetVecTy, targetPtr, length=sourceSize).result

    def __insertDbgStmt(self, value, dbgStmt):
        """
        Insert a debug print out statement if the programmer requested. Handles 
        statements like `cudaq.dbg.ast.print_i64(i)`.
        """
        value = self.ifPointerThenLoad(value)
        printFunc = None
        printStr = '[cudaq-ast-dbg] '
        argsTy = [cc.PointerType.get(self.getIntegerType(8))]
        if dbgStmt == 'print_i64':
            if not IntegerType.isinstance(value.type):
                self.emitFatalError(
                    f"print_i64 requested, but value is not of integer type (type was {value.type})."
                )

            currentST = SymbolTable(self.module.operation)
            argsTy += [self.getIntegerType()]
            # If `printf` is not in the module, or if it is but the last argument type is not an integer
            # then we have to add it
            if not 'print_i64' in currentST or not IntegerType.isinstance(
                    currentST['print_i64'].type.inputs[-1]):
                with InsertionPoint(self.module.body):
                    printOp = func.FuncOp('print_i64', (argsTy, []))
                    printOp.sym_visibility = StringAttr.get("private")
            currentST = SymbolTable(self.module.operation)
            printFunc = currentST['print_i64']
            printStr += '%ld\n'

        elif dbgStmt == 'print_f64':
            if not F64Type.isinstance(value.type):
                self.emitFatalError(
                    f"print_f64 requested, but value is not of float type (type was {value.type})."
                )

            currentST = SymbolTable(self.module.operation)
            argsTy += [self.getFloatType()]
            # If `printf` is not in the module, or if it is but the last argument type is not an float
            # then we have to add it
            if not 'print_f64' in currentST or not F64Type.isinstance(
                    currentST['print_f64'].type.inputs[-1]):
                with InsertionPoint(self.module.body):
                    printOp = func.FuncOp('print_f64', (argsTy, []))
                    printOp.sym_visibility = StringAttr.get("private")
            currentST = SymbolTable(self.module.operation)
            printFunc = currentST['print_f64']
            printStr += '%.12lf\n'
        else:
            raise self.emitFatalError(
                f"Invalid cudaq.dbg.ast statement - {dbgStmt}")

        strLitTy = cc.PointerType.get(
            cc.ArrayType.get(self.getIntegerType(8),
                             len(printStr) + 1))
        strLit = cc.CreateStringLiteralOp(strLitTy,
                                          StringAttr.get(printStr)).result
        strLit = cc.CastOp(cc.PointerType.get(self.getIntegerType(8)),
                           strLit).result
        func.CallOp(printFunc, [strLit, value])
        return

    def __get_vector_size(self, vector):
        """
        Get the size of a vector or array type.
        
        Args:
            vector: MLIR Value of vector/array type
            
        Returns:
            MLIR Value containing the size as an integer
        """
        if cc.StdvecType.isinstance(vector.type):
            return cc.StdvecSizeOp(self.getIntegerType(), vector).result
        return self.getConstantInt(
            cc.ArrayType.getSize(cc.PointerType.getElementType(vector.type)))

    def __load_vector_element(self, vector, index):
        """
        Load an element from a vector or array at the given index.
        
        Args:
            vector: MLIR Value of vector/array type
            index: MLIR Value containing integer index
            
        Returns:
            MLIR Value containing the loaded element
        """
        if cc.StdvecType.isinstance(vector.type):
            data_ptr = cc.StdvecDataOp(
                cc.PointerType.get(
                    cc.ArrayType.get(cc.StdvecType.getElementType(
                        vector.type))), vector).result
            return cc.LoadOp(
                cc.ComputePtrOp(
                    cc.PointerType.get(cc.StdvecType.getElementType(
                        vector.type)), data_ptr, [index],
                    DenseI32ArrayAttr.get([kDynamicPtrIndex]))).result
        return cc.LoadOp(
            cc.ComputePtrOp(
                cc.PointerType.get(
                    cc.ArrayType.getElementType(
                        cc.PointerType.getElementType(vector.type))), vector,
                [index], DenseI32ArrayAttr.get([kDynamicPtrIndex]))).result

    def __get_superior_type(self, a, b):
        """
        Get the superior numeric type between two MLIR Values.
        F64 > F32 > Integer, with integers promoting to the wider width.
        
        Args:
            a: First MLIR Value
            b: Second MLIR Value
            
        Returns:
            MLIR Type representing the superior type
        """
        if F64Type.isinstance(a.type) or F64Type.isinstance(b.type):
            return F64Type.get()
        if F32Type.isinstance(a.type) or F32Type.isinstance(b.type):
            return F32Type.get()
        return self.getIntegerType(
            max(IntegerType(a.type).width,
                IntegerType(b.type).width))

    def convertArithmeticToSuperiorType(self, values, type):
        """
        Assuming all values provided are arithmetic, convert each one to the 
        provided superior type. Float is superior to integer and complex is 
        superior to float (superior implies the inferior type can can be converted to the 
        superior type)
        """
        retValues = []
        for v in values:
            retValues.append(self.promoteOperandType(type, v))

        return retValues

    def isQuantumStructType(self, ty):
        """
        Return True if the given struct type has only quantum member variables.
        """
        return quake.StruqType.isinstance(ty)

    def mlirTypeFromAnnotation(self, annotation):
        """
        Return the MLIR Type corresponding to the given kernel function argument type annotation.
        Throws an exception if the programmer did not annotate function argument types. 
        """
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
        """
        Loop through the provided control and target qubit values and 
        assert that they are of quantum type. Emit a fatal error if not. 
        """
        [
            self.emitFatalError(f'control operand {i} is not of quantum type.')
            if not self.isQuantumType(control.type) else None
            for i, control in enumerate(controls)
        ]
        [
            self.emitFatalError(f'target operand {i} is not of quantum type.')
            if not self.isQuantumType(target.type) else None
            for i, target in enumerate(targets)
        ]

    def createInvariantForLoop(self,
                               endVal,
                               bodyBuilder,
                               startVal=None,
                               stepVal=None,
                               isDecrementing=False,
                               elseStmts=None):
        """
        Create an invariant loop using the CC dialect. 
        """
        startVal = self.getConstantInt(0) if startVal == None else startVal
        stepVal = self.getConstantInt(1) if stepVal == None else stepVal

        iTy = self.getIntegerType()
        inputs = [startVal]
        resultTys = [iTy]

        loop = cc.LoopOp(resultTys, inputs, BoolAttr.get(False))

        whileBlock = Block.create_at_start(loop.whileRegion, [iTy])
        with InsertionPoint(whileBlock):
            condPred = IntegerAttr.get(
                iTy, 2) if not isDecrementing else IntegerAttr.get(iTy, 4)
            cc.ConditionOp(
                arith.CmpIOp(condPred, whileBlock.arguments[0], endVal).result,
                whileBlock.arguments)

        bodyBlock = Block.create_at_start(loop.bodyRegion, [iTy])
        with InsertionPoint(bodyBlock):
            self.symbolTable.pushScope()
            self.pushForBodyStack(bodyBlock.arguments)
            bodyBuilder(bodyBlock.arguments[0])
            if not self.hasTerminator(bodyBlock):
                cc.ContinueOp(bodyBlock.arguments)
            self.popForBodyStack()
            self.symbolTable.popScope()

        stepBlock = Block.create_at_start(loop.stepRegion, [iTy])
        with InsertionPoint(stepBlock):
            incr = arith.AddIOp(stepBlock.arguments[0], stepVal).result
            cc.ContinueOp([incr])

        if elseStmts:
            elseBlock = Block.create_at_start(loop.elseRegion, [iTy])
            with InsertionPoint(elseBlock):
                self.symbolTable.pushScope()
                for stmt in elseStmts:
                    self.visit(stmt)
                if not self.hasTerminator(elseBlock):
                    cc.ContinueOp(elseBlock.arguments)
                self.symbolTable.popScope()

        loop.attributes.__setitem__('invariant', UnitAttr.get())
        return

    def __applyQuantumOperation(self, opName, parameters, targets):
        opCtor = getattr(quake, '{}Op'.format(opName.title()))
        for quantumValue in targets:
            if quake.VeqType.isinstance(quantumValue.type):

                def bodyBuilder(iterVal):
                    q = quake.ExtractRefOp(self.getRefType(),
                                           quantumValue,
                                           -1,
                                           index=iterVal).result
                    opCtor([], parameters, [], [q])

                veqSize = quake.VeqSizeOp(self.getIntegerType(),
                                          quantumValue).result
                self.createInvariantForLoop(veqSize, bodyBuilder)
            elif quake.RefType.isinstance(quantumValue.type):
                opCtor([], parameters, [], [quantumValue])
            else:
                self.emitFatalError(
                    f'quantum operation {opName} on incorrect quantum type {quantumValue.type}.'
                )
        return

    def __processRangeLoopIterationBounds(self, argumentNodes):
        """
        Analyze `range(...)` bounds and return the start, end, 
        and step values, as well as whether or not this a decrementing range.
        """
        iTy = self.getIntegerType(64)
        zero = arith.ConstantOp(iTy, IntegerAttr.get(iTy, 0))
        one = arith.ConstantOp(iTy, IntegerAttr.get(iTy, 1))
        isDecrementing = False
        if len(argumentNodes) == 3:
            # Find the step val and we need to
            # know if its decrementing
            # can be incrementing or decrementing
            stepVal = self.popValue()
            if isinstance(argumentNodes[2], ast.UnaryOp):
                self.debug_msg(lambda: f'[(Inline) Visit UnaryOp]',
                               argumentNodes[2])
                if isinstance(argumentNodes[2].op, ast.USub):
                    if isinstance(argumentNodes[2].operand, ast.Constant):
                        self.debug_msg(lambda: f'[(Inline) Visit Constant]',
                                       argumentNodes[2].operand)
                        if argumentNodes[2].operand.value > 0:
                            isDecrementing = True
                    else:
                        self.emitFatalError(
                            'CUDA-Q requires step value on range() to be a constant.'
                        )

            # exclusive end
            endVal = self.popValue()

            # inclusive start
            startVal = self.popValue()

        elif len(argumentNodes) == 2:
            stepVal = one
            endVal = self.popValue()
            startVal = self.popValue()
        else:
            stepVal = one
            endVal = self.popValue()
            startVal = zero

        startVal = self.ifPointerThenLoad(startVal)
        endVal = self.ifPointerThenLoad(endVal)
        stepVal = self.ifPointerThenLoad(stepVal)

        # Range expects integers
        if F64Type.isinstance(startVal.type):
            startVal = arith.FPToSIOp(self.getIntegerType(), startVal).result

        if F64Type.isinstance(endVal.type):
            endVal = arith.FPToSIOp(self.getIntegerType(), endVal).result

        if F64Type.isinstance(stepVal.type):
            stepVal = arith.FPToSIOp(self.getIntegerType(), stepVal).result

        return startVal, endVal, stepVal, isDecrementing

    def __visitStructAttribute(self, node, structValue):
        """
        Handle struct member extraction from either a pointer to struct or direct struct value.
        Uses the most efficient approach for each case.
        """
        if cc.PointerType.isinstance(structValue.type):
            # Handle pointer to struct - use ComputePtrOp
            eleType = cc.PointerType.getElementType(structValue.type)
            if cc.StructType.isinstance(eleType):
                structIdx, memberTy = self.getStructMemberIdx(
                    node.attr, eleType)
                eleAddr = cc.ComputePtrOp(cc.PointerType.get(memberTy),
                                          structValue, [],
                                          DenseI32ArrayAttr.get([structIdx
                                                                ])).result

                if self.attributePushPointerValue:
                    self.pushValue(eleAddr)
                    return

                # Load the value
                eleAddr = cc.LoadOp(eleAddr).result
                self.pushValue(eleAddr)
                return
        elif cc.StructType.isinstance(structValue.type):
            # Handle direct struct value - use ExtractValueOp (more efficient)
            structIdx, memberTy = self.getStructMemberIdx(
                node.attr, structValue.type)
            extractedValue = cc.ExtractValueOp(
                memberTy, structValue, [],
                DenseI32ArrayAttr.get([structIdx])).result

            if self.attributePushPointerValue:
                # If we need a pointer, we have to create a temporary slot
                tempSlot = cc.AllocaOp(cc.PointerType.get(memberTy),
                                       TypeAttr.get(memberTy)).result
                cc.StoreOp(extractedValue, tempSlot)
                self.pushValue(tempSlot)
                return

            self.pushValue(extractedValue)
            return
        else:
            self.emitFatalError(
                f"Cannot access attribute '{node.attr}' on type {structValue.type}"
            )

    def needsStackSlot(self, type):
        """
        Return true if this is a type that has been "passed by value" and 
        needs a stack slot created (i.e. a `cc.alloca`) for use throughout the 
        function. 
        """
        # FIXME add more as we need them
        return ComplexType.isinstance(type) or F64Type.isinstance(
            type) or F32Type.isinstance(type) or IntegerType.isinstance(
                type) or cc.StructType.isinstance(type)

    def visit(self, node):
        self.debug_msg(lambda: f'[Visit {type(node).__name__}]', node)
        self.indent_level += 1
        super().visit(node)
        self.indent_level -= 1

    def generic_visit(self, node):
        self.debug_msg(lambda: f'[Generic Visit]', node)
        for field, value in reversed(list(ast.iter_fields(node))):
            if isinstance(value, list):
                for item in value:
                    if isinstance(item, ast.AST):
                        self.visit(item)
            elif isinstance(value, ast.AST):
                self.visit(value)

    def visit_FunctionDef(self, node):
        """
        Create an MLIR `func.FuncOp` for the given FunctionDef AST node. For the top-level
        FunctionDef, this will add the `FuncOp` to the `ModuleOp` body, annotate the `FuncOp` with 
        `cudaq-entrypoint` if it is an Entry Point CUDA-Q kernel, and visit the rest of the 
        FunctionDef body. If this is an inner FunctionDef, this will treat the function as a CC 
        lambda function and add the cc.callable-typed value to the symbol table, keyed on the 
        FunctionDef name. 

        We keep track of the top-level function name as well as its internal MLIR name, prefixed 
        with the __nvqpp__mlirgen__ prefix. 
        """
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

            # Set this kernel as an entry point if the argument types are classical only
            def isQuantumTy(ty):
                return quake.RefType.isinstance(ty) or quake.VeqType.isinstance(
                    ty) or quake.StruqType.isinstance(ty)

            areQuantumTypes = [isQuantumTy(ty) for ty in self.argTypes]
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
                        stackSlot = cc.AllocaOp(cc.PointerType.get(b.type),
                                                TypeAttr.get(b.type)).result
                        cc.StoreOp(b, stackSlot)
                        self.symbolTable[argNames[i]] = stackSlot
                    else:
                        self.symbolTable[argNames[i]] = b

                # Visit the function
                startIdx = 0
                # Search for the potential documentation string, and
                # if found, start the body visitation after it.
                if len(node.body) and isinstance(node.body[0], ast.Expr):
                    self.debug_msg(lambda: f'[(Inline) Visit Expr]',
                                   node.body[0])
                    expr = node.body[0]
                    if hasattr(expr, 'value') and isinstance(
                            expr.value, ast.Constant):
                        self.debug_msg(lambda: f'[(Inline) Visit Constant]',
                                       expr.value)
                        constant = expr.value
                        if isinstance(constant.value, str):
                            startIdx = 1
                [self.visit(n) for n in node.body[startIdx:]]
                # Add the return operation
                if not self.hasTerminator(self.entry):
                    # If the function has a known (non-None) return type, emit
                    # an `undef` of that type and return it; else return void
                    if self.knownResultType is not None:
                        undef = cc.UndefOp(self.knownResultType).result
                        ret = func.ReturnOp([undef])
                    else:
                        ret = func.ReturnOp([])
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
        """
        Implement `ast.Expr` visitation to screen out all
        multi-line `docstrings`. These are differentiated from other strings
        at the node-type level. Strings we may care about will have been
        assigned to a variable (hence `ast.Assign` nodes), while other strings will exist
        as standalone expressions with no uses.
        """
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
        self.currentNode = node

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

    def visit_Assign(self, node):
        """
        Map an assign operation in the AST to an equivalent variable value assignment 
        in the MLIR. This method will first see if this is a tuple assignment, enabling one 
        to assign multiple values in a single statement.

        For all assignments, the variable name will be used as a key for the symbol table, 
        mapping to the corresponding MLIR Value. For values of `ref` / `veq`, `i1`, or `cc.callable`, 
        the values will be stored directly in the table. For all other values, the variable 
        will be allocated with a `cc.alloca` op, and the loaded value will be stored in the 
        symbol table.
        """
        self.currentNode = node

        # CUDA-Q does not yet support dynamic memory allocation
        if isinstance(node.value, ast.List) and len(node.value.elts) == 0:
            self.emitFatalError(
                "CUDA-Q does not support allocating lists of zero size (no dynamic memory allocation)",
                node)

        # Retain the variable name for potential children (like `mz(q, registerName=...)`)
        if len(node.targets) == 1 and not isinstance(node.targets[0],
                                                     ast.Tuple):
            # Handle simple `var = expr`
            if isinstance(node.targets[0], ast.Name):
                self.debug_msg(lambda: f'[(Inline) Visit Name]',
                               node.targets[0])
                self.currentAssignVariableName = str(node.targets[0].id)
                self.visit(node.value)
                self.currentAssignVariableName = None
            # Handle assignments like `listVar[IDX] = VALUE`, or `classVar.attr = VALUE`
            # `listVar`, `classVar` must be in the symbol table
            elif isinstance(
                    node.targets[0].value,
                    ast.Name) and node.targets[0].value.id in self.symbolTable:
                if isinstance(node.targets[0], ast.Subscript):
                    # Visit_Subscript will try to load any pointer and return it
                    # but here we want the pointer, so flip that flag
                    # FIXME: move loading from Visit_Subscript to the user instead.
                    self.subscriptPushPointerValue = True
                    # Visit the subscript node, get the pointer value
                    self.visit(node.targets[0])
                    # Reset the push pointer value flag
                    self.subscriptPushPointerValue = False
                    ptrVal = self.popValue()
                    if not cc.PointerType.isinstance(ptrVal.type):
                        self.emitFatalError(
                            "Invalid CUDA-Q subscript assignment, variable must be a pointer.",
                            node)
                    # See if this is a pointer to an array, if so cast it
                    # to a pointer on the array type
                    ptrEleType = cc.PointerType.getElementType(ptrVal.type)
                    if cc.ArrayType.isinstance(ptrEleType):
                        ptrVal = cc.CastOp(
                            cc.PointerType.get(
                                cc.ArrayType.getElementType(ptrEleType)),
                            ptrVal).result

                    # Visit the value being assigned
                    self.visit(node.value)
                    valueToStore = self.popValue()
                    # Store the value
                    cc.StoreOp(valueToStore, ptrVal)
                    return

                elif isinstance(node.targets[0], ast.Attribute):
                    self.attributePushPointerValue = True
                    # Visit the attribute node, get the pointer value
                    self.visit(node.targets[0])
                    # Reset the push pointer value flag
                    self.attributePushPointerValue = False
                    ptrVal = self.popValue()
                    if not cc.PointerType.isinstance(ptrVal.type):
                        self.emitFatalError(
                            "Invalid CUDA-Q attribute assignment, variable must be a pointer.",
                            node)
                    # Visit the value being assigned
                    self.visit(node.value)
                    valueToStore = self.popValue()
                    # Store the value
                    cc.StoreOp(valueToStore, ptrVal)
                    return
        else:
            if isinstance(node.value, ast.Tuple):
                for ele in node.value.elts:
                    self.visit(ele)
            else:
                self.visit(node.value)

        if len(self.valueStack) == 0:
            self.emitFatalError("invalid assignment detected.", node)

        varNames = []
        varValues = []

        # Can assign a, b, c, = Tuple...
        # or single assign a = something
        if isinstance(node.targets[0], ast.Tuple):
            self.debug_msg(lambda: f'[(Inline) Visit Tuple]', node.targets[0])
            assert len(self.valueStack) == len(node.targets[0].elts)
            varValues = [
                self.popValue() for _ in range(len(node.targets[0].elts))
            ]
            varValues.reverse()
            varNames = [name.id for name in node.targets[0].elts]
        else:
            self.debug_msg(lambda: f'[(Inline) Visit Name]', node.targets[0])
            varValues = [self.popValue()]
            varNames = [node.targets[0].id]

        for i, value in enumerate(varValues):
            if self.isQuantumType(value.type) or cc.CallableType.isinstance(
                    value.type):
                self.symbolTable[varNames[i]] = value
            elif self.isMeasureResultType(value.type, value):
                value = self.ifPointerThenLoad(value)
                if varNames[i] in self.symbolTable:
                    cc.StoreOp(
                        value,
                        self.ifNotPointerThenStore(
                            self.symbolTable[varNames[i]]))
                self.symbolTable[varNames[i]] = value
            elif varNames[i] in self.symbolTable:
                if varNames[i] in self.capturedVars:
                    self.emitFatalError(
                        f"CUDA-Q does not allow assignment to variables captured from parent scope.",
                        node)

                cc.StoreOp(value, self.symbolTable[varNames[i]])
            elif cc.PointerType.isinstance(value.type):
                self.symbolTable[varNames[i]] = value
            elif cc.StructType.isinstance(value.type) and isinstance(
                    value.owner.opview, cc.InsertValueOp):
                # If we have a new struct from `cc.undef` and `cc.insert_value`, we don't
                # want to allocate new memory.
                self.symbolTable[varNames[i]] = value
            else:
                # We should allocate and store
                alloca = cc.AllocaOp(cc.PointerType.get(value.type),
                                     TypeAttr.get(value.type)).result
                cc.StoreOp(value, alloca)
                self.symbolTable[varNames[i]] = alloca

    def visit_Attribute(self, node):
        """
        Visit an attribute node and map to valid MLIR code. This method specifically 
        looks for attributes like method calls, or common attributes we'll 
        see from ubiquitous external modules like `numpy`.
        """
        self.currentNode = node
        # Disallow list.append since we don't do dynamic memory allocation
        if isinstance(node.value,
                      ast.Name) and node.value.id in self.symbolTable:
            self.debug_msg(lambda: f'[(Inline) Visit Name]', node.value)
            value = self.symbolTable[node.value.id]
            if self.isQuantumStructType(value.type):
                # Here we have a quantum struct, need to use extract value instead
                # of load from compute pointer.
                structIdx, memberTy = self.getStructMemberIdx(
                    node.attr, value.type)
                self.pushValue(
                    quake.GetMemberOp(
                        memberTy, value,
                        IntegerAttr.get(self.getIntegerType(32),
                                        structIdx)).result)
                return

            if cc.PointerType.isinstance(value.type):
                eleType = cc.PointerType.getElementType(value.type)
                if cc.StructType.isinstance(eleType):
                    # Handle the case where we have a struct member extraction, memory semantics
                    self.__visitStructAttribute(node, value)
                    return

            if node.attr == 'append':
                type = value.type
                if cc.PointerType.isinstance(type):
                    type = cc.PointerType.getElementType(type)
                if cc.StdvecType.isinstance(type) or cc.ArrayType.isinstance(
                        type):
                    self.emitFatalError(
                        "CUDA-Q does not allow dynamic list resizing.", node)
                return

            if node.attr == 'size' and quake.VeqType.isinstance(value.type):
                self.pushValue(
                    quake.VeqSizeOp(self.getIntegerType(64),
                                    self.symbolTable[node.value.id]).result)
                return

        if node.attr in ['imag', 'real']:
            if isinstance(node.value,
                          ast.Name) and node.value.id in self.symbolTable:
                self.debug_msg(lambda: f'[(Inline) Visit Name]', node.value)
                value = self.symbolTable[node.value.id]
            else:
                self.visit(node.value)
                value = self.popValue()

            value = self.ifPointerThenLoad(value)

            if ComplexType.isinstance(value.type):
                if (node.attr == 'real'):
                    self.pushValue(complex.ReOp(value).result)
                    return

                if (node.attr == 'imag'):
                    self.pushValue(complex.ImOp(value).result)
                    return

        if isinstance(node.value, ast.Name):
            self.debug_msg(lambda: f'[(Inline) Visit Name]', node.value)
            if node.value.id in ['np', 'numpy', 'math']:
                if node.attr == 'complex64':
                    self.pushValue(self.getComplexType(width=32))
                    return
                if node.attr == 'complex128':
                    self.pushValue(self.getComplexType(width=64))
                    return
                if node.attr == 'pi':
                    self.pushValue(self.getConstantFloat(np.pi))
                    return
                if node.attr == 'e':
                    self.pushValue(self.getConstantFloat(np.e))
                    return
                if node.attr == 'euler_gamma':
                    self.pushValue(self.getConstantFloat(np.euler_gamma))
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
                    self.pushValue(
                        self.getConstantInt(channel_class.num_parameters))
                    self.pushValue(self.getConstantInt(hash(channel_class)))
                    return

        # Handle attribute access on call results (e.g., M(6, 9).i)
        if isinstance(node.value, ast.Call):
            # Visit the call first to get the result on the stack
            self.visit(node.value)
            if len(self.valueStack) == 0:
                self.emitFatalError("Call expression did not produce a value",
                                    node)

            # Get the call result from the stack
            call_result = self.popValue()

            # Handle attribute access on the call result
            self.__visitStructAttribute(node, call_result)

    def visit_Call(self, node):
        """
        Map a Python Call operation to equivalent MLIR. This method will first check 
        for call operations that are `ast.Name` nodes in the tree (the name of a function to call). 
        It will handle the Python `range(start, stop, step)` function by creating an array of 
        integers to loop through via an invariant CC loop operation. Subsequent users of the 
        `range()` result can iterate through the elements of the returned `cc.array`. It will handle the 
        Python `enumerate(iterable)` function by constructing another invariant loop that builds up and 
        array of `cc.struct<i64, T>`, representing the counter and the element. 

        It will next handle any quantum operation (optionally with a rotation parameter). 
        Single target operations can be represented that take a single qubit reference,
        multiple single qubits, or a vector of qubits, where the latter two 
        will apply the operation to every qubit in the vector: 

        Valid single qubit operations are `h`, `x`, `y`, `z`, `s`, `t`, `rx`, `ry`, `rz`, `r1`. 

        Measurements `mx`, `my`, `mz` are mapped to corresponding quake operations and the return i1 
        value is added to the value stack. Measurements of single qubit reference and registers of 
        qubits are supported. 

        General calls to previously seen CUDA-Q kernels are supported. By this we mean that 
        an kernel can not be invoked from a kernel unless it was defined before the current kernel.
        Kernels can also be reversed or controlled with `cudaq.adjoint(kernel, ...)` and `cudaq.control(kernel, ...)`.

        Finally, general operation modifiers are supported, specifically `OPERATION.adj` and `OPERATION.ctrl` 
        for adjoint and control synthesis of the operation.  

        ```python
            q, r = cudaq.qubit(), cudaq.qubit()
            qubits = cudaq.qubit(2)

            x(q) # apply x to q
            x(q, r) # apply x to q and r
            x(qubits) # for q in qubits: apply x 
            ry(np.pi, qubits)
        ```
        """
        global globalRegisteredOperations

        self.currentNode = node

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

                    [self.visit(arg) for arg in node.args]
                    values = [self.popValue() for _ in node.args]
                    values.reverse()
                    values = [self.ifPointerThenLoad(v) for v in values]
                    func.CallOp(otherKernel, values)
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
                [self.visit(arg) for arg in node.args]
                if len(self.valueStack) != 1:
                    self.emitFatalError(
                        f"cudaq.dbg.ast.{node.func.attr} call invalid - too many arguments passed.",
                        node)

                self.__insertDbgStmt(self.popValue(), node.func.attr)
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

                [self.visit(arg) for arg in node.args]
                values = [self.popValue() for _ in node.args]
                values.reverse()
                func.CallOp(otherKernel, values)
                return

        if isinstance(node.func, ast.Name):
            # Just visit the arguments, we know the name
            [self.visit(arg) for arg in node.args]

            namedArgs = {}
            for keyword in node.keywords:
                self.visit(keyword.value)
                namedArgs[keyword.arg] = self.popValue()

            self.debug_msg(lambda: f'[(Inline) Visit Name]', node.func)
            if node.func.id == 'len':
                listVal = self.ifPointerThenLoad(self.popValue())
                if cc.StdvecType.isinstance(listVal.type):
                    self.pushValue(
                        cc.StdvecSizeOp(self.getIntegerType(), listVal).result)
                    return
                if quake.VeqType.isinstance(listVal.type):
                    self.pushValue(
                        quake.VeqSizeOp(self.getIntegerType(), listVal).result)
                    return

                self.emitFatalError(
                    "__len__ not supported on variables of this type.", node)

            if node.func.id == 'range':
                startVal, endVal, stepVal, isDecrementing = self.__processRangeLoopIterationBounds(
                    node.args)

                iTy = self.getIntegerType(64)
                zero = arith.ConstantOp(iTy, IntegerAttr.get(iTy, 0))
                one = arith.ConstantOp(iTy, IntegerAttr.get(iTy, 1))

                # The total number of elements in the iterable
                # we are generating should be `N == endVal - startVal`
                actualSize = arith.SubIOp(endVal, startVal).result
                totalSize = math.AbsIOp(actualSize).result

                # If the step is not == 1, then we also have
                # to update the total size for the range iterable
                actualSize = arith.DivSIOp(actualSize,
                                           math.AbsIOp(stepVal).result).result
                totalSize = arith.DivSIOp(totalSize,
                                          math.AbsIOp(stepVal).result).result

                # Create an array of i64 of the total size
                arrTy = cc.ArrayType.get(iTy)
                iterable = cc.AllocaOp(cc.PointerType.get(arrTy),
                                       TypeAttr.get(iTy),
                                       seqSize=totalSize).result

                # Logic here is as follows:
                # We are building an array like this
                # array = [start, start +- step, start +- 2*step, start +- 3*step, ...]
                # So we need to know the start and step (already have them),
                # but we also need to keep track of a counter
                counter = cc.AllocaOp(cc.PointerType.get(iTy),
                                      TypeAttr.get(iTy)).result
                cc.StoreOp(zero, counter)

                def bodyBuilder(iterVar):
                    loadedCounter = cc.LoadOp(counter).result
                    tmp = arith.MulIOp(loadedCounter, stepVal).result
                    arrElementVal = arith.AddIOp(startVal, tmp).result
                    eleAddr = cc.ComputePtrOp(
                        cc.PointerType.get(iTy), iterable, [loadedCounter],
                        DenseI32ArrayAttr.get([kDynamicPtrIndex],
                                              context=self.ctx))
                    cc.StoreOp(arrElementVal, eleAddr)
                    incrementedCounter = arith.AddIOp(loadedCounter, one).result
                    cc.StoreOp(incrementedCounter, counter)

                self.createInvariantForLoop(endVal,
                                            bodyBuilder,
                                            startVal=startVal,
                                            stepVal=stepVal,
                                            isDecrementing=isDecrementing)

                self.pushValue(iterable)
                self.pushValue(actualSize)
                return

            if node.func.id == 'enumerate':
                # We have to have something "iterable" on the stack,
                # could be coming from `range()` or an iterable like `qvector`
                totalSize = None
                iterable = None
                iterEleTy = None
                extractFunctor = None
                if len(self.valueStack) == 1:
                    # `qreg`-like or `stdvec`-like thing thing
                    iterable = self.ifPointerThenLoad(self.popValue())
                    # Create a new iterable, `alloca cc.struct<i64, T>`
                    totalSize = None
                    if quake.VeqType.isinstance(iterable.type):
                        iterEleTy = self.getRefType()
                        totalSize = quake.VeqSizeOp(self.getIntegerType(),
                                                    iterable).result

                        def extractFunctor(idxVal):
                            return quake.ExtractRefOp(iterEleTy,
                                                      iterable,
                                                      -1,
                                                      index=idxVal).result
                    elif cc.StdvecType.isinstance(iterable.type):
                        iterEleTy = cc.StdvecType.getElementType(iterable.type)
                        totalSize = cc.StdvecSizeOp(self.getIntegerType(),
                                                    iterable).result

                        def extractFunctor(idxVal):
                            arrEleTy = cc.ArrayType.get(iterEleTy)
                            elePtrTy = cc.PointerType.get(iterEleTy)
                            arrPtrTy = cc.PointerType.get(arrEleTy)
                            vecPtr = cc.StdvecDataOp(arrPtrTy, iterable).result
                            eleAddr = cc.ComputePtrOp(
                                elePtrTy, vecPtr, [idxVal],
                                DenseI32ArrayAttr.get([kDynamicPtrIndex],
                                                      context=self.ctx)).result
                            return cc.LoadOp(eleAddr).result
                    else:
                        self.emitFatalError(
                            "could not infer enumerate tuple type ({})".format(
                                iterable.type), node)
                else:
                    if len(self.valueStack) != 2:
                        msg = 'Error in AST processing, should have 2 values on the stack for enumerate {}'.format(
                            ast.unparse(node) if hasattr(ast, 'unparse'
                                                        ) else node)
                        self.emitFatalError(msg)

                    totalSize = self.popValue()
                    iterable = self.popValue()
                    arrTy = cc.PointerType.getElementType(iterable.type)
                    iterEleTy = cc.ArrayType.getElementType(arrTy)

                    def localFunc(idxVal):
                        eleAddr = cc.ComputePtrOp(
                            cc.PointerType.get(iterEleTy), iterable, [idxVal],
                            DenseI32ArrayAttr.get([kDynamicPtrIndex],
                                                  context=self.ctx)).result
                        return cc.LoadOp(eleAddr).result

                    extractFunctor = localFunc

                # Enumerate returns a iterable of tuple(i64, T) for type T
                # Allocate an array of struct<i64, T> == tuple (for us)
                structTy = cc.StructType.get([self.getIntegerType(), iterEleTy])
                arrTy = cc.ArrayType.get(structTy)
                enumIterable = cc.AllocaOp(cc.PointerType.get(arrTy),
                                           TypeAttr.get(structTy),
                                           seqSize=totalSize).result

                # Now we need to loop through `enumIterable` and set the elements
                def bodyBuilder(iterVar):
                    # Create the struct
                    element = cc.UndefOp(structTy)
                    # Get the element from the iterable
                    extracted = extractFunctor(iterVar)
                    # Get the pointer to the enumeration iterable so we can set it
                    eleAddr = cc.ComputePtrOp(
                        cc.PointerType.get(structTy), enumIterable, [iterVar],
                        DenseI32ArrayAttr.get([kDynamicPtrIndex],
                                              context=self.ctx))
                    # Set the index value
                    element = cc.InsertValueOp(
                        structTy, element, iterVar,
                        DenseI64ArrayAttr.get([0], context=self.ctx)).result
                    # Set the extracted element value
                    element = cc.InsertValueOp(
                        structTy, element, extracted,
                        DenseI64ArrayAttr.get([1], context=self.ctx)).result
                    cc.StoreOp(element, eleAddr)

                self.createInvariantForLoop(totalSize, bodyBuilder)
                self.pushValue(enumIterable)
                self.pushValue(totalSize)
                return

            if node.func.id == 'complex':
                if len(namedArgs) == 0:
                    imag = self.popValue()
                    real = self.popValue()
                else:
                    imag = namedArgs['imag']
                    real = namedArgs['real']
                imag = self.promoteOperandType(self.getFloatType(), imag)
                real = self.promoteOperandType(self.getFloatType(), real)
                self.pushValue(
                    complex.CreateOp(self.getComplexType(), real, imag).result)
                return

            if node.func.id in ['h', 'x', 'y', 'z', 's', 't']:
                # Here we enable application of the op on all the
                # provided arguments, e.g. `x(qubit)`, `x(qvector)`, `x(q, r)`, etc.
                numValues = len(self.valueStack)
                qubitTargets = [self.popValue() for _ in range(numValues)]
                qubitTargets.reverse()
                self.checkControlAndTargetTypes([], qubitTargets)
                self.__applyQuantumOperation(node.func.id, [], qubitTargets)
                return

            if node.func.id in ['ch', 'cx', 'cy', 'cz', 'cs', 'ct']:
                # These are single target controlled quantum operations
                MAX_ARGS = 2
                numValues = len(self.valueStack)
                if numValues != MAX_ARGS:
                    raise RuntimeError(
                        "invalid number of arguments passed to callable {} ({} vs required {})"
                        .format(node.func.id, len(node.args), MAX_ARGS))
                target = self.popValue()
                control = self.popValue()
                negatedControlQubits = None
                if len(self.controlNegations):
                    negCtrlBool = control in self.controlNegations
                    negatedControlQubits = DenseBoolArrayAttr.get(negCtrlBool)
                    self.controlNegations.clear()
                self.checkControlAndTargetTypes([control], [target])
                # Map `cx` to `XOp`...
                opCtor = getattr(
                    quake, '{}Op'.format(node.func.id.title()[1:].upper()))
                opCtor([], [], [control], [target],
                       negated_qubit_controls=negatedControlQubits)
                return

            if node.func.id in ['rx', 'ry', 'rz', 'r1']:
                numValues = len(self.valueStack)
                if numValues < 2:
                    self.emitFatalError(
                        f'invalid number of arguments ({numValues}) passed to {node.func.id} (requires at least 2 arguments)',
                        node)
                qubitTargets = [self.popValue() for _ in range(numValues - 1)]
                qubitTargets.reverse()
                param = self.popValue()
                if IntegerType.isinstance(param.type):
                    param = arith.SIToFPOp(self.getFloatType(), param).result
                elif not F64Type.isinstance(param.type):
                    self.emitFatalError(
                        'rotational parameter must be a float, or int.', node)
                self.checkControlAndTargetTypes([], qubitTargets)
                self.__applyQuantumOperation(node.func.id, [param],
                                             qubitTargets)
                return

            if node.func.id in ['crx', 'cry', 'crz', 'cr1']:
                ## These are single target, one parameter, controlled quantum operations
                MAX_ARGS = 3
                numValues = len(self.valueStack)
                if numValues != MAX_ARGS:
                    raise RuntimeError(
                        "invalid number of arguments passed to callable {} ({} vs required {})"
                        .format(node.func.id, len(node.args), MAX_ARGS))
                target = self.popValue()
                control = self.popValue()
                self.checkControlAndTargetTypes([control], [target])
                param = self.popValue()
                if IntegerType.isinstance(param.type):
                    param = arith.SIToFPOp(self.getFloatType(), param).result
                elif not F64Type.isinstance(param.type):
                    self.emitFatalError(
                        'rotational parameter must be a float, or int.', node)
                # Map `crx` to `RxOp`...
                opCtor = getattr(
                    quake, '{}Op'.format(node.func.id.title()[1:].capitalize()))
                opCtor([], [param], [control], [target])
                return

            if node.func.id in ['sdg', 'tdg']:
                target = self.popValue()
                self.checkControlAndTargetTypes([], [target])
                # Map `sdg` to `SOp`...
                opCtor = getattr(quake, '{}Op'.format(node.func.id.title()[0]))
                if quake.VeqType.isinstance(target.type):

                    def bodyBuilder(iterVal):
                        q = quake.ExtractRefOp(self.getRefType(),
                                               target,
                                               -1,
                                               index=iterVal).result
                        opCtor([], [], [], [q], is_adj=True)

                    veqSize = quake.VeqSizeOp(self.getIntegerType(),
                                              target).result
                    self.createInvariantForLoop(veqSize, bodyBuilder)
                    return
                elif quake.RefType.isinstance(target.type):
                    opCtor([], [], [], [target], is_adj=True)
                    return
                else:
                    self.emitFatalError(
                        'adj quantum operation on incorrect type {}.'.format(
                            target.type), node)
                return

            if node.func.id in ['mx', 'my', 'mz']:
                registerName = self.currentAssignVariableName
                # If `registerName` is None, then we know that we
                # are not assigning this measure result to anything
                # so we therefore should not push it on the stack
                pushResultToStack = registerName != None or self.walkingReturnNode

                # By default we set the `register_name` for the measurement
                # to the assigned variable name (if there is one). But
                # the use could have manually specified `register_name='something'`
                # check for that here and use it there
                if len(node.keywords) == 1 and hasattr(node.keywords[0], 'arg'):
                    if node.keywords[0].arg == 'register_name':
                        userProvidedRegName = node.keywords[0]
                        if not isinstance(userProvidedRegName.value,
                                          ast.Constant):
                            self.emitFatalError(
                                "measurement register_name keyword must be a constant string literal.",
                                node)
                        self.debug_msg(lambda: f'[(Inline) Visit Constant]',
                                       userProvidedRegName.value)
                        registerName = userProvidedRegName.value.value
                qubits = [self.popValue() for _ in range(len(self.valueStack))]
                self.checkControlAndTargetTypes([], qubits)
                opCtor = getattr(quake, '{}Op'.format(node.func.id.title()))
                i1Ty = self.getIntegerType(1)
                resTy = i1Ty if len(qubits) == 1 and quake.RefType.isinstance(
                    qubits[0].type) else cc.StdvecType.get(i1Ty)
                measTy = quake.MeasureType.get(
                ) if len(qubits) == 1 and quake.RefType.isinstance(
                    qubits[0].type) else cc.StdvecType.get(
                        quake.MeasureType.get())
                label = registerName
                if not label:
                    label = None
                measureResult = opCtor(measTy, [], qubits,
                                       registerName=label).result
                if pushResultToStack:
                    self.pushValue(
                        quake.DiscriminateOp(resTy, measureResult).result)
                return

            if node.func.id == 'swap':
                # should have 1 value on the stack if
                # this is a vanilla Hadamard
                qubitB = self.popValue()
                qubitA = self.popValue()
                self.checkControlAndTargetTypes([], [qubitA, qubitB])
                opCtor = getattr(quake, '{}Op'.format(node.func.id.title()))
                opCtor([], [], [], [qubitA, qubitB])
                return

            if node.func.id == 'reset':
                target = self.popValue()
                self.checkControlAndTargetTypes([], [target])
                if quake.RefType.isinstance(target.type):
                    quake.ResetOp([], target)
                    return
                if quake.VeqType.isinstance(target.type):

                    def bodyBuilder(iterVal):
                        q = quake.ExtractRefOp(
                            self.getRefType(),
                            target,
                            -1,  # `kDynamicIndex`
                            index=iterVal).result
                        quake.ResetOp([], q)

                    veqSize = quake.VeqSizeOp(self.getIntegerType(),
                                              target).result
                    self.createInvariantForLoop(veqSize, bodyBuilder)
                    return
                self.emitFatalError(
                    'reset quantum operation on incorrect type {}.'.format(
                        target.type), node)

            if node.func.id == 'u3':
                # Single target, three parameters `u3(θ,φ,λ)`
                all_args = [
                    self.popValue() for _ in range(len(self.valueStack))
                ]
                if len(all_args) < 4:
                    self.emitFatalError(
                        f'invalid number of arguments ({len(all_args)}) passed to {node.func.id} (requires at least 4 arguments)',
                        node)
                qubitTargets = all_args[:-3]
                qubitTargets.reverse()
                self.checkControlAndTargetTypes([], qubitTargets)
                params = all_args[-3:]
                params.reverse()
                for idx, val in enumerate(params):
                    if IntegerType.isinstance(val.type):
                        params[idx] = arith.SIToFPOp(self.getFloatType(),
                                                     val).result
                    elif not F64Type.isinstance(val.type):
                        self.emitFatalError(
                            'rotational parameter must be a float, or int.',
                            node)
                self.__applyQuantumOperation(node.func.id, params, qubitTargets)
                return

            if node.func.id in globalRegisteredOperations:
                unitary = globalRegisteredOperations[node.func.id]
                numTargets = int(np.log2(np.sqrt(unitary.size)))

                numValues = len(self.valueStack)
                if numValues != numTargets:
                    self.emitFatalError(
                        f'invalid number of arguments ({numValues}) passed to {node.func.id} (requires {numTargets} arguments)',
                        node)

                targets = [self.popValue() for _ in range(numTargets)]
                targets.reverse()

                for i, t in enumerate(targets):
                    if not quake.RefType.isinstance(t.type):
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
                    targets=targets,
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
                if len(fType.inputs) != len(node.args):
                    self.emitFatalError(
                        "invalid number of arguments passed to callable {} ({} vs required {})"
                        .format(node.func.id, len(node.args),
                                len(fType.inputs)), node)

                values = [self.popValue() for _ in node.args]
                values.reverse()
                values = [self.ifPointerThenLoad(v) for v in values]
                if len(fType.results) == 0:
                    func.CallOp(otherKernel, values)
                else:
                    result = func.CallOp(otherKernel, values).result
                    self.pushValue(result)
                return

            elif node.func.id in self.symbolTable:
                val = self.symbolTable[node.func.id]
                if cc.CallableType.isinstance(val.type):
                    numVals = len(self.valueStack)
                    values = [self.popValue() for _ in range(numVals)]
                    callableTy = cc.CallableType.getFunctionType(val.type)
                    if not self.argumentsValidForFunction(values, callableTy):
                        self.emitFatalError(
                            "invalid argument types for callable function ({} vs {})"
                            .format([v.type for v in values], callableTy), node)

                    callable = cc.CallableFuncOp(callableTy, val).result
                    func.CallIndirectOp([], callable, values)
                    return
                else:
                    self.emitFatalError(
                        f"`{node.func.id}` object is not callable, found symbol of type {val.type}",
                        node)

            elif node.func.id == 'exp_pauli':
                pauliWord = self.popValue()
                qubits = self.popValue()
                self.checkControlAndTargetTypes([], [qubits])
                theta = self.popValue()
                if IntegerType.isinstance(theta.type):
                    theta = arith.SIToFPOp(self.getFloatType(), theta).result
                quake.ExpPauliOp([], [theta], [], [qubits], pauli=pauliWord)
                return

            elif node.func.id == 'int':
                # cast operation
                value = self.popValue()
                casted = self.promoteOperandType(IntegerType.get_signless(64),
                                                 value)
                self.pushValue(casted)
                if not IntegerType.isinstance(casted.type):
                    self.emitFatalError(
                        f'Invalid cast to integer: {value.type}', node)

            elif node.func.id == 'list':
                if len(self.valueStack) == 2:
                    maybeIterableSize = self.popValue()
                    maybeIterable = self.popValue()

                    # Make sure that we have a list + size
                    if IntegerType.isinstance(maybeIterableSize.type):
                        if cc.PointerType.isinstance(maybeIterable.type):
                            ptrEleTy = cc.PointerType.getElementType(
                                maybeIterable.type)
                            if cc.ArrayType.isinstance(ptrEleTy):
                                # We're good, just pass this back through.
                                self.pushValue(maybeIterable)
                                self.pushValue(maybeIterableSize)
                                return
                if len(self.valueStack) == 1:
                    arrayTy = self.valueStack[0].type
                    if cc.PointerType.isinstance(arrayTy):
                        arrayTy = cc.PointerType.getElementType(arrayTy)
                    if cc.StdvecType.isinstance(arrayTy):
                        return
                    if cc.ArrayType.isinstance(arrayTy):
                        return

                self.emitFatalError('Invalid list() cast requested.', node)

            elif node.func.id in ['print_i64', 'print_f64']:
                self.__insertDbgStmt(self.popValue(), node.func.id)
                return

            elif node.func.id in globalRegisteredTypes.classes:
                # Handle User-Custom Struct Constructor
                cls, annotations = globalRegisteredTypes.getClassAttributes(
                    node.func.id)

                if '__slots__' not in cls.__dict__:
                    self.emitWarning(
                        f"Adding new fields in data classes is not yet supported. The dataclass must be declared with @dataclass(slots=True) or @dataclasses.dataclass(slots=True).",
                        node)

                # Alloca the struct
                structTys = [
                    mlirTypeFromPyType(v, self.ctx)
                    for _, v in annotations.items()
                ]
                # Ensure we don't use hybrid data types
                numQuantumMemberTys = sum(
                    [1 if self.isQuantumType(ty) else 0 for ty in structTys])
                if numQuantumMemberTys != 0:  # we have quantum member types
                    if numQuantumMemberTys != len(structTys):
                        self.emitFatalError(
                            f'hybrid quantum-classical data types not allowed in kernel code',
                            node)

                isStruq = not (not structTys)
                for fieldTy in structTys:
                    if not self.isQuantumType(fieldTy):
                        isStruq = False
                if isStruq:
                    structTy = quake.StruqType.getNamed(node.func.id, structTys)
                    # Disallow recursive quantum struct types.
                    for fieldTy in structTys:
                        if self.isQuantumStructType(fieldTy):
                            self.emitFatalError(
                                'recursive quantum struct types not allowed.',
                                node)
                else:
                    structTy = cc.StructType.getNamed(node.func.id, structTys)
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

                nArgs = len(self.valueStack)
                ctorArgs = [self.popValue() for _ in range(nArgs)]
                ctorArgs.reverse()

                if isStruq:
                    # If we have a quantum struct. We cannot allocate classical
                    # memory and load / store quantum type values to that memory
                    # space, so use `quake.MakeStruqOp`.
                    self.pushValue(quake.MakeStruqOp(structTy, ctorArgs).result)
                    return

                stackSlot = cc.AllocaOp(cc.PointerType.get(structTy),
                                        TypeAttr.get(structTy)).result

                if len(ctorArgs) != len(structTys):
                    self.emitFatalError(
                        f'constructor struct type {node.func.id} requires '
                        f'{len(structTys)} arguments, but was given {len(ctorArgs)}.',
                        node)

                # loop over each type and `compute_ptr` / store

                for i, ty in enumerate(structTys):
                    eleAddr = cc.ComputePtrOp(
                        cc.PointerType.get(ty), stackSlot, [],
                        DenseI32ArrayAttr.get([i], context=self.ctx)).result
                    cc.StoreOp(ctorArgs[i], eleAddr)
                self.pushValue(stackSlot)
                return

            else:
                self.emitFatalError(
                    "unhandled function call - {}, known kernels are {}".format(
                        node.func.id, globalKernelRegistry.keys()), node)

        elif isinstance(node.func, ast.Attribute):
            self.debug_msg(lambda: f'[(Inline) Visit Attribute]', node.func)
            self.debug_msg(lambda: f'[(Inline) Visit Name]', node.func.value)
            if node.func.value.id in ['numpy', 'np']:
                [self.visit(arg) for arg in node.args]

                namedArgs = {}
                for keyword in node.keywords:
                    self.visit(keyword.value)
                    namedArgs[keyword.arg] = self.popValue()

                value = self.popValue()

                if node.func.attr == 'array':
                    # `np.array(vec, <dtype = ty>)`
                    arrayType = value.type
                    if cc.PointerType.isinstance(value.type):
                        arrayType = cc.PointerType.getElementType(value.type)

                    if cc.StdvecType.isinstance(arrayType):
                        eleTy = cc.StdvecType.getElementType(arrayType)
                        dTy = eleTy
                        if len(namedArgs) > 0:
                            dTy = namedArgs['dtype']

                        # Convert the vector to the provided data type if needed.
                        self.pushValue(
                            self.__copyVectorAndCastElements(value, dTy))
                        return

                    raise self.emitFatalError(
                        f"unexpected numpy array initializer type: {value.type}",
                        node)

                value = self.ifPointerThenLoad(value)

                if node.func.attr in ['complex128', 'complex64']:
                    if node.func.attr == 'complex128':
                        ty = self.getComplexType()
                        eleTy = self.getFloatType()
                    if node.func.attr == 'complex64':
                        ty = self.getComplexType(width=32)
                        eleTy = self.getFloatType(width=32)

                    value = self.promoteOperandType(ty, value)
                    if (ty == value.type):
                        self.pushValue(value)
                        return

                    real = complex.ReOp(value).result
                    imag = complex.ImOp(value).result
                    real = self.promoteOperandType(eleTy, real)
                    imag = self.promoteOperandType(eleTy, imag)

                    self.pushValue(complex.CreateOp(ty, real, imag).result)
                    return

                if node.func.attr in ['float64', 'float32']:
                    if node.func.attr == 'float64':
                        ty = self.getFloatType()
                    if node.func.attr == 'float32':
                        ty = self.getFloatType(width=32)

                    value = self.promoteOperandType(ty, value)
                    self.pushValue(value)
                    return

                # Promote argument's types for `numpy.func` calls to match python's semantics
                if node.func.attr in ['sin', 'cos', 'sqrt', 'ceil', 'exp']:
                    if ComplexType.isinstance(value.type):
                        value = self.promoteOperandType(self.getComplexType(),
                                                        value)
                    if IntegerType.isinstance(value.type):
                        value = self.promoteOperandType(self.getFloatType(),
                                                        value)

                if node.func.attr == 'cos':
                    if ComplexType.isinstance(value.type):
                        self.pushValue(complex.CosOp(value).result)
                        return
                    self.pushValue(math.CosOp(value).result)
                    return
                if node.func.attr == 'sin':
                    if ComplexType.isinstance(value.type):
                        self.pushValue(complex.SinOp(value).result)
                        return
                    self.pushValue(math.SinOp(value).result)
                    return
                if node.func.attr == 'sqrt':
                    if ComplexType.isinstance(value.type):
                        self.pushValue(complex.SqrtOp(value).result)
                        return
                    self.pushValue(math.SqrtOp(value).result)
                    return
                if node.func.attr == 'exp':
                    if ComplexType.isinstance(value.type):
                        # Note: using `complex.ExpOp` results in a
                        # "can't legalize `complex.exp`" error.
                        # Using Euler's' formula instead:
                        #
                        # "e^(x+i*y) = (e^x) * (cos(y)+i*sin(y))"
                        complexType = ComplexType(value.type)
                        floatType = complexType.element_type
                        real = complex.ReOp(value).result
                        imag = complex.ImOp(value).result
                        left = self.promoteOperandType(complexType,
                                                       math.ExpOp(real).result)
                        re2 = math.CosOp(imag).result
                        im2 = math.SinOp(imag).result
                        right = complex.CreateOp(ComplexType.get(floatType),
                                                 re2, im2).result
                        res = complex.MulOp(left, right).result
                        self.pushValue(res)
                        return
                    self.pushValue(math.ExpOp(value).result)
                    return
                if node.func.attr == 'ceil':
                    if ComplexType.isinstance(value.type):
                        self.emitFatalError(
                            f"numpy call ({node.func.attr}) is not supported for complex numbers",
                            node)
                        return
                    self.pushValue(math.CeilOp(value).result)
                    return

                self.emitFatalError(
                    f"unsupported NumPy call ({node.func.attr})", node)

            self.generic_visit(node)

            if node.func.value.id == 'cudaq':
                if node.func.attr == 'complex':
                    self.pushValue(self.simulationDType())
                    return

                if node.func.attr == 'amplitudes':
                    value = self.popValue()
                    arrayType = value.type
                    if cc.PointerType.isinstance(value.type):
                        arrayType = cc.PointerType.getElementType(value.type)
                    if cc.StdvecType.isinstance(arrayType):
                        self.pushValue(value)
                        return

                    self.emitFatalError(
                        f"unsupported amplitudes argument type: {value.type}",
                        node)

                if node.func.attr == 'qvector':
                    if len(self.valueStack) == 0:
                        self.emitFatalError(
                            'qvector does not have default constructor. Init from size or existing state.',
                            node)

                    valueOrPtr = self.popValue()
                    initializerTy = valueOrPtr.type

                    if cc.PointerType.isinstance(initializerTy):
                        initializerTy = cc.PointerType.getElementType(
                            initializerTy)

                    if (IntegerType.isinstance(initializerTy)):
                        # handle `cudaq.qvector(n)`
                        value = self.ifPointerThenLoad(valueOrPtr)
                        ty = self.getVeqType()
                        qubits = quake.AllocaOp(ty, size=value).result
                        self.pushValue(qubits)
                        return
                    if cc.StdvecType.isinstance(initializerTy):
                        # handle `cudaq.qvector(initState)`

                        # Validate the length in case of a constant initializer:
                        # `cudaq.qvector([1., 0., ...])`
                        # `cudaq.qvector(np.array([1., 0., ...]))`
                        value = self.ifPointerThenLoad(valueOrPtr)
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

                        eleTy = cc.StdvecType.getElementType(value.type)
                        size = cc.StdvecSizeOp(self.getIntegerType(),
                                               value).result
                        numQubits = math.CountTrailingZerosOp(size).result

                        # TODO: Dynamically check if number of qubits is power of 2
                        # and if the state is normalized

                        ptrTy = cc.PointerType.get(eleTy)
                        arrTy = cc.ArrayType.get(eleTy)
                        ptrArrTy = cc.PointerType.get(arrTy)
                        veqTy = quake.VeqType.get()

                        qubits = quake.AllocaOp(veqTy, size=numQubits).result
                        data = cc.StdvecDataOp(ptrArrTy, value).result
                        init = quake.InitializeStateOp(veqTy, qubits,
                                                       data).result
                        self.pushValue(init)
                        return

                    if cc.StateType.isinstance(initializerTy):
                        # handle `cudaq.qvector(state)`
                        statePtr = self.ifNotPointerThenStore(valueOrPtr)

                        i64Ty = self.getIntegerType()
                        numQubits = quake.GetNumberOfQubitsOp(i64Ty,
                                                              statePtr).result

                        veqTy = quake.VeqType.get()
                        qubits = quake.AllocaOp(veqTy, size=numQubits).result
                        init = quake.InitializeStateOp(veqTy, qubits,
                                                       statePtr).result

                        self.pushValue(init)
                        return

                    self.emitFatalError(
                        f"unsupported qvector argument type: {value.type}",
                        node)
                    return

                if node.func.attr == "qubit":
                    if len(self.valueStack) == 1 and IntegerType.isinstance(
                            self.valueStack[0].type):
                        self.emitFatalError(
                            'cudaq.qubit() constructor does not take any arguments. To construct a vector of qubits, use `cudaq.qvector(N)`.'
                        )
                    self.pushValue(quake.AllocaOp(self.getRefType()).result)
                    return

                if node.func.attr == 'adjoint':
                    # Handle cudaq.adjoint(kernel, ...)
                    otherFuncName = node.args[0].id
                    if otherFuncName in self.symbolTable:
                        # This is a callable block argument
                        values = [
                            self.popValue()
                            for _ in range(len(self.valueStack) - 2)
                        ]
                        quake.ApplyOp([], [self.popValue()], [], values)
                        return

                    if otherFuncName not in globalKernelRegistry:
                        self.emitFatalError(
                            f"{otherFuncName} is not a known quantum kernel (was it annotated?)."
                        )

                    values = [
                        self.popValue() for _ in range(len(self.valueStack))
                    ]
                    values.reverse()
                    if len(values) != len(
                            globalKernelRegistry[otherFuncName].arguments):
                        self.emitFatalError(
                            f"incorrect number of runtime arguments for cudaq.control({otherFuncName},..) call.",
                            node)
                    quake.ApplyOp([], [], [],
                                  values,
                                  callee=FlatSymbolRefAttr.get(nvqppPrefix +
                                                               otherFuncName),
                                  is_adj=True)
                    return

                if node.func.attr == 'control':
                    # Handle cudaq.control(kernel, ...)
                    otherFuncName = node.args[0].id
                    if otherFuncName in self.symbolTable:
                        # This is a callable argument
                        values = [
                            self.popValue()
                            for _ in range(len(self.valueStack) - 2)
                        ]
                        controls = self.popValue()
                        a = quake.ApplyOp([], [self.popValue()], [controls],
                                          values)
                        return

                    if otherFuncName not in globalKernelRegistry:
                        self.emitFatalError(
                            f"{otherFuncName} is not a known quantum kernel (was it annotated?).",
                            node)
                    values = [
                        self.popValue()
                        for _ in range(len(self.valueStack) - 1)
                    ]
                    values.reverse()
                    if len(values) != len(
                            globalKernelRegistry[otherFuncName].arguments):
                        self.emitFatalError(
                            f"incorrect number of runtime arguments for cudaq.control({otherFuncName},..) call.",
                            node)
                    controls = self.popValue()
                    self.checkControlAndTargetTypes([controls], [])
                    quake.ApplyOp([], [], [controls],
                                  values,
                                  callee=FlatSymbolRefAttr.get(nvqppPrefix +
                                                               otherFuncName))
                    return

                if node.func.attr == 'apply_noise':
                    # Pop off all the arguments we need
                    values = [
                        self.popValue() for _ in range(len(self.valueStack))
                    ]
                    # They are in reverse order
                    values.reverse()
                    # First one should be the number of Kraus channel parameters
                    numParamsVal = values[0]
                    # Shrink the arguments down
                    values = values[1:]

                    # Need to get the number of parameters as an integer
                    concreteIntAttr = IntegerAttr(
                        numParamsVal.owner.attributes['value'])
                    numParams = concreteIntAttr.value

                    # Next Value is our generated key for the channel
                    # Get it and shrink the list
                    key = values[0]
                    values = values[1:]

                    # Now we know the next `numParams` arguments are
                    # our Kraus channel parameters
                    params = values[:numParams]
                    for i, p in enumerate(params):
                        # If we have a F64 value, we want to
                        # store it to a pointer
                        if F64Type.isinstance(p.type):
                            alloca = cc.AllocaOp(cc.PointerType.get(p.type),
                                                 TypeAttr.get(p.type)).result
                            cc.StoreOp(p, alloca)
                            params[i] = alloca

                    # The remaining arguments are the qubits
                    asVeq = quake.ConcatOp(self.getVeqType(),
                                           values[numParams:]).result
                    quake.ApplyNoiseOp(params, [asVeq], key=key)
                    return

                if node.func.attr == 'compute_action':
                    # There can only be 2 arguments here.
                    action = None
                    compute = None
                    actionArg = node.args[1]
                    if isinstance(actionArg, ast.Name):
                        actionName = actionArg.id
                        if actionName in self.symbolTable:
                            action = self.symbolTable[actionName]
                        else:
                            self.emitFatalError(
                                "could not find action lambda / function in the symbol table.",
                                node)
                    else:
                        action = self.popValue()

                    computeArg = node.args[0]
                    if isinstance(computeArg, ast.Name):
                        computeName = computeArg.id
                        if computeName in self.symbolTable:
                            compute = self.symbolTable[computeName]
                        else:
                            self.emitFatalError(
                                "could not find compute lambda / function in the symbol table.",
                                node)
                    else:
                        compute = self.popValue()

                    quake.ComputeActionOp(compute, action)
                    return

                self.emitFatalError(
                    f'Invalid function or class type requested from the cudaq module ({node.func.attr})',
                    node)

            if node.func.value.id in self.symbolTable:
                # Method call on one of our variables
                var = self.symbolTable[node.func.value.id]
                if quake.VeqType.isinstance(var.type):
                    if node.func.attr == 'size':
                        # Handled already in the Attribute visit
                        return

                    # `qreg` or `qview` method call
                    if node.func.attr == 'back':
                        qrSize = quake.VeqSizeOp(self.getIntegerType(),
                                                 var).result
                        one = self.getConstantInt(1)
                        endOff = arith.SubIOp(qrSize, one)
                        if len(node.args):
                            # extract the `subveq`
                            startOff = arith.SubIOp(qrSize, self.popValue())
                            dyna = IntegerAttr.get(self.getIntegerType(), -1)
                            self.pushValue(
                                quake.SubVeqOp(self.getVeqType(),
                                               var,
                                               dyna,
                                               dyna,
                                               lower=startOff,
                                               upper=endOff).result)
                        else:
                            # extract the qubit...
                            self.pushValue(
                                quake.ExtractRefOp(self.getRefType(),
                                                   var,
                                                   -1,
                                                   index=endOff).result)
                        return
                    if node.func.attr == 'front':
                        zero = self.getConstantInt(0)
                        if len(node.args):
                            # extract the `subveq`
                            qrSize = self.popValue()
                            one = self.getConstantInt(1)
                            offset = arith.SubIOp(qrSize, one)
                            dyna = IntegerAttr.get(self.getIntegerType(), -1)
                            self.pushValue(
                                quake.SubVeqOp(self.getVeqType(),
                                               var,
                                               dyna,
                                               dyna,
                                               lower=zero,
                                               upper=offset).result)
                        else:
                            # extract the qubit...
                            self.pushValue(
                                quake.ExtractRefOp(self.getRefType(),
                                                   var,
                                                   -1,
                                                   index=zero).result)
                        return

            def maybeProposeOpAttrFix(opName, attrName):
                """
                Check the quantum operation attribute name and 
                propose a smart fix message if possible. For example, 
                if we have `x.control(...)` then remind the programmer the 
                correct attribute is `x.ctrl(...)`.
                """
                # TODO Add more possibilities in the future...
                if attrName in ['control'
                               ] or 'control' in attrName or 'ctrl' in attrName:
                    return f'Did you mean {opName}.ctrl(...)?'

                if attrName in ['adjoint'
                               ] or 'adjoint' in attrName or 'adj' in attrName:
                    return f'Did you mean {opName}.adj(...)?'

                return ''

            # We have a `func_name.ctrl`
            if node.func.value.id in ['h', 'x', 'y', 'z', 's', 't']:
                if node.func.attr == 'ctrl':
                    target = self.popValue()
                    # Should be number of arguments minus one for the controls
                    controls = [
                        self.popValue() for i in range(len(node.args) - 1)
                    ]
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

                    opCtor = getattr(quake,
                                     '{}Op'.format(node.func.value.id.title()))
                    self.checkControlAndTargetTypes(controls, [target])
                    opCtor([], [],
                           controls, [target],
                           negated_qubit_controls=negatedControlQubits)
                    return
                if node.func.attr == 'adj':
                    target = self.popValue()
                    self.checkControlAndTargetTypes([], [target])
                    opCtor = getattr(quake,
                                     '{}Op'.format(node.func.value.id.title()))
                    if quake.VeqType.isinstance(target.type):

                        def bodyBuilder(iterVal):
                            q = quake.ExtractRefOp(self.getRefType(),
                                                   target,
                                                   -1,
                                                   index=iterVal).result
                            opCtor([], [], [], [q], is_adj=True)

                        veqSize = quake.VeqSizeOp(self.getIntegerType(),
                                                  target).result
                        self.createInvariantForLoop(veqSize, bodyBuilder)
                        return
                    elif quake.RefType.isinstance(target.type):
                        opCtor([], [], [], [target], is_adj=True)
                        return
                    else:
                        self.emitFatalError(
                            'adj quantum operation on incorrect type {}.'.
                            format(target.type), node)

                self.emitFatalError(
                    f'Unknown attribute on quantum operation {node.func.value.id} ({node.func.attr}). {maybeProposeOpAttrFix(node.func.value.id, node.func.attr)}'
                )

            # We have a `func_name.ctrl`
            if node.func.value.id == 'swap' and node.func.attr == 'ctrl':
                targetB = self.popValue()
                targetA = self.popValue()
                controls = [
                    self.popValue() for i in range(len(self.valueStack))
                ]
                if not controls:
                    self.emitFatalError(
                        'controlled operation requested without any control argument(s).',
                        node)
                opCtor = getattr(quake,
                                 '{}Op'.format(node.func.value.id.title()))
                self.checkControlAndTargetTypes(controls, [targetA, targetB])
                opCtor([], [], controls, [targetA, targetB])
                return

            if node.func.value.id in ['rx', 'ry', 'rz', 'r1']:
                if node.func.attr == 'ctrl':
                    target = self.popValue()
                    controls = [
                        self.popValue() for i in range(len(self.valueStack))
                    ]
                    param = controls[-1]
                    controls = controls[:-1]
                    if not controls:
                        self.emitFatalError(
                            'controlled operation requested without any control argument(s).',
                            node)
                    if IntegerType.isinstance(param.type):
                        param = arith.SIToFPOp(self.getFloatType(),
                                               param).result
                    elif not F64Type.isinstance(param.type):
                        self.emitFatalError(
                            'rotational parameter must be a float, or int.',
                            node)
                    opCtor = getattr(quake,
                                     '{}Op'.format(node.func.value.id.title()))
                    self.checkControlAndTargetTypes(controls, [target])
                    opCtor([], [param], controls, [target])
                    return

                if node.func.attr == 'adj':
                    target = self.popValue()
                    param = self.popValue()
                    if IntegerType.isinstance(param.type):
                        param = arith.SIToFPOp(self.getFloatType(),
                                               param).result
                    elif not F64Type.isinstance(param.type):
                        self.emitFatalError(
                            'rotational parameter must be a float, or int.',
                            node)
                    opCtor = getattr(quake,
                                     '{}Op'.format(node.func.value.id.title()))
                    self.checkControlAndTargetTypes([], [target])
                    if quake.VeqType.isinstance(target.type):

                        def bodyBuilder(iterVal):
                            q = quake.ExtractRefOp(self.getRefType(),
                                                   target,
                                                   -1,
                                                   index=iterVal).result
                            opCtor([], [param], [], [q], is_adj=True)

                        veqSize = quake.VeqSizeOp(self.getIntegerType(),
                                                  target).result
                        self.createInvariantForLoop(veqSize, bodyBuilder)
                        return
                    elif quake.RefType.isinstance(target.type):
                        opCtor([], [param], [], [target], is_adj=True)
                        return
                    else:
                        self.emitFatalError(
                            'adj quantum operation on incorrect type {}.'.
                            format(target.type), node)

                self.emitFatalError(
                    f'Unknown attribute on quantum operation {node.func.value.id} ({node.func.attr}). {maybeProposeOpAttrFix(node.func.value.id, node.func.attr)}'
                )

            if node.func.value.id == 'u3':
                numValues = len(self.valueStack)
                target = self.popValue()
                other_args = [self.popValue() for _ in range(numValues - 1)]

                opCtor = getattr(quake,
                                 '{}Op'.format(node.func.value.id.title()))

                if node.func.attr == 'ctrl':
                    controls = other_args[:-3]
                    if not controls:
                        self.emitFatalError(
                            'controlled operation requested without any control argument(s).',
                            node)
                    params = other_args[-3:]
                    params.reverse()
                    for idx, val in enumerate(params):
                        if IntegerType.isinstance(val.type):
                            params[idx] = arith.SIToFPOp(
                                self.getFloatType(), val).result
                        elif not F64Type.isinstance(val.type):
                            self.emitFatalError(
                                'rotational parameter must be a float, or int.',
                                node)
                    negatedControlQubits = None
                    if len(self.controlNegations):
                        negCtrlBools = [None] * len(controls)
                        for i, c in enumerate(controls):
                            negCtrlBools[i] = c in self.controlNegations
                        negatedControlQubits = DenseBoolArrayAttr.get(
                            negCtrlBools)
                        self.controlNegations.clear()

                    self.checkControlAndTargetTypes(controls, [target])
                    opCtor([],
                           params,
                           controls, [target],
                           negated_qubit_controls=negatedControlQubits)
                    return

                if node.func.attr == 'adj':
                    params = other_args
                    params.reverse()
                    for idx, val in enumerate(params):
                        if IntegerType.isinstance(val.type):
                            params[idx] = arith.SIToFPOp(
                                self.getFloatType(), val).result
                        elif not F64Type.isinstance(val.type):
                            self.emitFatalError(
                                'rotational parameter must be a float, or int.',
                                node)

                    self.checkControlAndTargetTypes([], [target])
                    if quake.VeqType.isinstance(target.type):

                        def bodyBuilder(iterVal):
                            q = quake.ExtractRefOp(self.getRefType(),
                                                   target,
                                                   -1,
                                                   index=iterVal).result
                            opCtor([], params, [], [q], is_adj=True)

                        veqSize = quake.VeqSizeOp(self.getIntegerType(),
                                                  target).result
                        self.createInvariantForLoop(veqSize, bodyBuilder)
                        return
                    elif quake.RefType.isinstance(target.type):
                        opCtor([], params, [], [target], is_adj=True)
                        return
                    else:
                        self.emitFatalError(
                            'adj quantum operation on incorrect type {}.'.
                            format(target.type), node)

            # custom `ctrl` and `adj`
            if node.func.value.id in globalRegisteredOperations:
                if not node.func.attr == 'ctrl' and not node.func.attr == 'adj':
                    self.emitFatalError(
                        f'Unknown attribute on custom operation {node.func.value.id} ({node.func.attr}).'
                    )

                unitary = globalRegisteredOperations[node.func.value.id]
                numTargets = int(np.log2(np.sqrt(unitary.size)))
                numValues = len(self.valueStack)
                targets = [self.popValue() for _ in range(numTargets)]
                targets.reverse()

                for i, t in enumerate(targets):
                    if not quake.RefType.isinstance(t.type):
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
                    controls = [
                        self.popValue() for _ in range(numValues - numTargets)
                    ]
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
        """
        This method currently supports lowering simple list comprehensions 
        to the MLIR. By simple, we mean expressions like 
        `[expr(iter) for iter in iterable]` or 
        `myList = [exprThatReturns(iter) for iter in iterable]`.
        """
        self.currentNode = node

        if len(node.generators) > 1:
            self.emitFatalError(
                "CUDA-Q only supports single generators for list comprehension.",
                node)

        if not isinstance(node.generators[0].target, ast.Name):
            self.emitFatalError(
                "only support named targets in list comprehension", node)

        # Handle the case of `[qOp(q) for q in veq]`
        if isinstance(
                node.generators[0].iter,
                ast.Name) and node.generators[0].iter.id in self.symbolTable:
            self.debug_msg(lambda: f'[(Inline) Visit Name]',
                           node.generators[0].iter)
            if quake.VeqType.isinstance(
                    self.symbolTable[node.generators[0].iter.id].type):
                # now we know we have `[expr(r) for r in iterable]`
                # reuse what we do in `visit_For()`
                forNode = ast.For()
                forNode.iter = node.generators[0].iter
                forNode.target = node.generators[0].target
                forNode.body = [node.elt]
                forNode.orelse = []
                self.visit_For(forNode)
                return

        # General case of
        # `listVar = [expr(i) for i in iterable]`
        # Need to think of this as
        # `listVar = stdvec(iterable.size)`
        # `for i, r in enumerate(listVar):`
        # `   listVar[i] = expr(r)`

        # Let's handle the following `listVar` types
        # `   %9 = cc.alloca !cc.array<!cc.stdvec<T> x 2> -> ptr<array<stdvec<T> x N>`
        # or
        # `    %3 = cc.alloca T[%2 : i64] -> ptr<array<T>>`
        self.visit(node.generators[0].iter)

        if len(self.valueStack) != 2:
            self.emitFatalError(
                "Invalid CUDA-Q list creation via list comprehension - valid iterable not detected.",
                node)

        iterableSize = self.popValue()
        iterable = self.popValue()
        # We require that the iterable is a pointer to an `array<T>`
        # FIXME revisit this
        if not cc.PointerType.isinstance(iterable.type):
            self.emitFatalError(
                "CUDA-Q only considers general list comprehension on iterables from range(...)",
                node)

        arrayTy = cc.PointerType.getElementType(iterable.type)
        if not cc.ArrayType.isinstance(arrayTy):
            self.emitFatalError(
                "CUDA-Q only considers general list comprehension on iterables from range(...)",
                node)

        arrayEleTy = cc.ArrayType.getElementType(arrayTy)

        # If `node.elt` is `ast.List`, then we want to allocate a
        # `cc.array<cc.stdvec<i64> x len(node.elt.elts)>`
        # otherwise we just want to allocate an `array<T>`
        listValue = None
        listComputePtrTy = arrayEleTy
        if not isinstance(node.elt, ast.List):
            listValue = cc.AllocaOp(cc.PointerType.get(arrayTy),
                                    TypeAttr.get(arrayEleTy),
                                    seqSize=iterableSize).result
        else:
            listComputePtrTy = cc.StdvecType.get(arrayEleTy)
            arrOfStdvecTy = cc.ArrayType.get(listComputePtrTy)
            listValue = cc.AllocaOp(cc.PointerType.get(arrOfStdvecTy),
                                    TypeAttr.get(listComputePtrTy),
                                    seqSize=iterableSize).result

        def bodyBuilder(iterVar):
            self.symbolTable.pushScope()
            eleAddr = cc.ComputePtrOp(
                cc.PointerType.get(arrayEleTy), iterable, [iterVar],
                DenseI32ArrayAttr.get([kDynamicPtrIndex], context=self.ctx))
            loadedEle = cc.LoadOp(eleAddr).result
            self.symbolTable[node.generators[0].target.id] = loadedEle
            self.visit(node.elt)
            result = self.popValue()
            listValueAddr = cc.ComputePtrOp(
                cc.PointerType.get(listComputePtrTy), listValue, [iterVar],
                DenseI32ArrayAttr.get([kDynamicPtrIndex], context=self.ctx))
            cc.StoreOp(result, listValueAddr)
            self.symbolTable.popScope()

        self.createInvariantForLoop(iterableSize, bodyBuilder)
        self.pushValue(
            cc.StdvecInitOp(cc.StdvecType.get(listComputePtrTy),
                            listValue,
                            length=iterableSize).result)
        return

    def visit_List(self, node):
        """
        This method will visit the `ast.List` node and represent lists of 
        quantum typed values as a concatenated `quake.ConcatOp` producing a 
        single `veq` instances. 
        """
        self.generic_visit(node)

        self.currentNode = node

        listElementValues = [self.popValue() for _ in range(len(node.elts))]
        listElementValues.reverse()
        valueTys = [
            quake.VeqType.isinstance(v.type) or quake.RefType.isinstance(v.type)
            for v in listElementValues
        ]
        if False not in valueTys:
            # this is a list of quantum types,
            # concatenate them into a `veq`
            if len(listElementValues) == 1:
                self.pushValue(listElementValues[0])
            else:
                self.pushValue(
                    quake.ConcatOp(self.getVeqType(), listElementValues).result)
            return

        # We do not store lists of pointers
        listElementValues = [
            cc.LoadOp(ele).result
            if cc.PointerType.isinstance(ele.type) else ele
            for ele in listElementValues
        ]

        # not a list of quantum types
        # Get the first element
        firstTy = listElementValues[0].type
        # Is this a list of homogenous types?
        isHomogeneous = False not in [
            firstTy == v.type for v in listElementValues
        ]

        # If not, see if the types are arithmetic and if so, find
        # the superior type and convert all to it.
        if not isHomogeneous:
            # this list does not contain all the same types of elements
            # check if they are at least all arithmetic
            isArithmetic = False not in [
                self.isArithmeticType(v.type) for v in listElementValues
            ]
            if isArithmetic:
                # Find the "superior type" (int < float < complex)
                superiorType = self.getIntegerType()
                for t in [v.type for v in listElementValues]:
                    if F32Type.isinstance(t):
                        superiorType = t
                    if F64Type.isinstance(t):
                        superiorType = t
                    if ComplexType.isinstance(t):
                        superiorType = t
                        break  # can do no better

                # Convert the values to the superior arithmetic type
                listElementValues = self.convertArithmeticToSuperiorType(
                    listElementValues, superiorType)

                # The list is now homogeneous
                isHomogeneous = True

        # If we are still not homogenous
        if not isHomogeneous:
            self.emitFatalError(
                "non-homogenous list not allowed - must all be same type: {}".
                format([v.type for v in listElementValues]), node)

        # Turn this List into a StdVec<T>
        self.pushValue(
            self.__createStdvecWithKnownValues(len(node.elts),
                                               listElementValues))

    def visit_Constant(self, node):
        """
        Convert constant values in the code to constant values in the MLIR. 
        """
        self.currentNode = node
        if isinstance(node.value, bool):
            boolValue = 0 if node.value == 0 else 1
            self.pushValue(self.getConstantInt(boolValue, 1))
            return

        if isinstance(node.value, int):
            self.pushValue(self.getConstantInt(node.value))
            return

        if isinstance(node.value, float):
            self.pushValue(self.getConstantFloat(node.value))
            return

        if isinstance(node.value, str):
            # Do not process the function doc string
            if self.docstring != None:
                if node.value.strip() == self.docstring.strip():
                    return

            strLitTy = cc.PointerType.get(
                cc.ArrayType.get(self.getIntegerType(8),
                                 len(node.value) + 1))
            self.pushValue(
                cc.CreateStringLiteralOp(strLitTy,
                                         StringAttr.get(node.value)).result)
            return

        if isinstance(node.value, type(1j)):
            self.pushValue(
                complex.CreateOp(ComplexType.get(self.getFloatType()),
                                 self.getConstantFloat(node.value.real),
                                 self.getConstantFloat(node.value.imag)).result)
            return

        self.emitFatalError("unhandled constant value", node)

    def visit_Subscript(self, node):
        """
        Convert element extractions (`__getitem__`, `operator[](idx)`, `q[1:3]`) to 
        corresponding extraction or slice code in the MLIR. This method handles 
        extraction for `veq` types and `stdvec` types. 
        """
        self.currentNode = node

        # handle complex slice, VAR[lower:upper]
        if isinstance(node.slice, ast.Slice):
            self.debug_msg(lambda: f'[(Inline) Visit Slice]', node.slice)
            self.visit(node.value)
            var = self.ifPointerThenLoad(self.popValue())

            vectorSize = None
            if quake.VeqType.isinstance(var.type):
                vectorSize = quake.VeqSizeOp(self.getIntegerType(64),
                                             var).result
            elif cc.StdvecType.isinstance(var.type):
                vectorSize = cc.StdvecSizeOp(self.getIntegerType(), var).result

            def fix_negative(val):
                if hasattr(val.owner, 'opview') and isinstance(
                        val.owner.opview, arith.ConstantOp):
                    if 'value' in val.owner.attributes:
                        attr = IntegerAttr(val.owner.attributes['value'])
                        upperInt = attr.value
                        if upperInt < 0:
                            return arith.AddIOp(
                                vectorSize,
                                self.getConstantInt(upperInt)).result
                return val

            lowerVal, upperVal, stepVal = (None, None, None)
            if node.slice.lower is not None:
                self.visit(node.slice.lower)
                lowerVal = fix_negative(self.popValue())
            else:
                lowerVal = self.getConstantInt(0)
            if node.slice.upper is not None:
                self.visit(node.slice.upper)
                upperVal = fix_negative(self.popValue())
            else:
                if not quake.VeqType.isinstance(
                        var.type) and not cc.StdvecType.isinstance(var.type):
                    self.emitFatalError(
                        f"unhandled upper slice == None, can't handle type {var.type}",
                        node)
                else:
                    upperVal = vectorSize

            if node.slice.step is not None:
                self.emitFatalError("step value in slice is not supported.",
                                    node)

            if quake.VeqType.isinstance(var.type):
                # Upper bound is exclusive
                upperVal = arith.SubIOp(upperVal, self.getConstantInt(1)).result
                dyna = IntegerAttr.get(self.getIntegerType(), -1)
                self.pushValue(
                    quake.SubVeqOp(self.getVeqType(),
                                   var,
                                   dyna,
                                   dyna,
                                   lower=lowerVal,
                                   upper=upperVal).result)
            elif cc.StdvecType.isinstance(var.type):
                eleTy = cc.StdvecType.getElementType(var.type)
                ptrTy = cc.PointerType.get(eleTy)
                arrTy = cc.ArrayType.get(eleTy)
                ptrArrTy = cc.PointerType.get(arrTy)
                nElementsVal = arith.SubIOp(upperVal, lowerVal).result
                # need to compute the distance between `upperVal` and `lowerVal`
                # then slice is `stdvecdataOp + computeptr[lower] + stdvecinit[ptr,distance]`
                vecPtr = cc.StdvecDataOp(ptrArrTy, var).result
                ptr = cc.ComputePtrOp(
                    ptrTy, vecPtr, [lowerVal],
                    DenseI32ArrayAttr.get([kDynamicPtrIndex],
                                          context=self.ctx)).result
                self.pushValue(
                    cc.StdvecInitOp(var.type, ptr, length=nElementsVal).result)
            else:
                self.emitFatalError(
                    f"unhandled slice operation, cannot handle type {var.type}",
                    node)

            return

        self.generic_visit(node)

        assert len(self.valueStack) > 1

        # get the last name, should be name of var being subscripted
        var = self.ifPointerThenLoad(self.popValue())
        idx = self.popValue()

        # Support `VAR[-1]` as the last element of `VAR`
        if quake.VeqType.isinstance(var.type):
            if hasattr(idx.owner, 'opview') and isinstance(
                    idx.owner.opview, arith.ConstantOp):
                if 'value' in idx.owner.attributes:
                    concreteIntAttr = IntegerAttr(idx.owner.attributes['value'])
                    idxConcrete = concreteIntAttr.value
                    if idxConcrete == -1:
                        qrSize = quake.VeqSizeOp(self.getIntegerType(),
                                                 var).result
                        one = self.getConstantInt(1)
                        endOff = arith.SubIOp(qrSize, one)
                        self.pushValue(
                            quake.ExtractRefOp(self.getRefType(),
                                               var,
                                               -1,
                                               index=endOff).result)
                        return

            # Made it here, general VAR[idx], handle `veq` and `stdvec`
            qrefTy = self.getRefType()
            if not IntegerType.isinstance(idx.type):
                self.emitFatalError(
                    f'invalid index variable type used for qvector extraction ({idx.type})',
                    node)

            self.pushValue(
                quake.ExtractRefOp(qrefTy, var, -1, index=idx).result)
            return

        if cc.StdvecType.isinstance(var.type):
            eleTy = cc.StdvecType.getElementType(var.type)
            elePtrTy = cc.PointerType.get(eleTy)
            arrTy = cc.ArrayType.get(eleTy)
            ptrArrTy = cc.PointerType.get(arrTy)
            vecPtr = cc.StdvecDataOp(ptrArrTy, var).result
            eleAddr = cc.ComputePtrOp(
                elePtrTy, vecPtr, [idx],
                DenseI32ArrayAttr.get([kDynamicPtrIndex],
                                      context=self.ctx)).result
            if self.subscriptPushPointerValue:
                self.pushValue(eleAddr)
                return
            self.pushValue(cc.LoadOp(eleAddr).result)
            return

        if cc.PointerType.isinstance(var.type):
            ptrEleTy = cc.PointerType.getElementType(var.type)
            # Return the pointer if someone asked for it
            if self.subscriptPushPointerValue:
                self.pushValue(var)
                return
            if cc.ArrayType.isinstance(ptrEleTy):
                # Here we want subscript on `ptr<array<>>`
                arrayEleTy = cc.ArrayType.getElementType(ptrEleTy)
                ptrEleTy = cc.PointerType.get(arrayEleTy)
                casted = cc.CastOp(ptrEleTy, var).result
                eleAddr = cc.ComputePtrOp(
                    ptrEleTy, casted, [idx],
                    DenseI32ArrayAttr.get([kDynamicPtrIndex],
                                          context=self.ctx)).result
                self.pushValue(cc.LoadOp(eleAddr).result)
                return

        if cc.StructType.isinstance(var.type):
            # Handle the case where we have a tuple member extraction, memory semantics
            idxValue = None
            if hasattr(idx.owner, 'opview') and isinstance(
                    idx.owner.opview, arith.ConstantOp):
                if 'value' in idx.owner.attributes:
                    attr = IntegerAttr(idx.owner.attributes['value'])
                    idxValue = attr.value

            if idxValue == None:
                self.emitFatalError(
                    "non-constant subscript value on a tuple is not supported",
                    node)

            memberTys = cc.StructType.getTypes(var.type)
            if idxValue >= len(memberTys):
                self.emitFatalError(f'tuple index is out of range: {idxValue}',
                                    node)

            structPtr = self.ifNotPointerThenStore(var)
            eleAddr = cc.ComputePtrOp(
                cc.PointerType.get(memberTys[idxValue]), structPtr, [],
                DenseI32ArrayAttr.get([idxValue], context=self.ctx)).result

            # Return the pointer if someone asked for it
            if self.subscriptPushPointerValue:
                self.pushValue(eleAddr)
                return
            self.pushValue(cc.LoadOp(eleAddr).result)
            return

        self.emitFatalError("unhandled subscript", node)

    def visit_For(self, node):
        """
        Visit the For node. This node represents the typical 
        Python for statement, `for VAR in ITERABLE`. Currently supported 
        ITERABLEs are the `veq` type, the `stdvec` type, and the result of 
        range() and enumerate(). 
        """
        self.currentNode = node

        # We can simplify `for i in range(N)` MLIR code immensely
        # by just building a for loop with N as the upper value,
        # no need to generate an array from the `range` call.
        if isinstance(node.iter, ast.Call):
            self.debug_msg(lambda: f'[(Inline) Visit Call]', node.iter)
            if node.iter.func.id == 'range':
                # This is a range(N) for loop, we just need
                # the upper bound N for this loop
                [self.visit(arg) for arg in node.iter.args]
                startVal, endVal, stepVal, isDecrementing = self.__processRangeLoopIterationBounds(
                    node.iter.args)

                def bodyBuilder(iterVar):
                    self.symbolTable.pushScope()
                    self.symbolTable.add(node.target.id, iterVar)
                    [self.visit(b) for b in node.body]
                    self.symbolTable.popScope()

                self.createInvariantForLoop(endVal,
                                            bodyBuilder,
                                            startVal=startVal,
                                            stepVal=stepVal,
                                            isDecrementing=isDecrementing,
                                            elseStmts=node.orelse)

                return

        # We can simplify `for i,j in enumerate(L)` MLIR code immensely
        # by just building a for loop over the iterable object L and using
        # the index into that iterable and the element.
        if isinstance(node.iter, ast.Call):
            self.debug_msg(lambda: f'[(Inline) Visit Call]', node.iter)
            if node.iter.func.id == 'enumerate':
                [self.visit(arg) for arg in node.iter.args]
                if len(self.valueStack) == 2:
                    iterable = self.popValue()
                    self.popValue()
                else:
                    assert len(self.valueStack) == 1
                    iterable = self.popValue()
                iterable = self.ifPointerThenLoad(iterable)
                totalSize = None
                extractFunctor = None
                varNames = []
                for elt in node.target.elts:
                    varNames.append(elt.id)

                beEfficient = False
                if quake.VeqType.isinstance(iterable.type):
                    totalSize = quake.VeqSizeOp(self.getIntegerType(),
                                                iterable).result

                    def functor(seq, idx):
                        q = quake.ExtractRefOp(self.getRefType(),
                                               seq,
                                               -1,
                                               index=idx).result
                        return [idx, q]

                    extractFunctor = functor
                    beEfficient = True
                elif cc.StdvecType.isinstance(iterable.type):
                    totalSize = cc.StdvecSizeOp(self.getIntegerType(),
                                                iterable).result

                    def functor(seq, idx):
                        vecTy = cc.StdvecType.getElementType(seq.type)
                        dataTy = cc.PointerType.get(vecTy)
                        arrTy = vecTy
                        if not cc.ArrayType.isinstance(arrTy):
                            arrTy = cc.ArrayType.get(vecTy)
                        dataArrTy = cc.PointerType.get(arrTy)
                        data = cc.StdvecDataOp(dataArrTy, seq).result
                        v = cc.ComputePtrOp(
                            dataTy, data, [idx],
                            DenseI32ArrayAttr.get([kDynamicPtrIndex],
                                                  context=self.ctx)).result
                        return [idx, v]

                    extractFunctor = functor
                    beEfficient = True

                if beEfficient:

                    def bodyBuilder(iterVar):
                        self.symbolTable.pushScope()
                        values = extractFunctor(iterable, iterVar)
                        for i, v in enumerate(values):
                            self.symbolTable[varNames[i]] = v
                        [self.visit(b) for b in node.body]
                        self.symbolTable.popScope()

                    self.createInvariantForLoop(totalSize,
                                                bodyBuilder,
                                                elseStmts=node.orelse)
                    return

        self.visit(node.iter)
        assert len(self.valueStack) > 0 and len(self.valueStack) < 3

        totalSize = None
        iterable = None
        extractFunctor = None

        # It could be that its the only value we have,
        # in which case we know we have for var in iterable,
        # but we could also have another value on the stack,
        # the total size of the iterable, produced by range() / enumerate()
        if len(self.valueStack) == 1:
            # Get the iterable from the stack
            iterable = self.ifPointerThenLoad(self.popValue())
            # we currently handle `veq` and `stdvec` types
            if quake.VeqType.isinstance(iterable.type):
                size = quake.VeqType.getSize(iterable.type)
                if quake.VeqType.hasSpecifiedSize(iterable.type):
                    totalSize = self.getConstantInt(size)
                else:
                    totalSize = quake.VeqSizeOp(self.getIntegerType(64),
                                                iterable).result

                def functor(iter, idx):
                    return [
                        quake.ExtractRefOp(self.getRefType(),
                                           iter,
                                           -1,
                                           index=idx).result
                    ]

                extractFunctor = functor
            elif cc.StdvecType.isinstance(iterable.type):
                iterEleTy = cc.StdvecType.getElementType(iterable.type)
                totalSize = cc.StdvecSizeOp(self.getIntegerType(),
                                            iterable).result

                def functor(iter, idxVal):
                    elePtrTy = cc.PointerType.get(iterEleTy)
                    arrTy = cc.ArrayType.get(iterEleTy)
                    ptrArrTy = cc.PointerType.get(arrTy)
                    vecPtr = cc.StdvecDataOp(ptrArrTy, iter).result
                    eleAddr = cc.ComputePtrOp(
                        elePtrTy, vecPtr, [idxVal],
                        DenseI32ArrayAttr.get([kDynamicPtrIndex],
                                              context=self.ctx)).result
                    return [cc.LoadOp(eleAddr).result]

                extractFunctor = functor

            else:
                self.emitFatalError('{} iterable type not supported.', node)

        else:
            # In this case, we are coming from range() or enumerate(),
            # and the iterable is a cc.array and the total size of the
            # array is on the stack, pop it here
            totalSize = self.popValue()
            # Get the iterable from the stack
            iterable = self.popValue()

            # Double check our types are right
            assert cc.PointerType.isinstance(iterable.type)
            arrayType = cc.PointerType.getElementType(iterable.type)
            assert cc.ArrayType.isinstance(arrayType)
            elementType = cc.ArrayType.getElementType(arrayType)

            def functor(iter, idx):
                eleAddr = cc.ComputePtrOp(
                    cc.PointerType.get(elementType), iter, [idx],
                    DenseI32ArrayAttr.get([kDynamicPtrIndex],
                                          context=self.ctx)).result
                loaded = cc.LoadOp(eleAddr).result
                if IntegerType.isinstance(elementType):
                    return [loaded]
                elif cc.StructType.isinstance(elementType):
                    # Get struct types
                    types = cc.StructType.getTypes(elementType)
                    ret = []
                    for i, ty in enumerate(types):
                        ret.append(
                            cc.ExtractValueOp(
                                ty, loaded, [],
                                DenseI32ArrayAttr.get([i],
                                                      context=self.ctx)).result)
                    return ret

            extractFunctor = functor

        # Get the name of the variable, VAR in for VAR in range(...),
        # could be a tuple of names too
        varNames = []
        if isinstance(node.target, ast.Name):
            self.debug_msg(lambda: f'[(Inline) Visit Name]', node.target)
            varNames.append(node.target.id)
        else:
            # has to be a `ast.Tuple`
            self.debug_msg(lambda: f'[(Inline) Visit Tuple]', node.target)
            for elt in node.target.elts:
                varNames.append(elt.id)

        # We'll need a zero and one value of integer type
        iTy = self.getIntegerType(64)
        zero = arith.ConstantOp(iTy, IntegerAttr.get(iTy, 0))
        one = arith.ConstantOp(iTy, IntegerAttr.get(iTy, 1))

        def bodyBuilder(iterVar):
            self.symbolTable.pushScope()
            # we set the extract functor above, use it here
            values = extractFunctor(iterable, iterVar)
            for i, v in enumerate(values):
                self.symbolTable[varNames[i]] = v
            [self.visit(b) for b in node.body]
            self.symbolTable.popScope()

        self.createInvariantForLoop(totalSize,
                                    bodyBuilder,
                                    elseStmts=node.orelse)

    def visit_While(self, node):
        """
        Convert Python while statements into the equivalent CC `LoopOp`. 
        """
        self.currentNode = node

        loop = cc.LoopOp([], [], BoolAttr.get(False))
        whileBlock = Block.create_at_start(loop.whileRegion, [])
        with InsertionPoint(whileBlock):
            # BUG you cannot print MLIR values while building the cc `LoopOp` while region.
            # verify will get called, no terminator yet, CCOps.cpp:520
            v = self.verbose
            self.verbose = False
            self.visit(node.test)
            condition = self.popValue()
            if self.getIntegerType(1) != condition.type:
                # not equal to 0, then compare with 1
                condPred = IntegerAttr.get(self.getIntegerType(), 1)
                condition = arith.CmpIOp(condPred, condition,
                                         self.getConstantInt(0)).result
            cc.ConditionOp(condition, [])
            self.verbose = v

        bodyBlock = Block.create_at_start(loop.bodyRegion, [])
        with InsertionPoint(bodyBlock):
            self.symbolTable.pushScope()
            self.pushForBodyStack([])
            [self.visit(b) for b in node.body]
            if not self.hasTerminator(bodyBlock):
                cc.ContinueOp([])
            self.popForBodyStack()
            self.symbolTable.popScope()

        stepBlock = Block.create_at_start(loop.stepRegion, [])
        with InsertionPoint(stepBlock):
            cc.ContinueOp([])

        if node.orelse:
            elseBlock = Block.create_at_start(loop.elseRegion, [])
            with InsertionPoint(elseBlock):
                self.symbolTable.pushScope()
                for stmt in node.orelse:
                    self.visit(stmt)
                if not self.hasTerminator(elseBlock):
                    cc.ContinueOp(elseBlock.arguments)
                self.symbolTable.popScope()

    def visit_BoolOp(self, node):
        """
        Convert boolean operations into equivalent MLIR operations using 
        the Arith Dialect.
        """
        self.currentNode = node
        shortCircuitWhenTrue = isinstance(node.op, ast.Or)
        if isinstance(node.op, ast.And) or isinstance(node.op, ast.Or):
            # Visit the LHS and pop the value
            # Note we want any `mz(q)` calls to push their
            # result value to the stack, so we set a non-None
            # variable name here.
            self.currentAssignVariableName = ''
            self.visit(node.values[0])
            lhs = self.popValue()
            zero = self.getConstantInt(0, IntegerType(lhs.type).width)

            cond = arith.CmpIOp(
                self.getIntegerAttr(self.getIntegerType(),
                                    1 if shortCircuitWhenTrue else 0), lhs,
                zero).result

            ifOp = cc.IfOp([cond.type], cond, [])
            thenBlock = Block.create_at_start(ifOp.thenRegion, [])
            with InsertionPoint(thenBlock):
                if isinstance(node.op, ast.And):
                    constantFalse = arith.ConstantOp(cond.type,
                                                     BoolAttr.get(False))
                    cc.ContinueOp([constantFalse])
                else:
                    cc.ContinueOp([cond])

            elseBlock = Block.create_at_start(ifOp.elseRegion, [])
            with InsertionPoint(elseBlock):
                self.symbolTable.pushScope()
                self.pushIfStmtBlockStack()
                self.visit(node.values[1])
                rhs = self.popValue()
                cc.ContinueOp([rhs])
                self.popIfStmtBlockStack()
                self.symbolTable.popScope()

            # Reset the assign variable name
            self.currentAssignVariableName = None

            self.pushValue(ifOp.result)
            return

    def visit_Compare(self, node):
        """
        Visit while loop compare operations and translate to equivalent MLIR. 
        Note, Python lets you construct expressions with multiple comparators, 
        here we limit ourselves to just a single comparator. 
        """
        if len(node.ops) > 1:
            self.emitFatalError("only single comparators are supported.", node)

        self.currentNode = node

        iTy = self.getIntegerType()

        if isinstance(node.left, ast.Name):
            self.debug_msg(lambda: f'[(Inline) Visit Name]', node.left)
            if node.left.id not in self.symbolTable:
                self.emitFatalError(
                    f"{node.left.id} was not initialized before use in compare expression.",
                    node)

        self.visit(node.left)
        left = self.popValue()
        self.visit(node.comparators[0])
        right = self.popValue()
        op = node.ops[0]

        left_type = left.type
        right_type = right.type

        if IntegerType.isinstance(left_type) and F64Type.isinstance(right_type):
            left = arith.SIToFPOp(self.getFloatType(), left).result
        elif F64Type.isinstance(left_type) and IntegerType.isinstance(
                right_type):
            right = arith.SIToFPOp(self.getFloatType(), right).result
        elif IntegerType.isinstance(left_type) and IntegerType.isinstance(
                right_type):
            if IntegerType(left_type).width < IntegerType(right_type).width:
                zeroext = IntegerType(left_type).width == 1
                left = cc.CastOp(right_type,
                                 left,
                                 sint=not zeroext,
                                 zint=zeroext).result
            elif IntegerType(left_type).width > IntegerType(right_type).width:
                zeroext = IntegerType(right_type).width == 1
                right = cc.CastOp(left_type,
                                  right,
                                  sint=not zeroext,
                                  zint=zeroext).result

        if isinstance(op, ast.Gt):
            if F64Type.isinstance(left.type):
                self.pushValue(
                    arith.CmpFOp(self.getIntegerAttr(iTy, 2), left,
                                 right).result)
            else:
                self.pushValue(
                    arith.CmpIOp(self.getIntegerAttr(iTy, 4), left,
                                 right).result)
            return

        if isinstance(op, ast.GtE):
            if F64Type.isinstance(left.type):
                self.pushValue(
                    arith.CmpFOp(self.getIntegerAttr(iTy, 3), left,
                                 right).result)
            else:
                self.pushValue(
                    arith.CmpIOp(self.getIntegerAttr(iTy, 5), left,
                                 right).result)
            return

        if isinstance(op, ast.Lt):
            if F64Type.isinstance(left.type):
                self.pushValue(
                    arith.CmpFOp(self.getIntegerAttr(iTy, 4), left,
                                 right).result)
            else:
                self.pushValue(
                    arith.CmpIOp(self.getIntegerAttr(iTy, 2), left,
                                 right).result)
            return

        if isinstance(op, ast.LtE):
            if F64Type.isinstance(left.type):
                self.pushValue(
                    arith.CmpFOp(self.getIntegerAttr(iTy, 5), left,
                                 right).result)
            else:
                self.pushValue(
                    arith.CmpIOp(self.getIntegerAttr(iTy, 7), left,
                                 right).result)
            return

        if isinstance(op, ast.NotEq):
            if F64Type.isinstance(left.type):
                self.pushValue(
                    arith.CmpFOp(self.getIntegerAttr(iTy, 6), left,
                                 right).result)
            else:
                self.pushValue(
                    arith.CmpIOp(self.getIntegerAttr(iTy, 1), left,
                                 right).result)
            return

        if isinstance(op, ast.Eq):
            if F64Type.isinstance(left.type):
                self.pushValue(
                    arith.CmpFOp(self.getIntegerAttr(iTy, 1), left,
                                 right).result)
            else:
                self.pushValue(
                    arith.CmpIOp(self.getIntegerAttr(iTy, 0), left,
                                 right).result)
            return

        if isinstance(op, (ast.In, ast.NotIn)):
            right_val = right
            left_val = left

            # Type validation and vector initialization
            if not (cc.StdvecType.isinstance(right_val.type) or
                    cc.ArrayType.isinstance(right_val.type)):
                self.emitFatalError(
                    "Right operand must be a list/vector for 'in' comparison")

            # Loop setup
            i1_type = self.getIntegerType(1)
            accumulator = cc.AllocaOp(cc.PointerType.get(i1_type),
                                      TypeAttr.get(i1_type)).result
            cc.StoreOp(self.getConstantInt(0, 1), accumulator)

            # Element comparison loop
            def check_element(idx):
                element = self.__load_vector_element(right_val, idx)
                promoted_left, promoted_element = self.convertArithmeticToSuperiorType(
                    [left_val, element],
                    self.__get_superior_type(left_val, element))

                iCondPred = IntegerAttr.get(self.getIntegerType(), 0)
                fCondPred = IntegerAttr.get(self.getIntegerType(), 0)
                cmp_result = (
                    arith.CmpIOp(iCondPred, promoted_left, promoted_element)
                    if IntegerType.isinstance(promoted_left.type) else
                    arith.CmpFOp(fCondPred, promoted_left, promoted_element))

                current = cc.LoadOp(accumulator).result
                cc.StoreOp(arith.OrIOp(current, cmp_result.result), accumulator)

            self.createInvariantForLoop(self.__get_vector_size(right_val),
                                        check_element)

            final_result = cc.LoadOp(accumulator).result
            if isinstance(op, ast.NotIn):
                final_result = arith.XOrIOp(final_result,
                                            self.getConstantInt(1, 1)).result
            self.pushValue(final_result)

            return

    def visit_AugAssign(self, node):
        """
        Visit augment-assign operations (e.g. +=). 
        """
        target = None
        self.currentNode = node

        if isinstance(node.target,
                      ast.Name) and node.target.id in self.symbolTable:
            self.debug_msg(lambda: f'[(Inline) Visit Name]', node.target)
            target = self.symbolTable[node.target.id]
        else:
            self.emitFatalError(
                "unable to get augment-assign target variable from symbol table.",
                node)

        self.visit(node.value)
        value = self.popValue()

        loaded = cc.LoadOp(target).result
        if isinstance(node.op, ast.Sub):
            # i -= 1 -> i = i - 1
            if IntegerType.isinstance(loaded.type):
                res = arith.SubIOp(loaded, value).result
                cc.StoreOp(res, target)
                return

            self.emitFatalError("unhandled AugAssign.Sub types.", node)

        if isinstance(node.op, ast.Add):
            # i += 1 -> i = i + 1
            if IntegerType.isinstance(loaded.type):
                res = arith.AddIOp(loaded, value).result
                cc.StoreOp(res, target)
                return
            if F64Type.isinstance(loaded.type):
                if IntegerType.isinstance(value.type):
                    value = arith.SIToFPOp(loaded.type, value).result
                res = arith.AddFOp(loaded, value).result
                cc.StoreOp(res, target)
                return

            self.emitFatalError("unhandled AugAssign.Add types.", node)

        if isinstance(node.op, ast.Mult):
            # i *= 3 -> i = i * 3
            if IntegerType.isinstance(loaded.type):
                res = arith.MulIOp(loaded, value).result
                cc.StoreOp(res, target)
                return
            elif F64Type.isinstance(loaded.type):
                if IntegerType.isinstance(value.type):
                    value = arith.SIToFPOp(self.getFloatType(), value).result
                res = arith.MulFOp(loaded, value).result
                cc.StoreOp(res, target)
                return

            self.emitFatalError("unhandled AugAssign.Mult types.", node)

        self.emitFatalError("unhandled aug-assign operation.", node)

    def visit_If(self, node):
        """
        Map a Python `ast.If` node to an if statement operation in the CC dialect. 
        """

        self.currentNode = node

        # Visit the conditional node, retain
        # measurement results by assigning a dummy variable name
        self.currentAssignVariableName = ''
        self.visit(node.test)
        self.currentAssignVariableName = None

        condition = self.popValue()
        condition = self.ifPointerThenLoad(condition)

        if self.getIntegerType(1) != condition.type:
            # not equal to 0, then compare with 1
            condPred = IntegerAttr.get(self.getIntegerType(), 1)
            condition = arith.CmpIOp(condPred, condition,
                                     self.getConstantInt(0)).result

        ifOp = cc.IfOp([], condition, [])
        thenBlock = Block.create_at_start(ifOp.thenRegion, [])
        with InsertionPoint(thenBlock):
            self.symbolTable.pushScope()
            self.pushIfStmtBlockStack()
            [self.visit(b) for b in node.body]
            if not self.hasTerminator(thenBlock):
                cc.ContinueOp([])
            self.popIfStmtBlockStack()
            self.symbolTable.popScope()

        if len(node.orelse) > 0:
            elseBlock = Block.create_at_start(ifOp.elseRegion, [])
            with InsertionPoint(elseBlock):
                self.symbolTable.pushScope()
                self.pushIfStmtBlockStack()
                [self.visit(b) for b in node.orelse]
                if not self.hasTerminator(elseBlock):
                    cc.ContinueOp([])
                self.popIfStmtBlockStack()
                self.symbolTable.popScope()

    def visit_Return(self, node):

        if node.value == None:
            return

        self.walkingReturnNode = True
        self.visit(node.value)
        self.walkingReturnNode = False

        if len(self.valueStack) == 0:
            return

        result = self.ifPointerThenLoad(self.popValue())
        result = self.ifPointerThenLoad(result)
        if result.type != self.knownResultType:
            # FIXME consider more auto-casting where possible
            result = self.promoteOperandType(self.knownResultType, result)

        if result.type != self.knownResultType:
            self.emitFatalError(
                f"Invalid return type, function was defined to return a {mlirTypeToPyType(self.knownResultType)} but the value being returned is of type {mlirTypeToPyType(result.type)}",
                node)

        if cc.StdvecType.isinstance(result.type):
            symName = '__nvqpp_vectorCopyCtor'
            load_intrinsic(self.module, symName)
            eleTy = cc.StdvecType.getElementType(result.type)
            ptrTy = cc.PointerType.get(self.getIntegerType(8))
            arrTy = cc.ArrayType.get(self.getIntegerType(8))
            ptrArrTy = cc.PointerType.get(arrTy)
            resBuf = cc.StdvecDataOp(ptrArrTy, result).result
            # TODO Revisit this calculation
            byteWidth = 16 if ComplexType.isinstance(eleTy) else 8
            eleSize = self.getConstantInt(byteWidth)
            dynSize = cc.StdvecSizeOp(self.getIntegerType(), result).result
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
        """
        Map tuples in the Python AST to equivalents in MLIR.
        """
        self.generic_visit(node)
        self.currentNode = node

        elementValues = [self.popValue() for _ in range(len(node.elts))]
        elementValues.reverse()

        # We do not store structs of pointers
        elementValues = [
            cc.LoadOp(ele).result
            if cc.PointerType.isinstance(ele.type) else ele
            for ele in elementValues
        ]

        structTys = [v.type for v in elementValues]

        if len(elementValues) != len(structTys):
            self.emitFatalError(
                f'tuple constructor requires {len(structTys)} arguments, '
                f'but was given {len(elementValues)}.', node)

        structTy = cc.StructType.getNamed("tuple", structTys)
        stackSlot = cc.AllocaOp(cc.PointerType.get(structTy),
                                TypeAttr.get(structTy)).result

        # loop over each type and `compute_ptr` / store

        for i, ty in enumerate(structTys):
            eleAddr = cc.ComputePtrOp(
                cc.PointerType.get(ty), stackSlot, [],
                DenseI32ArrayAttr.get([i], context=self.ctx)).result
            cc.StoreOp(elementValues[i], eleAddr)
        self.pushValue(stackSlot)
        return

    def visit_UnaryOp(self, node):
        """
        Map unary operations in the Python AST to equivalents in MLIR.
        """

        self.currentNode = node

        self.generic_visit(node)
        operand = self.popValue()
        # Handle qubit negations
        if isinstance(node.op, ast.Invert):
            if quake.RefType.isinstance(operand.type):
                self.controlNegations.append(operand)
                self.pushValue(operand)
                return

        if isinstance(node.op, ast.USub):
            # Make our lives easier for -1 used in variable subscript extraction
            if isinstance(node.operand,
                          ast.Constant) and node.operand.value == 1:
                self.pushValue(self.getConstantInt(-1))
                return

            if F64Type.isinstance(operand.type):
                self.pushValue(arith.NegFOp(operand).result)
            elif ComplexType.isinstance(operand.type):
                # `complex.NegOp` does not seem to work
                self.pushValue(
                    complex.MulOp(
                        complex.CreateOp(operand.type,
                                         self.getConstantFloat(-1.),
                                         self.getConstantFloat(0.)).result,
                        operand).result)
            else:
                negOne = self.getConstantInt(-1)
                self.pushValue(arith.MulIOp(negOne, operand).result)
            return

        if isinstance(node.op, ast.Not):
            if not IntegerType.isinstance(operand.type):
                self.emitFatalError("UnaryOp Not() on non-integer value.", node)

            zero = self.getConstantInt(0, IntegerType(operand.type).width)
            self.pushValue(
                arith.CmpIOp(IntegerAttr.get(self.getIntegerType(), 0), operand,
                             zero).result)
            return

        self.emitFatalError("unhandled UnaryOp.", node)

    def visit_Break(self, node):

        self.currentNode = node

        if not self.isInForBody():
            self.emitFatalError("break statement outside of for loop body.",
                                node)

        if self.isInIfStmtBlock():
            # Get the innermost enclosing `for` or `while` loop
            inArgs = [b for b in self.inForBodyStack[-1]]
            cc.UnwindBreakOp(inArgs)
        else:
            cc.BreakOp([])

        return

    def visit_Continue(self, node):

        self.currentNode = node

        if not self.isInForBody():
            self.emitFatalError("continue statement outside of for loop body.",
                                node)

        if self.isInIfStmtBlock():
            # Get the innermost enclosing `for` or `while` loop
            inArgs = [b for b in self.inForBodyStack[-1]]
            cc.UnwindContinueOp(inArgs)
        else:
            cc.ContinueOp([])

    def visit_BinOp(self, node):
        """
        Visit binary operation nodes in the AST and map them to equivalents in the 
        MLIR. This method handles arithmetic operations between values. 
        """

        self.currentNode = node

        # Get the left and right parts of this expression
        self.visit(node.left)
        left = self.popValue()
        self.visit(node.right)
        right = self.popValue()

        if cc.PointerType.isinstance(left.type):
            left = cc.LoadOp(left).result
        if cc.PointerType.isinstance(right.type):
            right = cc.LoadOp(right).result

        if not IntegerType.isinstance(left.type) and not F64Type.isinstance(
                left.type) and not ComplexType.isinstance(left.type):
            raise RuntimeError("Invalid type for Binary Op {} ({}, {})".format(
                type(node.op), left, right))

        if not IntegerType.isinstance(right.type) and not F64Type.isinstance(
                right.type) and not ComplexType.isinstance(right.type):
            raise RuntimeError("Invalid type for Binary Op {} ({}, {})".format(
                type(node.op), right, right))

        # Type promotion for addition, subtraction, multiplication, or division
        if isinstance(node.op, (ast.Add, ast.Sub, ast.Mult, ast.Div)):
            right = self.promoteOperandType(left.type, right)
            left = self.promoteOperandType(right.type, left)

        # Based on the op type and the leaf types, create the MLIR operator
        if isinstance(node.op, ast.Add):
            if IntegerType.isinstance(left.type):
                self.pushValue(arith.AddIOp(left, right).result)
                return
            elif F64Type.isinstance(left.type):
                self.pushValue(arith.AddFOp(left, right).result)
                return
            elif ComplexType.isinstance(left.type):
                self.pushValue(complex.AddOp(left, right).result)
                return
            else:
                self.emitFatalError("unhandled BinOp.Add types.", node)

        if isinstance(node.op, ast.Sub):
            if IntegerType.isinstance(left.type):
                self.pushValue(arith.SubIOp(left, right).result)
                return
            if F64Type.isinstance(left.type):
                self.pushValue(arith.SubFOp(left, right).result)
                return
            if ComplexType.isinstance(left.type):
                self.pushValue(complex.SubOp(left, right).result)
            else:
                self.emitFatalError("unhandled BinOp.Sub types.", node)
        if isinstance(node.op, ast.FloorDiv):
            if IntegerType.isinstance(left.type):
                self.pushValue(arith.FloorDivSIOp(left, right).result)
                return
            else:
                self.emitFatalError("unhandled BinOp.FloorDiv types.", node)
        if isinstance(node.op, ast.Div):
            if ComplexType.isinstance(left.type):
                self.pushValue(complex.DivOp(left, right).result)
                return

            if IntegerType.isinstance(left.type):
                left = arith.SIToFPOp(self.getFloatType(), left).result
            if IntegerType.isinstance(right.type):
                right = arith.SIToFPOp(self.getFloatType(), right).result

            self.pushValue(arith.DivFOp(left, right).result)
            return
        if isinstance(node.op, ast.Pow):
            if IntegerType.isinstance(left.type) and IntegerType.isinstance(
                    right.type):
                # `math.ipowi` does not lower to LLVM as is
                # workaround, use math to function conversion
                self.pushValue(math.IPowIOp(left, right).result)
                return

            if F64Type.isinstance(left.type) and IntegerType.isinstance(
                    right.type):
                self.pushValue(math.FPowIOp(left, right).result)
                return

            # now we know the types are different, default to float
            if IntegerType.isinstance(left.type):
                left = arith.SIToFPOp(self.getFloatType(), left).result
            if IntegerType.isinstance(right.type):
                right = arith.SIToFPOp(self.getFloatType(), right).result

            self.pushValue(math.PowFOp(left, right).result)
            return
        if isinstance(node.op, ast.Mult):
            if ComplexType.isinstance(left.type):
                self.pushValue(complex.MulOp(left, right).result)
                return

            if F64Type.isinstance(left.type):
                self.pushValue(arith.MulFOp(left, right).result)
                return

            if IntegerType.isinstance(left.type):
                self.pushValue(arith.MulIOp(left, right).result)
                return
            return
        if isinstance(node.op, ast.Mod):
            if F64Type.isinstance(left.type):
                left = arith.FPToSIOp(self.getIntegerType(), left).result
            if F64Type.isinstance(right.type):
                right = arith.FPToSIOp(self.getIntegerType(), right).result

            self.pushValue(arith.RemUIOp(left, right).result)
            return

        if isinstance(node.op, ast.LShift):
            if IntegerType.isinstance(left.type) and IntegerType.isinstance(
                    right.type):
                left = self.promoteOperandType(self.getIntegerType(), left)
                right = self.promoteOperandType(self.getIntegerType(), right)
                self.pushValue(arith.ShLIOp(left, right).result)
                return
            else:
                self.emitFatalError(
                    "unsupported operand type(s) for '<<'; only integers supported.",
                    node)

        if isinstance(node.op, ast.RShift):
            if IntegerType.isinstance(left.type) and IntegerType.isinstance(
                    right.type):
                left = self.promoteOperandType(self.getIntegerType(), left)
                right = self.promoteOperandType(self.getIntegerType(), right)
                self.pushValue(arith.ShRSIOp(left, right).result)
                return
            else:
                self.emitFatalError(
                    "unsupported operand type(s) for '>>'; only integers supported.",
                    node)

        if isinstance(node.op, ast.BitAnd):
            if IntegerType.isinstance(left.type) and IntegerType.isinstance(
                    right.type):
                left = self.promoteOperandType(self.getIntegerType(), left)
                right = self.promoteOperandType(self.getIntegerType(), right)
                self.pushValue(arith.AndIOp(left, right).result)
                return
            else:
                self.emitFatalError(
                    "unsupported operand type(s) for '&'; only integers supported.",
                    node)

        if isinstance(node.op, ast.BitOr):
            if IntegerType.isinstance(left.type) and IntegerType.isinstance(
                    right.type):
                left = self.promoteOperandType(self.getIntegerType(), left)
                right = self.promoteOperandType(self.getIntegerType(), right)
                self.pushValue(arith.OrIOp(left, right).result)
                return
            else:
                self.emitFatalError(
                    "unsupported operand type(s) for '|'; only integers supported.",
                    node)

        if isinstance(node.op, ast.BitXor):
            if IntegerType.isinstance(left.type) and IntegerType.isinstance(
                    right.type):
                left = self.promoteOperandType(self.getIntegerType(), left)
                right = self.promoteOperandType(self.getIntegerType(), right)
                self.pushValue(arith.XOrIOp(left, right).result)
                return
            else:
                self.emitFatalError(
                    "unsupported operand type(s) for '^'; only integers supported.",
                    node)

        self.emitFatalError(f"unhandled binary operator - {node.op}", node)

    def visit_Name(self, node):
        """
        Visit `ast.Name` nodes and extract the correct value from the symbol table.
        """
        self.currentNode = node

        if node.id in globalKernelRegistry:
            return

        if node.id == 'complex':
            self.pushValue(self.getComplexType())
            return

        if node.id == 'float':
            self.pushValue(self.getFloatType())
            return

        if node.id in self.symbolTable:
            value = self.symbolTable[node.id]
            if cc.PointerType.isinstance(value.type):
                eleTy = cc.PointerType.getElementType(value.type)
                if cc.ArrayType.isinstance(eleTy):
                    self.pushValue(value)
                    return
                # Retain `ptr<i8>`
                if IntegerType.isinstance(eleTy) and IntegerType(
                        eleTy).width == 8:
                    self.pushValue(value)
                    return
                if cc.StdvecType.isinstance(eleTy):
                    self.pushValue(value)
                    return
                if cc.StateType.isinstance(eleTy):
                    self.pushValue(value)
                    return
                loaded = cc.LoadOp(value).result
                self.pushValue(loaded)
            elif cc.CallableType.isinstance(
                    value.type) and not BlockArgument.isinstance(value):
                return
            else:
                self.pushValue(self.symbolTable[node.id])
            return

        if node.id in self.capturedVars:
            # Only support a small subset of types here
            complexType = type(1j)
            value = self.capturedVars[node.id]

            if isinstance(value, State):
                self.pushValue(self.capturedDataStorage.storeCudaqState(value))
                return

            if isinstance(value, (list, np.ndarray)) and isinstance(
                    value[0], (int, bool, float, np.float32, np.float64,
                               complexType, np.complex64, np.complex128)):
                elementValues = None
                if isinstance(value[0], (float, np.float64)):
                    elementValues = [self.getConstantFloat(el) for el in value]
                elif isinstance(value[0], np.float32):
                    elementValues = [
                        self.getConstantFloat(el, width=32) for el in value
                    ]
                elif isinstance(value[0], int):
                    elementValues = [self.getConstantInt(el) for el in value]
                elif isinstance(value[0], bool):
                    elementValues = [self.getConstantInt(el, 1) for el in value]
                elif isinstance(value[0], complexType) or isinstance(
                        value[0], np.complex128):
                    elementValues = [
                        self.getConstantComplex(el, width=64) for el in value
                    ]
                elif isinstance(value[0], np.complex64):
                    elementValues = [
                        self.getConstantComplex(el, width=32) for el in value
                    ]

                if elementValues != None:
                    # Save the copy of the captured list so we can compare
                    # it to the scope to detect changes on recompilation.
                    self.dependentCaptureVars[node.id] = value.copy()
                    mlirVal = self.__createStdvecWithKnownValues(
                        len(value), elementValues)
                    self.symbolTable.add(node.id, mlirVal, 0)
                    self.pushValue(mlirVal)
                    return

            mlirValCreator = None
            self.dependentCaptureVars[node.id] = value
            if isinstance(value, int):
                mlirValCreator = lambda: self.getConstantInt(value)
            elif isinstance(value, bool):
                mlirValCreator = lambda: self.getConstantInt(value, 1)
            elif isinstance(value, (float, np.float64)):
                mlirValCreator = lambda: self.getConstantFloat(value)
            elif isinstance(value, np.float32):
                mlirValCreator = lambda: self.getConstantFloat(value, width=32)
            elif isinstance(value, complexType) or isinstance(
                    value, np.complex128):
                mlirValCreator = lambda: self.getConstantComplex(value,
                                                                 width=64)
            elif isinstance(value, np.complex64):
                mlirValCreator = lambda: self.getConstantComplex(value,
                                                                 width=32)

            if mlirValCreator != None:
                with InsertionPoint.at_block_begin(self.entry):
                    mlirVal = mlirValCreator()
                    stackSlot = cc.AllocaOp(cc.PointerType.get(mlirVal.type),
                                            TypeAttr.get(mlirVal.type)).result
                    cc.StoreOp(mlirVal, stackSlot)
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

                    self.pushValue(self.getConstantInt(value.num_parameters))
                    self.pushValue(self.getConstantInt(hash(value)))
                    return
            except TypeError:
                pass

            self.emitFatalError(
                f"Invalid type for variable ({node.id}) captured from parent scope (only int, bool, float, complex, cudaq.State, and list/np.ndarray[int|bool|float|complex] accepted, type was {errorType}).",
                node)

        # Throw an exception for the case that the name is not
        # in the symbol table
        self.emitFatalError(
            f"Invalid variable name requested - '{node.id}' is not defined within the quantum kernel it is used in.",
            node)


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

    # First validate the arguments, make sure they are annotated
    bridge.validateArgumentAnnotations(astModule)

    # First we need to find any dependent kernels, they have to be
    # built as part of this ModuleOp...
    vis = FindDepKernelsVisitor(bridge.ctx)
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
            localVis = FindDepKernelsVisitor(bridge.ctx)
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
        "builtin.module(func.func(canonicalize,cse,quake-add-metadata),quake-propagate-metadata)",
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
