# ============================================================================ #
# Copyright (c) 2022 - 2025 NVIDIA Corporation & Affiliates.                   #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

import ast
import ctypes

from cudaq.mlir.dialects import arith
from cudaq.mlir.ir import (
    Context,
    F32Type,
    F64Type,
    FunctionType,
    IntegerType,
    Type,
    Value
)
from .diagnostic_emitter import DiagnosticEmitter

class TypeConverter:
    def __init__(self, diagnostic: DiagnosticEmitter, context: Context):
        self.diagnostic = diagnostic
        self.context = context
        self.name_to_type_map: dict[str, Type] = {
            'bool': IntegerType.get_signless(1, context),
            'int': IntegerType.get_signless(64, context),
            'float': F64Type.get(context)
        }
        self.type_to_type_map: dict[type, Type] = {
            bool: IntegerType.get_signless(1, context),
            int: IntegerType.get_signless(64, context),
            float: F64Type.get(context)
        }

    def _convert_name(self, type_hint: ast.Name) -> Type:
        type_name = type_hint.id
        if type_name in self.name_to_type_map:
            return self.name_to_type_map[type_name]

        # TODO: Handle TypeAlias
        # TODO: Handle User-defined ?
        self.diagnostic.emit_error(
            type_hint.lineno,
            type_hint.col_offset,
            f"Unknown type hint for type '{type_name}'.",
        )

    def _convert_mlir_to_ctype(self, type_):
        # Feels kind of a hack
        if isinstance(type_, F32Type):
            return ctypes.c_float * 1
        if isinstance(type_, F64Type):
            return ctypes.c_double * 1
        if isinstance(type_, IntegerType):
            return ctypes.c_int32 * 1

        self.diagnostic.emit_error(
            type_hint.lineno,
            type_hint.col_offset,
            f"Unknown type hint for type '{type_name}'.",
        )

    def convert_type_hint(self, type_hint: ast.expr) -> Type:
        if isinstance(type_hint, ast.Subscript):
            self.diagnostic.emit_error(
                type_hint.lineno,
                type_hint.col_offset,
                "Converting subscript type hints is not supported.",
            )

        if isinstance(type_hint, ast.Name):
            return self._convert_name(type_hint)

        self.diagnostic.emit_error(
            type_hint.lineno,
            type_hint.col_offset,
            f"Unknown type hint AST node '{type_hint}'.",
        )

    def convert_type(self, ty: type) -> Type:
        if ty in self.type_to_type_map:
            return self.type_to_type_map[ty]

        if isinstance(ty, FunctionType):
            inputs = [self._convert_mlir_to_ctype(t) for t in ty.inputs]
            result = self._convert_mlir_to_ctype(ty.results[0])
            return ctypes.CFUNCTYPE(result, *inputs)
            

        self.diagnostic.emit_error(
            type_hint.lineno,
            type_hint.col_offset,
            f"Unknown type '{ty}'.",
        )

    def deref_pointer(self, ty: type) -> Type:
        if ty in self.type_to_type_map:
            return self.type_to_type_map[ty]

        self.diagnostic.emit_error(
            type_hint.lineno,
            type_hint.col_offset,
            f"Unknown type '{ty}'.",
        )

    # Perhaps put this somewhere else
    def coerce_value_type(self, ty: Type, value: Value) -> Value:
        result = value
        # TODO: Complex type
        # TODO: Integer types of different sizes
        # TODO: Maybe emit warning when narrowing, e.g. F64 -> F32
        if F64Type.isinstance(ty):
            if F32Type.isinstance(value.type):
                result = arith.ExtFOp(ty, value).result
            elif IntegerType.isinstance(value.type):
                result = arith.SIToFPOp(ty, value).result
        elif F32Type.isinstance(ty):
            if F64Type.isinstance(value.type):
                result = arith.TruncFOp(ty, value).result
            elif IntegerType.isinstance(value.type):
                result = arith.SIToFPOp(ty, value).result

        return result
