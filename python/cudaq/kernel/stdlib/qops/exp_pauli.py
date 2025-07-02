# ============================================================================ #
# Copyright (c) 2022 - 2025 NVIDIA Corporation & Affiliates.                   #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

from cudaq.mlir.dialects import quake
from cudaq.mlir.ir import Value
from ..literals.string_literal import StringLiteral
from ..float32 import ImplicitFloat32able
from ..float64 import ImplicitFloat64able


def exp_pauli(rotation: ImplicitFloat32able | ImplicitFloat64able,
              qubits: list[Value], pauli_word: str | StringLiteral):
    if isinstance(pauli_word, StringLiteral):
        pauli_word = pauli_word.materialize()
    rotation = rotation.__as_float64__().value
    quake.ExpPauliOp([], [rotation], [], [qubits], pauli=pauli_word)
