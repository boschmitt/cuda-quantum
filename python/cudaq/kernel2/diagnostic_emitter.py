# ============================================================================ #
# Copyright (c) 2022 - 2025 NVIDIA Corporation & Affiliates.                   #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

class CodeGenerationException(Exception):
    def __init__(
        self,
        file: str | None,
        line: int,
        column: int,
        msg: str,
    ):
        super().__init__()
        self.msg = msg
        self.file = file
        self.line = line
        self.column = column

    def __str__(self) -> str:
        str = "Kernel code generation exception at "
        if self.file:
            return str + f'{self.file}:{self.line}: {self.msg}'
        else:
            return str + f"<unknown>:{self.line}:{self.column}: {self.msg}"


class DiagnosticEmitter:
    def __init__(self, filename: str, first_line: int):
        self.filename = filename
        self.first_line = first_line

    def emit_error(self, line: int, column: int, msg: str):
        raise CodeGenerationException(
            self.filename,
            line + self.first_line,
            column,
            msg,
        )

    def emit_warning(self, line: int, column: int, msg: str):
        str = "Kernel code generation warning at "
        if self.filename:
            return str + f'{self.filename}:{line}:{column}: {msg}'
        else:
            return str + f"<unknown>:{line}:{column}: {msg}"

