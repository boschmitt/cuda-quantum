# ============================================================================ #
# Copyright (c) 2022 - 2025 NVIDIA Corporation & Affiliates.                   #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

from pathlib import Path
from typing import Dict, List, Optional
from .diagnostic import Diagnostic, DiagnosticSeverity, DiagnosticSpan, DiagnosticSuggestion

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
    def __init__(self, filename: str, first_line: int, source: str | None):
        self.filename = filename
        self.source = source
        self.first_line = first_line
        self.source_lines: Dict[str, List[str]] = {}
        self.source_lines[filename] = source.split('\n') if source else []

    def _get_source_line(self, file: str, line_num: int) -> Optional[str]:
        """Get a specific line from the source file.
        
        Args:
            file: The source file path
            line_num: The 1-based line number to get
            
        Returns:
            The line content if available, None otherwise
        """
        # Convert to 0-based index for list access
        idx = line_num - 1
        
        # If we don't have the file's source yet, try to read it
        if file not in self.source_lines and Path(file).exists():
            try:
                with open(file, 'r') as f:
                    self.source_lines[file] = f.readlines()
            except Exception:
                return None
                
        # Return the line if we have it
        if file in self.source_lines and 0 <= idx < len(self.source_lines[file]):
            return self.source_lines[file][idx].rstrip('\n')
        return None

    def _format_severity(self, severity: DiagnosticSeverity) -> str:
        """Format the severity level"""
        return {
            DiagnosticSeverity.ERROR: "error",
            DiagnosticSeverity.WARNING: "warning",
            DiagnosticSeverity.NOTE: "note",
            DiagnosticSeverity.HELP: "help"
        }[severity]

    def _format_span(self, span: DiagnosticSpan, margin_width: int) -> str:
        """Format a single span"""
        result = []
        
        # Add the file and line information
        result.append(f"  --> {span.file}:{span.start_line}:{span.start_col}")

        # Create a consistent margin for lines without numbers
        margin = " " * margin_width

        # Always add the kernel signature line with proper alignment
        result.append(f"{margin} |")
        
        # Add the code snippet if we have a primary span
        if span.is_primary:
            # Get the source line
            source_line = self._get_source_line(span.file, span.start_line)
            
            # Add the code line with proper alignment
            if source_line is not None:
                line_num = span.start_line + self.first_line - 1
                result.append(f"{str(line_num).rjust(margin_width)} | {source_line}")
            else:
                # Fallback if we can't get the source
                result.append(f"{str(span.start_line).rjust(margin_width)} |")
            
            # Add the underline pointing to the specific column
            underline = "^" * (span.end_col - span.start_col)
            result.append(f"{margin} | {' ' * span.start_col}{underline} {span.label if span.label else ''}")
            
        return "\n".join(result)

    def _format_suggestion(self, suggestion: DiagnosticSuggestion) -> str:
        """Format a suggestion"""
        result = []
        
        # Add the suggestion message
        result.append(f"   = help: {suggestion.message}")
        
        # Add the code if present
        if suggestion.code:
            result.append(f"   = note: {suggestion.code}")
            
        # Add the applicability if present
        if suggestion.applicability:
            result.append(f"   = note: {suggestion.applicability}")
            
        return "\n".join(result)

    def format_diagnostic(self, diagnostic: Diagnostic) -> str:
        """Format the complete diagnostic message"""
        result = []
        
        # Add the severity and message
        result.append(f"{self._format_severity(diagnostic.severity)}: {diagnostic.message}")

        # Calculate margin width based on line numbers in spans with code
        margin_width = 0
        for span in diagnostic.spans:
            if span.is_primary:
                # Account for the span's line number
                line_width = len(str(span.start_line + self.first_line - 1))
                margin_width = max(margin_width, line_width)
        margin_width += 1

        # Add all spans
        for span in diagnostic.spans:
            result.append(self._format_span(span, margin_width))
            
        # Add all suggestions
        for suggestion in diagnostic.suggestions:
            result.append(self._format_suggestion(suggestion))
            
        # Add all notes
        for note in diagnostic.notes:
            result.append(f"note: {note}")
            
        # Add all help messages
        for help_msg in diagnostic.help:
            result.append(f"help: {help_msg}")
            
        return "\n".join(result)

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

    def emit(self, diagnostic: Diagnostic) -> None:
        """Emit a diagnostic message.
        
        Args:
            diagnostic: The Diagnostic object to emit
            
        Raises:
            CodeGenerationException: If the diagnostic severity is ERROR
        """
        # For warnings and other severities, print the formatted diagnostic
        print(self.format_diagnostic(diagnostic))

        # For errors, raise an exception with the first span's information
        if diagnostic.severity == DiagnosticSeverity.ERROR and diagnostic.spans:
            span = diagnostic.spans[0]
            raise CodeGenerationException(
                span.file,
                span.start_line,
                span.start_col,
                diagnostic.message
            )

