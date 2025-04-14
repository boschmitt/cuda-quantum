from dataclasses import dataclass
from enum import Enum, auto
from typing import List, Optional, Tuple, Union, Any
from pathlib import Path
import ast

class DiagnosticSeverity(Enum):
    ERROR = auto()
    WARNING = auto()
    NOTE = auto()
    HELP = auto()

@dataclass
class DiagnosticSpan:
    """Represents a span of code in a file"""
    file: str
    start_line: int
    start_col: int
    end_line: int
    end_col: int
    label: Optional[str] = None
    is_primary: bool = False

    @staticmethod
    def from_ast_node(node: ast.AST, file: str, label: Optional[str] = None, is_primary: bool = False) -> 'DiagnosticSpan':
        """Create a DiagnosticSpan from an AST node.
        
        Args:
            node: The AST node to create the span from
            file: The source file path
            label: Optional label for the span
            is_primary: Whether this is a primary span
            
        Returns:
            A new DiagnosticSpan instance
            
        Note:
            This method requires Python 3.10+ as it uses the lineno/end_lineno
            and col_offset/end_col_offset attributes introduced in that version.
        """
        start_line = getattr(node, 'lineno', 0)
        end_line = getattr(node, 'end_lineno', start_line)
        start_col = getattr(node, 'col_offset', 0)
        end_col = getattr(node, 'end_col_offset', start_col)
        
        return DiagnosticSpan(
            file=file,
            start_line=start_line,
            start_col=start_col,
            end_line=end_line,
            end_col=end_col,
            label=label,
            is_primary=is_primary
        )

@dataclass
class DiagnosticSuggestion:
    """Represents a suggested fix for a diagnostic"""
    message: str
    code: Optional[str] = None
    applicability: Optional[str] = None

class Diagnostic:
    """A class for generating diagnostic messages"""
    
    def __init__(self, severity: DiagnosticSeverity, message: str):
        self.severity = severity
        self.message = message
        self.spans: List[DiagnosticSpan] = []
        self.suggestions: List[DiagnosticSuggestion] = []
        self.notes: List[str] = []
        self.help: List[str] = []

    def add_span(self, span: DiagnosticSpan) -> 'Diagnostic':
        """Add a code span to the diagnostic"""
        self.spans.append(span)
        return self

    def add_suggestion(self, suggestion: DiagnosticSuggestion) -> 'Diagnostic':
        """Add a suggestion to the diagnostic"""
        self.suggestions.append(suggestion)
        return self

    def add_note(self, note: str) -> 'Diagnostic':
        """Add a note to the diagnostic"""
        self.notes.append(note)
        return self

    def add_help(self, help_msg: str) -> 'Diagnostic':
        """Add a help message to the diagnostic"""
        self.help.append(help_msg)
        return self

    def __repr__(self) -> str:
        return (f"Diagnostic(\n"
                f"  severity: {self.severity.name.lower()},\n"
                f"  message: {self.message},\n"
                f"  spans: {self.spans},\n"
                f"  suggestions: {self.suggestions},\n"
                f"  notes: {self.notes},\n"
                f"  help: {self.help}\n)")
