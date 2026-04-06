"""
LLM-as-annotators for digital agents.
"""

from .version import __version__
from . import prompts, utils

__all__ = [
    "__version__",
    # Add your public API exports here
    "prompts",
    "utils",
]
