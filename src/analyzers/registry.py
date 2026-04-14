"""Analyzer registry.

This class currently aliases the existing model registry so migration can
move callers from model-centric naming to analyzer-centric naming.
"""

from ..models.registry import ModelRegistry


class AnalyzerRegistry(ModelRegistry):
    """Drop-in analyzer registry backed by the current model implementations."""
