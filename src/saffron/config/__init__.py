"""
Configuration management for saffron package.

This module provides configuration classes for:
- Path management and directory structure
- Analysis parameters and settings  
- Experiment setup and metadata
- Configuration file I/O (YAML/JSON)
"""

from .paths import (
    PathConfig,
    AnalysisConfig, 
    ExperimentConfig,
    create_default_configs,
    load_or_create_config,
)

__all__ = [
    'PathConfig',
    'AnalysisConfig',
    'ExperimentConfig', 
    'create_default_configs',
    'load_or_create_config',
]