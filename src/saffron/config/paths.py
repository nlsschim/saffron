"""
Configuration management for saffron image analysis package.

This module provides configuration classes and utilities for managing:
- File paths and directory structure
- Analysis parameters
- Experimental conditions
- Output locations
"""

import os
from pathlib import Path
from typing import Dict, List, Optional, Union, Any
from dataclasses import dataclass, field
import json
import yaml
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


@dataclass
class PathConfig:
    """
    Configuration class for managing file paths and directory structure.
    
    This replaces all the hardcoded paths from the original mito_threshold.py script.
    """
    
    # Base directories
    base_data_dir: Union[str, Path] = "/Users/nelsschimek/Documents/nancelab/Data/mito_images/brendan_full_analysis"
    condition: str = "OGD_only"
    cell_type: str = "cd11b" 
    threshold_method: str = "li_thresh"
    
    # Derived paths (automatically calculated)
    condition_dir: Path = field(init=False)
    tifs_dir: Path = field(init=False)
    converted_tiffs_dir: Path = field(init=False)
    
    # Mask directories  
    microglia_masks_dir: Path = field(init=False)
    mitochondria_masks_dir: Path = field(init=False)
    nuclei_masks_dir: Path = field(init=False)
    
    # External data paths
    properties_csv_path: Union[str, Path] = "/Users/nelsschimek/Downloads/All_Properties.csv"
    vampire_model_path: Union[str, Path] = "/Users/nelsschimek/Documents/nancelab/Data/mito_images/brendan_full_analysis/tifs/training/vampire_data/model_li_(50_5_39)__.pickle"
    
    def __post_init__(self):
        """Calculate derived paths after initialization."""
        self.base_data_dir = Path(self.base_data_dir)
        
        # Build directory structure
        self.condition_dir = self.base_data_dir / "tifs" / self.condition / self.cell_type
        self.tifs_dir = self.condition_dir / self.threshold_method
        self.converted_tiffs_dir = self.tifs_dir / "converted_tiffs"
        
        # Mask directories
        self.microglia_masks_dir = self.converted_tiffs_dir / "microglia_masks"
        self.mitochondria_masks_dir = self.converted_tiffs_dir / "mitochondria_masks"  
        self.nuclei_masks_dir = self.converted_tiffs_dir / "nuclei_masks"
    
    def create_directories(self) -> None:
        """Create all necessary directories if they don't exist."""
        directories = [
            self.condition_dir,
            self.tifs_dir, 
            self.converted_tiffs_dir,
            self.microglia_masks_dir,
            self.mitochondria_masks_dir,
            self.nuclei_masks_dir
        ]
        
        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)
            logger.info(f"Ensured directory exists: {directory}")
    
    def get_output_path(self, filename: str, subdirectory: Optional[str] = None) -> Path:
        """
        Get standardized output path for results.
        
        Parameters
        ----------
        filename : str
            Name of the output file
        subdirectory : str, optional
            Subdirectory within the condition directory
            
        Returns
        -------
        Path
            Full path for the output file
        """
        base_output = self.condition_dir
        
        if subdirectory:
            base_output = base_output / subdirectory
            base_output.mkdir(parents=True, exist_ok=True)
        
        return base_output / filename
    
    def validate_paths(self) -> bool:
        """
        Validate that all required paths exist.
        
        Returns
        -------
        bool
            True if all paths are valid
        """
        required_paths = [
            self.base_data_dir,
            Path(self.properties_csv_path),
            Path(self.vampire_model_path)
        ]
        
        optional_paths = [
            self.microglia_masks_dir,
            self.mitochondria_masks_dir, 
            self.nuclei_masks_dir
        ]
        
        # Check required paths
        for path in required_paths:
            if not path.exists():
                logger.error(f"Required path does not exist: {path}")
                return False
        
        # Warn about missing optional paths
        for path in optional_paths:
            if not path.exists():
                logger.warning(f"Optional path does not exist (will be created): {path}")
        
        logger.info("Path validation completed successfully")
        return True
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return {
            'base_data_dir': str(self.base_data_dir),
            'condition': self.condition,
            'cell_type': self.cell_type,
            'threshold_method': self.threshold_method,
            'properties_csv_path': str(self.properties_csv_path),
            'vampire_model_path': str(self.vampire_model_path)
        }
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'PathConfig':
        """Create configuration from dictionary."""
        return cls(**config_dict)


@dataclass  
class AnalysisConfig:
    """
    Configuration class for analysis parameters.
    
    This centralizes all the magic numbers and parameters from the original script.
    """
    
    # Distance analysis parameters
    max_distance: float = 37.0  # Maximum distance for mito-nuclei pairing
    
    # Mask creation parameters
    mitochondria_percentile: float = 99.0  # Percentile threshold for mito mask
    mitochondria_min_size: int = 10  # Minimum object size for mitochondria
    microglia_min_size_large: int = 8590  # Minimum size for large microglia objects
    microglia_min_size_small: int = 71  # Minimum size for small microglia objects
    
    # Image processing parameters
    nuclei_count_offset: int = 0  # Offset for nuclei labeling
    image_shape: tuple = (1024, 1024)  # Standard image dimensions
    
    # VAMPIRE model parameters
    vampire_cluster_params: str = "(50_5_39)"  # VAMPIRE clustering parameters
    
    # Visualization parameters
    plot_figsize: tuple = (10, 4)  # Default figure size
    plot_alpha: float = 0.1  # Transparency for distance plots
    circle_alpha: float = 0.5  # Transparency for distance circles
    
    # Colors for overlays (RGB)
    nuclei_color: List[int] = field(default_factory=lambda: [0, 0, 255])  # Blue
    microglia_color: List[int] = field(default_factory=lambda: [0, 255, 0])  # Green  
    mitochondria_color: List[int] = field(default_factory=lambda: [255, 0, 255])  # Magenta
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return {
            'max_distance': self.max_distance,
            'mitochondria_percentile': self.mitochondria_percentile,
            'mitochondria_min_size': self.mitochondria_min_size,
            'microglia_min_size_large': self.microglia_min_size_large,
            'microglia_min_size_small': self.microglia_min_size_small,
            'nuclei_count_offset': self.nuclei_count_offset,
            'image_shape': self.image_shape,
            'vampire_cluster_params': self.vampire_cluster_params,
            'plot_figsize': self.plot_figsize,
            'plot_alpha': self.plot_alpha,
            'circle_alpha': self.circle_alpha,
            'nuclei_color': self.nuclei_color,
            'microglia_color': self.microglia_color,
            'mitochondria_color': self.mitochondria_color
        }
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'AnalysisConfig':
        """Create configuration from dictionary."""
        return cls(**config_dict)


@dataclass
class ExperimentConfig:
    """
    Complete experiment configuration combining paths and analysis parameters.
    """
    
    paths: PathConfig = field(default_factory=PathConfig)
    analysis: AnalysisConfig = field(default_factory=AnalysisConfig)
    
    # Experiment metadata
    experiment_name: str = "mitochondria_analysis"
    researcher: str = "Unknown"
    date_created: str = field(default_factory=lambda: datetime.now().isoformat())
    description: str = "Automated mitochondria and microglia spatial analysis"
    
    def save_config(self, filepath: Union[str, Path], format: str = "yaml") -> None:
        """
        Save configuration to file.
        
        Parameters
        ----------
        filepath : str or Path
            Where to save the configuration
        format : str, default "yaml"  
            Format to save in ("yaml", "json")
        """
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        config_data = {
            'experiment_name': self.experiment_name,
            'researcher': self.researcher, 
            'date_created': self.date_created,
            'description': self.description,
            'paths': self.paths.to_dict(),
            'analysis': self.analysis.to_dict()
        }
        
        if format.lower() == "yaml":
            with open(filepath, 'w') as f:
                yaml.safe_dump(config_data, f, default_flow_style=False)
        elif format.lower() == "json":
            with open(filepath, 'w') as f:
                json.dump(config_data, f, indent=2)
        else:
            raise ValueError(f"Unsupported format: {format}. Use 'yaml' or 'json'")
        
        logger.info(f"Configuration saved to {filepath}")
    
    @classmethod
    def load_config(cls, filepath: Union[str, Path]) -> 'ExperimentConfig':
        """
        Load configuration from file.
        
        Parameters
        ----------
        filepath : str or Path
            Path to configuration file
            
        Returns
        -------
        ExperimentConfig
            Loaded configuration
        """
        filepath = Path(filepath)
        
        if not filepath.exists():
            raise FileNotFoundError(f"Configuration file not found: {filepath}")
        
        # Determine format from extension
        if filepath.suffix.lower() in ['.yml', '.yaml']:
            with open(filepath, 'r') as f:
                config_data = yaml.safe_load(f)
        elif filepath.suffix.lower() == '.json':
            with open(filepath, 'r') as f:
                config_data = json.load(f)
        else:
            raise ValueError(f"Unsupported file format: {filepath.suffix}")
        
        # Extract components
        paths_data = config_data.pop('paths', {})
        analysis_data = config_data.pop('analysis', {})
        
        # Create configuration
        config = cls(
            paths=PathConfig.from_dict(paths_data),
            analysis=AnalysisConfig.from_dict(analysis_data),
            **config_data
        )
        
        logger.info(f"Configuration loaded from {filepath}")
        return config
    
    def setup_experiment(self) -> None:
        """Set up experiment by creating directories and validating paths."""
        logger.info(f"Setting up experiment: {self.experiment_name}")
        
        # Create directories
        self.paths.create_directories()
        
        # Validate paths  
        if not self.paths.validate_paths():
            raise RuntimeError("Path validation failed")
        
        logger.info("Experiment setup completed successfully")


def create_default_configs() -> Dict[str, ExperimentConfig]:
    """
    Create default configurations for common experimental conditions.
    
    Returns
    -------
    Dict[str, ExperimentConfig]
        Dictionary mapping condition names to configurations
    """
    conditions = ["OGD_only", "HC", "control"]
    configs = {}
    
    for condition in conditions:
        config = ExperimentConfig(
            paths=PathConfig(condition=condition),
            experiment_name=f"mitochondria_analysis_{condition}",
            description=f"Mitochondria and microglia analysis for {condition} condition"
        )
        configs[condition] = config
    
    return configs


def load_or_create_config(config_path: Union[str, Path], 
                         condition: str = "OGD_only") -> ExperimentConfig:
    """
    Load configuration from file, or create default if file doesn't exist.
    
    Parameters
    ----------
    config_path : str or Path
        Path to configuration file
    condition : str, default "OGD_only"
        Experimental condition for default config
        
    Returns
    -------
    ExperimentConfig
        Loaded or default configuration
    """
    config_path = Path(config_path)
    
    if config_path.exists():
        logger.info(f"Loading existing configuration from {config_path}")
        return ExperimentConfig.load_config(config_path)
    else:
        logger.info(f"Creating default configuration for condition: {condition}")
        config = ExperimentConfig(
            paths=PathConfig(condition=condition),
            experiment_name=f"mitochondria_analysis_{condition}"
        )
        
        # Save the default configuration
        config.save_config(config_path)
        return config