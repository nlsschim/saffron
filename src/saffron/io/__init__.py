"""
I/O operations for saffron image analysis package.

This module provides functions for:
- File format conversions (ND2 to TIFF)
- Loading and saving masks in various formats
- CSV data management
- File path utilities and validation
"""

from .file_io import (
    # File conversion
    nd2_to_tif,
    batch_convert_nd2_to_tif,
    
    # File path utilities
    get_file_paths,
    validate_file_correspondence,
    
    # Mask loading/saving
    load_masks,
    load_microglia_masks,
    load_mitochondria_masks, 
    load_nuclei_masks,
    save_mask,
    
    # CSV operations
    load_properties_csv,
    save_results_csv,
)

__all__ = [
    # File conversion
    'nd2_to_tif',
    'batch_convert_nd2_to_tif',
    
    # File path utilities
    'get_file_paths',
    'validate_file_correspondence',
    
    # Mask loading/saving
    'load_masks',
    'load_microglia_masks',
    'load_mitochondria_masks',
    'load_nuclei_masks', 
    'save_mask',
    
    # CSV operations
    'load_properties_csv',
    'save_results_csv',
]