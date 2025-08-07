"""
File I/O operations for mitochondria and microglia image analysis.

This module handles all file reading/writing operations including:
- Converting ND2 files to TIFF format
- Loading various mask types (microglia, mitochondria, nuclei)
- Saving processed masks
- File path management and validation
"""

import os
import csv
from pathlib import Path
from typing import List, Union, Optional, Tuple
import logging

import numpy as np
import tifffile as tiff
import pandas as pd
from nd2 import ND2File

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def nd2_to_tif(path: Union[str, Path], file_name: str) -> Path:
    """
    Convert ND2 file to TIFF format.
    
    Parameters
    ----------
    path : str or Path
        Directory containing the ND2 file
    file_name : str
        Name of the ND2 file to convert
        
    Returns
    -------
    Path
        Path to the created TIFF file
        
    Raises
    ------
    FileNotFoundError
        If the ND2 file doesn't exist
    """
    path = Path(path)
    nd2_path = path / file_name
    tif_path = nd2_path.with_suffix(".tif")
    
    if not nd2_path.exists():
        raise FileNotFoundError(f"ND2 file not found: {nd2_path}")
    
    try:
        with ND2File(nd2_path) as nd2_file:
            nd2_data = nd2_file.asarray()
            tiff.imwrite(tif_path, nd2_data)
        
        logger.info(f"Converted {nd2_path} to {tif_path}")
        return tif_path
        
    except Exception as e:
        logger.error(f"Failed to convert {nd2_path}: {e}")
        raise


def get_file_paths(directory: Union[str, Path], 
                  pattern: str = "*.tif",
                  sort: bool = True) -> List[str]:
    """
    Get list of file paths matching a pattern in a directory.
    
    Parameters
    ----------
    directory : str or Path
        Directory to search in
    pattern : str, default "*.tif"
        File pattern to match (e.g., "*.tif", "*.npy")
    sort : bool, default True
        Whether to sort the file paths
        
    Returns
    -------
    List[str]
        List of file paths as strings
        
    Raises
    ------
    FileNotFoundError
        If directory doesn't exist
    """
    directory = Path(directory)
    
    if not directory.exists():
        raise FileNotFoundError(f"Directory not found: {directory}")
    
    files = list(directory.glob(pattern))
    
    if sort:
        files = sorted(files)
    
    file_paths = [str(f) for f in files]
    logger.info(f"Found {len(file_paths)} files matching '{pattern}' in {directory}")
    
    return file_paths


def load_masks(mask_directory: Union[str, Path], 
               mask_type: str = "tif",
               sort: bool = True) -> Tuple[List[np.ndarray], List[str]]:
    """
    Load mask files from a directory.
    
    Parameters
    ----------
    mask_directory : str or Path
        Directory containing mask files
    mask_type : str, default "tif"
        Type of mask files ("tif", "npy")
    sort : bool, default True
        Whether to sort files before loading
        
    Returns
    -------
    Tuple[List[np.ndarray], List[str]]
        Tuple of (loaded_masks, file_paths)
        
    Raises
    ------
    ValueError
        If unsupported mask_type is provided
    FileNotFoundError
        If directory doesn't exist
    """
    mask_directory = Path(mask_directory)
    
    if not mask_directory.exists():
        raise FileNotFoundError(f"Mask directory not found: {mask_directory}")
    
    # Get file paths based on mask type
    if mask_type.lower() == "tif":
        pattern = "*.tif"
        load_func = tiff.imread
    elif mask_type.lower() == "npy":
        pattern = "*.npy"
        load_func = np.load
    else:
        raise ValueError(f"Unsupported mask_type: {mask_type}. Use 'tif' or 'npy'")
    
    file_paths = get_file_paths(mask_directory, pattern, sort=sort)
    
    if not file_paths:
        logger.warning(f"No {mask_type} files found in {mask_directory}")
        return [], []
    
    # Load masks
    loaded_masks = []
    valid_paths = []
    
    for file_path in file_paths:
        try:
            mask = load_func(file_path)
            loaded_masks.append(mask)
            valid_paths.append(file_path)
        except Exception as e:
            logger.error(f"Failed to load {file_path}: {e}")
            continue
    
    logger.info(f"Successfully loaded {len(loaded_masks)} {mask_type} masks from {mask_directory}")
    
    return loaded_masks, valid_paths


def load_microglia_masks(mask_directory: Union[str, Path]) -> Tuple[List[np.ndarray], List[str]]:
    """
    Load microglia mask files (TIFF format).
    
    Parameters
    ----------
    mask_directory : str or Path
        Directory containing microglia mask files
        
    Returns
    -------
    Tuple[List[np.ndarray], List[str]]
        Tuple of (loaded_masks, file_paths)
    """
    return load_masks(mask_directory, mask_type="tif")


def load_mitochondria_masks(mask_directory: Union[str, Path]) -> Tuple[List[np.ndarray], List[str]]:
    """
    Load mitochondria mask files (NPY format).
    
    Parameters
    ----------
    mask_directory : str or Path
        Directory containing mitochondria mask files
        
    Returns
    -------
    Tuple[List[np.ndarray], List[str]]
        Tuple of (loaded_masks, file_paths)
    """
    return load_masks(mask_directory, mask_type="npy")


def load_nuclei_masks(mask_directory: Union[str, Path]) -> Tuple[List[np.ndarray], List[str]]:
    """
    Load nuclei mask files (NPY format).
    
    Parameters
    ----------
    mask_directory : str or Path
        Directory containing nuclei mask files
        
    Returns
    -------
    Tuple[List[np.ndarray], List[str]]
        Tuple of (loaded_masks, file_paths)
    """
    return load_masks(mask_directory, mask_type="npy")


def save_mask(mask: np.ndarray, 
              output_path: Union[str, Path],
              mask_format: str = "npy") -> None:
    """
    Save a mask to file.
    
    Parameters
    ----------
    mask : np.ndarray
        Mask array to save
    output_path : str or Path
        Path where to save the mask
    mask_format : str, default "npy"
        Format to save in ("npy", "tif")
        
    Raises
    ------
    ValueError
        If unsupported mask_format is provided
    """
    output_path = Path(output_path)
    
    # Ensure output directory exists
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    if mask_format.lower() == "npy":
        np.save(output_path, mask)
    elif mask_format.lower() == "tif":
        tiff.imwrite(output_path, mask)
    else:
        raise ValueError(f"Unsupported mask_format: {mask_format}. Use 'npy' or 'tif'")
    
    logger.info(f"Saved mask to {output_path}")


def load_properties_csv(csv_path: Union[str, Path]) -> pd.DataFrame:
    """
    Load cell properties from CSV file.
    
    Parameters
    ----------
    csv_path : str or Path
        Path to the CSV file containing cell properties
        
    Returns
    -------
    pd.DataFrame
        DataFrame containing the properties data
        
    Raises
    ------
    FileNotFoundError
        If CSV file doesn't exist
    """
    csv_path = Path(csv_path)
    
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV file not found: {csv_path}")
    
    try:
        df = pd.read_csv(csv_path)
        logger.info(f"Loaded properties from {csv_path}: {len(df)} rows")
        return df
    except Exception as e:
        logger.error(f"Failed to load CSV {csv_path}: {e}")
        raise


def save_results_csv(df: pd.DataFrame, 
                    output_path: Union[str, Path],
                    index: bool = False) -> None:
    """
    Save analysis results to CSV file.
    
    Parameters
    ----------
    df : pd.DataFrame
        Results dataframe to save
    output_path : str or Path
        Path where to save the CSV
    index : bool, default False
        Whether to save the index
    """
    output_path = Path(output_path)
    
    # Ensure output directory exists
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    try:
        df.to_csv(output_path, index=index)
        logger.info(f"Saved results to {output_path}: {len(df)} rows")
    except Exception as e:
        logger.error(f"Failed to save CSV {output_path}: {e}")
        raise


def batch_convert_nd2_to_tif(directory: Union[str, Path]) -> List[Path]:
    """
    Convert all ND2 files in a directory to TIFF format.
    
    Parameters
    ----------
    directory : str or Path
        Directory containing ND2 files
        
    Returns
    -------
    List[Path]
        List of paths to created TIFF files
    """
    directory = Path(directory)
    nd2_files = get_file_paths(directory, "*.nd2")
    
    converted_files = []
    for nd2_file in nd2_files:
        try:
            tif_path = nd2_to_tif(directory, Path(nd2_file).name)
            converted_files.append(tif_path)
        except Exception as e:
            logger.error(f"Failed to convert {nd2_file}: {e}")
            continue
    
    logger.info(f"Converted {len(converted_files)} ND2 files to TIFF")
    return converted_files


def validate_file_correspondence(file_paths_list: List[List[str]], 
                               file_types: List[str]) -> bool:
    """
    Validate that file lists have corresponding files (same base names).
    
    Parameters
    ----------
    file_paths_list : List[List[str]]
        List of file path lists to validate
    file_types : List[str]
        Names of file types for logging
        
    Returns
    -------
    bool
        True if all lists have corresponding files
    """
    if not file_paths_list or len(set(len(paths) for paths in file_paths_list)) > 1:
        logger.error("File lists have different lengths")
        return False
    
    # Extract base names for comparison
    base_names_list = []
    for file_paths in file_paths_list:
        base_names = [Path(f).stem.split('_')[0] for f in file_paths]  # Adjust based on naming convention
        base_names_list.append(base_names)

    # Check if all base names match
    for i, (base_names, file_type) in enumerate(zip(base_names_list, file_types)):
        if i == 0:
            reference_names = set(base_names)
        else:
            current_names = set(base_names)
            if reference_names != current_names:
                logger.error(f"File correspondence mismatch between {file_types[0]} and {file_type}")
                logger.error(f"Missing in {file_type}: {reference_names - current_names}")
                logger.error(f"Extra in {file_type}: {current_names - reference_names}")
                return False

    logger.info(f"File correspondence validated for {len(file_types)} file types")
    return True
