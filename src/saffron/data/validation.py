"""
Shape validation utilities for ensuring consistent image dimensions
throughout the saffron pipeline.
"""

import numpy as np
from typing import List, Dict, Tuple, Optional
import logging

logger = logging.getLogger(__name__)


def validate_image_shapes(
    images: List,
    expected_ndim: int = 2,
    expected_shape: Optional[Tuple[int, ...]] = None,
    raise_on_error: bool = True
) -> Dict[str, any]:
    """
    Validate that all images in a list have consistent shapes.
    
    This function checks images after channel extraction and max projection
    to ensure they're ready for patch extraction. It's critical for catching
    shape mismatches before they cause tensor errors in DataLoaders.
    
    Parameters
    ----------
    images : List[ImageData] or List[np.ndarray]
        List of images to validate. Can be ImageData objects or numpy arrays.
    expected_ndim : int, default 2
        Expected number of dimensions (2 for grayscale patches, 3 for RGB)
    expected_shape : Tuple[int, ...], optional
        If provided, all images must match this exact shape
    raise_on_error : bool, default True
        If True, raise ValueError on validation failure
        If False, return validation results without raising
        
    Returns
    -------
    Dict[str, any]
        Dictionary containing:
        - 'valid': bool, whether all images pass validation
        - 'num_images': int, total number of images checked
        - 'shape_distribution': dict mapping shapes to counts
        - 'invalid_indices': list of indices with wrong shapes
        - 'invalid_files': list of file paths with issues (if available)
        - 'error_messages': list of specific error messages
        
    Raises
    ------
    ValueError
        If raise_on_error=True and validation fails
        
    Examples
    --------
    >>> # After channel extraction and max projection
    >>> images = load_images_from_directory("path/to/images")
    >>> dataset = ImageDataset(images)
    >>> dataset.extract_single_channel(channel=0)
    >>> dataset.apply_max_projection()
    >>> 
    >>> # Validate before creating patches
    >>> results = validate_image_shapes(
    ...     dataset.images, 
    ...     expected_ndim=2,
    ...     raise_on_error=True
    ... )
    >>> print(f"All {results['num_images']} images validated successfully!")
    """
    
    # Initialize results
    results = {
        'valid': True,
        'num_images': len(images),
        'shape_distribution': {},
        'invalid_indices': [],
        'invalid_files': [],
        'error_messages': []
    }
    
    if len(images) == 0:
        results['valid'] = False
        results['error_messages'].append("No images provided for validation")
        if raise_on_error:
            raise ValueError("No images provided for validation")
        return results
    
    # Extract arrays and file paths from ImageData objects if needed
    arrays = []
    file_paths = []
    
    for i, img in enumerate(images):
        if hasattr(img, 'data'):  # ImageData object
            arrays.append(img.data)
            file_paths.append(getattr(img, 'file_path', f'image_{i}'))
        else:  # Already numpy array
            arrays.append(img)
            file_paths.append(f'image_{i}')
    
    # Check each image
    for i, (arr, file_path) in enumerate(zip(arrays, file_paths)):
        
        # Track shape distribution
        shape_key = str(arr.shape)
        results['shape_distribution'][shape_key] = \
            results['shape_distribution'].get(shape_key, 0) + 1
        
        # Check dimensionality
        if arr.ndim != expected_ndim:
            results['valid'] = False
            results['invalid_indices'].append(i)
            results['invalid_files'].append(file_path)
            error_msg = (
                f"Image {i} ({file_path}): "
                f"Expected {expected_ndim}D, got {arr.ndim}D with shape {arr.shape}"
            )
            results['error_messages'].append(error_msg)
            logger.error(error_msg)
            continue

        # Check exact shape if specified
        if expected_shape is not None:
            if arr.shape != expected_shape:
                results['valid'] = False
                results['invalid_indices'].append(i)
                results['invalid_files'].append(file_path)
                error_msg = (
                    f"Image {i} ({file_path}): "
                    f"Expected shape {expected_shape}, got {arr.shape}"
                )
                results['error_messages'].append(error_msg)
                logger.error(error_msg)

    # Log summary
    if results['valid']:
        logger.info(
            f"✓ Shape validation passed for {results['num_images']} images "
            f"(all {expected_ndim}D)"
        )
        if expected_shape:
            logger.info(f"  All images have shape: {expected_shape}")
    else:
        logger.error(
            f"✗ Shape validation failed for {len(results['invalid_indices'])} "
            f"out of {results['num_images']} images"
        )
        logger.error(f"  Shape distribution: {results['shape_distribution']}")

    # Raise error if requested
    if not results['valid'] and raise_on_error:
        error_summary = (
            f"\nShape validation failed!\n"
            f"  Total images: {results['num_images']}\n"
            f"  Invalid images: {len(results['invalid_indices'])}\n"
            f"  Expected: {expected_ndim}D"
        )
        if expected_shape:
            error_summary += f" with shape {expected_shape}"
        error_summary += f"\n  Shape distribution found:\n"
        for shape, count in results['shape_distribution'].items():
            error_summary += f"    {shape}: {count} images\n"
        error_summary += f"\n  First 5 errors:\n"
        for msg in results['error_messages'][:5]:
            error_summary += f"    - {msg}\n"

        raise ValueError(error_summary)

    return results
