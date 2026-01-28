import numpy as np
from pathlib import Path
from saffron.io import data_io
from saffron.data import datasets, data_processing
from saffron.models import torch_models
import matplotlib.pyplot as plt
import numpy as np
import cv2
from typing import List

def to_grayscale(image: np.ndarray, channel: int = 0) -> np.ndarray:
    """
    Convert image to 2D grayscale regardless of input dimensions.
    
    Handles:
    - 2D: (H, W) -> return as-is
    - 3D: (C, H, W) or (H, W, C) -> extract channel
    - 4D: (C, Z, H, W) or (Z, H, W, C) -> max projection + extract channel
    """
    # Case 1: Already 2D grayscale
    if image.ndim == 2:
        return image
    
    # Case 2: 3D with channels first (C, H, W)
    elif image.ndim == 3:
        if image.shape[0] in [1, 3, 4] and image.shape[0] < image.shape[1]:
            return image[channel]
        # Channels last (H, W, C)
        elif image.shape[2] in [1, 3, 4] and image.shape[2] < image.shape[0]:
            return image[:, :, channel]
        # Z-stack without channel dimension (Z, H, W) - max project
        else:
            return np.max(image, axis=0)
    
    # Case 3: 4D with channels first (C, Z, H, W) - max project Z
    elif image.ndim == 4:
        if image.shape[0] in [1, 3, 4] and image.shape[0] < image.shape[1]:
            max_proj = np.max(image[channel], axis=0)
            return max_proj
        # Channels last (Z, H, W, C) - max project Z
        elif image.shape[3] in [1, 3, 4] and image.shape[3] < image.shape[0]:
            max_proj = np.max(image[:, :, :, channel], axis=0)
            return max_proj
    
    # Unsupported shape
    raise ValueError(f"Cannot determine channel dimension for shape {image.shape}")


def split_into_quadrants(image: np.ndarray) -> list:
    """
    Split a 2D image into 4 quadrants.
    
    Returns list of dicts with keys: 'image', 'quadrant', 'position'
    Quadrants: 'TL' (top-left), 'TR' (top-right), 'BL' (bottom-left), 'BR' (bottom-right)
    """
    if image.ndim != 2:
        raise ValueError(f"Expected 2D image, got shape {image.shape}")
    
    h, w = image.shape
    mid_h = h // 2
    mid_w = w // 2
    
    quadrants = []
    
    # Top-left
    quadrants.append({
        'image': image[:mid_h, :mid_w],
        'quadrant': 'TL',
        'position': (0, 0)
    })
    
    # Top-right
    quadrants.append({
        'image': image[:mid_h, mid_w:],
        'quadrant': 'TR',
        'position': (0, mid_w)
    })
    
    # Bottom-left
    quadrants.append({
        'image': image[mid_h:, :mid_w],
        'quadrant': 'BL',
        'position': (mid_h, 0)
    })
    
    # Bottom-right
    quadrants.append({
        'image': image[mid_h:, mid_w:],
        'quadrant': 'BR',
        'position': (mid_h, mid_w)
    })
    
    return quadrants

def normalize_to_float32(image: np.ndarray) -> np.ndarray:
    """
    Normalize image to float32 in range [0, 1].
    
    Handles uint8, uint16, and float inputs.
    """
    # Convert to float32
    image_float = image.astype(np.float32)
    
    # Min-max normalization to [0, 1]
    img_min = image_float.min()
    img_max = image_float.max()
    
    if img_max - img_min == 0:
        return np.zeros_like(image_float)
    
    normalized = (image_float - img_min) / (img_max - img_min)
    return normalized

def resize_image(image: np.ndarray, target_size: tuple = (512, 512)) -> np.ndarray:
    """
    Resize image to target size.
    
    Parameters
    ----------
    image : np.ndarray
        Input 2D image
    target_size : tuple
        Target (height, width)
    """
    if image.ndim != 2:
        raise ValueError(f"Expected 2D image, got shape {image.shape}")
    
    # Already correct size
    if image.shape == target_size:
        return image
    
    # Resize using bilinear interpolation
    # cv2.resize expects (width, height) not (height, width)
    resized = cv2.resize(image, (target_size[1], target_size[0]), interpolation=cv2.INTER_LINEAR)
    return resized

def apply_clahe(image: np.ndarray, clip_limit: float = 2.0, tile_grid_size: tuple = (8, 8)) -> np.ndarray:
    """
    Apply Contrast Limited Adaptive Histogram Equalization.
    
    Parameters
    ----------
    image : np.ndarray
        Input image (should be in 0-1 range, float32)
    clip_limit : float
        Threshold for contrast limiting
    tile_grid_size : tuple
        Size of grid for histogram equalization
    """
    # CLAHE requires uint8, so convert from [0, 1] float to [0, 255] uint8
    image_uint8 = (image * 255).astype(np.uint8)
    
    # Create CLAHE object
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    
    # Apply CLAHE
    enhanced = clahe.apply(image_uint8)
    
    # Convert back to float32 [0, 1]
    enhanced_float = enhanced.astype(np.float32) / 255.0
    return enhanced_float

def preprocess_image(image: np.ndarray, 
                     channel: int = 0,
                     target_size: tuple = (512, 512),
                     use_clahe: bool = True,
                     split_quadrants: bool = False) -> list:
    """
    Complete preprocessing pipeline for a single image.
    
    Returns list of dicts with keys: 'image', 'quadrant' (if split), 'position' (if split)
    If not split, returns list with single dict containing 'image'.
    """
    # Step 1: Convert to grayscale 2D

    gray = to_grayscale(image, channel=channel)
    
    # Step 2: Normalize to [0, 1]
    normalized = normalize_to_float32(gray)
    
    # Step 3: Resize to target size
    resized = resize_image(normalized, target_size=target_size)
    
    # Step 4: Apply CLAHE if requested
    if use_clahe:
        processed = apply_clahe(resized)
    else:
        processed = resized
    
    # Step 5: Split into quadrants if requested
    if split_quadrants:
        return split_into_quadrants(processed)
    else:
        return [{'image': processed, 'quadrant': None, 'position': None}]


def plot_examples(processed_list: list, n_examples: int = 4, save_path: str = None):
    """
    Plot a few examples from preprocessed output.
    
    Parameters
    ----------
    processed_list : list
        Output from preprocess_image()
    n_examples : int
        Number of examples to show (default: 4)
    save_path : str, optional
        Path to save figure
    """
    n_show = min(n_examples, len(processed_list))
    
    fig, axes = plt.subplots(1, n_show, figsize=(4 * n_show, 4))
    if n_show == 1:
        axes = [axes]
    
    for i in range(n_show):
        axes[i].imshow(processed_list[i]['image'], cmap='gray', vmin=0, vmax=1)
        title = f"{processed_list[i]['quadrant']}" if processed_list[i]['quadrant'] else f"Image {i}"
        axes[i].set_title(title)
        axes[i].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved to {save_path}")
    
    plt.show()

def save_processed_images(processed_list: list, output_dir: str, base_filename: str):
    """
    Save each processed image/quadrant as a separate .npy file.
    
    Parameters
    ----------
    processed_list : list
        Output from preprocess_image()
    output_dir : str
        Directory to save files
    base_filename : str
        Base name for files (e.g., 'image_01')
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    saved_files = []
    
    for item in processed_list:
        # Create filename
        if item['quadrant']:
            filename = f"{base_filename}_{item['quadrant']}.npy"
        else:
            filename = f"{base_filename}.npy"
        
        filepath = output_path / filename
        
        # Save as .npy
        np.save(filepath, item['image'])
        saved_files.append(str(filepath))
    
    print(f"Saved {len(saved_files)} files to {output_dir}")
    return saved_files

def preprocess_directory(input_dir: str, 
                         output_dir: str,
                         channel: int = 0,
                         target_size: tuple = (512, 512),
                         use_clahe: bool = True,
                         split_quadrants: bool = False,
                         file_extensions: list = ['.tif', '.tiff']) -> List[str]:
    """
    Preprocess all images in a directory and save results.
    
    Returns list of all saved file paths.
    """
    import tifffile
    
    input_path = Path(input_dir)
    all_saved_files = []
    
    # Find all image files
    image_files = []
    for ext in file_extensions:
        image_files.extend(input_path.glob(f"*{ext}"))
    
    print(f"Found {len(image_files)} images to process")
    
    # Process each image
    for img_file in image_files:
        try:
            # Load image
            image = tifffile.imread(str(img_file))
            
            # Get base filename (without extension)
            base_name = img_file.stem
            
            # Preprocess
            result = preprocess_image(image, channel, target_size, use_clahe, split_quadrants)
            
            # Save results
            saved = save_processed_images(result, output_dir, base_name)
            all_saved_files.extend(saved)
            
        except Exception as e:
            print(f"Error processing {img_file}: {e}")
            continue
    
    print(f"\nProcessed {len(image_files)} images -> {len(all_saved_files)} output files")
    return all_saved_files