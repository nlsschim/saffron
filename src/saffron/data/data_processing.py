import numpy as np
from typing import List, Tuple, Dict, Optional, Union
from dataclasses import dataclass
import random
from pathlib import Path
import logging
from collections import defaultdict
import cv2
from sklearn.model_selection import train_test_split as sklearn_split

# Assuming the ImageData class from the previous module
from saffron.io.data_io import ImageData

logger = logging.getLogger(__name__)

@dataclass
class DataSplit:
    """Container for train/test/validation splits."""
    train: List[ImageData]
    test: List[ImageData]
    val: List[ImageData]
    
    def __len__(self):
        return len(self.train) + len(self.test) + len(self.val)
    
    def get_split_info(self) -> Dict[str, int]:
        return {
            'train': len(self.train),
            'test': len(self.test),
            'val': len(self.val),
            'total': len(self)
        }


@dataclass
class PatchPair:
    """Container for patch pairs used in contrastive learning."""
    masked_image: np.ndarray
    candidate_patch: np.ndarray
    is_correct: bool
    original_position: Tuple[int, int]  # (row, col) of patch in original image
    patch_size: int
    source_image_path: str


def extract_metadata_from_path(file_path: str) -> Dict[str, str]:
    """
    Extract metadata from file path (animal ID, slice number, condition, etc.)
    Assumes naming convention like: animal01_slice05_condition_A.tif
    
    Args:
        file_path: Path to the image file
        
    Returns:
        Dictionary with extracted metadata
    """
    path = Path(file_path)
    filename = path.stem
    
    metadata = {}
    parts = filename.split('_')
    
    for part in parts:
        if part.startswith('animal'):
            metadata['animal'] = part
        elif part.startswith('slice'):
            metadata['slice'] = part
        elif 'condition' in part.lower():
            metadata['condition'] = part
        elif part.startswith('rep'):
            metadata['replicate'] = part

    return metadata


def train_test_split(images: List[ImageData],
                     test_size: float = 0.2,
                     val_size: float = 0.1,
                     split_criteria: str = "random",
                     random_state: int = 42) -> DataSplit:
    """
    Method to split images into training, testing, and validation based on specified criteria.

    Args:
        images: List of ImageData objects
        test_size: Proportion for test set
        val_size: Proportion for validation set  
        split_criteria: "random", "by_slice", or "by_animal"
        random_state: Random seed for reproducibility
        
    Returns:
        DataSplit object containing train/test/val splits
    """
    if not images:
        raise ValueError("No images provided for splitting")
    
    np.random.seed(random_state)
    random.seed(random_state)
    
    if split_criteria == "random":
        return _random_split(images, test_size, val_size, random_state)
    
    elif split_criteria == "by_slice":
        return _split_by_slice(images, test_size, val_size, random_state)
    
    elif split_criteria == "by_animal":
        return _split_by_animal(images, test_size, val_size, random_state)
    
    else:
        raise ValueError(f"Unknown split criteria: {split_criteria}")


def _random_split(images: List[ImageData], test_size: float, val_size: float, 
                 random_state: int) -> DataSplit:
    """Random split implementation."""
    # First split: separate out test set
    train_val, test = sklearn_split(images, test_size=test_size, random_state=random_state)
    
    # Second split: separate train and validation from remaining data
    val_size_adjusted = val_size / (1 - test_size)  # Adjust val_size for remaining data
    train, val = sklearn_split(train_val, test_size=val_size_adjusted, random_state=random_state)
    
    logger.info(f"Random split: Train={len(train)}, Val={len(val)}, Test={len(test)}")
    return DataSplit(train=train, test=test, val=val)


def _split_by_slice(images: List[ImageData], test_size: float, val_size: float,
                   random_state: int) -> DataSplit:
    """Split by slice to ensure no slice appears in multiple sets."""
    # Group images by slice
    slice_groups = defaultdict(list)
    for img in images:
        metadata = extract_metadata_from_path(img.file_path)
        slice_id = metadata.get('slice', 'unknown_slice')
        slice_groups[slice_id].append(img)
    
    # Split slices into sets
    slice_ids = list(slice_groups.keys())
    random.shuffle(slice_ids)
    
    n_test = max(1, int(len(slice_ids) * test_size))
    n_val = max(1, int(len(slice_ids) * val_size))
    
    test_slices = slice_ids[:n_test]
    val_slices = slice_ids[n_test:n_test + n_val]
    train_slices = slice_ids[n_test + n_val:]
    
    # Collect images for each set
    train = [img for slice_id in train_slices for img in slice_groups[slice_id]]
    val = [img for slice_id in val_slices for img in slice_groups[slice_id]]
    test = [img for slice_id in test_slices for img in slice_groups[slice_id]]
    
    logger.info(f"Slice-based split: Train={len(train)}, Val={len(val)}, Test={len(test)}")
    logger.info(f"Train slices: {train_slices}")
    logger.info(f"Val slices: {val_slices}")
    logger.info(f"Test slices: {test_slices}")
    
    return DataSplit(train=train, test=test, val=val)


def _split_by_animal(images: List[ImageData], test_size: float, val_size: float,
                    random_state: int) -> DataSplit:
    """Split by animal to ensure no animal appears in multiple sets."""
    # Group images by animal
    animal_groups = defaultdict(list)
    for img in images:
        metadata = extract_metadata_from_path(img.file_path)
        animal_id = metadata.get('animal', 'unknown_animal')
        animal_groups[animal_id].append(img)
    
    # Check for condition balance if conditions exist
    condition_balance = _check_condition_balance(animal_groups)
    if condition_balance:
        logger.info(f"Condition distribution: {condition_balance}")
    
    # Split animals into sets
    animal_ids = list(animal_groups.keys())
    random.shuffle(animal_ids)
    
    n_test = max(1, int(len(animal_ids) * test_size))
    n_val = max(1, int(len(animal_ids) * val_size))
    
    test_animals = animal_ids[:n_test]
    val_animals = animal_ids[n_test:n_test + n_val]
    train_animals = animal_ids[n_test + n_val:]
    
    # Collect images for each set
    train = [img for animal_id in train_animals for img in animal_groups[animal_id]]
    val = [img for animal_id in val_animals for img in animal_groups[animal_id]]
    test = [img for animal_id in test_animals for img in animal_groups[animal_id]]
    
    logger.info(f"Animal-based split: Train={len(train)}, Val={len(val)}, Test={len(test)}")
    logger.info(f"Train animals: {train_animals}")
    logger.info(f"Val animals: {val_animals}")
    logger.info(f"Test animals: {test_animals}")
    
    return DataSplit(train=train, test=test, val=val)


def _check_condition_balance(animal_groups: Dict[str, List[ImageData]]) -> Dict[str, int]:
    """Check condition distribution across animals."""
    condition_counts = defaultdict(int)
    for animal_id, images in animal_groups.items():
        if images:
            metadata = extract_metadata_from_path(images[0].file_path)
            condition = metadata.get('condition', 'unknown')
            condition_counts[condition] += 1
    return dict(condition_counts)


def create_masked_image(image: np.ndarray, patch_position: Tuple[int, int], 
                       patch_size: int, mask_value: float = 0.0) -> np.ndarray:
    """
    Create a masked version of the image by removing a patch.
    
    Args:
        image: Original image
        patch_position: (row, col) position of top-left corner of patch
        patch_size: Size of the square patch to remove
        mask_value: Value to fill the masked region with
        
    Returns:
        Masked image
    """
    masked = image.copy()
    row, col = patch_position
    
    # Ensure patch doesn't go outside image bounds
    end_row = min(row + patch_size, image.shape[0])
    end_col = min(col + patch_size, image.shape[1])
    
    masked[row:end_row, col:end_col] = mask_value
    return masked


def extract_patch(image: np.ndarray, patch_position: Tuple[int, int], 
                 patch_size: int) -> np.ndarray:
    """
    Extract a patch from the image.
    
    Args:
        image: Source image
        patch_position: (row, col) position of top-left corner
        patch_size: Size of the square patch
        
    Returns:
        Extracted patch
    """
    row, col = patch_position
    end_row = min(row + patch_size, image.shape[0])
    end_col = min(col + patch_size, image.shape[1])
    
    patch = image[row:end_row, col:end_col]
    
    # Pad if patch is smaller than requested size (edge case)
    if patch.shape[0] < patch_size or patch.shape[1] < patch_size:
        padded = np.zeros((patch_size, patch_size), dtype=image.dtype)
        padded[:patch.shape[0], :patch.shape[1]] = patch
        return padded
    
    return patch


def generate_random_patch_positions(image_shape: Tuple[int, ...], patch_size: int,
                                    num_positions: int, min_distance: int = 0) -> List[Tuple[int, int]]:
    """
    Generate random patch positions ensuring they fit within image bounds.

    Args:
        image_shape: Shape of the image
        patch_size: Size of patches
        num_positions: Number of positions to generate
        min_distance: Minimum distance between patches

    Returns:
        List of (row, col) positions
    """
    height, width = image_shape[:2]
    max_row = height - patch_size
    max_col = width - patch_size
    
    if max_row <= 0 or max_col <= 0:
        raise ValueError(f"Image too small for patch size {patch_size}")
    
    positions = []
    attempts = 0
    max_attempts = num_positions * 100
    
    while len(positions) < num_positions and attempts < max_attempts:
        row = random.randint(0, max_row)
        col = random.randint(0, max_col)
        
        # Check minimum distance constraint
        if min_distance > 0:
            valid = True
            for existing_row, existing_col in positions:
                distance = np.sqrt((row - existing_row)**2 + (col - existing_col)**2)
                if distance < min_distance:
                    valid = False
                    break
            if not valid:
                attempts += 1
                continue
        
        positions.append((row, col))
        attempts += 1
    
    return positions


def create_positive_pairs(images: List[ImageData], patch_size: int = 64,
                          patches_per_image: int = 5) -> List[PatchPair]:
    """
    Create positive pairs for contrastive learning.
    Each positive pair consists of a masked image and its correct patch.
    
    Args:
        images: List of ImageData objects
        patch_size: Size of patches to extract
        patches_per_image: Number of patches per image
        
    Returns:
        List of positive PatchPair objects
    """
    positive_pairs = []
    
    for img_data in images:
        image = img_data.data
        
        # Handle different image dimensions
        if len(image.shape) == 3:
            # If multichannel, use first channel or convert to grayscale
            if image.shape[2] == 3:  # RGB
                image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            else:
                image = image[:, :, 0]
        
        try:
            positions = generate_random_patch_positions(
                image.shape, patch_size, patches_per_image,
                min_distance=patch_size//2
            )
            
            for pos in positions:
                # Create masked image
                masked_img = create_masked_image(image, pos, patch_size)
                
                # Extract the correct patch
                correct_patch = extract_patch(image, pos, patch_size)
                
                # Create positive pair
                pair = PatchPair(
                    masked_image=masked_img,
                    candidate_patch=correct_patch,
                    is_correct=True,
                    original_position=pos,
                    patch_size=patch_size,
                    source_image_path=img_data.file_path
                )
                
                positive_pairs.append(pair)
                
        except ValueError as e:
            logger.warning(f"Skipping image {img_data.file_path}: {e}")
            continue
    
    logger.info(f"Created {len(positive_pairs)} positive pairs")
    return positive_pairs


def create_negative_pairs(positive_pairs: List[PatchPair], 
                          all_images: List[ImageData],
                          negatives_per_positive: int = 3) -> List[PatchPair]:
    """
    Create negative pairs for contrastive learning.
    Each negative pair consists of a masked image and an incorrect patch.
    
    Args:
        positive_pairs: List of positive pairs to create negatives for
        all_images: All available images to sample negative patches from
        negatives_per_positive: Number of negative patches per positive
        
    Returns:
        List of negative PatchPair objects
    """
    negative_pairs = []
    
    # Collect all available patches from all images
    all_patches = []
    for img_data in all_images:
        image = img_data.data
        
        # Handle different image dimensions
        if len(image.shape) == 3:
            if image.shape[2] == 3:
                image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            else:
                image = image[:, :, 0]
        
        # Sample random patches from this image
        try:
            positions = generate_random_patch_positions(
                image.shape, positive_pairs[0].patch_size, 5
            )
            
            for pos in positions:
                patch = extract_patch(image, pos, positive_pairs[0].patch_size)
                all_patches.append((patch, img_data.file_path))
                
        except ValueError:
            continue
    
    # Create negative pairs
    for pos_pair in positive_pairs:
        for _ in range(negatives_per_positive):
            # Sample a random patch that's not from the same location
            negative_patch, source_path = random.choice(all_patches)
            
            # Ensure it's not from the same position in the same image
            while (source_path == pos_pair.source_image_path and 
                   np.array_equal(negative_patch, pos_pair.candidate_patch)):
                negative_patch, source_path = random.choice(all_patches)
            
            neg_pair = PatchPair(
                masked_image=pos_pair.masked_image,
                candidate_patch=negative_patch,
                is_correct=False,
                original_position=pos_pair.original_position,
                patch_size=pos_pair.patch_size,
                source_image_path=pos_pair.source_image_path
            )
            
            negative_pairs.append(neg_pair)
    
    logger.info(f"Created {len(negative_pairs)} negative pairs")
    return negative_pairs


if __name__ == "__main__":
    # Example usage
    print("Data Preprocessing module loaded successfully!")
 
    # Example of how to use the functions:
    # 1. Load images using the data_io module
    # 2. Split them using train_test_split()
    # 3. Create positive and negative pairs for contrastive learning

    """
    # Example workflow:
    from data_io import load_images_from_directory

    # Load images
    images = load_images_from_directory("path/to/microglia/images")

    # Split into train/test/val
    data_splits = train_test_split(images, split_criteria="by_animal")

    # Create pairs for contrastive learning
    positive_pairs = create_positive_pairs(data_splits.train)
    negative_pairs = create_negative_pairs(positive_pairs, data_splits.train)

    print(f"Training data: {len(data_splits.train)} images")
    print(f"Positive pairs: {len(positive_pairs)}")
    print(f"Negative pairs: {len(negative_pairs)}")
    """
