"""
Adapted dataset and dataloader creation using existing functions from:
- saffron.io.data_io (ImageData, load_tif, load_npy)
- saffron.io.data_processing (train_test_split, create_positive_pairs, create_negative_pairs, PatchPair)
"""

import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from typing import List, Tuple, Optional
from pathlib import Path
import logging

# Import from saffron modules
from saffron.io.data_io import ImageData, load_images_from_directory
from saffron.data.data_processing import (
    train_test_split,
    create_positive_pairs,
    create_negative_pairs,
    PatchPair,
    DataSplit
)

logger = logging.getLogger(__name__)


class PatchPairDataset(Dataset):
    """
    Dataset that uses pre-generated PatchPair objects from data_processing.py
    
    This wraps the PatchPair objects created by create_positive_pairs() and
    create_negative_pairs() into a PyTorch Dataset.
    """
    
    def __init__(
        self,
        positive_pairs: List[PatchPair],
        negative_pairs: List[PatchPair],
        transform=None
    ):
        """
        Args:
            positive_pairs: List of positive PatchPair objects
            negative_pairs: List of negative PatchPair objects
            transform: Optional transforms to apply
        """
        self.positive_pairs = positive_pairs
        self.negative_pairs = negative_pairs
        self.transform = transform
        
        # Create mapping: each positive gets its corresponding negatives
        # Assuming negatives_per_positive ratio is consistent
        self.negatives_per_positive = len(negative_pairs) // len(positive_pairs)
        
        logger.info(f"Dataset created with {len(positive_pairs)} positive pairs")
        logger.info(f"Dataset created with {len(negative_pairs)} negative pairs")
        logger.info(f"Ratio: {self.negatives_per_positive} negatives per positive")
    
    def __len__(self):
        return len(self.positive_pairs)
    
    def __getitem__(self, idx: int):
        """
        Get a training sample consisting of:
        - Masked image
        - Positive patch (correct)
        - Negative patches (incorrect)
        
        Returns:
            dict with keys:
                - masked_image: [1, H, W]
                - positive_patch: [1, pH, pW]
                - negative_patches: [N, 1, pH, pW]
                - patch_position: (row, col)
                - source_path: path to source image
        """
        # Get positive pair
        pos_pair = self.positive_pairs[idx]
        
        # Get corresponding negative pairs
        neg_start_idx = idx * self.negatives_per_positive
        neg_end_idx = neg_start_idx + self.negatives_per_positive
        neg_pairs = self.negative_pairs[neg_start_idx:neg_end_idx]
        
        # Convert to tensors
        masked_image = torch.from_numpy(pos_pair.masked_image).unsqueeze(0).float()
        positive_patch = torch.from_numpy(pos_pair.candidate_patch).unsqueeze(0).float()
        
        # Stack negative patches
        negative_patches = torch.stack([
            torch.from_numpy(neg.candidate_patch).unsqueeze(0).float()
            for neg in neg_pairs
        ])
        
        # Apply transforms if provided
        if self.transform:
            masked_image = self.transform(masked_image)
            positive_patch = self.transform(positive_patch)
            negative_patches = torch.stack([
                self.transform(neg) for neg in negative_patches
            ])
        
        return {
            'masked_image': masked_image,
            'positive_patch': positive_patch,
            'negative_patches': negative_patches,
            'patch_position': torch.tensor(pos_pair.original_position),
            'source_path': pos_pair.source_image_path
        }


def create_contrastive_datasets(
    image_directory: str,
    patch_size: int = 64,
    patches_per_image: int = 5,
    negatives_per_positive: int = 3,
    split_criteria: str = "random",
    test_size: float = 0.2,
    val_size: float = 0.1,
    random_state: int = 42
) -> Tuple[PatchPairDataset, PatchPairDataset, PatchPairDataset]:
    """
    Create train, validation, and test datasets using data_processing functions.
    
    This is the main function that orchestrates everything:
    1. Load images from directory
    2. Split into train/val/test
    3. Generate positive and negative pairs
    4. Create PyTorch datasets
    
    Args:
        image_directory: Path to directory containing images
        patch_size: Size of patches to extract
        patches_per_image: Number of patches per image
        negatives_per_positive: Number of negative samples per positive
        split_criteria: "random", "by_slice", or "by_animal"
        test_size: Proportion for test set
        val_size: Proportion for validation set
        random_state: Random seed
        
    Returns:
        train_dataset, val_dataset, test_dataset
    """
    logger.info("="*80)
    logger.info("Creating contrastive learning datasets")
    logger.info("="*80)
    
    # ========================================
    # STEP 1: Load images
    # ========================================
    logger.info(f"\nStep 1: Loading images from {image_directory}")
    
    images = load_images_from_directory(
        image_directory,
        file_extensions=['.tif', '.tiff', '.npy']
    )
    
    logger.info(f"Loaded {len(images)} images")
    
    if len(images) == 0:
        raise ValueError(f"No images found in {image_directory}")
    
    # ========================================
    # STEP 2: Split into train/val/test
    # ========================================
    logger.info(f"\nStep 2: Splitting data using '{split_criteria}' strategy")
    
    data_splits = train_test_split(
        images,
        test_size=test_size,
        val_size=val_size,
        split_criteria=split_criteria,
        random_state=random_state
    )
    
    logger.info(f"Split info: {data_splits.get_split_info()}")
    
    # ========================================
    # STEP 3: Generate pairs for each split
    # ========================================
    logger.info(f"\nStep 3: Generating positive and negative pairs")
    
    datasets = []
    
    for split_name, split_images in [
        ('train', data_splits.train),
        ('val', data_splits.val),
        ('test', data_splits.test)
    ]:
        logger.info(f"\nProcessing {split_name} split ({len(split_images)} images)...")
        
        # Generate positive pairs
        positive_pairs = create_positive_pairs(
            split_images,
            patch_size=patch_size,
            patches_per_image=patches_per_image
        )
        
        # Generate negative pairs
        negative_pairs = create_negative_pairs(
            positive_pairs,
            all_images=split_images,
            negatives_per_positive=negatives_per_positive
        )
        
        # Create dataset
        dataset = PatchPairDataset(
            positive_pairs=positive_pairs,
            negative_pairs=negative_pairs
        )
        
        datasets.append(dataset)
        
        logger.info(f"{split_name} dataset: {len(dataset)} samples")
    
    train_dataset, val_dataset, test_dataset = datasets
    
    logger.info("\n" + "="*80)
    logger.info("Dataset creation complete!")
    logger.info("="*80)
    
    return train_dataset, val_dataset, test_dataset


def create_contrastive_dataloaders(
    image_directory: str,
    batch_size: int = 16,
    patch_size: int = 64,
    patches_per_image: int = 5,
    negatives_per_positive: int = 3,
    split_criteria: str = "random",
    test_size: float = 0.2,
    val_size: float = 0.1,
    num_workers: int = 4,
    random_state: int = 42
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Create train, validation, and test DataLoaders.
    
    This is the high-level convenience function you'll typically use.
    
    Args:
        image_directory: Path to directory containing images
        batch_size: Batch size for dataloaders
        patch_size: Size of patches to extract
        patches_per_image: Number of patches per image
        negatives_per_positive: Number of negative samples per positive
        split_criteria: "random", "by_slice", or "by_animal"
        test_size: Proportion for test set
        val_size: Proportion for validation set
        num_workers: Number of dataloader workers
        random_state: Random seed
        
    Returns:
        train_loader, val_loader, test_loader
    """
    # Create datasets
    train_dataset, val_dataset, test_dataset = create_contrastive_datasets(
        image_directory=image_directory,
        patch_size=patch_size,
        patches_per_image=patches_per_image,
        negatives_per_positive=negatives_per_positive,
        split_criteria=split_criteria,
        test_size=test_size,
        val_size=val_size,
        random_state=random_state
    )
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True  # Drop last incomplete batch for consistent batch sizes
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    logger.info(f"\nDataLoaders created:")
    logger.info(f"  Train: {len(train_loader)} batches")
    logger.info(f"  Val: {len(val_loader)} batches")
    logger.info(f"  Test: {len(test_loader)} batches")
    
    return train_loader, val_loader, test_loader


# ============================================================================
# INTEGRATION WITH SAFFRON CONFIG SYSTEM
# ============================================================================

def create_dataloaders_from_saffron_config(
    condition: str = "OGD_only",
    mask_type: str = "microglia",
    batch_size: int = 16,
    patch_size: int = 64,
    patches_per_image: int = 5,
    negatives_per_positive: int = 3,
    split_criteria: str = "random",
    num_workers: int = 4
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Create dataloaders using saffron's PathConfig system.
    
    This integrates with your existing saffron configuration.
    
    Args:
        condition: Experimental condition (e.g., "OGD_only", "HC")
        mask_type: Type of masks to use ("microglia", "mitochondria", "nuclei")
        batch_size: Batch size
        patch_size: Size of patches
        patches_per_image: Number of patches per image
        negatives_per_positive: Number of negatives per positive
        split_criteria: Split strategy
        num_workers: Number of workers
        
    Returns:
        train_loader, val_loader, test_loader
    """
    from saffron.config import PathConfig
    
    # Create path configuration
    path_config = PathConfig(condition=condition)
    
    # Get appropriate directory based on mask type
    if mask_type == "microglia":
        image_directory = path_config.microglia_masks_dir
    elif mask_type == "mitochondria":
        image_directory = path_config.mitochondria_masks_dir
    elif mask_type == "nuclei":
        image_directory = path_config.nuclei_masks_dir
    else:
        raise ValueError(f"Unknown mask_type: {mask_type}")
    
    logger.info(f"Loading {mask_type} masks from {image_directory}")
    logger.info(f"Condition: {condition}")
    
    # Create dataloaders
    return create_contrastive_dataloaders(
        image_directory=str(image_directory),
        batch_size=batch_size,
        patch_size=patch_size,
        patches_per_image=patches_per_image,
        negatives_per_positive=negatives_per_positive,
        split_criteria=split_criteria,
        num_workers=num_workers
    )


# ============================================================================
# EXAMPLE USAGE
# ============================================================================

if __name__ == "__main__":
    # Example 1: Using direct path
    print("\n" + "="*80)
    print("EXAMPLE 1: Direct path")
    print("="*80)
    
    train_loader, val_loader, test_loader = create_contrastive_dataloaders(
        image_directory="/path/to/your/microglia/masks",
        batch_size=16,
        patch_size=64,
        patches_per_image=5,
        negatives_per_positive=3,
        split_criteria="random",
        num_workers=4
    )
    
    # Test the dataloader
    batch = next(iter(train_loader))
    print(f"\nBatch shapes:")
    print(f"  Masked image: {batch['masked_image'].shape}")
    print(f"  Positive patch: {batch['positive_patch'].shape}")
    print(f"  Negative patches: {batch['negative_patches'].shape}")
    
    # Example 2: Using saffron config
    print("\n" + "="*80)
    print("EXAMPLE 2: Using saffron config")
    print("="*80)
    
    train_loader, val_loader, test_loader = create_dataloaders_from_saffron_config(
        condition="OGD_only",
        mask_type="microglia",
        batch_size=16,
        patch_size=64,
        split_criteria="by_slice"  # Ensures no slice in train appears in val/test
    )
    
    # Example 3: Advanced - using different split criteria
    print("\n" + "="*80)
    print("EXAMPLE 3: Split by animal (for biological independence)")
    print("="*80)
    
    train_loader, val_loader, test_loader = create_dataloaders_from_saffron_config(
        condition="HC",
        mask_type="microglia",
        batch_size=8,
        split_criteria="by_animal"  # Ensures no animal in train appears in val/test
    )
    
    # Example 4: Create datasets only (without DataLoaders)
    print("\n" + "="*80)
    print("EXAMPLE 4: Just datasets (for custom DataLoader)")
    print("="*80)
    
    train_ds, val_ds, test_ds = create_contrastive_datasets(
        image_directory="/path/to/images",
        patch_size=64,
        patches_per_image=5,
        negatives_per_positive=3
    )
    
    print(f"Train dataset size: {len(train_ds)}")
    print(f"Val dataset size: {len(val_ds)}")
    print(f"Test dataset size: {len(test_ds)}")
    
    # You can then create custom DataLoaders
    custom_train_loader = DataLoader(
        train_ds,
        batch_size=32,
        shuffle=True,
        collate_fn=custom_collate_function  # Your custom function
    )


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def inspect_dataset_sample(dataset: PatchPairDataset, idx: int = 0):
    """
    Inspect a single sample from the dataset for debugging.
    
    Args:
        dataset: PatchPairDataset instance
        idx: Index of sample to inspect
    """
    import matplotlib.pyplot as plt
    
    sample = dataset[idx]
    
    print(f"\nSample {idx} details:")
    print(f"  Masked image shape: {sample['masked_image'].shape}")
    print(f"  Positive patch shape: {sample['positive_patch'].shape}")
    print(f"  Negative patches shape: {sample['negative_patches'].shape}")
    print(f"  Patch position: {sample['patch_position']}")
    print(f"  Source: {sample['source_path']}")
    
    # Visualize
    fig, axes = plt.subplots(1, 3 + sample['negative_patches'].shape[0], 
                             figsize=(15, 3))
    
    # Masked image
    axes[0].imshow(sample['masked_image'].squeeze(), cmap='gray')
    axes[0].set_title('Masked Image')
    axes[0].axis('off')
    
    # Positive patch
    axes[1].imshow(sample['positive_patch'].squeeze(), cmap='gray')
    axes[1].set_title('Correct Patch')
    axes[1].axis('off')
    axes[1].set_facecolor('lightgreen')
    
    # Negative patches
    for i in range(sample['negative_patches'].shape[0]):
        axes[2 + i].imshow(sample['negative_patches'][i].squeeze(), cmap='gray')
        axes[2 + i].set_title(f'Wrong Patch {i+1}')
        axes[2 + i].axis('off')
        axes[2 + i].set_facecolor('lightcoral')
    
    plt.tight_layout()
    plt.savefig('dataset_sample_visualization.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    print("\nVisualization saved to 'dataset_sample_visualization.png'")


def get_dataset_statistics(dataset: PatchPairDataset):
    """
    Compute and display statistics about the dataset.
    
    Args:
        dataset: PatchPairDataset instance
    """
    print("\nDataset Statistics:")
    print(f"  Total samples: {len(dataset)}")
    print(f"  Positive pairs: {len(dataset.positive_pairs)}")
    print(f"  Negative pairs: {len(dataset.negative_pairs)}")
    print(f"  Negatives per positive: {dataset.negatives_per_positive}")
    
    # Sample to check shapes
    sample = dataset[0]
    print(f"\nSample shapes:")
    print(f"  Masked image: {sample['masked_image'].shape}")
    print(f"  Positive patch: {sample['positive_patch'].shape}")
    print(f"  Negative patches: {sample['negative_patches'].shape}")
    
    # Source files
    source_files = set([pair.source_image_path for pair in dataset.positive_pairs])
    print(f"\nUnique source images: {len(source_files)}")
    
    # Patch positions distribution
    positions = [pair.original_position for pair in dataset.positive_pairs]
    positions_array = np.array(positions)
    print(f"\nPatch position statistics:")
    print(f"  Mean position: ({positions_array[:, 0].mean():.1f}, "
          f"{positions_array[:, 1].mean():.1f})")
    print(f"  Std position: ({positions_array[:, 0].std():.1f}, "
          f"{positions_array[:, 1].std():.1f})")