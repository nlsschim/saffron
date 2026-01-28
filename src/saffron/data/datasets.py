import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from typing import List, Tuple
from pathlib import Path


def get_random_patch_position(image_shape: Tuple[int, int], patch_size: int) -> Tuple[int, int]:
    """
    Generate random top-left position for a patch.
    
    Returns (row, col) ensuring patch fits within image.
    """
    h, w = image_shape
    max_row = h - patch_size
    max_col = w - patch_size
    
    row = np.random.randint(0, max_row + 1)
    col = np.random.randint(0, max_col + 1)
    
    return (row, col)


def extract_patch(image: np.ndarray, position: Tuple[int, int], patch_size: int) -> np.ndarray:
    """
    Extract a patch from an image.
    
    Returns patch of shape (patch_size, patch_size)
    """
    row, col = position
    patch = image[row:row + patch_size, col:col + patch_size]
    return patch


def create_masked_image(image: np.ndarray, position: Tuple[int, int], patch_size: int) -> np.ndarray:
    """
    Create masked image by zeroing out a patch.
    
    Returns copy of image with patch region set to 0.
    """
    masked = image.copy()
    row, col = position
    masked[row:row + patch_size, col:col + patch_size] = 0.0
    return masked


class PatchPairDataset(Dataset):
    """Dataset that generates masked images and positive/negative patch pairs."""
    
    def __init__(self, file_paths: List[str], patch_size: int = 64, n_negatives: int = 3):
        self.file_paths = file_paths
        self.patch_size = patch_size
        self.n_negatives = n_negatives
    
    def __len__(self):
        return len(self.file_paths)
    
    def __getitem__(self, idx: int):
        # Load image
        image_path = self.file_paths[idx]
        image = np.load(image_path)  # Shape: (512, 512), float32, [0, 1]
        
        # Generate random position for positive patch
        pos_position = get_random_patch_position(image.shape, self.patch_size)
        
        # Extract positive patch
        positive_patch = extract_patch(image, pos_position, self.patch_size)
        
        # Create masked image
        masked_image = create_masked_image(image, pos_position, self.patch_size)
        
        # Generate negative patches from random positions
        negative_patches = []
        for _ in range(self.n_negatives):
            neg_position = get_random_patch_position(image.shape, self.patch_size)
            neg_patch = extract_patch(image, neg_position, self.patch_size)
            negative_patches.append(neg_patch)
        
        # Convert to tensors and add channel dimension
        masked_image = torch.from_numpy(masked_image).unsqueeze(0)  # (1, H, W)
        positive_patch = torch.from_numpy(positive_patch).unsqueeze(0)  # (1, pH, pW)
        negative_patches = torch.stack([torch.from_numpy(p).unsqueeze(0) for p in negative_patches])  # (N, 1, pH, pW)
        
        return {
            'masked_image': masked_image,
            'positive_patch': positive_patch,
            'negative_patches': negative_patches
        }


def create_dataloaders(train_files: List[str],
                       val_files: List[str], 
                       test_files: List[str],
                       batch_size: int = 16,
                       patch_size: int = 64,
                       n_negatives: int = 3,
                       num_workers: int = 4) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Create train, val, and test DataLoaders."""
    # Create datasets
    train_dataset = PatchPairDataset(train_files, patch_size=patch_size, n_negatives=n_negatives)
    val_dataset = PatchPairDataset(val_files, patch_size=patch_size, n_negatives=n_negatives)
    test_dataset = PatchPairDataset(test_files, patch_size=patch_size, n_negatives=n_negatives)
    
    # Create DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
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
    
    return train_loader, val_loader, test_loader


def load_split_files(split_dir: str) -> Tuple[List[str], List[str], List[str]]:
    """
    Load train/val/test file lists from split directories.
    
    Expects structure:
        split_dir/
            train/*.npy
            val/*.npy
            test/*.npy
    """
    split_path = Path(split_dir)
    
    train_files = [str(f) for f in (split_path / 'train').glob('*.npy')]
    val_files = [str(f) for f in (split_path / 'val').glob('*.npy')]
    test_files = [str(f) for f in (split_path / 'test').glob('*.npy')]
    
    return train_files, val_files, test_files


if __name__ == "__main__":
    # Test the dataset
    print("Testing PatchPairDataset...")
    
    # Create a fake .npy file for testing
    test_image = np.random.rand(512, 512).astype(np.float32)
    np.save('/tmp/test_image.npy', test_image)
    
    # Create dataset
    dataset = PatchPairDataset(['/tmp/test_image.npy'], patch_size=64, n_negatives=3)
    
    print(f"Dataset length: {len(dataset)}")
    
    # Get a sample
    sample = dataset[0]
    print(f"\nSample shapes:")
    print(f"  Masked image: {sample['masked_image'].shape}")
    print(f"  Positive patch: {sample['positive_patch'].shape}")
    print(f"  Negative patches: {sample['negative_patches'].shape}")
    
    print("\n✓ Dataset test passed!")
