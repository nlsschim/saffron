"""
Minimal smoke test suite for saffron.data.datasets module.

These tests verify that PyTorch datasets and dataloaders work correctly.
Run with: pytest tests/test_datasets.py -v
"""

import pytest
import torch
import numpy as np
from pathlib import Path
import tempfile
import shutil
import tifffile

# Import from saffron
from saffron.io.data_io import ImageData
from saffron.data.data_processing import PatchPair, create_positive_pairs, create_negative_pairs
from saffron.data.datasets import (
    PatchPairDataset,
    create_contrastive_datasets,
    create_contrastive_dataloaders
)


# ============================================================================
# FIXTURES
# ============================================================================

@pytest.fixture
def temp_image_dir():
    """Create temporary directory with test images."""
    temp_dir = tempfile.mkdtemp()
    
    # Create 10 test images
    for i in range(10):
        img = np.random.rand(128, 128).astype(np.float32)
        tifffile.imwrite(
            Path(temp_dir) / f"animal0{i % 3}_slice0{i % 5}_test.tif",
            img
        )
    
    yield temp_dir
    shutil.rmtree(temp_dir)


@pytest.fixture
def sample_image_data_list():
    """Create sample ImageData objects."""
    images = []
    for i in range(10):
        img = np.random.rand(128, 128).astype(np.float32)
        images.append(ImageData(
            data=img,
            file_path=f"test_images/img_{i:03d}.tif",
            shape=img.shape,
            dtype=str(img.dtype)
        ))
    return images


@pytest.fixture
def sample_patch_pairs(sample_image_data_list):
    """Create sample positive and negative patch pairs."""
    positive_pairs = create_positive_pairs(
        sample_image_data_list,
        patch_size=32,
        patches_per_image=2
    )
    
    negative_pairs = create_negative_pairs(
        positive_pairs,
        all_images=sample_image_data_list,
        negatives_per_positive=3
    )
    
    return positive_pairs, negative_pairs


# ============================================================================
# DATASET TESTS
# ============================================================================

class TestPatchPairDataset:
    """Test PatchPairDataset class."""
    
    def test_dataset_creation(self, sample_patch_pairs):
        """Test creating dataset from patch pairs."""
        positive_pairs, negative_pairs = sample_patch_pairs
        
        dataset = PatchPairDataset(positive_pairs, negative_pairs)
        
        assert len(dataset) == len(positive_pairs)
        assert dataset.negatives_per_positive == 3
    
    def test_dataset_getitem(self, sample_patch_pairs):
        """Test getting item from dataset."""
        positive_pairs, negative_pairs = sample_patch_pairs
        dataset = PatchPairDataset(positive_pairs, negative_pairs)
        
        sample = dataset[0]
        
        # Check keys
        assert 'masked_image' in sample
        assert 'positive_patch' in sample
        assert 'negative_patches' in sample
        assert 'patch_position' in sample
        assert 'source_path' in sample
    
    def test_dataset_shapes(self, sample_patch_pairs):
        """Test that dataset returns correct tensor shapes."""
        positive_pairs, negative_pairs = sample_patch_pairs
        dataset = PatchPairDataset(positive_pairs, negative_pairs)
        
        sample = dataset[0]
        
        # Check shapes
        assert sample['masked_image'].shape == (1, 128, 128)
        assert sample['positive_patch'].shape == (1, 32, 32)
        assert sample['negative_patches'].shape == (3, 1, 32, 32)
    
    def test_dataset_types(self, sample_patch_pairs):
        """Test that dataset returns PyTorch tensors."""
        positive_pairs, negative_pairs = sample_patch_pairs
        dataset = PatchPairDataset(positive_pairs, negative_pairs)
        
        sample = dataset[0]
        
        # Check types
        assert isinstance(sample['masked_image'], torch.Tensor)
        assert isinstance(sample['positive_patch'], torch.Tensor)
        assert isinstance(sample['negative_patches'], torch.Tensor)
    
    def test_dataset_iteration(self, sample_patch_pairs):
        """Test iterating through dataset."""
        positive_pairs, negative_pairs = sample_patch_pairs
        dataset = PatchPairDataset(positive_pairs, negative_pairs)
        
        count = 0
        for sample in dataset:
            count += 1
            assert 'masked_image' in sample
        
        assert count == len(dataset)
    
    def test_dataset_indexing(self, sample_patch_pairs):
        """Test random access indexing."""
        positive_pairs, negative_pairs = sample_patch_pairs
        dataset = PatchPairDataset(positive_pairs, negative_pairs)
        
        # Access different indices
        first = dataset[0]
        last = dataset[-1]
        middle = dataset[len(dataset) // 2]
        
        # Should not crash
        assert first is not None
        assert last is not None
        assert middle is not None


# ============================================================================
# DATASET CREATION TESTS
# ============================================================================

class TestCreateContrastiveDatasets:
    """Test create_contrastive_datasets function."""
    
    def test_create_datasets(self, temp_image_dir):
        """Test creating train/val/test datasets."""
        train_ds, val_ds, test_ds = create_contrastive_datasets(
            image_directory=temp_image_dir,
            patch_size=32,
            patches_per_image=2,
            negatives_per_positive=3,
            split_criteria="random",
            test_size=0.2,
            val_size=0.1,
            random_state=42
        )
        
        # Check datasets created
        assert train_ds is not None
        assert val_ds is not None
        assert test_ds is not None
        
        # Check datasets have samples
        assert len(train_ds) > 0
        assert len(val_ds) > 0
        assert len(test_ds) > 0
    
    def test_dataset_split_proportions(self, temp_image_dir):
        """Test that split proportions are approximately correct."""
        train_ds, val_ds, test_ds = create_contrastive_datasets(
            image_directory=temp_image_dir,
            patch_size=32,
            patches_per_image=2,
            negatives_per_positive=3,
            test_size=0.2,
            val_size=0.1,
            random_state=42
        )
        
        total = len(train_ds) + len(val_ds) + len(test_ds)
        
        # Train should be largest
        assert len(train_ds) > len(val_ds)
        assert len(train_ds) > len(test_ds)
        
        # All should contribute to total
        assert total > 0
    
    def test_different_split_criteria(self, temp_image_dir):
        """Test different split criteria work."""
        for criteria in ['random', 'by_slice', 'by_animal']:
            train_ds, val_ds, test_ds = create_contrastive_datasets(
                image_directory=temp_image_dir,
                patch_size=32,
                patches_per_image=1,
                split_criteria=criteria,
                random_state=42
            )
            
            assert len(train_ds) > 0
            assert len(val_ds) > 0
            assert len(test_ds) > 0
    
    def test_different_patch_sizes(self, temp_image_dir):
        """Test creating datasets with different patch sizes."""
        for patch_size in [16, 32, 64]:
            train_ds, val_ds, test_ds = create_contrastive_datasets(
                image_directory=temp_image_dir,
                patch_size=patch_size,
                patches_per_image=1,
                random_state=42
            )
            
            sample = train_ds[0]
            assert sample['positive_patch'].shape == (1, patch_size, patch_size)
    
    def test_empty_directory(self):
        """Test that empty directory raises error."""
        with tempfile.TemporaryDirectory() as empty_dir:
            with pytest.raises(ValueError):
                create_contrastive_datasets(
                    image_directory=empty_dir,
                    patch_size=32
                )


# ============================================================================
# DATALOADER TESTS
# ============================================================================

class TestCreateContrastiveDataloaders:
    """Test create_contrastive_dataloaders function."""
    
    def test_create_dataloaders(self, temp_image_dir):
        """Test creating dataloaders."""
        train_loader, val_loader, test_loader = create_contrastive_dataloaders(
            image_directory=temp_image_dir,
            batch_size=4,
            patch_size=32,
            patches_per_image=2,
            num_workers=0,  # Use 0 for testing
            random_state=42
        )
        
        assert train_loader is not None
        assert val_loader is not None
        assert test_loader is not None
    
    def test_dataloader_batching(self, temp_image_dir):
        """Test that dataloader produces correct batch sizes."""
        batch_size = 4
        train_loader, _, _ = create_contrastive_dataloaders(
            image_directory=temp_image_dir,
            batch_size=batch_size,
            patch_size=32,
            num_workers=0,
            random_state=42
        )
        
        batch = next(iter(train_loader))
        
        # Check batch dimensions
        assert batch['masked_image'].shape[0] == batch_size
        assert batch['positive_patch'].shape[0] == batch_size
        assert batch['negative_patches'].shape[0] == batch_size
    
    def test_dataloader_iteration(self, temp_image_dir):
        """Test iterating through dataloader."""
        train_loader, _, _ = create_contrastive_dataloaders(
            image_directory=temp_image_dir,
            batch_size=2,
            patch_size=32,
            num_workers=0,
            random_state=42
        )
        
        batch_count = 0
        for batch in train_loader:
            batch_count += 1
            assert 'masked_image' in batch
            assert 'positive_patch' in batch
            assert 'negative_patches' in batch
        
        assert batch_count > 0
    
    def test_dataloader_shapes(self, temp_image_dir):
        """Test dataloader batch shapes are correct."""
        batch_size = 4
        patch_size = 32
        negatives = 3
        
        train_loader, _, _ = create_contrastive_dataloaders(
            image_directory=temp_image_dir,
            batch_size=batch_size,
            patch_size=patch_size,
            negatives_per_positive=negatives,
            num_workers=0
        )
        
        batch = next(iter(train_loader))
        
        assert batch['masked_image'].shape == (batch_size, 1, 128, 128)
        assert batch['positive_patch'].shape == (batch_size, 1, patch_size, patch_size)
        assert batch['negative_patches'].shape == (batch_size, negatives, 1, patch_size, patch_size)
    
    def test_dataloader_device_transfer(self, temp_image_dir):
        """Test transferring batch to device."""
        train_loader, _, _ = create_contrastive_dataloaders(
            image_directory=temp_image_dir,
            batch_size=2,
            patch_size=32,
            num_workers=0
        )
        
        batch = next(iter(train_loader))
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Transfer to device
        masked = batch['masked_image'].to(device)
        positive = batch['positive_patch'].to(device)
        negative = batch['negative_patches'].to(device)
        
        assert masked.device.type == device.type
        assert positive.device.type == device.type
        assert negative.device.type == device.type
    
    def test_different_batch_sizes(self, temp_image_dir):
        """Test creating dataloaders with different batch sizes."""
        for batch_size in [2, 4, 8]:
            train_loader, _, _ = create_contrastive_dataloaders(
                image_directory=temp_image_dir,
                batch_size=batch_size,
                patch_size=32,
                num_workers=0
            )
            
            batch = next(iter(train_loader))
            assert batch['masked_image'].shape[0] == batch_size


# ============================================================================
# INTEGRATION TESTS
# ============================================================================

class TestDatasetIntegration:
    """Integration tests for complete workflows."""
    
    def test_end_to_end_pipeline(self, temp_image_dir):
        """Test complete pipeline from directory to batches."""
        # Create dataloaders
        train_loader, val_loader, test_loader = create_contrastive_dataloaders(
            image_directory=temp_image_dir,
            batch_size=4,
            patch_size=32,
            patches_per_image=2,
            negatives_per_positive=3,
            num_workers=0,
            random_state=42
        )
        
        # Get batches from each loader
        train_batch = next(iter(train_loader))
        val_batch = next(iter(val_loader))
        test_batch = next(iter(test_loader))
        
        # All should have same structure
        for batch in [train_batch, val_batch, test_batch]:
            assert 'masked_image' in batch
            assert 'positive_patch' in batch
            assert 'negative_patches' in batch
    
    def test_training_loop_simulation(self, temp_image_dir):
        """Test simulating a training loop."""
        train_loader, _, _ = create_contrastive_dataloaders(
            image_directory=temp_image_dir,
            batch_size=2,
            patch_size=32,
            num_workers=0
        )
        
        # Simulate a few training iterations
        num_iterations = 3
        for i, batch in enumerate(train_loader):
            if i >= num_iterations:
                break
            
            # Simulate forward pass
            masked = batch['masked_image']
            positive = batch['positive_patch']
            negative = batch['negative_patches']
            
            # Check we can do basic operations
            assert masked.mean() >= 0
            assert positive.std() >= 0
            assert negative.shape[0] > 0
    
    def test_dataloader_reproducibility(self, temp_image_dir):
        """Test that same seed gives same batches."""
        # Create two dataloaders with same seed
        loader1, _, _ = create_contrastive_dataloaders(
            image_directory=temp_image_dir,
            batch_size=4,
            patch_size=32,
            num_workers=0,
            random_state=42
        )
        
        loader2, _, _ = create_contrastive_dataloaders(
            image_directory=temp_image_dir,
            batch_size=4,
            patch_size=32,
            num_workers=0,
            random_state=42
        )
        
        # Get first batches
        batch1 = next(iter(loader1))
        batch2 = next(iter(loader2))
        
        # Positions should be same (deterministic)
        assert torch.equal(batch1['patch_position'], batch2['patch_position'])


# ============================================================================
# EDGE CASES
# ============================================================================

class TestEdgeCases:
    """Test edge cases and error handling."""
    
    def test_small_batch_size(self, temp_image_dir):
        """Test with batch size of 1."""
        train_loader, _, _ = create_contrastive_dataloaders(
            image_directory=temp_image_dir,
            batch_size=1,
            patch_size=32,
            num_workers=0
        )
        
        batch = next(iter(train_loader))
        assert batch['masked_image'].shape[0] == 1
    
    def test_large_patch_size(self, temp_image_dir):
        """Test with large patch size (but still fits in image)."""
        train_loader, _, _ = create_contrastive_dataloaders(
            image_directory=temp_image_dir,
            batch_size=2,
            patch_size=64,  # Large but fits in 128x128 image
            num_workers=0
        )
        
        batch = next(iter(train_loader))
        assert batch['positive_patch'].shape[2:] == (64, 64)
    
    def test_many_negatives(self, temp_image_dir):
        """Test with many negative samples."""
        train_loader, _, _ = create_contrastive_dataloaders(
            image_directory=temp_image_dir,
            batch_size=2,
            patch_size=32,
            negatives_per_positive=10,
            num_workers=0
        )
        
        batch = next(iter(train_loader))
        assert batch['negative_patches'].shape[1] == 10
    
    def test_single_patch_per_image(self, temp_image_dir):
        """Test with only 1 patch per image."""
        train_loader, _, _ = create_contrastive_dataloaders(
            image_directory=temp_image_dir,
            batch_size=2,
            patch_size=32,
            patches_per_image=1,
            num_workers=0
        )
        
        batch = next(iter(train_loader))
        assert batch is not None


# ============================================================================
# PERFORMANCE TESTS
# ============================================================================

class TestPerformance:
    """Basic performance smoke tests."""
    
    @pytest.mark.slow
    def test_large_batch_performance(self, temp_image_dir):
        """Test that large batches work without crashing."""
        train_loader, _, _ = create_contrastive_dataloaders(
            image_directory=temp_image_dir,
            batch_size=16,  # Larger batch
            patch_size=32,
            num_workers=0
        )
        
        # Should be able to get a batch
        batch = next(iter(train_loader))
        assert batch['masked_image'].shape[0] <= 16
    
    def test_multiple_epochs(self, temp_image_dir):
        """Test iterating through multiple epochs."""
        train_loader, _, _ = create_contrastive_dataloaders(
            image_directory=temp_image_dir,
            batch_size=2,
            patch_size=32,
            num_workers=0
        )
        
        # Iterate through 2 epochs
        for epoch in range(2):
            batch_count = 0
            for batch in train_loader:
                batch_count += 1
            assert batch_count > 0


# ============================================================================
# RUN TESTS
# ============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])