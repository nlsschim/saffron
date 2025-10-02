"""
Minimal smoke test suite for saffron.data.preprocessing module.

These tests verify that the core functions work without errors on simple inputs.
Run with: pytest tests/test_data_preprocessing.py -v
"""

import pytest
import numpy as np
from pathlib import Path
import tempfile
import shutil

# Assuming the reorganized structure
from saffron.io.data_io import ImageData
from saffron.data.data_processing import (
    extract_metadata_from_path,
    train_test_split,
    create_masked_image,
    extract_patch,
    generate_random_patch_positions,
    create_positive_pairs,
    create_negative_pairs,
    DataSplit,
    PatchPair
)


# ============================================================================
# FIXTURES
# ============================================================================

@pytest.fixture
def dummy_image():
    """Create a simple test image."""
    return np.random.rand(128, 128).astype(np.float32)


@pytest.fixture
def dummy_image_data_list():
    """Create a list of ImageData objects for testing."""
    images = []
    for i in range(10):
        img = np.random.rand(128, 128).astype(np.float32)
        images.append(ImageData(
            data=img,
            file_path=f"test_images/animal0{i % 3}_slice0{i % 5}_conditionA.tif",
            shape=img.shape,
            dtype=str(img.dtype)
        ))
    return images


@pytest.fixture
def temp_image_dir():
    """Create temporary directory with test images."""
    temp_dir = tempfile.mkdtemp()
    
    # Create some dummy TIFF files
    for i in range(5):
        img = np.random.rand(128, 128).astype(np.float32)
        import tifffile
        tifffile.imwrite(
            Path(temp_dir) / f"animal0{i % 2}_slice0{i}_control.tif",
            img
        )
    
    yield temp_dir
    
    # Cleanup
    shutil.rmtree(temp_dir)


# ============================================================================
# METADATA EXTRACTION TESTS
# ============================================================================

class TestMetadataExtraction:
    """Test metadata extraction from file paths."""
    
    def test_extract_animal_id(self):
        """Test extracting animal ID from filename."""
        path = "data/animal01_slice05_conditionA.tif"
        metadata = extract_metadata_from_path(path)
        assert 'animal' in metadata
        assert metadata['animal'] == 'animal01'
    
    def test_extract_slice_number(self):
        """Test extracting slice number from filename."""
        path = "data/animal01_slice05_conditionA.tif"
        metadata = extract_metadata_from_path(path)
        assert 'slice' in metadata
        assert metadata['slice'] == 'slice05'
    
    def test_extract_condition(self):
        """Test extracting condition from filename."""
        path = "data/animal01_slice05_conditionA.tif"
        metadata = extract_metadata_from_path(path)
        assert 'condition' in metadata
        assert metadata['condition'] == 'conditionA'
    
    def test_missing_metadata(self):
        """Test handling files without metadata."""
        path = "data/random_file.tif"
        metadata = extract_metadata_from_path(path)
        # Should not crash, just return empty dict or partial metadata
        assert isinstance(metadata, dict)


# ============================================================================
# TRAIN/TEST SPLIT TESTS
# ============================================================================

class TestTrainTestSplit:
    """Test train/validation/test splitting functionality."""
    
    def test_random_split_proportions(self, dummy_image_data_list):
        """Test that random split creates correct proportions."""
        splits = train_test_split(
            dummy_image_data_list,
            test_size=0.2,
            val_size=0.1,
            split_criteria="random",
            random_state=42
        )
        
        assert isinstance(splits, DataSplit)
        assert len(splits.train) > 0
        assert len(splits.val) > 0
        assert len(splits.test) > 0
        assert len(splits) == len(dummy_image_data_list)
    
    def test_split_by_slice(self, dummy_image_data_list):
        """Test splitting by slice ensures no slice overlap."""
        splits = train_test_split(
            dummy_image_data_list,
            test_size=0.2,
            val_size=0.1,
            split_criteria="by_slice",
            random_state=42
        )
        
        # Extract slice IDs from each split
        train_slices = {extract_metadata_from_path(img.file_path).get('slice') 
                       for img in splits.train}
        val_slices = {extract_metadata_from_path(img.file_path).get('slice') 
                     for img in splits.val}
        test_slices = {extract_metadata_from_path(img.file_path).get('slice') 
                      for img in splits.test}
        
        # Check no overlap
        assert len(train_slices & val_slices) == 0
        assert len(train_slices & test_slices) == 0
        assert len(val_slices & test_slices) == 0
    
    def test_split_by_animal(self, dummy_image_data_list):
        """Test splitting by animal ensures no animal overlap."""
        splits = train_test_split(
            dummy_image_data_list,
            test_size=0.2,
            val_size=0.1,
            split_criteria="by_animal",
            random_state=42
        )
        
        # Extract animal IDs from each split
        train_animals = {extract_metadata_from_path(img.file_path).get('animal') 
                        for img in splits.train}
        val_animals = {extract_metadata_from_path(img.file_path).get('animal') 
                      for img in splits.val}
        test_animals = {extract_metadata_from_path(img.file_path).get('animal') 
                       for img in splits.test}
        
        # Check no overlap
        assert len(train_animals & val_animals) == 0
        assert len(train_animals & test_animals) == 0
        assert len(val_animals & test_animals) == 0
    
    def test_split_reproducibility(self, dummy_image_data_list):
        """Test that splits are reproducible with same random seed."""
        splits1 = train_test_split(
            dummy_image_data_list,
            test_size=0.2,
            val_size=0.1,
            split_criteria="random",
            random_state=42
        )
        
        splits2 = train_test_split(
            dummy_image_data_list,
            test_size=0.2,
            val_size=0.1,
            split_criteria="random",
            random_state=42
        )
        
        # Check same images in train set
        train_paths1 = {img.file_path for img in splits1.train}
        train_paths2 = {img.file_path for img in splits2.train}
        assert train_paths1 == train_paths2
    
    def test_split_with_empty_list(self):
        """Test that splitting empty list raises error."""
        with pytest.raises(ValueError):
            train_test_split([], test_size=0.2, val_size=0.1)
    
    def test_data_split_get_split_info(self, dummy_image_data_list):
        """Test DataSplit.get_split_info() returns correct counts."""
        splits = train_test_split(
            dummy_image_data_list,
            test_size=0.2,
            val_size=0.1,
            split_criteria="random"
        )
        
        info = splits.get_split_info()
        assert info['total'] == len(dummy_image_data_list)
        assert info['train'] == len(splits.train)
        assert info['val'] == len(splits.val)
        assert info['test'] == len(splits.test)


# ============================================================================
# PATCH OPERATIONS TESTS
# ============================================================================

class TestPatchOperations:
    """Test patch extraction and masking operations."""
    
    def test_create_masked_image(self, dummy_image):
        """Test creating masked image sets patch to 0."""
        patch_pos = (10, 10)
        patch_size = 32
        
        masked = create_masked_image(dummy_image, patch_pos, patch_size)
        
        # Check patch region is masked (set to 0)
        assert np.all(masked[10:42, 10:42] == 0.0)
        
        # Check rest is unchanged
        assert np.array_equal(masked[0:10, :], dummy_image[0:10, :])
        assert np.array_equal(masked[42:, :], dummy_image[42:, :])
    
    def test_extract_patch(self, dummy_image):
        """Test extracting patch from image."""
        patch_pos = (20, 20)
        patch_size = 32
        
        patch = extract_patch(dummy_image, patch_pos, patch_size)
        
        assert patch.shape == (32, 32)
        assert np.array_equal(patch, dummy_image[20:52, 20:52])
    
    def test_extract_patch_at_edge(self, dummy_image):
        """Test extracting patch at image edge (with padding)."""
        patch_pos = (110, 110)  # Near edge of 128x128 image
        patch_size = 32
        
        patch = extract_patch(dummy_image, patch_pos, patch_size)
        
        # Should return 32x32 patch (padded if needed)
        assert patch.shape == (32, 32)
    
    def test_generate_random_patch_positions(self, dummy_image):
        """Test generating valid random patch positions."""
        positions = generate_random_patch_positions(
            dummy_image.shape,
            patch_size=32,
            num_positions=5
        )
        
        assert len(positions) == 5
        
        # Check all positions are valid (patch fits in image)
        for row, col in positions:
            assert 0 <= row <= 128 - 32
            assert 0 <= col <= 128 - 32
    
    def test_generate_positions_with_min_distance(self, dummy_image):
        """Test generating positions with minimum distance constraint."""
        positions = generate_random_patch_positions(
            dummy_image.shape,
            patch_size=32,
            num_positions=3,
            min_distance=40
        )
        
        # Check all positions respect min distance
        for i, (r1, c1) in enumerate(positions):
            for r2, c2 in positions[i+1:]:
                distance = np.sqrt((r1 - r2)**2 + (c1 - c2)**2)
                assert distance >= 40
    
    def test_patch_size_too_large(self):
        """Test that too large patch size raises error."""
        small_image = np.random.rand(32, 32)
        
        with pytest.raises(ValueError):
            generate_random_patch_positions(
                small_image.shape,
                patch_size=64,  # Larger than image
                num_positions=1
            )


# ============================================================================
# PAIR GENERATION TESTS
# ============================================================================

class TestPairGeneration:
    """Test positive and negative pair generation."""
    
    def test_create_positive_pairs(self, dummy_image_data_list):
        """Test creating positive pairs from images."""
        positive_pairs = create_positive_pairs(
            dummy_image_data_list,
            patch_size=32,
            patches_per_image=3
        )
        
        assert len(positive_pairs) > 0
        assert len(positive_pairs) == len(dummy_image_data_list) * 3
        
        # Check PatchPair structure
        pair = positive_pairs[0]
        assert isinstance(pair, PatchPair)
        assert pair.masked_image.shape == (128, 128)
        assert pair.candidate_patch.shape == (32, 32)
        assert pair.is_correct == True
        assert pair.patch_size == 32
    
    def test_positive_pair_masking_correct(self, dummy_image_data_list):
        """Test that positive pairs have correctly masked images."""
        positive_pairs = create_positive_pairs(
            dummy_image_data_list,
            patch_size=32,
            patches_per_image=1
        )
        
        pair = positive_pairs[0]
        row, col = pair.original_position
        
        # Check that masked region is all zeros
        assert np.all(pair.masked_image[row:row+32, col:col+32] == 0.0)
    
    def test_create_negative_pairs(self, dummy_image_data_list):
        """Test creating negative pairs from positive pairs."""
        positive_pairs = create_positive_pairs(
            dummy_image_data_list,
            patch_size=32,
            patches_per_image=2
        )
        
        negative_pairs = create_negative_pairs(
            positive_pairs,
            all_images=dummy_image_data_list,
            negatives_per_positive=3
        )
        
        assert len(negative_pairs) > 0
        assert len(negative_pairs) == len(positive_pairs) * 3
        
        # Check structure
        neg_pair = negative_pairs[0]
        assert isinstance(neg_pair, PatchPair)
        assert neg_pair.is_correct == False
        assert neg_pair.candidate_patch.shape == (32, 32)
    
    def test_negative_pairs_different_from_positive(self, dummy_image_data_list):
        """Test that negative patches are different from positive."""
        positive_pairs = create_positive_pairs(
            dummy_image_data_list,
            patch_size=32,
            patches_per_image=1
        )
        
        negative_pairs = create_negative_pairs(
            positive_pairs,
            all_images=dummy_image_data_list,
            negatives_per_positive=3
        )
        
        # For each positive, check its negatives are different
        for i, pos_pair in enumerate(positive_pairs):
            pos_patch = pos_pair.candidate_patch
            
            for j in range(3):
                neg_pair = negative_pairs[i * 3 + j]
                neg_patch = neg_pair.candidate_patch
                
                # Patches should not be identical
                # (unless very unlucky with random sampling)
                assert not np.array_equal(pos_patch, neg_patch)
    
    def test_pair_generation_empty_list(self):
        """Test that empty image list returns empty pairs."""
        pairs = create_positive_pairs([], patch_size=32, patches_per_image=5)
        assert len(pairs) == 0


# ============================================================================
# INTEGRATION TESTS
# ============================================================================

class TestIntegration:
    """Integration tests for complete workflows."""
    
    def test_full_pipeline(self, dummy_image_data_list):
        """Test complete pipeline from images to pairs."""
        # Split data
        splits = train_test_split(
            dummy_image_data_list,
            test_size=0.2,
            val_size=0.1,
            split_criteria="random",
            random_state=42
        )
        
        # Generate positive pairs
        train_positive = create_positive_pairs(
            splits.train,
            patch_size=32,
            patches_per_image=3
        )
        
        # Generate negative pairs
        train_negative = create_negative_pairs(
            train_positive,
            all_images=splits.train,
            negatives_per_positive=2
        )
        
        # Verify counts
        assert len(train_positive) == len(splits.train) * 3
        assert len(train_negative) == len(train_positive) * 2
        
        # Verify structure
        assert all(p.is_correct for p in train_positive)
        assert all(not p.is_correct for p in train_negative)
    
    def test_different_patch_sizes(self, dummy_image_data_list):
        """Test pipeline works with different patch sizes."""
        for patch_size in [16, 32, 64]:
            pairs = create_positive_pairs(
                dummy_image_data_list[:3],  # Use subset
                patch_size=patch_size,
                patches_per_image=2
            )
            
            assert len(pairs) > 0
            assert all(p.patch_size == patch_size for p in pairs)
            assert all(p.candidate_patch.shape == (patch_size, patch_size) 
                      for p in pairs)


# ============================================================================
# EDGE CASES AND ERROR HANDLING
# ============================================================================

class TestEdgeCases:
    """Test edge cases and error handling."""
    
    def test_single_image(self):
        """Test with single image."""
        img = ImageData(
            data=np.random.rand(128, 128).astype(np.float32),
            file_path="test.tif",
            shape=(128, 128),
            dtype="float32"
        )
        
        pairs = create_positive_pairs([img], patch_size=32, patches_per_image=2)
        assert len(pairs) == 2
    
    def test_very_small_image(self):
        """Test with image smaller than typical patch size."""
        img = ImageData(
            data=np.random.rand(40, 40).astype(np.float32),
            file_path="small.tif",
            shape=(40, 40),
            dtype="float32"
        )
        
        # Should work with smaller patch size
        pairs = create_positive_pairs([img], patch_size=16, patches_per_image=1)
        assert len(pairs) == 1
    
    def test_invalid_split_criteria(self, dummy_image_data_list):
        """Test that invalid split criteria raises error."""
        with pytest.raises(ValueError):
            train_test_split(
                dummy_image_data_list,
                test_size=0.2,
                val_size=0.1,
                split_criteria="invalid_method"
            )
    
    def test_invalid_test_size(self, dummy_image_data_list):
        """Test that invalid test_size is handled."""
        # test_size + val_size > 1.0 should raise error or be handled
        with pytest.raises((ValueError, AssertionError)):
            train_test_split(
                dummy_image_data_list,
                test_size=0.8,
                val_size=0.5,  # Total > 1.0
                split_criteria="random"
            )


# ============================================================================
# PERFORMANCE/SMOKE TESTS
# ============================================================================

class TestPerformance:
    """Basic performance smoke tests."""
    
    def test_large_batch_performance(self):
        """Test that processing many images doesn't crash."""
        # Create 50 dummy images
        images = []
        for i in range(50):
            img = np.random.rand(128, 128).astype(np.float32)
            images.append(ImageData(
                data=img,
                file_path=f"test_{i}.tif",
                shape=img.shape,
                dtype=str(img.dtype)
            ))
        
        # Should complete without errors
        pairs = create_positive_pairs(
            images,
            patch_size=32,
            patches_per_image=3
        )
        
        assert len(pairs) == 50 * 3
    
    def test_many_negatives_per_positive(self, dummy_image_data_list):
        """Test generating many negative samples."""
        positive_pairs = create_positive_pairs(
            dummy_image_data_list,
            patch_size=32,
            patches_per_image=2
        )
        
        # Generate 10 negatives per positive
        negative_pairs = create_negative_pairs(
            positive_pairs,
            all_images=dummy_image_data_list,
            negatives_per_positive=10
        )
        
        assert len(negative_pairs) == len(positive_pairs) * 10


# ============================================================================
# RUN TESTS
# ============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])