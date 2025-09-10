import pytest
import numpy as np
import tempfile
import os
from unittest.mock import patch, MagicMock
from pathlib import Path
import tifffile
from PIL import Image

# Assuming the module is named data_io
# from data_io import load_tif, load_npy, validate_im


class TestDataIO:
    """Test suite for the Data IO module functions."""
    
    @pytest.fixture
    def temp_dir(self):
        """Create a temporary directory for test files."""
        with tempfile.TemporaryDirectory() as temp_dir:
            yield temp_dir
    
    @pytest.fixture
    def valid_image_array(self):
        """Create a valid test image array."""
        return np.random.randint(0, 255, (512, 512, 3), dtype=np.uint8)
    
    @pytest.fixture
    def valid_grayscale_array(self):
        """Create a valid grayscale test image array."""
        return np.random.randint(0, 255, (512, 512), dtype=np.uint8)
    
    @pytest.fixture
    def multichannel_array(self):
        """Create a multi-channel test image array (common in microscopy)."""
        return np.random.randint(0, 65535, (4, 512, 512), dtype=np.uint16)


class TestLoadTif:
    """Tests for the load_tif function."""
    
    def test_load_valid_tif_file(self, temp_dir, valid_image_array):
        """Test loading a valid TIF file."""
        # Create a test TIF file
        tif_path = os.path.join(temp_dir, "test_image.tif")
        tifffile.imwrite(tif_path, valid_image_array)
        
        # Test loading
        loaded_image = load_tif(tif_path)
        
        assert loaded_image is not None
        assert isinstance(loaded_image, np.ndarray)
        np.testing.assert_array_equal(loaded_image, valid_image_array)
    
    def test_load_grayscale_tif(self, temp_dir, valid_grayscale_array):
        """Test loading a grayscale TIF file."""
        tif_path = os.path.join(temp_dir, "test_grayscale.tif")
        tifffile.imwrite(tif_path, valid_grayscale_array)
        
        loaded_image = load_tif(tif_path)
        
        assert loaded_image.shape == valid_grayscale_array.shape
        np.testing.assert_array_equal(loaded_image, valid_grayscale_array)
    
    def test_load_multichannel_tif(self, temp_dir, multichannel_array):
        """Test loading a multi-channel TIF file (common in microscopy)."""
        tif_path = os.path.join(temp_dir, "test_multichannel.tif")
        tifffile.imwrite(tif_path, multichannel_array)
        
        loaded_image = load_tif(tif_path)
        
        assert loaded_image.shape == multichannel_array.shape
        np.testing.assert_array_equal(loaded_image, multichannel_array)
    
    def test_load_nonexistent_file(self):
        """Test loading a file that doesn't exist."""
        with pytest.raises((FileNotFoundError, IOError)):
            load_tif("nonexistent_file.tif")
    
    def test_load_invalid_file_extension(self, temp_dir):
        """Test loading a file with wrong extension."""
        invalid_path = os.path.join(temp_dir, "test.txt")
        with open(invalid_path, 'w') as f:
            f.write("not an image")
        
        with pytest.raises((ValueError, IOError)):
            load_tif(invalid_path)
    
    def test_load_corrupted_tif(self, temp_dir):
        """Test loading a corrupted TIF file."""
        corrupted_path = os.path.join(temp_dir, "corrupted.tif")
        with open(corrupted_path, 'wb') as f:
            f.write(b"corrupted tif data")
        
        with pytest.raises((ValueError, IOError)):
            load_tif(corrupted_path)
    
    def test_load_empty_string_path(self):
        """Test loading with empty string path."""
        with pytest.raises((ValueError, FileNotFoundError)):
            load_tif("")
    
    def test_load_none_path(self):
        """Test loading with None path."""
        with pytest.raises((TypeError, ValueError)):
            load_tif(None)
    
    @patch('data_io.validate_im')
    def test_validate_im_called(self, mock_validate, temp_dir, valid_image_array):
        """Test that validate_im is called during loading."""
        tif_path = os.path.join(temp_dir, "test_image.tif")
        tifffile.imwrite(tif_path, valid_image_array)
        mock_validate.return_value = True
        
        load_tif(tif_path)
        
        mock_validate.assert_called_once()
    
    @patch('data_io.validate_im')
    def test_load_fails_validation(self, mock_validate, temp_dir, valid_image_array):
        """Test that loading fails when validation fails."""
        tif_path = os.path.join(temp_dir, "test_image.tif")
        tifffile.imwrite(tif_path, valid_image_array)
        mock_validate.side_effect = ValueError("Image validation failed")
        
        with pytest.raises(ValueError, match="Image validation failed"):
            load_tif(tif_path)


class TestLoadNpy:
    """Tests for the load_npy function."""
    
    def test_load_valid_npy_file(self, temp_dir, valid_image_array):
        """Test loading a valid NPY file."""
        npy_path = os.path.join(temp_dir, "test_array.npy")
        np.save(npy_path, valid_image_array)
        
        loaded_array = load_npy(npy_path)
        
        assert loaded_array is not None
        assert isinstance(loaded_array, np.ndarray)
        np.testing.assert_array_equal(loaded_array, valid_image_array)
    
    def test_load_different_dtypes(self, temp_dir):
        """Test loading NPY files with different data types."""
        test_arrays = {
            'uint8': np.random.randint(0, 255, (100, 100), dtype=np.uint8),
            'uint16': np.random.randint(0, 65535, (100, 100), dtype=np.uint16),
            'float32': np.random.random((100, 100)).astype(np.float32),
            'float64': np.random.random((100, 100)).astype(np.float64),
        }
        
        for dtype_name, test_array in test_arrays.items():
            npy_path = os.path.join(temp_dir, f"test_{dtype_name}.npy")
            np.save(npy_path, test_array)
            
            loaded_array = load_npy(npy_path)
            
            assert loaded_array.dtype == test_array.dtype
            np.testing.assert_array_equal(loaded_array, test_array)
    
    def test_load_different_shapes(self, temp_dir):
        """Test loading NPY files with different shapes."""
        test_shapes = [
            (512, 512),          # 2D grayscale
            (512, 512, 1),       # 2D with single channel
            (512, 512, 3),       # 2D RGB
            (4, 512, 512),       # Multi-channel microscopy format
            (10, 512, 512, 3),   # Stack of RGB images
        ]
        
        for i, shape in enumerate(test_shapes):
            test_array = np.random.randint(0, 255, shape, dtype=np.uint8)
            npy_path = os.path.join(temp_dir, f"test_shape_{i}.npy")
            np.save(npy_path, test_array)
            
            loaded_array = load_npy(npy_path)
            
            assert loaded_array.shape == shape
            np.testing.assert_array_equal(loaded_array, test_array)
    
    def test_load_nonexistent_npy_file(self):
        """Test loading a nonexistent NPY file."""
        with pytest.raises(FileNotFoundError):
            load_npy("nonexistent_file.npy")
    
    def test_load_corrupted_npy(self, temp_dir):
        """Test loading a corrupted NPY file."""
        corrupted_path = os.path.join(temp_dir, "corrupted.npy")
        with open(corrupted_path, 'wb') as f:
            f.write(b"corrupted numpy data")
        
        with pytest.raises((ValueError, IOError)):
            load_npy(corrupted_path)
    
    def test_load_empty_npy_file(self, temp_dir):
        """Test loading an empty NPY file."""
        empty_path = os.path.join(temp_dir, "empty.npy")
        empty_array = np.array([])
        np.save(empty_path, empty_array)
        
        loaded_array = load_npy(empty_path)
        
        assert loaded_array.size == 0
        np.testing.assert_array_equal(loaded_array, empty_array)
    
    @patch('data_io.validate_im')
    def test_validate_im_called(self, mock_validate, temp_dir, valid_image_array):
        """Test that validate_im is called during NPY loading."""
        npy_path = os.path.join(temp_dir, "test_array.npy")
        np.save(npy_path, valid_image_array)
        mock_validate.return_value = True
        
        load_npy(npy_path)
        
        mock_validate.assert_called_once()
    
    @patch('data_io.validate_im')
    def test_load_npy_fails_validation(self, mock_validate, temp_dir, valid_image_array):
        """Test that NPY loading fails when validation fails."""
        npy_path = os.path.join(temp_dir, "test_array.npy")
        np.save(npy_path, valid_image_array)
        mock_validate.side_effect = ValueError("Image validation failed")
        
        with pytest.raises(ValueError, match="Image validation failed"):
            load_npy(npy_path)


class TestValidateIm:
    """Tests for the validate_im function."""
    
    def test_validate_valid_2d_image(self, valid_grayscale_array):
        """Test validation of a valid 2D grayscale image."""
        result = validate_im(valid_grayscale_array)
        assert result is True
    
    def test_validate_valid_3d_image(self, valid_image_array):
        """Test validation of a valid 3D color image."""
        result = validate_im(valid_image_array)
        assert result is True
    
    def test_validate_valid_multichannel_image(self, multichannel_array):
        """Test validation of a valid multi-channel image."""
        result = validate_im(multichannel_array)
        assert result is True
    
    def test_validate_empty_array(self):
        """Test validation of an empty array."""
        empty_array = np.array([])
        
        with pytest.raises(ValueError, match="Image is empty"):
            validate_im(empty_array)
    
    def test_validate_none_input(self):
        """Test validation of None input."""
        with pytest.raises((TypeError, ValueError)):
            validate_im(None)
    
    def test_validate_wrong_dimensions(self):
        """Test validation of arrays with wrong dimensions."""
        # 1D array
        array_1d = np.random.randint(0, 255, (100,), dtype=np.uint8)
        with pytest.raises(ValueError, match="Invalid image dimensions"):
            validate_im(array_1d)
        
        # 4D array (too many dimensions for typical use)
        array_4d = np.random.randint(0, 255, (10, 3, 100, 100), dtype=np.uint8)
        with pytest.raises(ValueError, match="Invalid image dimensions"):
            validate_im(array_4d)
    
    def test_validate_zero_dimension(self):
        """Test validation of arrays with zero dimensions."""
        arrays_with_zero_dim = [
            np.zeros((0, 512)),      # Height = 0
            np.zeros((512, 0)),      # Width = 0
            np.zeros((0, 0)),        # Both = 0
            np.zeros((512, 512, 0)), # Channels = 0
        ]
        
        for array in arrays_with_zero_dim:
            with pytest.raises(ValueError, match="Image has zero dimensions"):
                validate_im(array)
    
    def test_validate_minimum_size(self):
        """Test validation of images that are too small."""
        # Very small images that might be problematic for ML
        small_arrays = [
            np.ones((1, 1), dtype=np.uint8),
            np.ones((2, 2), dtype=np.uint8),
            np.ones((5, 5), dtype=np.uint8),
        ]
        
        for array in small_arrays:
            with pytest.raises(ValueError, match="Image is too small"):
                validate_im(array)
    
    def test_validate_non_numpy_array(self):
        """Test validation of non-numpy array inputs."""
        invalid_inputs = [
            [1, 2, 3],           # Python list
            "not an array",      # String
            123,                 # Integer
            {"not": "array"},    # Dictionary
        ]
        
        for invalid_input in invalid_inputs:
            with pytest.raises(TypeError, match="Input must be a numpy array"):
                validate_im(invalid_input)
    
    def test_validate_unsupported_dtype(self):
        """Test validation of arrays with unsupported data types."""
        # String array
        string_array = np.array([["a", "b"], ["c", "d"]])
        with pytest.raises(ValueError, match="Unsupported data type"):
            validate_im(string_array)
        
        # Complex numbers
        complex_array = np.array([[1+2j, 3+4j], [5+6j, 7+8j]])
        with pytest.raises(ValueError, match="Unsupported data type"):
            validate_im(complex_array)
    
    def test_validate_extreme_values(self):
        """Test validation with extreme values."""
        # Test with inf and nan values
        array_with_inf = np.array([[1.0, 2.0], [np.inf, 4.0]])
        with pytest.raises(ValueError, match="Image contains invalid values"):
            validate_im(array_with_inf)
        
        array_with_nan = np.array([[1.0, 2.0], [3.0, np.nan]])
        with pytest.raises(ValueError, match="Image contains invalid values"):
            validate_im(array_with_nan)
    
    def test_validate_different_channel_counts(self):
        """Test validation of images with different channel counts."""
        valid_channel_counts = [1, 3, 4]  # Grayscale, RGB, RGBA
        
        for channels in valid_channel_counts:
            if channels == 1:
                array = np.random.randint(0, 255, (100, 100), dtype=np.uint8)
            else:
                array = np.random.randint(0, 255, (100, 100, channels), dtype=np.uint8)
            
            result = validate_im(array)
            assert result is True
        
        # Test invalid channel count
        array_invalid_channels = np.random.randint(0, 255, (100, 100, 7), dtype=np.uint8)
        with pytest.raises(ValueError, match="Invalid number of channels"):
            validate_im(array_invalid_channels)


class TestIntegrationDataIO:
    """Integration tests for the Data IO module."""
    
    def test_full_workflow_tif_to_validation(self, temp_dir, valid_image_array):
        """Test the complete workflow from TIF loading to validation."""
        tif_path = os.path.join(temp_dir, "test_workflow.tif")
        tifffile.imwrite(tif_path, valid_image_array)
        
        # This should work without raising any exceptions
        loaded_image = load_tif(tif_path)
        validation_result = validate_im(loaded_image)
        
        assert validation_result is True
        assert loaded_image.shape == valid_image_array.shape
    
    def test_full_workflow_npy_to_validation(self, temp_dir, valid_image_array):
        """Test the complete workflow from NPY loading to validation."""
        npy_path = os.path.join(temp_dir, "test_workflow.npy")
        np.save(npy_path, valid_image_array)
        
        # This should work without raising any exceptions
        loaded_array = load_npy(npy_path)
        validation_result = validate_im(loaded_array)
        
        assert validation_result is True
        assert loaded_array.shape == valid_image_array.shape
    
    def test_round_trip_consistency(self, temp_dir, valid_image_array):
        """Test that saving and loading maintains data integrity."""
        # Test TIF round trip
        tif_path = os.path.join(temp_dir, "round_trip.tif")
        tifffile.imwrite(tif_path, valid_image_array)
        loaded_tif = load_tif(tif_path)
        
        # Test NPY round trip
        npy_path = os.path.join(temp_dir, "round_trip.npy")
        np.save(npy_path, valid_image_array)
        loaded_npy = load_npy(npy_path)
        
        # Both should be identical to original
        np.testing.assert_array_equal(loaded_tif, valid_image_array)
        np.testing.assert_array_equal(loaded_npy, valid_image_array)


# Pytest configuration for running tests
if __name__ == "__main__":
    pytest.main([__file__, "-v"])