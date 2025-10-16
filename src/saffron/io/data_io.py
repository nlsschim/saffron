import numpy as np
import tifffile
from pathlib import Path
from typing import Union, Tuple, Optional
import logging
from dataclasses import dataclass

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class ImageData:
    """
    Dataclass to track images of interest for cell analysis.
    """
    data: np.ndarray
    file_path: str
    shape: Tuple[int, ...]
    dtype: str
    metadata: Optional[dict] = None

    def __post_init__(self):
        """Validate the image data after initialization."""
        if not self._is_valid():
            raise ValueError(f"Invalid image data from {self.file_path}")

    def _is_valid(self) -> bool:
        """Check if the image data is valid."""
        return (
            self.data is not None and
            self.data.size > 0 and
            len(self.shape) >= 2
        )


def validate_im(image: np.ndarray, file_path: str, verbose = False) -> bool:
    """
    Function to validate the input image will work for the given workflow.
    Checks the shape of the image, ensures it isn't empty, etc.

    Args:
        image (np.ndarray): Image array to validate
        file_path (str): Path to the file (for logging)

    Returns:
        bool: True if image is valid, False otherwise
    """
    if image is None:
        logger.error(f"Image is None: {file_path}")
        return False

    if image.size == 0:
        logger.error(f"Image is empty: {file_path}")
        return False

    if len(image.shape) < 2:
        logger.error(
            f"Image must be at least 2D, got shape {image.shape}: {file_path}"
            )
        return False

    if len(image.shape) > 4:
        logger.error(
            f"Image has too many dimensions {image.shape}: {file_path}"
            )
        return False

    # Check for reasonable image dimensions
    min_size, max_size = 32, 10000  # Reasonable bounds for microglia images
    if any(dim < min_size or dim > max_size for dim in image.shape[:2]):
        logger.error(
            f"Image dimensions out of reasonable range {image.shape}: {file_path}"
            )
        return False

    # Check for valid data types
    valid_dtypes = [np.uint8, np.uint16, np.float32, np.float64]
    if image.dtype not in valid_dtypes:
        logger.warning(
            f"Unusual dtype {image.dtype}, converting to float32: {file_path}"
            )

    # Check for NaN or infinite values
    if np.isnan(image).any():
        logger.error(f"Image contains NaN values: {file_path}")
        return False

    if np.isinf(image).any():
        logger.error(f"Image contains infinite values: {file_path}")
        return False

    if verbose:
        logger.info(
            f"Image validation passed: {file_path} - Shape: {image.shape}, Dtype: {image.dtype}"
            )
    return True


def load_tif(file_path: Union[str, Path]) -> ImageData:
    """
    Function to load in tif files to be used for ML.
    Part of loading includes using the validate_im function
    to ensure the image can be used.

    Args:
        file_path (Union[str, Path]): Path to the TIF file

    Returns:
        ImageData: Loaded and validated image data

    Raises:
        FileNotFoundError: If file doesn't exist
        ValueError: If image validation fails
        Exception: For other loading errors
    """
    file_path = Path(file_path)

    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    if not file_path.suffix.lower() in ['.tif', '.tiff']:
        raise ValueError(f"File must be a TIF/TIFF file: {file_path}")

    try:
        # Load the TIF file
        image = tifffile.imread(str(file_path))

        # Convert to numpy array if needed
        if not isinstance(image, np.ndarray):
            image = np.array(image)

        # Validate the loaded image
        if not validate_im(image, str(file_path)):
            raise ValueError(f"Image validation failed for: {file_path}")

        # Normalize dtype if needed
        if image.dtype not in [np.uint8, np.uint16, np.float32, np.float64]:
            image = image.astype(np.float32)

        # Extract metadata if available
        metadata = {}
        try:
            with tifffile.TiffFile(str(file_path)) as tif:
                if tif.pages[0].tags:
                    metadata = {tag.name: tag.value for tag in tif.pages[0].tags.values()}
        except Exception as e:
            logger.warning(f"Could not extract metadata from {file_path}: {e}")

        #logger.info(f"Successfully loaded TIF: {file_path}")
        return ImageData(
            data=image,
            file_path=str(file_path),
            shape=image.shape,
            dtype=str(image.dtype),
            metadata=metadata
        )

    except Exception as e:
        logger.error(f"Error loading TIF file {file_path}: {e}")
        raise


def load_npy(file_path: Union[str, Path]) -> ImageData:
    """
    Function to load in npy files to be used for ML.
    Part of loading includes using the validate_im function
    to ensure the image can be used.

    Args:
        file_path (Union[str, Path]): Path to the NPY file

    Returns:
        ImageData: Loaded and validated image data

    Raises:
        FileNotFoundError: If file doesn't exist
        ValueError: If image validation fails
        Exception: For other loading errors
    """
    file_path = Path(file_path)

    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    if not file_path.suffix.lower() == '.npy':
        raise ValueError(f"File must be a NPY file: {file_path}")

    try:
        # Load the NPY file
        image = np.load(str(file_path))

        # Validate the loaded image
        if not validate_im(image, str(file_path)):
            raise ValueError(f"Image validation failed for: {file_path}")

        logger.info(f"Successfully loaded NPY: {file_path}")
        return ImageData(
            data=image,
            file_path=str(file_path),
            shape=image.shape,
            dtype=str(image.dtype),
            metadata=None  # NPY files don't typically have metadata
        )

    except Exception as e:
        logger.error(f"Error loading NPY file {file_path}: {e}")
        raise


# Utility function for batch loading
def load_images_from_directory(directory: Union[str, Path],
                               file_extensions: list =
                               ['.tif', '.tiff', '.npy']) -> list:
    """
    Load all images from a directory with specified extensions.

    Args:
        directory (Union[str, Path]): Directory containing images
        file_extensions (list): List of file extensions to load

    Returns:
        list: List of ImageData objects
    """
    directory = Path(directory)
    if not directory.exists():
        raise FileNotFoundError(f"Directory not found: {directory}")

    loaded_images = []

    for ext in file_extensions:
        for file_path in directory.glob(f"*{ext}"):
            try:
                if ext.lower() in ['.tif', '.tiff']:
                    img_data = load_tif(file_path)
                elif ext.lower() == '.npy':
                    img_data = load_npy(file_path)
                else:
                    continue

                loaded_images.append(img_data)

            except Exception as e:
                logger.warning(f"Failed to load {file_path}: {e}")
                continue

    logger.info(f"Successfully loaded {len(loaded_images)} images from {directory}")
    return loaded_images


class ImageDataset:
    def __init__(self, image_data_list: list[ImageData]):
        self.images = image_data_list

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        return self.images[idx]

    def get_shapes(self):
        return [img.shape for img in self.images]

    def get_batch(self, indices):
        return [self.images[i].data for i in indices]


if __name__ == "__main__":
    # Example usage
    try:
        # Load a single TIF file
        # img_data = load_tif("path/to/your/image.tif")
        # print(f"Loaded image: {img_data.shape}")

        # Load all images from a directory
        # images = load_images_from_directory("path/to/your/directory")
        # print(f"Loaded {len(images)} images")

        print("Data IO module loaded successfully!")

    except Exception as e:
        print(f"Error in example usage: {e}")
