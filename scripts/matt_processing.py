import os
import csv
import cv2
from skimage import io
import numpy as np
from skimage.measure import block_reduce, label, regionprops
from skimage.color import label2rgb
import tifffile as tiff
import matplotlib.pyplot as plt
from nd2 import ND2File
from pathlib import Path
from skimage.morphology import remove_small_objects
from skimage.segmentation import clear_border
from skimage import filters
from scipy import ndimage

from scipy.spatial import cKDTree

import pandas as pd
from matplotlib.patches import Circle

from sklearn.metrics.pairwise import paired_distances

from typing import Optional
import warnings

import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage


__all__ = [
    "apply_gaussian_filter",
    "remove_baseline",
    "binarize",
]


def apply_gaussian_filter(
    img: np.ndarray, sigma: int, radius: Optional[int] = None, debug: bool = False
) -> tuple[np.ndarray, np.ndarray]:
    """applies a 2D highpass filter to remove baseline drift and detect edges, or to blur image

    Parameters:
    ----------
    img: image

    sigma: sigma for gaussian kernel

    radius: Optional radius for gaussian kernel

    debug: whether to display figures intended for development

    Returns:
    -------
    lowpass filtered image, highpass filtered image
    """
    # normalize and discretize pixel intensities
    img = img.copy().astype(np.float64)
    img -= np.min(img.flatten())
    img *= 256 / np.max(img.flatten())
    img = np.round(img, 0).astype(np.int64)

    # filter image
    gauss_lowpass = ndimage.gaussian_filter(img, sigma, radius=radius)
    gauss_highpass = np.array(img, dtype=np.int64) - np.array(
        gauss_lowpass, dtype=np.int64
    )
    min_highpass = np.min(np.min(gauss_highpass))
    if min_highpass < 0:
        gauss_highpass -= min_highpass

    if debug:
        # gauss_highpass = np.max(np.max(gauss_highpass)) - gauss_highpass
        print(f"original range: ({np.min(np.min(img))}, {np.max(np.max(img))})")
        print(
            f"lowpass range:  ({np.min(np.min(gauss_lowpass))}, {np.max(np.max(gauss_lowpass))})"
        )
        print(
            f"highpass range: ({np.min(np.min(gauss_highpass))}, {np.max(np.max(gauss_highpass))})"
        )

        fig0 = plt.figure()
        ax0 = plt.gca()
        fig0.suptitle("Original")
        ax0.imshow(img, cmap="gray")

        fig1 = plt.figure()
        ax1 = plt.gca()
        fig1.suptitle("Lowpass")
        ax1.imshow(gauss_lowpass, cmap="gray")

        fig2 = plt.figure()
        ax2 = plt.gca()
        fig2.suptitle("Highpass")
        ax2.imshow(gauss_highpass, cmap="gray")

        plt.show()
        plt.close()

    return (gauss_lowpass, gauss_highpass)


def remove_baseline(img: np.ndarray, factor: int | float = 4) -> np.ndarray:
    """uses a highpass filter to remove baseline"""
    dim = min([*np.shape(img)])
    baseline, no_baseline = apply_gaussian_filter(img, sigma=int(np.ceil(dim / factor)))

    return no_baseline


def remove_baseline_DEBUGGING(img: np.ndarray, factor: int | float = 4) -> np.ndarray:
    """uses a highpass filter to remove baseline"""
    dim = min([*np.shape(img)])
    import matplotlib.pyplot as plt

    factors = 5 * np.logspace(0, 2, num=5, endpoint=True, base=10)
    for f in [*factors, 128]:
        baseline, no_baseline = apply_gaussian_filter(img, sigma=int(f), debug=False)
        fig, ax = plt.subplots(1, 3)
        fig.suptitle(f"{f}\n{np.min(no_baseline):.3f}, {np.max(no_baseline):.3f}")
        ax[0].imshow(img)
        ax[1].imshow(baseline)
        ax[2].imshow(no_baseline)

    plt.show()

    import sys

    sys.exit()

    return no_baseline


def binarize(
    highpass_img: np.ndarray,
    opt_thresh: bool = False,
    thresh: int | float = 0.5,
    show_hist: bool = False,
) -> np.ndarray:
    """masks image

    Parameters:
        img: should be highpass-filtered to remove baseline and enhance edges
        opt_thresh: whether the program should decide the optimal pixel intensity threshold
        thresh: float between 0 and 1. Sets threshold for binarizing image
        show_hist: for development only. shows histogram of intensities

    Returns:
        binarized image of same dimensions

    Raises:
        ValueError if thresh is out of range
    """
    if 0 > thresh or thresh > 1:
        raise ValueError(f"thresh must be between 0 and 1. Received value of {thresh}")

    max_val = int(np.max(np.max(highpass_img)))
    min_val = int(np.min(np.min(highpass_img)))
    thresh_val = thresh * max_val + (1 - thresh) * min_val

    if opt_thresh:
        try:
            hist, bins = np.histogram(
                highpass_img.flatten(),
                bins=round(max_val - min_val + 1),
            )
            peak = np.argmax(hist)
            hist_high = hist[peak:]
            bins_high = bins[peak:-1]
            thresh_val = np.min(bins_high[hist_high < 0.5 * hist[peak]])
            # thresh_val = bins_high[np.argmin(np.diff(hist_high))]
        except:
            warnings.warn(
                "Unable to use optimal threshold. Using default threshold.",
                category=UserWarning,
            )

    if show_hist:
        plt.figure()
        plt.title("Pixel Intensities")
        plt.hist(highpass_img.copy().flatten(), bins=100)
        plt.axvline(thresh_val, color="r")
        plt.show()
        plt.close()

    bin_img = np.zeros_like(highpass_img, dtype=np.int64)
    bin_img[highpass_img > thresh_val] = 1

    return bin_img

from math import sqrt

import numpy as np

# from reader import nd2_img_reader, get_stain
# from .filters import (
#     apply_gaussian_filter,
#     remove_baseline,
#     binarize,
#     remove_baseline_DEBUGGING,
# )



def mask_img(img: np.ndarray, **kwargs) -> tuple[np.ndarray, np.ndarray]:
    """Masks image

    Parameters:
    ----------
    img: np.ndarray of image (N x M)

    bin_thresh: intensity threshold to binarize image (range: 0-1, default = 0.1)

    filter_coeff: coefficient to filter image. Smaller numbers remove baseline more effectively (default = 4)


    Returns:
    -------
    masked image containing filtered values within cell bounds

    mask used for this calculation
    """
    params = {
        "bin_thresh": 0.1,
        "filter_coeff": 4,
        "opt_thresh": False,
    }
    params.update(kwargs)

    filtered = remove_baseline(img, params["filter_coeff"])
    mask = binarize(
        filtered, opt_thresh=params["opt_thresh"], thresh=params["bin_thresh"]
    )

    masked_img = mask * filtered

    values = masked_img.copy().flatten()
    nonzero = values[values > 0]
    shift = mask * float(np.min(nonzero)) if (len(nonzero) > 0) else 0
    masked_img_shifted = masked_img - shift

    return (masked_img_shifted, mask)


def highlight_mask_edges(mask: np.ndarray) -> np.ndarray:
    """uses gaussian filter to highlight edges"""
    centers, edges = apply_gaussian_filter(mask, sigma=1, radius=2)
    edges_bin = np.zeros_like(edges, dtype=np.int64)
    edges_bin[edges > 0.5] = 1

    return edges_bin



def preprocess_image(
    img: np.ndarray, filter_coeff: int, bin_thresh: float = 0.1, opt_thresh: bool = False
) -> tuple[np.ndarray, ...]:
    """Preprocesses images

    Arguments:
        img: DAPI-stained microscopy image

    Results:
        preprocessed image,
        mask,
        mask_edges,
    """
    # normalize
    norm_img = img.copy().astype(np.float64)
    norm_img -= np.min(norm_img.flatten())
    norm_img /= np.max(norm_img.flatten())

    # develop masks
    masked_img, mask = mask_img(
        norm_img, bin_thresh=bin_thresh, filter_coeff=filter_coeff, opt_thresh=opt_thresh
    )
    #mask_edges = highlight_mask_edges(mask)

    return (masked_img, mask)#, mask_edges)

def nd2_to_tif(path, file_name):
    nd2_path = Path(path) / file_name
    tif_path = nd2_path.with_suffix(".tif")

    with ND2File(nd2_path) as nd2_file:
        nd2_data = nd2_file.asarray()
        tiff.imwrite(tif_path, nd2_data)

def create_microglia_mask(image, threshold_methold=filters.threshold_li):

    thresh_li = threshold_methold(image)
    binary_li = image > thresh_li
    binary_li = remove_small_objects(binary_li, min_size=100)
    binary_li = ndimage.binary_fill_holes(binary_li)
    return binary_li

def create_mitochondria_mask(image, percentile=99, min_size=10):

    perc = np.percentile(image, percentile)
    mito_mask = remove_small_objects(image>perc, min_size=min_size)
    return mito_mask

def create_nuclei_mask(image):
    
    thresh_li = filters.threshold_li(image)
    binary_li = image > thresh_li
    binary_li = ndimage.binary_fill_holes(binary_li)
    nuclei_mask = remove_small_objects(binary_li)
    return nuclei_mask

directories = [
    '/Users/nelsschimek/Documents/nancelab/Data/mito_images/brendan_full_analysis/tifs/ORST/cd11b',]
#     '/Users/nelsschimek/Documents/nancelab/Data/mito_images/brendan_full_analysis/tifs/OGD_only/hif1a',
#     '/Users/nelsschimek/Documents/nancelab/Data/mito_images/brendan_full_analysis/tifs/HC/hif1a',
#     '/Users/nelsschimek/Documents/nancelab/Data/mito_images/brendan_full_analysis/tifs/30mR_control/hif1a',
#     '/Users/nelsschimek/Documents/nancelab/Data/mito_images/brendan_full_analysis/tifs/24R_OGD/hif1a',
#     '/Users/nelsschimek/Documents/nancelab/Data/mito_images/brendan_full_analysis/tifs/24R_control/hif1a',
#     '/Users/nelsschimek/Documents/nancelab/Data/mito_images/brendan_full_analysis/tifs/2R_control/hif1a',
#     '/Users/nelsschimek/Documents/nancelab/Data/mito_images/brendan_full_analysis/tifs/2R_OGD/hif1a'
# ]

for dir in directories:
    directory = Path(dir)

    # Get all .nd2 files
    nd2_files = list(directory.glob("*.nd2"))

    # If you want full paths as strings:
    nd2_file_paths = [str(f) for f in nd2_files]

    for file in nd2_files:

        nd2_to_tif(directory, file)

    # Get all .nd2 files
    tif_files = list(directory.glob("*.tif"))

    # If you want full paths as strings:
    tif_file_paths = [str(f) for f in tif_files]

    images = [tiff.imread(f) for f in tif_file_paths]
    len(images)

    for i, image in enumerate(images):
        print(i)
        out = preprocess_image(image[0,:,:], filter_coeff=4, bin_thresh=0.1)
        print(f"processing {tif_file_paths[i]}")
        np.save(f'/Users/nelsschimek/Documents/nancelab/Data/mito_images/brendan_full_analysis/tifs/ORST/cd11b/{str(tif_file_paths[i]).split("/")[-1]}'.replace(".tif", ".npy"), out)
        print(f"Succeeded in processing {tif_file_paths[i]}")
        print()

