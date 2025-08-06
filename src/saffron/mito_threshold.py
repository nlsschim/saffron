#!/usr/bin/env python
# coding: utf-8

# In[ ]:


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


# In[ ]:


def nd2_to_tif(path, file_name):
    nd2_path = Path(path) / file_name
    tif_path = nd2_path.with_suffix(".tif")

    with ND2File(nd2_path) as nd2_file:
        nd2_data = nd2_file.asarray()
        tiff.imwrite(tif_path, nd2_data)


# ### Start code

# In[ ]:


condition = "OGD_only"


# In[ ]:


all_prop = pd.read_csv("/Users/nelsschimek/Downloads/All_Properties.csv")


# In[ ]:


directory = Path(f'/Users/nelsschimek/Documents/nancelab/Data/mito_images/brendan_full_analysis/tifs/{condition}/cd11b')

# # Get all .nd2 files
# nd2_files = list(directory.glob("*.nd2"))

# # If you want full paths as strings:
# nd2_file_paths = [str(f) for f in nd2_files]


# In[ ]:


# for file in nd2_files:

#     nd2_to_tif(directory, file)

# # Get all .nd2 files
# tif_files = list(directory.glob("*.tif"))

# # If you want full paths as strings:
# tif_file_paths = [str(f) for f in tif_files]


# In[ ]:


microglia_masks = Path(f"/Users/nelsschimek/Documents/nancelab/Data/mito_images/brendan_full_analysis/tifs/{condition}/cd11b/li_thresh/converted_tiffs/microglia_masks")

microg_npys = sorted(list(microglia_masks.glob("*li_thresh.tif")))
microglia_mask_filepaths = ([str(f) for f in microg_npys])

loaded_microglia_masks = [tiff.imread(f) for f in sorted(microglia_mask_filepaths)]

len(loaded_microglia_masks)


# In[ ]:


mitochondria_masks = Path(f"/Users/nelsschimek/Documents/nancelab/Data/mito_images/brendan_full_analysis/tifs/{condition}/cd11b/li_thresh/converted_tiffs/mitochondria_masks")
mito_npys = sorted(list(mitochondria_masks.glob("*.npy")))
mito_mask_filepaths = ([str(f) for f in mito_npys])
loaded_mitochondria_masks = [np.load(f) for f in (mito_mask_filepaths)]

len(loaded_mitochondria_masks)


# In[ ]:


nuclei_masks = Path(f"/Users/nelsschimek/Documents/nancelab/Data/mito_images/brendan_full_analysis/tifs/{condition}/cd11b/li_thresh/converted_tiffs/nuclei_masks")
nuclei_npys = sorted(list(nuclei_masks.glob("*.npy")))
nuclei_mask_filepaths = ([str(f) for f in nuclei_npys])
loaded_nuclei_masks = [np.load(f) for f in (nuclei_mask_filepaths)]

len(loaded_nuclei_masks)


# In[ ]:


print((nuclei_mask_filepaths)[11])
print((mito_mask_filepaths)[11])
print((microglia_mask_filepaths)[11])


# In[ ]:


import numpy as np
from skimage.draw import disk

def create_mask_from_centroids(df, image_shape):
    mask = np.zeros(image_shape, dtype=np.uint8)



    for _, row in df.iterrows():
        center = ((row['i']*2), (row['j']*2))  # (row, col) = (y, x)
        radius = (row['ideal_radius']*2)

        rr, cc = disk(center, radius, shape=image_shape)
        mask[rr, cc] = 1

    return mask


# ### Microglia

# ### Overlay

# In[ ]:


def create_microglia_mask(image, threshold_methold=filters.threshold_li):

    thresh_li = filters.threshold_li(image)
    binary_li = image > thresh_li

    objects = label(binary_li)
    objects = clear_border(objects)
    large_objects = remove_small_objects(objects, min_size=8590)
    small_objects = label((objects ^ large_objects) > thresh_li)

    binary_li = ndimage.binary_fill_holes(remove_small_objects(small_objects > thresh_li, min_size=71))

    scaled_img = ((image - image.min()) * (1/(image.max() - image.min()) * 255)).astype('uint8')
    hist = np.histogram(scaled_img.flatten(), range=[0,50], bins=50)

    # if hist[0][0] > (hist[0][1] + hist[0][2]):

    #     # Save the binary mask as .npy
    #     #np.save(output_path, binary_li)
    #     #print(f"Processed: {input_path} -> {output_path}")

    # else:
    #     print("Too much background, not using image")

    return binary_li

def create_mitochondria_mask(image, percentile=99, min_size=10):

    perc = np.percentile(image, percentile)
    mito_mask = remove_small_objects(image>perc, min_size=min_size)
    return mito_mask

def create_nuclei_mask(image):

    thresh_li = filters.threshold_li(image)
    binary_li = image > thresh_li
    nuclei_mask = remove_small_objects(binary_li)
    nuclei_mask = ndimage.binary_fill_holes(nuclei_mask)
    return nuclei_mask

def create_matt_mask(image):
    thresh_li = filters.threshold_li(image)
    binary_li = image > thresh_li
    nuclei_mask = remove_small_objects(image)
    nuclei_mask = ndimage.binary_fill_holes(nuclei_mask)
    return nuclei_mask


# In[ ]:


directory


# In[ ]:


# mito_masks = []
# microg_masks = []
# pyknotic_nuclei_masks = []
# non_pyknotic_nuclei_masks = []

nuclei_masks = []

#fig, axes = plt.subplots(len(tif_file_paths), 1, figsize=(10,5*len(tif_file_paths)))

for file_name in microglia_mask_filepaths:

    #file_idx = tif_file_paths.index(file_name)

    microglia_im = tiff.imread(file_name)
    tif_name = file_name.split("/")[-1].replace("_li_thresh.tif", ".tif")

    image = tiff.imread(str(directory)+"/"+tif_name)


    file_name = (tif_name.split("/")[-1].split(".")[0])
    print(file_name)

    # scaled_img = ((image[1,:,:] - image[1,:,:].min()) * (1/(image[1,:,:].max() - image[1,:,:].min()) * 255)).astype('uint8')
    # hist = np.histogram(scaled_img.flatten(), range=[0,50], bins=50)
    # #ax.hist(scaled_img.flatten(), range=[0,50], bins=50)

    #mito_masks.append(create_mitochondria_mask(image[2,:,:]))
    # microg_masks.append(create_microglia_mask(image[1,:,:]))

    mito_mask = create_mitochondria_mask(image[2,:,:])
    np.save(Path(f"/Users/nelsschimek/Documents/nancelab/Data/mito_images/brendan_full_analysis/tifs/{condition}/cd11b/li_thresh/converted_tiffs/mitochondria_masks/mito_{file_name}"), arr=mito_mask)

    # pyknotic_mask = pyknotic_df[pyknotic_df["file_name"] == file_name]
    # non_pyknotic_mask = non_pyknotic_df[non_pyknotic_df["file_name"] == file_name]

    # pyknotic_nuclei_mask = create_mask_from_centroids(pyknotic_mask, (1024,1024))
    # non_pyknotic_nuclei_mask = create_mask_from_centroids(non_pyknotic_mask, (1024, 1024))
    # pyknotic_nuclei_masks.append(pyknotic_nuclei_mask)
    # non_pyknotic_nuclei_masks.append(non_pyknotic_nuclei_mask)

    nuclei_mask = all_prop[all_prop["file_name"] == file_name]
    nuclei_mask = create_mask_from_centroids(nuclei_mask, (1024, 1024))
    # nuclei_masks.append(nuclei_mask)
    np.save(Path(f"/Users/nelsschimek/Documents/nancelab/Data/mito_images/brendan_full_analysis/tifs/{condition}/cd11b/li_thresh/converted_tiffs/nuclei_masks/nuclei_{file_name}"), arr=nuclei_mask)



# print(len(microg_masks))
# print(len(mito_masks))
# print(len(pyknotic_nuclei_masks))
# print(len(non_pyknotic_nuclei_masks))

print(len(nuclei_masks))


# In[ ]:


#grayscale_background = np.mean(images[0][:3, :, :], axis=2)
#overlay = np.stack([grayscale_background]*3, axis=-1).astype(np.uint8)


def create_overlay(mito_mask, nuclei_mask, microglia_mask):
    # Initialize overlay image
    overlay = np.zeros([mito_mask.shape[0], mito_mask.shape[1], 3], dtype=np.uint8)

    # Define colors
    nuclei_color = [0, 0, 255]        # Blue
    microglia_color = [0, 255, 0]     # green
    mito_color = [255, 0, 255]        # Magenta

    # Create composite masks
    nuclei_within_microglia = np.logical_and(nuclei_mask, microglia_mask)

    # Assign colors
    overlay[microglia_mask > 0] = microglia_color
    overlay[mito_mask > 0] = mito_color
    overlay[nuclei_within_microglia > 0] = nuclei_color
    #overlay[nuclei_mask > 0] = nuclei_color

    return overlay#, nuclei_mask#, nuclei_within_microglia


# In[ ]:


fig, axes = plt.subplots(40, 1, figsize=(10, 40*4))

for ax, micro_mask, mito_mask, nuc_mask in zip (axes, loaded_microglia_masks, loaded_mitochondria_masks, loaded_nuclei_masks):

    overlay = create_overlay((mito_mask.astype(int)), nuc_mask, micro_mask)
    ax.imshow(overlay)


# In[ ]:


import pandas as pd
import numpy as np
from scipy.spatial import cKDTree
from skimage.measure import regionprops, label

def get_distance_dictionary(file_name, nuclei_mask, mitochondria_mask, microglia_mask, max_distance, 
                           microglia_shape_modes, nuclei_count=0):
    """
    Calculate distances between mitochondria and nuclei.
    Only considers nuclei that overlap with microglia.

    Parameters:
    -----------
    file_name : str
        Name of the file being processed
    nuclei_mask : np.ndarray
        Labeled mask of nuclei
    mitochondria_mask : np.ndarray  
        Labeled mask of mitochondria
    microglia_mask : np.ndarray
        Labeled mask of microglia
    max_distance : float
        Maximum distance to consider for mito-nuclei pairing
    microglia_shape_modes : dict
        Mapping of microglia_label -> shape_mode_integer (e.g., {1: 0, 2: 3, 3: 1, ...})
        where shape_mode is an integer 0-4 representing different morphological states
    nuclei_count : int
        Offset for nuclei labeling

    Returns:
    --------
    tuple
        (distance_dict, df) containing distance data and summary DataFrame
    """

    # Convert microglia_shape_modes to dict if it's a tuple/list of tuples
    if isinstance(microglia_shape_modes, (tuple, list)):
        microglia_shape_modes = dict(microglia_shape_modes)

    # Get region properties for all masks
    nuclei_props = regionprops(label(nuclei_mask))
    mitochondria_props = regionprops(label(mitochondria_mask))
    microglia_props = regionprops(label(microglia_mask))

    # Find nuclei that overlap with microglia and track which microglia
    overlapping_nuclei = []
    nuclei_microglia_mapping = {}  # Maps nucleus_label -> list of microglia_labels

    # Create a mapping from pixel values to regionprops labels for microglia
    microglia_pixel_to_label = {}
    for microglia_obj in microglia_props:
        # For each microglia object, map all its pixel values to its regionprops label
        for coord in microglia_obj.coords:
            y, x = coord
            pixel_value = microglia_mask[y, x]
            microglia_pixel_to_label[pixel_value] = microglia_obj.label

    for nucleus in nuclei_props:
        nucleus_label = nucleus.label
        nucleus_coords = nucleus.coords  # Get all pixel coordinates of this nucleus

        # Track which microglia this nucleus overlaps with (using regionprops labels)
        overlapping_microglia_labels = set()

        for coord in nucleus_coords:
            y, x = coord
            microglia_pixel_value = microglia_mask[y, x]
            if microglia_pixel_value > 0:  # Non-background microglia pixel
                # Convert pixel value to regionprops label
                microglia_label = microglia_pixel_to_label.get(microglia_pixel_value)
                if microglia_label is not None:
                    overlapping_microglia_labels.add(microglia_label)

        if overlapping_microglia_labels:
            overlapping_nuclei.append(nucleus)
            nuclei_microglia_mapping[nucleus_label] = list(overlapping_microglia_labels)

    print(f"Found {len(overlapping_nuclei)} nuclei overlapping with microglia out of {len(nuclei_props)} total nuclei")

    # If no overlapping nuclei, return empty results
    if len(overlapping_nuclei) == 0:
        print("No nuclei overlap with microglia - returning empty results")
        empty_df = pd.DataFrame()
        return {}, empty_df

    df = pd.DataFrame()

    # Precompute centroids only for overlapping nuclei
    nuclei_centroids = np.array([obj.centroid for obj in overlapping_nuclei])
    nuclei_labels = [obj.label for obj in overlapping_nuclei]
    nuclei_radii = np.sqrt(np.array([obj.area for obj in overlapping_nuclei]) / np.pi)

    distance_dict = {}

    # Build KDTree only with overlapping nuclei
    tree = cKDTree(nuclei_centroids)

    # For each mito centroid, query the closest overlapping nucleus
    for mito_object in mitochondria_props:
        mito_centroid = np.array(mito_object.centroid)
        mito_radius = np.sqrt(mito_object.area)/np.pi

        dist, idx = tree.query(mito_centroid)
        closest_label = nuclei_labels[idx]
        closest_centroid = nuclei_centroids[idx]
        closest_radii = nuclei_radii[idx]


        if dist < max_distance or dist == max_distance:

            # Get the microglia labels that this nucleus overlaps with
            overlapping_microglia_labels = nuclei_microglia_mapping[closest_label]

            # Get shape modes for overlapping microglia (integers 0-4)
            overlapping_shape_modes = []
            for microglia_label in overlapping_microglia_labels:
                shape_mode = microglia_shape_modes.get(microglia_label, -1)  # -1 for unknown/missing
                overlapping_shape_modes.append(shape_mode)

            # Primary shape mode (from first/primary microglia)
            primary_shape_mode = overlapping_shape_modes[0] if overlapping_shape_modes else -1

            if closest_label in distance_dict.keys():
                distance_dict[closest_label].append([closest_label, dist, closest_centroid, mito_centroid])
            else: 
                distance_dict[closest_label] = [closest_label, dist, closest_centroid, mito_centroid]

            new_row = pd.DataFrame(
                    {'filename': [file_name.split("/")[-1]],
                    'nuclei': [closest_label + nuclei_count],
                    'nuclei_label': [closest_label],  # Added original nucleus label
                    'overlapping_microglia_labels': [overlapping_microglia_labels],  # Added microglia labels
                    'primary_microglia_label': [overlapping_microglia_labels[0] if overlapping_microglia_labels else None],  # Primary microglia
                    'overlapping_shape_modes': [overlapping_shape_modes],  # All shape modes
                    'primary_shape_mode': [primary_shape_mode],  # Primary shape mode
                    'num_overlapping_microglia': [len(overlapping_microglia_labels)],  # Count of overlapping microglia
                    'centroid_distance': [dist], 
                    'nuclei_ideal_radius': [closest_radii],
                    'nuc_centroid_x': [closest_centroid[0]], 
                    'nuc_centroid_y': [closest_centroid[1]], 
                    'mito_ideal_radius': [mito_radius],
                    'mito_centroid_x': [mito_centroid[0]],
                    'mito_centroid_y': [mito_centroid[1]],
                    'total_nuclei': [len(nuclei_props)],
                    'overlapping_nuclei': [len(overlapping_nuclei)],  # Added this
                    'total_microglia': [len(microglia_props)]}
                )

            df = pd.concat([df, new_row], ignore_index=True)

    if len(df) > 0:
        df["total_mito_objects"] = len(mitochondria_props)
        df["paired_mito_objects"] = len(df)

    return distance_dict, df


# In[ ]:


import vampire
import logging
from pathlib import Path
from typing import Union, List, Optional

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def apply_vampire_model(model_path: Union[str, Path], 
                       image_paths: Union[str, Path, List[Union[str, Path]]], 
                       output_paths: Optional[Union[str, Path, List[Union[str, Path]]]] = None,
                       image_set_names: Optional[Union[str, List[str]]] = None) -> None:
    """
    Apply a trained VAMPIRE model to a set of images.

    Parameters:
    -----------
    model_path : str or Path
        Path to the trained VAMPIRE model (.pickle file)
    image_paths : str, Path, or list
        Path(s) to image dataset(s) to analyze. Can be:
        - Single path (str or Path)
        - List of paths
    output_paths : str, Path, list, or None
        Output path(s) for results. If None, uses same as image_paths.
        Must match the length of image_paths if provided as list.
    image_set_names : str, list, or None
        Name(s) for the image sets. If None, uses directory names.
        Must match the length of image_paths if provided as list.

    Returns:
    --------
    None
        Results are saved to the specified output paths

    Example:
    --------
    # Single image set
    apply_vampire_model(
        model_path="path/to/model.pickle",
        image_paths="path/to/images/",
        output_paths="path/to/results/"
    )

    # Multiple image sets
    apply_vampire_model(
        model_path="path/to/model.pickle",
        image_paths=["path/to/images1/", "path/to/images2/"],
        output_paths=["path/to/results1/", "path/to/results2/"],
        image_set_names=["dataset1", "dataset2"]
    )
    """

    # Convert inputs to lists for consistent processing
    if isinstance(image_paths, (str, Path)):
        image_paths = [image_paths]

    if output_paths is None:
        output_paths = image_paths
    elif isinstance(output_paths, (str, Path)):
        output_paths = [output_paths]

    if image_set_names is None:
        image_set_names = [Path(path).name for path in image_paths]
    elif isinstance(image_set_names, str):
        image_set_names = [image_set_names]

    # Validate inputs
    model_path = Path(model_path)
    if not model_path.exists():
        raise FileNotFoundError(f"Model file does not exist: {model_path}")

    if not (len(image_paths) == len(output_paths) == len(image_set_names)):
        raise ValueError("image_paths, output_paths, and image_set_names must have the same length")

    # Validate image paths exist
    valid_entries = []
    for img_path, out_path, name in zip(image_paths, output_paths, image_set_names):
        img_path = Path(img_path)
        if img_path.exists():
            valid_entries.append({
                'img_set_path': str(img_path),
                'model_path': str(model_path),
                'output_path': str(out_path),
                'img_set_name': name
            })
            logger.info(f"Added to processing queue: {name} ({img_path})")
        else:
            logger.warning(f"Path does not exist, skipping: {img_path}")

    if not valid_entries:
        raise ValueError("No valid image paths found")

    # Create DataFrame for VAMPIRE
    apply_info_df = pd.DataFrame(valid_entries)

    logger.info(f"Applying VAMPIRE model to {len(apply_info_df)} image sets")
    logger.info(f"Using model: {model_path}")

    try:
        # Apply the model using VAMPIRE
        vampire.quickstart.transform_datasets(apply_info_df)
        logger.info("Model application completed successfully")

        # Log output locations
        for entry in valid_entries:
            logger.info(f"Results saved for {entry['img_set_name']}: {entry['output_path']}")

    except Exception as e:
        logger.error(f"Error during model application: {e}")
        raise


# In[ ]:


apply_vampire_model(model_path="/Users/nelsschimek/Documents/nancelab/Data/mito_images/brendan_full_analysis/tifs/training/vampire_data/model_li_(50_5_39)__.pickle",
                    image_paths=f"/Users/nelsschimek/Documents/nancelab/Data/mito_images/brendan_full_analysis/tifs/{condition}/cd11b/li_thresh/converted_tiffs/",
                    output_paths=f"/Users/nelsschimek/Documents/nancelab/Data/mito_images/brendan_full_analysis/tifs/{condition}/cd11b/li_thresh/converted_tiffs/")


# In[ ]:


vamp_df = pd.read_csv(f"/Users/nelsschimek/Documents/nancelab/Data/mito_images/brendan_full_analysis/tifs/{condition}/cd11b/li_thresh/converted_tiffs/apply-properties_li_on_converted_tiffs_(50_5_39)__.csv")
first_file = vamp_df[vamp_df["filename"] == microglia_mask_filepaths[0].split("/")[-1]]
first_file


# In[ ]:


distances = []
dfs = pd.DataFrame()
nuclei_count = 0
for i in range(len(loaded_microglia_masks)):
    print()
    print(microglia_mask_filepaths[i].split("/")[-1])
    print(mito_mask_filepaths[i].split("/")[-1])
    print(nuclei_mask_filepaths[i].split("/")[-1])

    image_shape_modes = vamp_df[vamp_df["filename"] == microglia_mask_filepaths[i].split("/")[-1]]

    shape_modes = dict(zip(
        (range(len((image_shape_modes["label"])))), image_shape_modes["cluster_id"]
    ))
    cur_distance_dict, cur_df = get_distance_dictionary(microglia_mask_filepaths[i], loaded_nuclei_masks[i], loaded_mitochondria_masks[i],
                                                        loaded_microglia_masks[i], max_distance=37, nuclei_count=nuclei_count, microglia_shape_modes=shape_modes)
    distances.append(cur_distance_dict)
    dfs = pd.concat([dfs, cur_df], ignore_index=True)
    nuclei_count = len(dfs)

title = str(directory).split("/")
treatment_condition = f"{title[-2]}_{title[-1]}"
dfs['treatment_condition'] = treatment_condition
dfs.to_csv(f'/Users/nelsschimek/Documents/nancelab/Data/mito_images/brendan_full_analysis/tifs/{condition}/{condition}_triple_cell_data.csv')
len(dfs)


# In[ ]:


dfs[dfs["filename"] == microglia_mask_filepaths[0].split("/")[-1]]


# In[ ]:


distances = []
dfs = pd.DataFrame()
nuclei_count = 0
for i in range(len(subset_non_pyk_masks)):
    X = label(subset_non_pyk_masks[i])
    Y = label(mito_masks[i])
    X_props = regionprops(X)
    Y_props = regionprops(Y)

    cur_distance_dict, cur_df = get_distance_dictionary(file_name=tif_file_paths[i], X_props=X_props, Y_props=Y_props, max_distance=37, nuclei_count=nuclei_count)
    distances.append(cur_distance_dict)
    dfs = pd.concat([dfs, cur_df], ignore_index=True)
    nuclei_count = len(dfs)

title = str(directory).split("/")
treatment_condition = f"{title[-2]}_{title[-1]}"
dfs['treatment_condition'] = treatment_condition
dfs.to_csv(f'/Users/nelsschimek/Documents/nancelab/saffron/distances_data/non_pyknotic_{title[-2]}_Nuclei_Mito_distances_{title[-1]}.csv')
len(dfs)


# In[ ]:


len(subset_pyk_masks)


# In[ ]:





# In[ ]:


# def plot_distances(distance_dict_list):

#     plt.figure(figsize=(4, 4))

#     for key, value in distance_dict.items():
#         # Process the primary vector
#         if isinstance(value[2], np.ndarray) and isinstance(value[3], np.ndarray):
#             start = value[2]
#             end = value[3]
#             dx = end[0] - start[0]
#             dy = end[1] - start[1]
#             plt.arrow(start[0], start[1], dx, dy, head_width=3, head_length=3, fc='blue', ec='blue')
#             plt.plot(start[0], start[1], 'g.')  # Start point
#             plt.plot(end[0], end[1], 'r.')      # End point
#             #plt.text(start[0], start[1], str(key), fontsize=8, color='black')

#         # Process sub-vectors (if any)
#         for sub in value[4:]:
#             if isinstance(sub, list) and len(sub) >= 4:
#                 sub_start = sub[2]
#                 sub_end = sub[3]
#                 dx = sub_end[0] - sub_start[0]
#                 dy = sub_end[1] - sub_start[1]
#                 plt.arrow(sub_start[0], sub_start[1], dx, dy, head_width=3, head_length=3, fc='gray', ec='gray')
#                 plt.plot(sub_end[0], sub_end[1], 'r.')  # Smaller marker for sub-end

#     #plt.gca().invert_xaxis()  # Optional for image-style coordinates
#     #plt.gca().invert_yaxis()  # Optional for image-style coordinates
#     plt.axis('equal')
#     plt.xlabel('X')
#     plt.ylabel('Y')
#     plt.title('All Centroid Vectors (Primary and Sublists)')
#     #plt.grid(True)
#     plt.show()


# In[ ]:


def make_distance_plot(distance_dicts, max_distance=20, title=None):


    circle_one = Circle((max_distance, max_distance), 3, facecolor='none',
                        edgecolor=(0, 0.8, 0.8), linewidth=3, alpha=0.5)
    circle_two = Circle((max_distance, max_distance), max_distance, facecolor='none',
                    edgecolor=(0, 0.2, 0.2), linewidth=3, alpha=0.5)
    fig, ax = plt.subplots(figsize=(4,4))
    ax.add_patch(circle_one)
    ax.add_patch(circle_two)

    for distance_dict in distance_dicts:



        for key, value in distance_dict.items():
            # Process the primary vector
            if isinstance(value[2], np.ndarray) and isinstance(value[3], np.ndarray):
                start = value[2]
                end = value[3]
                dx = end[0] - start[0]
                dy = end[1] - start[1]
                ax.arrow(max_distance, max_distance, dx, dy, head_width=0.1, head_length=0.1, fc='blue', ec='blue', alpha=0.1)
                #ax.plot(start[0], start[1], 'g.')  # Start point
                #ax.plot(end[0], end[1], 'r.')      # End point
                #ax.text(start[0], start[1], str(key), fontsize=8, color='black')

            # Process sub-vectors (if any)
            for sub in value[4:]:
                if isinstance(sub, list) and len(sub) >= 4:
                    sub_start = sub[2]
                    sub_end = sub[3]
                    dx = sub_end[0] - sub_start[0]
                    dy = sub_end[1] - sub_start[1]
                    ax.arrow(max_distance, max_distance, dx, dy, head_width=0.1, head_length=0.1, fc='blue', ec='blue', alpha=0.1)
                    #ax.plot(sub_end[0], sub_end[1], 'r.')  # Smaller marker for sub-end

    ax.set_xlim([0,(max_distance*2)])
    ax.set_ylim([0,(max_distance*2)])
    ax.set_title(title)
    plt.show()


# In[ ]:


title = str(directory).split("/")
make_distance_plot(distance_dicts=distances, title=f'{title[-2]} Nuclei Mito distances {title[-1]} (16uM)', max_distance=40)


# In[ ]:





# In[ ]:


mask_file = '/Users/nelsschimek/Documents/nancelab/Data/mito_images/brendan_full_analysis/tifs/HC/cd11b/li_thresh/060225_P10F_4DIV_OR10_control_HC_F24h_DAPI_CD11b_Blank_MT_Slice_A_40x_mb_3_li_thresh.npy'


# In[ ]:


import skimage
from skimage.segmentation import clear_border

binary_mask = np.load(mask_file)

# Label connected regions in the binary mask
label_image = label(binary_mask)
cleaned_mask = clear_border(labels=label_image)

# Measure properties
props = skimage.measure.regionprops_table((label_image))

# Create a DataFrame for the current file
props_df = pd.DataFrame(props)




# In[ ]:


plt.imshow(cleaned_mask, cmap="gray")

