import pandas as pd
import numpy as np
from scipy.spatial import cKDTree
from skimage.measure import regionprops, label
from skimage.segmentation import find_boundaries
import cv2

def get_distance_dictionary_with_microglia(file_name, nuclei_mask, mitochondria_mask, microglia_mask, 
                                         max_distance, nuclei_count=0):
    """
    Calculate distances between mitochondria and nuclei, but only when nuclei overlap with microglia.
    
    Parameters:
    -----------
    file_name : str
        Name/path of the file being processed
    nuclei_mask : np.ndarray
        Labeled mask of nuclei
    mitochondria_mask : np.ndarray
        Labeled mask of mitochondria
    microglia_mask : np.ndarray
        Labeled mask of microglia
    max_distance : float
        Maximum distance to consider for mito-nuclei pairing
    nuclei_count : int
        Offset for nuclei labeling (for batch processing)
    
    Returns:
    --------
    tuple
        (distance_dict, df) containing distance data and summary DataFrame
    """
    
    df = pd.DataFrame()
    distance_dict = {}
    
    # Get region properties for all masks
    nuclei_props = regionprops(nuclei_mask)
    mito_props = regionprops(mitochondria_mask)
    microglia_props = regionprops(microglia_mask)
    
    if len(nuclei_props) == 0 or len(mito_props) == 0 or len(microglia_props) == 0:
        print(f"Warning: No objects found in one or more masks for {file_name}")
        return distance_dict, df
    
    # Create microglia lookup for efficient overlap checking
    microglia_labels = set(np.unique(microglia_mask)[1:])  # Exclude background (0)
    
    # Find nuclei that overlap with microglia
    overlapping_nuclei = []
    nuclei_microglia_mapping = {}
    
    for nucleus in nuclei_props:
        nucleus_label = nucleus.label
        
        # Get nucleus coordinates
        nucleus_coords = nucleus.coords  # All pixel coordinates of this nucleus
        
        # Check which microglia this nucleus overlaps with
        overlapping_microglia = set()
        
        for coord in nucleus_coords:
            y, x = coord
            microglia_value = microglia_mask[y, x]
            if microglia_value > 0:  # Non-background
                overlapping_microglia.add(microglia_value)
        
        if overlapping_microglia:
            overlapping_nuclei.append(nucleus)
            nuclei_microglia_mapping[nucleus_label] = list(overlapping_microglia)
    
    if len(overlapping_nuclei) == 0:
        print(f"Warning: No nuclei overlap with microglia in {file_name}")
        return distance_dict, df
    
    print(f"Found {len(overlapping_nuclei)} nuclei overlapping with microglia out of {len(nuclei_props)} total nuclei")
    
    # Precompute centroids and radii for overlapping nuclei only
    nuclei_centroids = np.array([obj.centroid for obj in overlapping_nuclei])
    nuclei_labels = [obj.label for obj in overlapping_nuclei]
    nuclei_radii = np.sqrt(np.array([obj.area for obj in overlapping_nuclei]) / np.pi)
    
    # Build KDTree for overlapping nuclei only
    tree = cKDTree(nuclei_centroids)
    
    # For each mitochondria, find closest overlapping nucleus
    for mito_object in mito_props:
        mito_centroid = np.array(mito_object.centroid)
        mito_radius = np.sqrt(mito_object.area / np.pi)
        mito_label = mito_object.label
        
        # Query closest nucleus (from overlapping nuclei only)
        dist, idx = tree.query(mito_centroid)
        closest_nucleus_label = nuclei_labels[idx]
        closest_centroid = nuclei_centroids[idx]
        closest_radius = nuclei_radii[idx]
        
        # Get the microglia IDs that this nucleus overlaps with
        overlapping_microglia_ids = nuclei_microglia_mapping[closest_nucleus_label]
        
        if dist <= max_distance:
            # Store in distance dictionary
            if closest_nucleus_label in distance_dict.keys():
                distance_dict[closest_nucleus_label].append([
                    closest_nucleus_label, dist, closest_centroid, mito_centroid, 
                    overlapping_microglia_ids, mito_label
                ])
            else:
                distance_dict[closest_nucleus_label] = [[
                    closest_nucleus_label, dist, closest_centroid, mito_centroid, 
                    overlapping_microglia_ids, mito_label
                ]]
            
            # Create DataFrame row
            new_row = pd.DataFrame({
                'filename': [file_name.split("/")[-1]],
                'nuclei_id': [closest_nucleus_label + nuclei_count],
                'original_nuclei_id': [closest_nucleus_label],
                'mito_id': [mito_label],
                'centroid_distance': [dist],
                'nuclei_ideal_radius': [closest_radius],
                'nuc_centroid_x': [closest_centroid[0]],
                'nuc_centroid_y': [closest_centroid[1]],
                'mito_ideal_radius': [mito_radius],
                'mito_centroid_x': [mito_centroid[0]],
                'mito_centroid_y': [mito_centroid[1]],
                'overlapping_microglia_ids': [overlapping_microglia_ids],
                'num_overlapping_microglia': [len(overlapping_microglia_ids)],
                'primary_microglia_id': [overlapping_microglia_ids[0] if overlapping_microglia_ids else None],
                'total_nuclei': [len(nuclei_props)],
                'overlapping_nuclei': [len(overlapping_nuclei)],
            })
            
            df = pd.concat([df, new_row], ignore_index=True)
    
    # Add total mitochondria count
    if len(df) > 0:
        df["total_mito_objects"] = len(mito_props)
        df["paired_mito_objects"] = len(df)
    
    return distance_dict, df


def load_and_process_masks(nuclei_path, mito_path, microglia_path):
    """
    Load and process mask images for analysis.
    
    Parameters:
    -----------
    nuclei_path : str
        Path to nuclei mask image
    mito_path : str
        Path to mitochondria mask image  
    microglia_path : str
        Path to microglia mask image
        
    Returns:
    --------
    tuple
        (nuclei_mask, mito_mask, microglia_mask) as labeled arrays
    """
    
    # Load masks
    nuclei_img = cv2.imread(nuclei_path, cv2.IMREAD_GRAYSCALE)
    mito_img = cv2.imread(mito_path, cv2.IMREAD_GRAYSCALE)
    microglia_img = cv2.imread(microglia_path, cv2.IMREAD_GRAYSCALE)
    
    if nuclei_img is None or mito_img is None or microglia_img is None:
        raise ValueError("Could not load one or more mask images")
    
    # Convert to binary if needed, then label
    nuclei_binary = (nuclei_img > 0).astype(np.uint8)
    mito_binary = (mito_img > 0).astype(np.uint8)
    microglia_binary = (microglia_img > 0).astype(np.uint8)
    
    # Label connected components
    nuclei_labeled = label(nuclei_binary)
    mito_labeled = label(mito_binary)
    microglia_labeled = label(microglia_binary)
    
    return nuclei_labeled, mito_labeled, microglia_labeled


def analyze_nuclei_microglia_overlap(nuclei_mask, microglia_mask):
    """
    Analyze the overlap between nuclei and microglia masks.
    
    Parameters:
    -----------
    nuclei_mask : np.ndarray
        Labeled nuclei mask
    microglia_mask : np.ndarray
        Labeled microglia mask
        
    Returns:
    --------
    dict
        Dictionary with overlap analysis results
    """
    
    nuclei_props = regionprops(nuclei_mask)
    microglia_props = regionprops(microglia_mask)
    
    overlap_data = {
        'total_nuclei': len(nuclei_props),
        'total_microglia': len(microglia_props),
        'overlapping_nuclei': 0,
        'nuclei_overlap_details': {},
        'microglia_coverage': {}
    }
    
    # Analyze each nucleus
    for nucleus in nuclei_props:
        nucleus_label = nucleus.label
        nucleus_coords = nucleus.coords
        
        overlapping_microglia = set()
        overlap_pixel_count = 0
        
        for coord in nucleus_coords:
            y, x = coord
            microglia_value = microglia_mask[y, x]
            if microglia_value > 0:
                overlapping_microglia.add(microglia_value)
                overlap_pixel_count += 1
        
        if overlapping_microglia:
            overlap_data['overlapping_nuclei'] += 1
            overlap_data['nuclei_overlap_details'][nucleus_label] = {
                'overlapping_microglia_ids': list(overlapping_microglia),
                'nucleus_area': nucleus.area,
                'overlap_pixels': overlap_pixel_count,
                'overlap_fraction': overlap_pixel_count / nucleus.area
            }
    
    # Analyze microglia coverage
    for microglia in microglia_props:
        microglia_label = microglia.label
        microglia_coords = microglia.coords
        
        overlapping_nuclei = set()
        overlap_pixel_count = 0
        
        for coord in microglia_coords:
            y, x = coord
            nucleus_value = nuclei_mask[y, x]
            if nucleus_value > 0:
                overlapping_nuclei.add(nucleus_value)
                overlap_pixel_count += 1
        
        overlap_data['microglia_coverage'][microglia_label] = {
            'overlapping_nuclei_ids': list(overlapping_nuclei),
            'microglia_area': microglia.area,
            'overlap_pixels': overlap_pixel_count,
            'overlap_fraction': overlap_pixel_count / microglia.area
        }
    
    return overlap_data


# Example usage function
def process_single_image_with_microglia(nuclei_path, mito_path, microglia_path, 
                                      max_distance=50, nuclei_count=0):
    """
    Complete processing pipeline for a single image with microglia analysis.
    
    Parameters:
    -----------
    nuclei_path : str
        Path to nuclei mask
    mito_path : str  
        Path to mitochondria mask
    microglia_path : str
        Path to microglia mask
    max_distance : float
        Maximum distance for mito-nuclei pairing
    nuclei_count : int
        Offset for nuclei numbering
        
    Returns:
    --------
    tuple
        (distance_dict, results_df, overlap_analysis)
    """
    
    # Load masks
    nuclei_mask, mito_mask, microglia_mask = load_and_process_masks(
        nuclei_path, mito_path, microglia_path
    )
    
    # Analyze overlap
    overlap_analysis = analyze_nuclei_microglia_overlap(nuclei_mask, microglia_mask)
    
    print(f"Overlap Analysis Results:")
    print(f"  Total nuclei: {overlap_analysis['total_nuclei']}")
    print(f"  Total microglia: {overlap_analysis['total_microglia']}")
    print(f"  Overlapping nuclei: {overlap_analysis['overlapping_nuclei']}")
    
    # Get distance data
    distance_dict, results_df = get_distance_dictionary_with_microglia(
        nuclei_path, nuclei_mask, mito_mask, microglia_mask, 
        max_distance, nuclei_count
    )
    
    return distance_dict, results_df, overlap_analysis


# Example batch processing function
def process_multiple_images_with_microglia(image_list, max_distance=50):
    """
    Process multiple images with microglia analysis.
    
    Parameters:
    -----------
    image_list : list
        List of dictionaries with keys: 'nuclei_path', 'mito_path', 'microglia_path'
    max_distance : float
        Maximum distance for mito-nuclei pairing
        
    Returns:
    --------
    tuple
        (combined_df, all_overlap_analyses)
    """
    
    all_dfs = []
    all_overlap_analyses = {}
    nuclei_count = 0
    
    for i, image_info in enumerate(image_list):
        print(f"\nProcessing image {i+1}/{len(image_list)}: {image_info['nuclei_path']}")
        
        try:
            distance_dict, results_df, overlap_analysis = process_single_image_with_microglia(
                image_info['nuclei_path'], 
                image_info['mito_path'], 
                image_info['microglia_path'],
                max_distance, 
                nuclei_count
            )
            
            if len(results_df) > 0:
                all_dfs.append(results_df)
                nuclei_count += overlap_analysis['total_nuclei']
            
            all_overlap_analyses[image_info['nuclei_path']] = overlap_analysis
            
        except Exception as e:
            print(f"Error processing {image_info['nuclei_path']}: {e}")
            continue
    
    # Combine all results
    if all_dfs:
        combined_df = pd.concat(all_dfs, ignore_index=True)
    else:
        combined_df = pd.DataFrame()
    
    return combined_df, all_overlap_analyses