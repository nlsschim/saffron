"""
Example showing different ways to use the updated saffron package.
"""

# Method 1: Import the entire io module
import saffron
from saffron import io
import numpy as np

# Load masks using the module
microglia_masks, paths = io.load_microglia_masks("path/to/microglia")
properties = io.load_properties_csv("properties.csv")

print(f"Saffron version: {saffron.__version__}")
print(f"Loaded {len(microglia_masks)} microglia masks")

# Method 2: Import specific functions (convenience imports)
from saffron import (
    load_microglia_masks, 
    load_mitochondria_masks,
    load_nuclei_masks,
    save_results_csv
)

# Use functions directly
microglia_masks, microglia_paths = load_microglia_masks("path/to/microglia")
mito_masks, mito_paths = load_mitochondria_masks("path/to/mito") 
nuclei_masks, nuclei_paths = load_nuclei_masks("path/to/nuclei")

# Method 3: Full module path (most explicit)
import saffron.io as sio

masks = sio.load_masks("path/to/masks", mask_type="tif")
sio.save_mask(processed_mask, "output/processed_mask.npy")

# Method 4: Mixed approach (common in scientific code)
from saffron import io, predict, __version__
import pandas as pd

def analyze_sample(condition: str):
    """Example analysis function using the new I/O structure."""
    
    print(f"Running analysis with saffron v{__version__}")
    
    # Load all required data
    base_path = f"data/{condition}"
    
    microglia_masks, _ = io.load_microglia_masks(f"{base_path}/microglia")
    mito_masks, _ = io.load_mitochondria_masks(f"{base_path}/mito")
    nuclei_masks, _ = io.load_nuclei_masks(f"{base_path}/nuclei")
    
    # Validate data correspondence
    if not io.validate_file_correspondence(
        [microglia_masks, mito_masks, nuclei_masks],
        ['microglia', 'mito', 'nuclei']
    ):
        raise ValueError("Mask files don't correspond")
    
    # Run analysis (placeholder)
    results = []
    for i, (mg, mt, nc) in enumerate(zip(microglia_masks, mito_masks, nuclei_masks)):
        # Your analysis code here...
        sample_result = {
            'sample': i,
            'condition': condition,
            'microglia_area': mg.sum(),
            'mito_count': len(np.unique(mt)) - 1,  # subtract background
            'nuclei_count': len(np.unique(nc)) - 1
        }
        results.append(sample_result)
    
    # Save results
    results_df = pd.DataFrame(results)
    output_path = f"results/{condition}_analysis_results.csv"
    io.save_results_csv(results_df, output_path)
    
    print(f"Analysis complete. Results saved to {output_path}")
    return results_df

# Example usage
if __name__ == "__main__":
    # This would replace the main analysis loop in mito_threshold.py
    conditions = ["OGD_only", "HC", "control"]
    
    for condition in conditions:
        try:
            results = analyze_sample(condition)
            print(f"Processed {condition}: {len(results)} samples")
        except Exception as e:
            print(f"Error processing {condition}: {e}")