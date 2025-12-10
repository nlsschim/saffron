from PIL import Image
import os
from pathlib import Path
import argparse

def split_into_quadrants(image_path, output_dir):
    img = Image.open(image_path)
    width, height = img.size
    
    mid_w, mid_h = width // 2, height // 2
    
    # Get filename without extension
    basename = Path(image_path).stem
    
    # Define quadrants
    quadrants = {
        'top_left': (0, 0, mid_w, mid_h),
        'top_right': (mid_w, 0, width, mid_h),
        'bottom_left': (0, mid_h, mid_w, height),
        'bottom_right': (mid_w, mid_h, width, height)
    }
    
    for name, box in quadrants.items():
        quadrant = img.crop(box)
        output_path = os.path.join(output_dir, f"{basename}_{name}.tif")
        quadrant.save(output_path)
    
    print(f"Processed: {basename}")

def process_directory(input_dir, output_dir):
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Get all .tif and .tiff files
    tif_files = list(Path(input_dir).glob('*.tif')) + list(Path(input_dir).glob('*.tiff'))
    
    print(f"Found {len(tif_files)} TIFF files")
    
    for tif_file in tif_files:
        try:
            split_into_quadrants(str(tif_file), output_dir)
        except Exception as e:
            print(f"Error processing {tif_file.name}: {e}")
    
    print(f"\nDone! Quadrants saved to: {output_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Split TIFF images into quadrants')
    parser.add_argument('input_dir', type=str, help='Input directory containing TIFF files')
    parser.add_argument('output_dir', type=str, help='Output directory for quadrants')
    
    args = parser.parse_args()
    
    # Check if input directory exists
    if not os.path.exists(args.input_dir):
        print(f"Error: Input directory '{args.input_dir}' does not exist")
        exit(1)
    
    process_directory(args.input_dir, args.output_dir)
