"""
Example showing how to use the configuration system to replace 
hardcoded values from the original mito_threshold.py script.
"""

from pathlib import Path
from saffron.config import ExperimentConfig, PathConfig, AnalysisConfig, load_or_create_config
from saffron.io import load_microglia_masks, load_mitochondria_masks, load_nuclei_masks

def setup_experiment_from_config(condition: str = "OGD_only"):
    """
    Set up an experiment using the configuration system.
    
    This replaces all the hardcoded paths and parameters.
    """
    
    # Method 1: Create configuration programmatically
    config = ExperimentConfig(
        paths=PathConfig(
            condition=condition,
            base_data_dir="/Users/nelsschimek/Documents/nancelab/Data/mito_images/brendan_full_analysis"
        ),
        analysis=AnalysisConfig(
            max_distance=37.0,
            mitochondria_percentile=99.0
        ),
        experiment_name=f"mito_analysis_{condition}",
        researcher="Your Name Here"
    )
    
    # Set up directories and validate paths
    config.setup_experiment()
    
    return config


def load_experiment_from_file(config_file: str, condition: str = "OGD_only"):
    """
    Load experiment configuration from a saved file.
    
    This enables reproducible analysis across different systems.
    """
    
    # Load or create configuration
    config = load_or_create_config(config_file, condition)
    
    # Set up experiment
    config.setup_experiment()
    
    return config


def run_analysis_with_config(config: ExperimentConfig):
    """
    Run the main analysis using configuration instead of hardcoded values.
    
    This is how the main analysis loop from mito_threshold.py would be refactored.
    """
    
    print(f"Running experiment: {config.experiment_name}")
    print(f"Condition: {config.paths.condition}")
    print(f"Max distance: {config.analysis.max_distance}")
    
    # Load masks using configuration
    microglia_masks, microglia_paths = load_microglia_masks(
        config.paths.microglia_masks_dir
    )
    
    mito_masks, mito_paths = load_mitochondria_masks(
        config.paths.mitochondria_masks_dir
    )
    
    nuclei_masks, nuclei_paths = load_nuclei_masks(
        config.paths.nuclei_masks_dir
    )
    
    print(f"Loaded {len(microglia_masks)} mask sets")
    
    # Analysis parameters from configuration
    max_distance = config.analysis.max_distance
    image_shape = config.analysis.image_shape
    mito_percentile = config.analysis.mitochondria_percentile
    
    # Your analysis code would use these configured parameters...
    # results = analyze_spatial_relationships(
    #     microglia_masks, mito_masks, nuclei_masks,
    #     max_distance=max_distance,
    #     image_shape=image_shape
    # )
    
    # Save results using configuration
    output_path = config.paths.get_output_path(
        f"{config.paths.condition}_analysis_results.csv"
    )
    
    print(f"Results will be saved to: {output_path}")
    
    return config


def create_config_templates():
    """
    Create configuration templates for different experimental conditions.
    """
    
    conditions = ["OGD_only", "HC", "control"]
    
    for condition in conditions:
        # Create configuration
        config = ExperimentConfig(
            paths=PathConfig(condition=condition),
            experiment_name=f"mito_analysis_{condition}",
            description=f"Mitochondria analysis for {condition} condition"
        )
        
        # Save configuration template
        config_path = f"configs/{condition}_config.yaml"
        config.save_config(config_path)
        
        print(f"Created configuration template: {config_path}")


def demonstrate_config_flexibility():
    """
    Show how configuration makes the analysis flexible and adaptable.
    """
    
    # Example 1: Different researcher with different paths
    config_researcher1 = ExperimentConfig(
        paths=PathConfig(
            base_data_dir="/path/to/researcher1/data",
            condition="OGD_only"
        ),
        researcher="Researcher 1"
    )
    
    # Example 2: Different analysis parameters
    config_high_sensitivity = ExperimentConfig(
        paths=PathConfig(condition="HC"),
        analysis=AnalysisConfig(
            max_distance=50.0,  # Larger search radius
            mitochondria_percentile=95.0,  # Lower threshold
            mitochondria_min_size=5  # Smaller minimum size
        ),
        experiment_name="high_sensitivity_analysis"
    )
    
    # Example 3: Custom image dimensions
    config_high_res = ExperimentConfig(
        analysis=AnalysisConfig(
            image_shape=(2048, 2048),  # Higher resolution
            max_distance=74.0  # Scale distance accordingly
        ),
        experiment_name="high_resolution_analysis"
    )
    
    # Save all configurations
    for i, config in enumerate([config_researcher1, config_high_sensitivity, config_high_res]):
        config.save_config(f"configs/example_{i+1}_config.yaml")
        print(f"Saved configuration: {config.experiment_name}")


# Example usage replacing the main script execution
if __name__ == "__main__":
    
    # Method 1: Programmatic configuration
    print("=== Method 1: Programmatic Configuration ===")
    config1 = setup_experiment_from_config("OGD_only")
    run_analysis_with_config(config1)
    
    print("\n=== Method 2: File-based Configuration ===")
    # Method 2: File-based configuration  
    config2 = load_experiment_from_file("configs/OGD_only_config.yaml", "OGD_only")
    run_analysis_with_config(config2)
    
    print("\n=== Method 3: Create Templates ===")
    # Method 3: Create configuration templates
    create_config_templates()
    
    print("\n=== Method 4: Demonstrate Flexibility ===")
    # Method 4: Show flexibility
    demonstrate_config_flexibility()
    
    print("\nConfiguration system examples completed!")


# Example of how the original script's main loop would be refactored:
def refactored_main_analysis():
    """
    This shows how the main analysis loop from mito_threshold.py 
    would look after refactoring with configuration.
    """
    
    # Instead of hardcoded condition
    conditions = ["OGD_only", "HC", "control"]
    
    for condition in conditions:
        print(f"\nProcessing condition: {condition}")
        
        # Load configuration
        config_file = f"configs/{condition}_config.yaml"
        config = load_or_create_config(config_file, condition)
        
        try:
            # Run analysis with configuration
            results = run_analysis_with_config(config)
            print(f"✓ Successfully processed {condition}")
            
        except Exception as e:
            print(f"✗ Error processing {condition}: {e}")
            continue