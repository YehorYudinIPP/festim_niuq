#!/usr/bin/env python3
"""
Test script for the custom YAML encoder.
"""
import os
import sys

# Add parent directory to path
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, parent_dir)

from util.Encoder import YAMLEncoder, AdvancedYAMLEncoder

def test_yaml_encoder():
    """Test the YAML encoder functionality."""
    
    # Test parameters
    test_params = {
        "D_0": 1.5e-6,
        "E_D": 0.7,
        "T": 350.0,
        "total_time": 10.0,
        "left_bc_value": 2.0e20,
        "source_value": 1.5e18
    }
    
    print("Testing YAML Encoder...")
    print("Parameters:", test_params)
    
    # Test simple encoder
    try:
        encoder = YAMLEncoder(
            template_fname="festim_yaml.template",
            target_filename="test_config.yaml",
            delimiter="$"
        )
        encoder.encode(test_params, target_dir="./")
        print("✓ Simple YAML encoder test passed")
    except Exception as e:
        print(f"✗ Simple YAML encoder test failed: {e}")
    
    # Test advanced encoder
    try:
        advanced_encoder = AdvancedYAMLEncoder(
            template_fname="festim_yaml.template",
            target_filename="test_config_advanced.yaml",
            parameter_map={
                "D_0": "materials.D_0",
                "E_D": "materials.E_D",
                "T": "materials.T"
            }
        )
        advanced_encoder.encode(test_params, target_dir="./")
        print("✓ Advanced YAML encoder test passed")
    except Exception as e:
        print(f"✗ Advanced YAML encoder test failed: {e}")

if __name__ == "__main__":
    test_yaml_encoder()
