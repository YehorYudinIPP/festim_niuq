"""
Custom YAML Encoder for EasyVVUQ
"""
import os
import yaml
import math
#from easyvvuq.encoders.base import BaseEncoder


class YAMLEncoder:
    """
    Custom encoder for YAML configuration files.
    
    This encoder reads a YAML template file and replaces specified parameters
    with values from the EasyVVUQ campaign.
    """
    
    def __init__(self, template_fname, target_filename="config.yaml", delimiter="$"):
        """
        Initialize the YAML encoder.
        
        Parameters:
        -----------
        template_fname : str
            Path to the YAML template file
        target_filename : str
            Name of the output YAML file to be created
        delimiter : str
            Delimiter used in template for parameter substitution
        """
        self.template_fname = template_fname
        self.target_filename = target_filename
        self.delimiter = delimiter
        
        # Verify template file exists
        if not os.path.exists(template_fname):
            raise FileNotFoundError(f"Template file not found: {template_fname}")
    
    def encode(self, params=None, target_dir="./"):
        """
        Encode the parameters into a YAML file.
        
        Parameters:
        -----------
        params : dict
            Dictionary of parameters to substitute in the template
        target_dir : str
            Directory where the output file should be created
        """
        if params is None:
            params = {}
        
        # Read the template file
        with open(self.template_fname, 'r') as f:
            template_content = f.read()
        
        # Replace parameters in template
        for param_name, param_value in params.items():
            placeholder = f"{self.delimiter}{param_name}{self.delimiter}"
            template_content = template_content.replace(placeholder, str(param_value))
        
        # Write the processed content to target file
        target_path = os.path.join(target_dir, self.target_filename)
        with open(target_path, 'w') as f:
            f.write(template_content)
        
        print(f"YAML file created: {target_path}")
        
        # Verify the output is valid YAML
        try:
            with open(target_path, 'r') as f:
                yaml.safe_load(f)
            print("✓ Generated YAML file is valid")
        except yaml.YAMLError as e:
            print(f"⚠ Warning: Generated YAML may be invalid: {e}")
    
    def get_restart_dict(self):
        """Return restart dictionary for EasyVVUQ."""
        return {
            "template_fname": self.template_fname,
            "target_filename": self.target_filename,
            "delimiter": self.delimiter
        }
    
    @staticmethod
    def deserialize(serialized_encoder):
        """Deserialize the encoder from a dictionary."""
        return YAMLEncoder(
            template_fname=serialized_encoder["template_fname"],
            target_filename=serialized_encoder["target_filename"],
            delimiter=serialized_encoder["delimiter"]
        )


class AdvancedYAMLEncoder(YAMLEncoder):
    """
    Advanced YAML encoder that can handle nested parameters and type conversion.
    """
    
    def __init__(self, template_fname, target_filename="config.yaml", 
                 parameter_map=None, type_conversions=None, fixed_parameters=None):
        """
        Initialize the advanced YAML encoder.
        
        Parameters:
        -----------
        template_fname : str
            Path to the YAML template file
        target_filename : str
            Name of the output YAML file
        parameter_map : dict
            Mapping of parameter names to YAML paths (e.g., {"D_0": "materials.D_0"})
        type_conversions : dict
            Type conversions for parameters (e.g., {"D_0": float, "n_elements": int})
        """
        self.template_fname = template_fname
        self.target_filename = target_filename
        self.parameter_map = parameter_map or {}
        self.type_conversions = type_conversions or {}
        self.fixed_parameters = fixed_parameters or {}

        if not os.path.exists(template_fname):
            raise FileNotFoundError(f"Template file not found: {template_fname}")
    
    def encode(self, params=None, target_dir="./"):
        """
        Encode parameters into YAML file with advanced features.
        """
        if params is None:
            params = {}
        
        # Load the template as YAML
        with open(self.template_fname, 'r') as f:
            config = yaml.safe_load(f)
        
        # Update configuration with parameters
        for param_name, param_value in params.items():
            # Apply type conversion if specified
            if param_name in self.type_conversions:
                param_value = self.type_conversions[param_name](param_value)
            
            # Use parameter mapping if available
            if param_name in self.parameter_map:
                yaml_path = self.parameter_map[param_name]
                self._set_nested_value(config, yaml_path, param_value)
            else:
                # Try to find the parameter in the config structure
                self._update_config_recursive(config, param_name, param_value)
        
        # Update fixed parameters for every run
        # print(f" >> Updating fixed parameters in the configuration: {self.fixed_parameters}") ###DEUBUG
        if self.fixed_parameters:
            self._update_fixed_parameters(config)

        # Write the updated configuration
        target_path = os.path.join(target_dir, self.target_filename)
        with open(target_path, 'w') as f:
            yaml.dump(config, f, default_flow_style=False, indent=2)
        
        print(f"Advanced YAML file created: {target_path}")
    
    def _set_nested_value(self, config, path, value):
        """Set a nested value in the configuration using dot notation."""
        # Split the path into keys
        if not isinstance(path, str):
            raise ValueError("Path must be a string in dot notation.")
        if not path:
            raise ValueError("Path cannot be empty.")
        keys = path.split('.')
        current = config

        # print(f" >>>> Encoding via keys: {keys}") ###DEBUG

        # Traverse the nested structure and set the value
        for key in keys[:-1]:
            # Check if the key exists in the current level
            if key not in current:
                # Create a new dictionary if the key does not exist
                current[key] = {}
            print(f" >>>> Encoding @: {current} >> {key}") ###DEBUG
            current = current[key]
            # Check if the key is a list
            if isinstance(current, list):
                # ATTENTION: If current is a list, choose the first element
                current = current[0]
        
        # print(f" >>>> Setting a leave of the config @: {current} >> {key}") ###DEBUG
        # Set the final key to the value
        current[keys[-1]] = value
        # print(f"Set nested value at {path} to {value}") ###DEBUG
        # ATTENTION: will not handle list at the end of the path

        # Ensure the value is correctly set
        if not math.isclose(current[keys[-1]], value):
            # If the value is not set correctly, raise an error
            raise ValueError(f"Failed to set value at {path}. Expected {value}, got {current[keys[-1]]}")

    def _update_config_recursive(self, config, param_name, param_value):
        """Recursively search and update parameter in config."""
        if isinstance(config, dict):
            for key, value in config.items():
                if key == param_name:
                    config[key] = param_value
                    return True
                elif isinstance(value, dict):
                    if self._update_config_recursive(value, param_name, param_value):
                        return True
        return False
    
    def get_restart_dict(self):
        """Return restart dictionary for EasyVVUQ."""
        return {
            "template_fname": self.template_fname,
            "target_filename": self.target_filename,
            "parameter_map": self.parameter_map,
            "type_conversions": self.type_conversions
        }
    
    @staticmethod
    def deserialize(serialized_encoder):
        """Deserialize the encoder from a dictionary."""
        return AdvancedYAMLEncoder(
            template_fname=serialized_encoder["template_fname"],
            target_filename=serialized_encoder["target_filename"],
            parameter_map=serialized_encoder.get("parameter_map"),
            type_conversions=serialized_encoder.get("type_conversions")
        )

    def _update_fixed_parameters(self, config):
        """Update fixed parameters in the configuration."""
        for key, value in self.fixed_parameters.items():

            # Apply type conversion if specified
            if key in self.type_conversions:
                value = self.type_conversions[key](value)

            # Use parameter mapping if available
            if key in self.parameter_map:

                yaml_path = self.parameter_map[key]

                # print(f" >> Setting nested fixed parameter '{key}' to '{value}' at path '{yaml_path}'") ###DEBUG
                
                self._set_nested_value(config, yaml_path, value)
                
            else:
                # Try to find the parameter in the config structure
                self._update_config_recursive(config, key, value)

# Convenience function for simple usage
def create_yaml_encoder(template_file, output_file="config.yaml", 
                       advanced=False, **kwargs):
    """
    Create a YAML encoder with sensible defaults.
    
    Parameters:
    -----------
    template_file : str
        Path to template file
    output_file : str
        Output filename
    advanced : bool
        Whether to use advanced encoder
    **kwargs : dict
        Additional arguments for the encoder
    """
    if advanced:
        return AdvancedYAMLEncoder(template_file, output_file, **kwargs)
    else:
        return YAMLEncoder(template_file, output_file, **kwargs)
