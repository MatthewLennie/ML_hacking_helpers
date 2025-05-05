import torch
import torch.nn as nn
import inspect
import numpy as np
try:
    # Try relative import (when used as a module)
    from .trackers import ParameterTracker
except ImportError:
    # Fall back to direct import (when run directly)
    from trackers import ParameterTracker

def initialize_model(model, constant_val=1e-4, verbose=False):
    """
    Initialize model weights using He (Kaiming) initialization for convolutional and linear layers,
    and set constant weights/biases to a very small positive number.
    
    The function introspects each layer to determine the appropriate initialization strategy.
    
    Args:
        model (nn.Module): PyTorch model to initialize
        constant_val (float): Small positive value to initialize constant parameters (default: 1e-4)
        verbose (bool): Whether to print detailed initialization info (default: False)
        
    Returns:
        nn.Module: Initialized model
    """
    for name, module in model.named_modules():
        # Skip the model itself and containers
        if module is model or not list(module.children()):
            
            # Initialize weights using He/Kaiming initialization
            if hasattr(module, 'weight') and module.weight is not None:
                if isinstance(module, (nn.Conv1d, nn.Conv2d, nn.Conv3d, nn.Linear)):
                    # Check if ReLU-like activation follows this layer
                    relu_follow = False
                    if hasattr(module, 'activation') and inspect.isfunction(module.activation):
                        relu_follow = module.activation.__name__ in ['relu', 'leaky_relu', 'rrelu']
                    
                    # If we can't detect activation directly, assume ReLU for He init
                    fan_mode = 'fan_out' if isinstance(module, (nn.Conv1d, nn.Conv2d, nn.Conv3d)) else 'fan_in'
                    nonlinearity = 'relu'  # Default assumption
                    
                    # Initialize with He/Kaiming initialization
                    nn.init.kaiming_normal_(module.weight, mode=fan_mode, nonlinearity=nonlinearity)
                    
                    if verbose:
                        print(f"Applied Kaiming initialization to {name}.weight with mode={fan_mode}, nonlinearity={nonlinearity}")
                else:
                    # For other layer types with weights, initialize with a small positive constant
                    nn.init.constant_(module.weight, constant_val)
                    if verbose:
                        print(f"Initialized {name}.weight with constant value {constant_val}")
            
            # Initialize biases with small constant
            if hasattr(module, 'bias') and module.bias is not None:
                nn.init.constant_(module.bias, constant_val)
                if verbose:
                    print(f"Initialized {name}.bias with constant value {constant_val}")
                
    # Return the initialized model for method chaining
    return model

def check_initialization_quality(model, reference_ranges=None):
    """
    Analyze the initialization of a model and compare it to best practices.
    
    Args:
        model (nn.Module): PyTorch model to analyze
        reference_ranges (dict, optional): Reference ranges for good initialization
            
    Returns:
        bool: Whether the model follows initialization best practices
    """
    if reference_ranges is None:
        # Default reference ranges based on best practices
        reference_ranges = {
            'conv': {'mean': (-0.1, 0.1), 'std': (0.01, 1.0)},
            'linear': {'mean': (-0.1, 0.1), 'std': (0.01, 1.0)},
            'bias': {'mean': (-0.1, 0.1), 'std': (0.0, 0.1)}
        }
    
    # Track parameters
    tracker = ParameterTracker(model)
    all_params = {}
    
    # Collect statistics for analysis
    for name, param in model.named_parameters():
        param_type = None
        if 'weight' in name:
            if 'conv' in name:
                param_type = 'conv'
            else:
                param_type = 'linear'
        elif 'bias' in name:
            param_type = 'bias'
        
        if param_type:
            mean = param.data.mean().item()
            std = param.data.std().item()
            all_params[name] = {
                'mean': mean,
                'std': std,
                'type': param_type
            }
    
    # Print analysis header
    print("\n========== Initialization Quality Check ==========")
    print(f"{'Parameter':<40} | {'Mean':<10} | {'Std':<10} | {'Status':<10}")
    print("-" * 75)
    
    # Check if parameters are well-initialized
    well_initialized = True
    
    for name, stats in all_params.items():
        param_type = stats['type']
        mean, std = stats['mean'], stats['std']
        
        # Check if values are in recommended ranges
        mean_in_range = reference_ranges[param_type]['mean'][0] <= mean <= reference_ranges[param_type]['mean'][1]
        std_in_range = reference_ranges[param_type]['std'][0] <= std <= reference_ranges[param_type]['std'][1]
        
        status = "GOOD" if mean_in_range and std_in_range else "WARNING"
        if status == "WARNING":
            well_initialized = False
        
        # Print analysis
        print(f"{name:<40} | {mean:<10.4e} | {std:<10.4e} | {status:<10}")
    
    # Print summary
    print("\n=== Summary ===")
    if well_initialized:
        print("✓ All parameters appear well-initialized")
    else:
        print("⚠ Some parameters may not be optimally initialized")
        print("  - Conv/Linear weights should have mean near 0 and appropriate variance")
        print("  - Biases should have small values")
    
    print("=" * 50)
    
    return well_initialized

def test_initialization():
    """This is a test function to demonstrate the model initialization functionality."""
    # Create a simple model
    model = nn.Sequential(
        nn.Conv2d(3, 16, kernel_size=3, padding=1),
        nn.ReLU(),
        nn.MaxPool2d(2),
        nn.Conv2d(16, 32, kernel_size=3, padding=1),
        nn.ReLU(),
        nn.MaxPool2d(2),
        nn.Flatten(),
        nn.Linear(32 * 8 * 8, 128),
        nn.ReLU(),
        nn.Linear(128, 10)
    )
    
    # Print before initialization
    print("BEFORE INITIALIZATION:")
    check_initialization_quality(model)
    
    # Initialize the model
    model = initialize_model(model, constant_val=1e-4, verbose=True)
    
    # Print after initialization
    print("\nAFTER INITIALIZATION:")
    check_initialization_quality(model)
    
    return model

def example_usage():
    """
    Example usage of the model initialization functions
    
    # Basic usage
    model = YourModel()
    initialize_model(model)
    
    # With custom constant value and verbose output
    model = YourModel()
    initialize_model(model, constant_val=1e-3, verbose=True)
    
    # Check initialization quality
    model = YourModel()
    initialize_model(model)
    check_initialization_quality(model)
    
    # Combine with other trackers for comprehensive analysis
    try:
        from .trackers import ActivationTracker, GradientTracker
    except ImportError:
        from trackers import ActivationTracker, GradientTracker
    
    model = YourModel()
    initialize_model(model)
    
    # Check parameter initialization
    check_initialization_quality(model)
    
    # Monitor activations
    act_tracker = ActivationTracker(model)
    # Forward pass with sample input
    input_data = torch.randn(1, 3, 32, 32)
    model(input_data)
    act_tracker.describe()
    
    # Set up gradient tracking (for training)
    grad_tracker = GradientTracker(model)
    """
    pass

if __name__ == "__main__":
    test_initialization() 