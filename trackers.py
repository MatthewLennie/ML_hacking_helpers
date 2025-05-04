import matplotlib.pyplot as plt
import numpy as np
import torch
from collections import defaultdict

class ActivationTracker:
    def __init__(self, model):
        self.activations = {}
        self.model = model
        self.track_activations()

    def track_activations_hook(self, name):
        def hook(module, input, output):
            self.activations[name] = output
        return hook

    def track_activations(self):
        for name, param in self.model.named_modules():
            param.register_forward_hook(self.track_activations_hook(name))
    
    def plot_activations(self, save_name =None, bins = 30):
        plt.figure(figsize=(10, 8))
        for idx, (name, activation) in enumerate(self.activations.items()):
            plt.subplot(len(self.activations), 1, idx + 1)
            plt.hist(activation.cpu().detach().flatten().numpy(), bins = bins)
            plt.title(name)
        plt.tight_layout()
        plt.show()
        if save_name is not None:
            plt.savefig(save_name, dpi=300)

    def activations_describe(self, key_name = None):
        if key_name is not None and key_name not in self.activations:
            raise ValueError(f"Key {key_name} not found in activations which has keys {self.activations.keys()}")
        
        # Print header
        print(f"{'Name':<30} | {'Dtype':<13} | {'Min':>12} | {'Max':>12} | {'Mean':>12} | {'Std':>12}")
        print('-' * 90)
        
        for name, activation in self.activations.items():
            if key_name is not None and key_name not in name:
                continue
            
            # Format values using scientific notation with 5 decimal places
            min_val = f"{activation.min().item():.1e}"
            max_val = f"{activation.max().item():.1e}"
            mean_val = f"{activation.mean().item():.1e}"
            std_val = f"{activation.std().item():.1e}"
            
            print(f"{name:<30} | {activation.dtype!s:<10} | {min_val:>12} | {max_val:>12} | {mean_val:>12} | {std_val:>12}")

class GradientTracker:
    def __init__(self, model):
        self.gradients = {}
        self.model = model
        self.track_gradients()
        self.grad_norms_history = defaultdict(list)

    def track_gradients_hook(self, name):
        def hook(module, grad_input, grad_output):
            self.gradients[name] = grad_output[0]
        return hook

    def track_gradients(self):
        for name, param in self.model.named_modules():
            param.register_backward_hook(self.track_gradients_hook(name))
    
    def plot_gradients(self, save_name=None, bins=30):
        plt.figure(figsize=(10, 8))
        for idx, (name, gradient) in enumerate(self.gradients.items()):
            if gradient is None:
                continue
            plt.subplot(len(self.gradients), 1, idx + 1)
            plt.hist(gradient.cpu().detach().flatten().numpy(), bins=bins)
            plt.title(name)
        plt.tight_layout()
        plt.show()
        if save_name is not None:
            plt.savefig(save_name, dpi=300)

    def gradients_describe(self, key_name=None):
        if key_name is not None and key_name not in self.gradients:
            raise ValueError(f"Key {key_name} not found in gradients which has keys {self.gradients.keys()}")
        
        # Print header
        print(f"{'Name':<30} | {'Dtype':<13} | {'Min':>12} | {'Max':>12} | {'Mean':>12} | {'Std':>12}")
        print('-' * 90)
        
        for name, gradient in self.gradients.items():
            if key_name is not None and key_name not in name:
                continue
            if gradient is None:
                print(f"{name:<30} | {'N/A':<13} | {'N/A':>12} | {'N/A':>12} | {'N/A':>12} | {'N/A':>12}")
                continue
            
            # Format values using scientific notation with 1 decimal place
            min_val = f"{gradient.min().item():.1e}"
            max_val = f"{gradient.max().item():.1e}"
            mean_val = f"{gradient.mean().item():.1e}"
            std_val = f"{gradient.std().item():.1e}"
            grad_norm = f"{gradient.norm().item():.1e}"
            print(f"{name:<30} | {gradient.dtype!s:<13} | {min_val:>12} | {max_val:>12} | {mean_val:>12} | {std_val:>12} | {grad_norm:>12}")
            self.grad_norms_history[name].append(grad_norm)
    def plot_grad_norms(self, save_name=None):
        plt.figure(figsize=(10, 8))
        for idx, (name, norms) in enumerate(self.grad_norms_history.items()):
            plt.subplot(len(self.grad_norms_history), 1, idx + 1)
            plt.plot(norms)
            plt.title(name)
        plt.tight_layout()
        plt.show()
        if save_name is not None:
            plt.savefig(save_name, dpi=300)

class ParameterTracker:
    def __init__(self, model):
        self.model = model
        self.parameters = self._get_parameters()
    
    def _get_parameters(self):
        """Collect all named parameters from the model."""
        parameters = {}
        for name, param in self.model.named_parameters():
            parameters[name] = param.data
        return parameters
    
    def describe(self, key_name=None):
        """Describe the parameters with formatted output."""
        if key_name is not None and not any(key_name in name for name in self.parameters.keys()):
            raise ValueError(f"Key {key_name} not found in parameters which has keys {self.parameters.keys()}")
        
        # Print header
        print(f"{'Name':<50} | {'Dtype':<13} | {'Min':>12} | {'Max':>12} | {'Mean':>12} | {'Std':>12} | {'Norm':>12}")
        print('-' * 120)
        
        for name, param in self.parameters.items():
            if key_name is not None and key_name not in name:
                continue
            
            # Format values using scientific notation with 1 decimal place
            min_val = f"{param.min().item():.1e}"
            max_val = f"{param.max().item():.1e}"
            mean_val = f"{param.mean().item():.1e}"
            std_val = f"{param.std().item():.1e}"
            norm_val = f"{param.norm().item():.1e}"
            
            print(f"{name:<50} | {param.dtype!s:<13} | {min_val:>12} | {max_val:>12} | {mean_val:>12} | {std_val:>12} | {norm_val:>12}")
        
        return self  # Allow method chaining
    
    def plot_parameters(self, save_name=None, bins=30):
        """Plot histograms of parameter distributions."""
        plt.figure(figsize=(12, 10))
        
        sorted_params = sorted(self.parameters.items(), key=lambda x: x[0])
        num_params = len(sorted_params)
        
        for idx, (name, param) in enumerate(sorted_params):
            plt.subplot(num_params, 1, idx + 1)
            plt.hist(param.cpu().detach().flatten().numpy(), bins=bins)
            plt.title(name)
        
        plt.tight_layout()
        plt.show()
        
        if save_name is not None:
            plt.savefig(save_name, dpi=300)
        
        return self  # Allow method chaining
    


def test_activation_tracker():
    model = torch.nn.Sequential(
        torch.nn.Linear(10, 10),
        torch.nn.ReLU(),
        torch.nn.Linear(10, 10)
    )
    tracker = ActivationTracker(model)
    model(torch.randn(1, 10))
    tracker.plot_activations("activations.png")
    tracker.activations_describe()

def test_gradient_tracker():
    model = torch.nn.Sequential(
        torch.nn.Linear(10, 10),
        torch.nn.ReLU(),
        torch.nn.Linear(10, 10)
    )
    tracker = GradientTracker(model)
    
    # Create input and run forward pass
    x = torch.randn(1, 10, requires_grad=True)
    output = model(x)
    
    # Create a dummy loss and run backward pass
    loss = output.sum()
    loss.backward()
    
    # Display gradients
    tracker.plot_gradients("gradients.png")
    tracker.plot_grad_norms("grad_norms.png")
    tracker.gradients_describe()

def test_parameter_tracker():
    model = torch.nn.Sequential(
        torch.nn.Linear(10, 10),
        torch.nn.ReLU(),
        torch.nn.Linear(10, 10)
    )
    
    # Example of how to use the parameter tracker with method chaining
    ParameterTracker(model).describe().plot_parameters("parameters.png")

    # Compare parameters between the two models
    tracker1 = ParameterTracker(model)
    tracker1.describe().plot_parameters("parameters.png")

def example_usage():
    """
    ML Model Trackers API Usage Examples
    -----------------------------------
    
    # ActivationTracker - Monitor activation values through the network
    tracker = ActivationTracker(model)            # Initialize with a PyTorch model
    model(inputs)                                 # Run a forward pass to capture activations
    tracker.plot_activations(save_name="path.png")# Plot activation histograms
    tracker.activations_describe()                # Print stats for all activations
    tracker.activations_describe(key_name="conv") # Print stats for layers containing "conv"
    
    # GradientTracker - Monitor gradients during backpropagation
    tracker = GradientTracker(model)              # Initialize with a PyTorch model
    outputs = model(inputs)                       # Forward pass
    loss = loss_fn(outputs, targets)              # Compute loss
    loss.backward()                               # Backward pass to capture gradients
    tracker.plot_gradients(save_name="path.png")  # Plot gradient histograms
    tracker.plot_grad_norms(save_name="path.png") # Plot gradient norm history
    tracker.gradients_describe()                  # Print gradient statistics
    
    # ParameterTracker - Analyze model parameters
    tracker = ParameterTracker(model)             # Initialize with a PyTorch model
    tracker.describe()                            # Print all parameter statistics
    tracker.describe(key_name="weight")           # Print stats for parameters containing "weight"
    tracker.plot_parameters(save_name="path.png") # Plot parameter histograms
    
    # Method chaining is supported
    ParameterTracker(model).describe().plot_parameters("params.png")
    """
    pass

if __name__ == "__main__":
    test_activation_tracker()
    test_gradient_tracker()
    # Uncomment to test parameter tracker
    test_parameter_tracker()