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
    
    def plot(self, save_name =None, bins = 30):
        plt.figure(figsize=(10, 8))
        for idx, (name, activation) in enumerate(self.activations.items()):
            plt.subplot(len(self.activations), 1, idx + 1)
            plt.hist(activation.cpu().detach().flatten().numpy(), bins = bins)
            plt.title(name)
        plt.tight_layout()
        plt.show()
        if save_name is not None:
            plt.savefig(save_name, dpi=300)

    def describe(self, key_name = None):
        if key_name is not None and key_name not in self.activations:
            raise ValueError(f"Key {key_name} not found in activations which has keys {self.activations.keys()}")
        
        # Print header
        print("Activation Statistics:")
        print(f"{'Name':<30} | {'Dtype':<13} | {'Min':>12} | {'Max':>12} | {'Mean':>12} | {'Std':>12}")
        print('-' * 90)
        
        for name, activation in self.activations.items():
            # Skip entries with blank names from Sequential modules
            if not name.strip():
                continue
                
            if key_name is not None and key_name not in name:
                continue
            
            # Format values using scientific notation with 5 decimal places
            min_val = f"{activation.min().item():.1e}"
            max_val = f"{activation.max().item():.1e}"
            mean_val = f"{activation.mean().item():.1e}"
            std_val = f"{activation.std().item():.1e}"
            
            print(f"{name:<30} | {activation.dtype!s:<10} | {min_val:>12} | {max_val:>12} | {mean_val:>12} | {std_val:>12}")
        
        return self  # Allow method chaining

class GradientTracker:
    def __init__(self, model):
        self.gradients = {}
        self.model = model
        self.grad_norms_history = defaultdict(list)
        self.track_gradients()

    def track_gradients_hook(self, name):
        def hook(module, grad_input, grad_output):
            # Store the gradient
            gradient = grad_output[0]
            self.gradients[name] = gradient
            
            # Calculate and store norm automatically if gradient is not None
            if gradient is not None and name.strip():
                self.grad_norms_history[name].append(gradient.norm().item())
        return hook

    def track_gradients(self):
        for name, param in self.model.named_modules():
            param.register_backward_hook(self.track_gradients_hook(name))
    
    def plot(self, save_name=None, bins=30, plot_type="histogram"):
        """
        Plot gradient information.
        
        Args:
            save_name (str, optional): Path to save the figure. Defaults to None.
            bins (int, optional): Number of bins for histogram. Defaults to 30.
            plot_type (str, optional): Type of plot - "histogram" or "norms". Defaults to "histogram".
        """
        plt.figure(figsize=(10, 8))
        
        if plot_type == "histogram":
            filtered_gradients = {name: grad for name, grad in self.gradients.items() 
                               if grad is not None and name.strip()}
            
            if not filtered_gradients:
                print("No gradients available to plot histograms")
                return self
                
            for idx, (name, gradient) in enumerate(filtered_gradients.items()):
                plt.subplot(len(filtered_gradients), 1, idx + 1)
                plt.hist(gradient.cpu().detach().flatten().numpy(), bins=bins)
                plt.title(name)
        elif plot_type == "norms":
            filtered_norms = {name: norms for name, norms in self.grad_norms_history.items() 
                           if norms and len(norms) > 0}
            
            if not filtered_norms:
                print("No gradient norms history available to plot")
                return self
                
            for idx, (name, norms) in enumerate(filtered_norms.items()):
                plt.subplot(len(filtered_norms), 1, idx + 1)
                plt.plot(norms)
                plt.title(f"{name} - Gradient Norm History")
                plt.xlabel("Iteration")
                plt.ylabel("Gradient Norm")
        else:
            raise ValueError(f"Invalid plot_type: {plot_type}. Expected 'histogram' or 'norms'")
            
        plt.tight_layout()
        plt.show()
        if save_name is not None:
            plt.savefig(save_name, dpi=300)
        
        return self  # Allow method chaining

    def describe(self, key_name=None):
        if key_name is not None and key_name not in self.gradients:
            raise ValueError(f"Key {key_name} not found in gradients which has keys {self.gradients.keys()}")
        
        # Print header
        print("Gradient Statistics:")
        print(f"{'Name':<30} | {'Dtype':<13} | {'Min':>12} | {'Max':>12} | {'Mean':>12}  | {'Std':>12}| {'GNorm':>12}")
        print('-' * 90)
        
        for name, gradient in self.gradients.items():
            # Skip entries with blank names from Sequential modules
            if not name.strip():
                continue
                
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
        
        return self  # Allow method chaining

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
        print("Parameter Statistics:")
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
    
    def plot(self, save_name=None, bins=30):
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
    tracker.plot("activations.png")
    tracker.describe()

def test_gradient_tracker():
    model = torch.nn.Sequential(
        torch.nn.Linear(10, 10),
        torch.nn.ReLU(),
        torch.nn.Linear(10, 10)
    )
    tracker = GradientTracker(model)
    
    # Run multiple iterations to build up norm history
    for i in range(10):  # Run 10 iterations to build up history
        # Create input and run forward pass
        x = torch.randn(1, 10, requires_grad=True)
        output = model(x)
        
        # Create a dummy loss and run backward pass
        loss = output.sum()
        loss.backward()
        
        # In a real scenario, you would update the model parameters here
        # For test purposes, we need to zero out the gradients manually
        for param in model.parameters():
            param.grad.zero_()
    
    # Display gradient statistics
    tracker.describe()
    
    # Plot gradient histograms (most recent gradients)
    tracker.plot("gradients.png")
    
    # Plot the gradient norms history (all iterations)
    tracker.plot("gradient_norms.png", plot_type="norms")
    
    print("Gradient norm history captured automatically:", 
          {k: len(v) for k, v in tracker.grad_norms_history.items() if len(v) > 0})

def test_parameter_tracker():
    model = torch.nn.Sequential(
        torch.nn.Linear(10, 10),
        torch.nn.ReLU(),
        torch.nn.Linear(10, 10)
    )
    
    # Example of how to use the parameter tracker with method chaining
    ParameterTracker(model).describe().plot("parameters.png")

    # Compare parameters between the two models
    tracker1 = ParameterTracker(model)
    tracker1.describe().plot("parameters.png")

def example_usage():
    """
    ML Model Trackers API Usage Examples
    -----------------------------------
    
    # ActivationTracker - Monitor activation values through the network
    tracker = ActivationTracker(model)            # Initialize with a PyTorch model
    model(inputs)                                 # Run a forward pass to capture activations
    tracker.plot(save_name="path.png")            # Plot activation histograms
    tracker.describe()                            # Print stats for all activations
    tracker.describe(key_name="conv")             # Print stats for layers containing "conv"
    
    # GradientTracker - Monitor gradients during backpropagation
    tracker = GradientTracker(model)              # Initialize with a PyTorch model
    
    # During training loop:
    for epoch in range(epochs):
        for batch in dataloader:
            inputs, targets = batch
            optimizer.zero_grad()                 # Zero the gradients
            outputs = model(inputs)               # Forward pass
            loss = loss_fn(outputs, targets)      # Compute loss
            loss.backward()                       # Backward pass - gradient norms captured automatically
            optimizer.step()                      # Update model parameters
    
    # After training or during evaluation:
    tracker.plot(save_name="path.png")            # Plot gradient histograms
    tracker.plot(save_name="norms.png", plot_type="norms")  # Plot gradient norm history over iterations
    tracker.describe()                            # Print gradient statistics
    tracker.describe(key_name="conv")             # Print stats for gradients of layers containing "conv"
    
    # ParameterTracker - Analyze model parameters
    tracker = ParameterTracker(model)             # Initialize with a PyTorch model
    tracker.describe()                            # Print all parameter statistics
    tracker.describe(key_name="weight")           # Print stats for parameters containing "weight"
    tracker.plot(save_name="path.png")            # Plot parameter histograms
    
    # Method chaining is supported
    ParameterTracker(model).describe().plot("params.png")
    """
    pass

if __name__ == "__main__":
    test_activation_tracker()
    test_gradient_tracker()
    # Uncomment to test parameter tracker
    test_parameter_tracker()