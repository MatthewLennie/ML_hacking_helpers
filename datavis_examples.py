import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader, TensorDataset
from torchvision import datasets, transforms
from datavis import DataVis
import os
from pathlib import Path
import random

"""
This file contains examples of how to use the DataVis module with different
data types and input formats, using real datasets rather than random noise.
"""

# Create output directory for figures
figures_dir = Path("figures")
figures_dir.mkdir(exist_ok=True, parents=True)

# Helper function to create bad data on the fly
def create_bad_mnist_data(num_samples=5000, corruption_rate=0.3):
    """Create corrupted MNIST data with various artifacts."""
    print("Creating bad MNIST data on the fly...")
    
    # Load MNIST dataset
    mnist_transform = transforms.Compose([transforms.ToTensor()])
    try:
        mnist_train = datasets.MNIST('./data', train=True, download=True, transform=mnist_transform)
    except:
        # Fallback - create synthetic data
        print("Could not download MNIST, creating synthetic data")
        images = torch.randn(num_samples, 1, 28, 28) * 0.1 + 0.5
        labels = torch.randint(0, 10, (num_samples,))
        return images.numpy(), labels.numpy()
    
    # Get a subset of MNIST
    subset_size = min(num_samples, len(mnist_train))
    indices = random.sample(range(len(mnist_train)), subset_size)
    mnist_subset = [mnist_train[i] for i in indices]
    
    # Convert to numpy arrays
    images = torch.stack([img for img, _ in mnist_subset]).numpy()
    labels = np.array([label for _, label in mnist_subset])
    
    # Apply various corruptions
    corrupted_images = images.copy()
    num_to_corrupt = int(subset_size * corruption_rate)
    corrupt_indices = random.sample(range(subset_size), num_to_corrupt)
    
    for idx in corrupt_indices:
        corruption_type = random.randint(0, 5)
        
        if corruption_type == 0:
            # Add noise
            corrupted_images[idx] += np.random.normal(0, 0.5, corrupted_images[idx].shape)
        elif corruption_type == 1:
            # Add high-frequency noise using FFT
            ft = np.fft.fft2(corrupted_images[idx, 0])
            rows, cols = ft.shape
            crow, ccol = rows//2, cols//2
            ft[crow-5:crow+5, ccol-5:ccol+5] = 0  # Remove low frequencies
            corrupted_images[idx, 0] = np.fft.ifft2(ft).real
        elif corruption_type == 2:
            # Make image blank or near-blank
            corrupted_images[idx] = np.random.uniform(0, 0.1, corrupted_images[idx].shape)
        elif corruption_type == 3:
            # Make image all white with small artifacts
            corrupted_images[idx] = np.ones_like(corrupted_images[idx])
            corrupted_images[idx, 0, random.randint(5, 22), random.randint(5, 22)] = 0
        elif corruption_type == 4:
            # Add horizontal or vertical stripes
            if random.random() > 0.5:
                corrupted_images[idx, 0, ::2, :] = 1  # Horizontal stripes
            else:
                corrupted_images[idx, 0, :, ::2] = 1  # Vertical stripes
        else:
            # Invert the image
            corrupted_images[idx] = 1 - corrupted_images[idx]
    
    # Clip values to valid range
    corrupted_images = np.clip(corrupted_images, 0, 1)
    
    print(f"Created {subset_size} images with {num_to_corrupt} corrupted examples")
    return corrupted_images, labels


# Example 1: Use with numpy arrays - creating bad MNIST data on the fly
def example_numpy():
    print("\n=== Example 1: Using Bad MNIST data (created on the fly) with numpy arrays ===")
    
    # Create corrupted MNIST data
    train_images, train_labels = create_bad_mnist_data(num_samples=5000, corruption_rate=0.3)
    
    print(f"Bad MNIST data shape: {train_images.shape}, labels shape: {train_labels.shape}")
    
    # Initialize DataVis with numpy arrays
    vis = DataVis(train_images, train_labels, output_dir="figures/bad_mnist_numpy")
    
    # Plot a single random sample
    img, label, idx, fig_path = vis.plot_sample(name="bad_mnist_single_sample")
    print(f"Displayed sample {idx} with label {label}")
    
    # Plot multiple samples
    vis.plot_samples(n=16, nrow=4, name="bad_mnist_samples_grid")
    
    # Plot pixel statistics
    vis.plot_pixel_stats(stat='mean', name="bad_mnist_pixel_mean")
    vis.plot_pixel_stats(stat='max', name="bad_mnist_pixel_max")
    
    # Plot class distribution - use try-except to handle None return
    try:
        class_counts, _ = vis.plot_class_distribution(name="bad_mnist_class_distribution")
        if class_counts is not None:
            print(f"Class distribution: {class_counts}")
    except Exception as e:
        print(f"Error plotting class distribution: {str(e)}")
    
    # First apply a more lenient filter to avoid empty results
    filtered = vis.filter_blank(min_threshold=0.1, max_threshold=0.9)
    print(f"After blank filter: {len(filtered)} out of {len(vis)} images remain")
    
    # Then apply noise filter with an appropriate threshold for our data
    filtered = filtered.filter_noise(threshold=20, freq_cutoff=14)
    print(f"After noise filter: {len(filtered)} images remain")
    
    # Plot filtered samples
    filtered.plot_samples(n=16, nrow=4, name="bad_mnist_filtered_samples")
    
    # Plot FFT histogram of original and filtered data
    vis.plot_fft_histogram(n_samples=50, name="bad_mnist_fft_original")
    filtered.plot_fft_histogram(n_samples=50, name="bad_mnist_fft_filtered")
    
    # Create a custom filter for demonstration
    print("\nApplying custom filter")
    def custom_filter(img, label, idx):
        # Keep only images of a specific class (class 0) that aren't too dark
        is_class_zero = (label == 0 or label.item() == 0 if hasattr(label, 'item') else False)
        not_too_dark = img.mean() > 0.2
        return is_class_zero and not_too_dark
    
    class_filtered = vis.filter_custom(custom_filter)
    print(f"After custom filter for class 0: {len(class_filtered)} images remain")
    
    # Plot custom filtered samples
    if len(class_filtered) > 0:
        class_filtered.plot_samples(n=min(16, len(class_filtered)), nrow=4, name="bad_mnist_class0_filter")
    
    # Save the filtered dataset
    saved_paths = filtered.save("repaired_mnist_dataset")
    print(f"Saved repaired dataset to: {saved_paths}")


# Example 2: Use with torch tensors - CIFAR-10 dataset
def example_torch():
    print("\n=== Example 2: Using CIFAR-10 with torch tensors ===")
    
    try:
        # Load CIFAR-10 dataset
        cifar_transform = transforms.Compose([
            transforms.ToTensor()
        ])
        
        cifar_train = datasets.CIFAR10('./data', train=True, download=True, transform=cifar_transform)
        
        # Get a subset of CIFAR-10 as torch tensors
        cifar_subset = [cifar_train[i] for i in range(1000)]  # Get 1000 samples
        images = torch.stack([img for img, _ in cifar_subset])
        labels = torch.tensor([label for _, label in cifar_subset])
        
        print(f"CIFAR-10 subset shape: {images.shape}, labels shape: {labels.shape}")
        
        # Get class names for more informative visualization
        cifar_classes = ['airplane', 'automobile', 'bird', 'cat', 'deer', 
                         'dog', 'frog', 'horse', 'ship', 'truck']
        
        # Initialize DataVis with torch tensors
        vis = DataVis(images, labels, output_dir="figures/cifar10_torch")
        
        # Plot samples
        vis.plot_samples(n=25, nrow=5, name="cifar10_samples_grid")
        
        # Plot samples of a specific class (e.g., cats = class 3)
        cat_indices = [i for i, label in enumerate(labels) if label == 3]
        cat_images = images[cat_indices[:16]]
        cat_labels = labels[cat_indices[:16]]
        
        # Create cat-specific visualization
        cat_vis = DataVis(cat_images, cat_labels, output_dir="figures/cifar10_cats")
        cat_vis.plot_samples(n=16, nrow=4, name="cifar10_cats_grid")
        
        # Plot class distribution
        try:
            class_counts, _ = vis.plot_class_distribution(name="cifar10_class_distribution")
            
            if class_counts is not None:
                # Print class names with counts
                for class_idx, count in class_counts.items():
                    try:
                        # Try to access by integer index
                        class_name = cifar_classes[int(class_idx) if isinstance(class_idx, str) else class_idx]
                        print(f"Class {class_idx} ({class_name}): {count} images")
                    except (IndexError, ValueError, TypeError):
                        # If cannot access class name, just show the index
                        print(f"Class {class_idx}: {count} images")
            else:
                print("Could not generate class distribution (no labels available)")
        except Exception as e:
            print(f"Error plotting class distribution: {str(e)}")
            
        # Save a subset in torch format
        saved_paths = vis.save("cifar10_subset", format="torch")
        
    except Exception as e:
        print(f"Error in CIFAR-10 example: {str(e)}")
        
        # Fallback to random data if CIFAR fails
        print("Falling back to simple torch tensor example")
        images = torch.rand(100, 3, 32, 32)
        labels = torch.randint(0, 5, (100,))
        
        vis = DataVis(images, labels, output_dir="figures/torch_example")
        vis.plot_samples(n=9, nrow=3, name="torch_samples_grid")


# Example 3: Use with torchvision datasets - Fashion MNIST
def example_torchvision():
    print("\n=== Example 3: Using Fashion-MNIST with torchvision dataset ===")
    
    try:
        # Load Fashion-MNIST dataset
        fashion_transform = transforms.Compose([
            transforms.ToTensor(),
        ])
        
        fashion_train = datasets.FashionMNIST('./data', train=True, download=True, transform=fashion_transform)
        
        # Class names for Fashion-MNIST
        fashion_classes = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
                          'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
        
        print(f"Fashion-MNIST dataset loaded, length: {len(fashion_train)}")
        sample_img, sample_label = fashion_train[0]
        print(f"Sample image shape: {sample_img.shape}, Sample label: {sample_label} ({fashion_classes[sample_label]})")
        
        # Initialize DataVis with torchvision dataset
        vis = DataVis(fashion_train, output_dir="figures/fashion_mnist")
        
        # Plot random samples
        vis.plot_samples(n=25, nrow=5, name="fashion_mnist_samples_grid")
        
        # Plot class distribution - use try-except to handle None return
        try:
            class_counts, _ = vis.plot_class_distribution(name="fashion_mnist_class_distribution")
            if class_counts is not None:
                # Print class names with counts
                for class_idx, count in class_counts.items():
                    try:
                        # Try to access by integer index
                        class_name = fashion_classes[int(class_idx) if isinstance(class_idx, str) else class_idx]
                        print(f"Class {class_idx} ({class_name}): {count} images")
                    except (IndexError, ValueError, TypeError):
                        # If cannot access class name, just show the index
                        print(f"Class {class_idx}: {count} images")
            else:
                print("Could not generate class distribution (no labels available)")
        except Exception as e:
            print(f"Error plotting class distribution: {str(e)}")
            
        # Plot pixel statistics
        vis.plot_pixel_stats(stat='mean', name="fashion_mnist_pixel_mean")
        vis.plot_pixel_stats(stat='std', name="fashion_mnist_pixel_std")
        
        # FFT analysis
        vis.plot_fft_histogram(n_samples=50, name="fashion_mnist_fft")
        
    except Exception as e:
        print(f"Error in Fashion-MNIST example: {str(e)}")
        
        # Fallback to regular MNIST if Fashion-MNIST fails
        print("Falling back to regular MNIST")
        try:
            mnist_train = datasets.MNIST('./data', train=True, download=True, transform=transforms.ToTensor())
            vis = DataVis(mnist_train, output_dir="figures/mnist")
            vis.plot_samples(n=25, nrow=5, name="mnist_samples_grid")
        except Exception as e2:
            print(f"MNIST fallback also failed: {str(e2)}")


# Example 4: Use with DataLoader - SVHN dataset
def example_dataloader():
    print("\n=== Example 4: Using SVHN with DataLoader ===")
    
    try:
        # Load SVHN (Street View House Numbers) dataset
        svhn_transform = transforms.Compose([
            transforms.ToTensor(),
        ])
        
        svhn_train = datasets.SVHN('./data', split='train', download=True, transform=svhn_transform)
        
        # Create DataLoader
        svhn_loader = DataLoader(svhn_train, batch_size=32, shuffle=True, num_workers=2)
        
        print(f"SVHN dataset loaded, length: {len(svhn_train)}")
        sample_img, sample_label = svhn_train[0]
        print(f"Sample image shape: {sample_img.shape}, Sample label: {sample_label}")
        
        # Initialize DataVis with DataLoader
        vis = DataVis(svhn_loader, output_dir="figures/svhn")
        
        # Plot samples
        vis.plot_samples(n=36, nrow=6, name="svhn_samples_grid")
        
        # Plot class distribution
        try:
            class_counts, _ = vis.plot_class_distribution(name="svhn_class_distribution")
            if class_counts is not None:
                print(f"SVHN class distribution: {class_counts}")
            else:
                print("Could not generate class distribution (no labels available)")
        except Exception as e:
            print(f"Error plotting class distribution: {str(e)}")
        
    except Exception as e:
        print(f"Error in SVHN example: {str(e)}")
        
        # Fallback to subset of CIFAR-10 with DataLoader
        print("Falling back to CIFAR-10 subset with DataLoader")
        try:
            cifar_train = datasets.CIFAR10('./data', train=True, download=True, transform=transforms.ToTensor())
            cifar_loader = DataLoader(cifar_train, batch_size=32, shuffle=True, num_workers=2)
            
            vis = DataVis(cifar_loader, output_dir="figures/cifar_dataloader")
            vis.plot_samples(n=16, nrow=4, name="cifar_samples_grid")
        except Exception as e2:
            print(f"CIFAR-10 fallback also failed: {str(e2)}")
            
            # Last resort: create a simple dataset
            images = torch.rand(200, 3, 32, 32)
            labels = torch.randint(0, 10, (200,))
            dataset = TensorDataset(images, labels)
            dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
            
            vis = DataVis(dataloader, output_dir="figures/dataloader_example")
            vis.plot_samples(n=16, nrow=4, name="dataloader_samples_grid")


# Example 5: Creating and using a custom bad dataset
def example_custom_bad_dataset():
    print("\n=== Example 5: Creating and using a custom bad dataset ===")
    
    # Create a custom class to generate bad data on the fly
    class BadMNISTDataset(Dataset):
        def __init__(self, size=1000, transform=None):
            self.size = size
            self.transform = transform
            try:
                self.mnist = datasets.MNIST('./data', train=True, download=True)
                print(f"Loaded MNIST dataset with {len(self.mnist)} images")
            except Exception as e:
                print(f"Error loading MNIST: {str(e)}")
                # Create dummy data if MNIST can't be loaded
                self.use_dummy = True
                print("Using dummy data instead")
                return
            self.use_dummy = False
            
        def __len__(self):
            return self.size
            
        def __getitem__(self, idx):
            if self.use_dummy:
                # Return dummy data if MNIST couldn't be loaded
                img_tensor = torch.rand(1, 28, 28)
                return img_tensor, torch.tensor(0)
                
            # Get a random MNIST sample
            mnist_idx = random.randint(0, len(self.mnist) - 1)
            img, label = self.mnist[mnist_idx]
            
            # Convert PIL image to numpy array
            img_np = np.array(img) / 255.0
            
            # Apply random corruption with 30% probability
            if random.random() < 0.3:
                corruption_type = random.randint(0, 4)
                
                if corruption_type == 0:
                    # Add noise
                    img_np += np.random.normal(0, 0.3, img_np.shape)
                elif corruption_type == 1:
                    # Make image blank or near-blank
                    img_np = np.random.uniform(0, 0.1, img_np.shape)
                elif corruption_type == 2:
                    # Add stripes
                    img_np[::2, :] = 1.0
                elif corruption_type == 3:
                    # Add a border
                    img_np[0:3, :] = 1.0
                    img_np[-3:, :] = 1.0
                    img_np[:, 0:3] = 1.0
                    img_np[:, -3:] = 1.0
                else:
                    # Invert the image
                    img_np = 1.0 - img_np
            
            # Clip values and convert to tensor
            img_np = np.clip(img_np, 0, 1)
            img_tensor = torch.tensor(img_np, dtype=torch.float32).unsqueeze(0)  # Add channel dimension
            
            if self.transform:
                img_tensor = self.transform(img_tensor)
                
            # Ensure label is a tensor
            label_tensor = torch.tensor(label, dtype=torch.long)
            return img_tensor, label_tensor
    
    try:
        # Create the bad dataset
        bad_dataset = BadMNISTDataset(size=2000)
        
        # Initialize DataVis with the dataset
        vis = DataVis(bad_dataset, output_dir="figures/custom_bad_mnist")
        
        # Plot samples
        vis.plot_samples(n=16, nrow=4, name="custom_bad_mnist_samples")
        
        # Define a custom filter function that properly handles tensor formats
        def custom_filter(img, label, idx):
            # Check if label is a tensor and call item() if needed
            label_value = label.item() if hasattr(label, 'item') else label
            
            # Ensure img is a tensor with the right dimensionality
            if not isinstance(img, torch.Tensor):
                return False
            
            # Keep only images with enough variation (not too uniform)
            pixel_std = torch.std(img).item()
            
            # Keep only certain classes and images with sufficient variation
            is_valid_class = label_value in [0, 1, 2]  # Keep only digits 0, 1, 2
            has_variation = pixel_std > 0.15
            
            return is_valid_class and has_variation
        
        # Apply the filter
        print("Applying custom filter for classes 0-2 with sufficient variation...")
        filtered = vis.filter_custom(custom_filter)
        print(f"Original dataset size: {len(vis)}, Filtered dataset size: {len(filtered)}")
        
        # Plot filtered samples
        if len(filtered) > 0:
            filtered.plot_samples(n=min(16, len(filtered)), nrow=4, name="custom_bad_mnist_filtered")
            
            # FFT analysis before and after
            try:
                vis.plot_fft_histogram(n_samples=50, name="custom_bad_mnist_fft_original")
                filtered.plot_fft_histogram(n_samples=50, name="custom_bad_mnist_fft_filtered")
            except Exception as e:
                print(f"Error generating FFT histograms: {str(e)}")
        else:
            print("No images passed the filter, cannot generate visualizations")
        
    except Exception as e:
        print(f"Error in custom bad dataset example: {str(e)}")
        
        # Fallback to generated bad data
        print("Falling back to generated bad data")
        bad_images, bad_labels = create_bad_mnist_data(num_samples=1000, corruption_rate=0.5)
        vis = DataVis(bad_images, bad_labels, output_dir="figures/generated_bad_mnist")
        vis.plot_samples(n=16, nrow=4, name="generated_bad_mnist_samples")


# Example 6: Using with Hugging Face datasets
def example_huggingface():
    print("\n=== Example 6: Using with Hugging Face datasets ===")
    
    try:
        # Import required libraries
        try:
            from datasets import load_dataset, Dataset
            import numpy as np
            huggingface_available = True
        except ImportError:
            print("Hugging Face datasets package not found. Install with: pip install datasets")
            huggingface_available = False
        
        if huggingface_available:
            # Try loading a simple dataset (beans) that has images
            print("Loading the 'beans' dataset from Hugging Face...")
            try:
                beans_dataset = load_dataset("beans", split="train[:100]")  # Just take first 100 for speed
                print(f"Dataset loaded with {len(beans_dataset)} samples")
                print(f"Dataset features: {beans_dataset.features}")
                print(f"Dataset column names: {beans_dataset.column_names}")
                
                # Display a sample
                first_item = beans_dataset[0]
                print(f"Sample keys: {list(first_item.keys())}")
                
                # Create DataVis with proper image and label keys
                image_key = "image"
                label_key = "labels"
                vis = DataVis(beans_dataset, output_dir="figures/beans", 
                              image_key=image_key, label_key=label_key)
                
                # Visualize samples
                print("Plotting sample images from beans dataset...")
                vis.plot_samples(n=9, nrow=3, name="beans_samples")
                
                # Plot class distribution
                try:
                    class_counts, _ = vis.plot_class_distribution(name="beans_class_distribution")
                    if class_counts:
                        print("Class distribution:")
                        for class_id, count in class_counts.items():
                            # Check for class names in dataset features
                            if hasattr(beans_dataset.features[label_key], 'names'):
                                class_names = beans_dataset.features[label_key].names
                                class_name = class_names[int(class_id)] if isinstance(class_id, str) else class_names[class_id]
                                print(f"  Class {class_id} ({class_name}): {count}")
                            else:
                                print(f"  Class {class_id}: {count}")
                except Exception as e:
                    print(f"Error plotting class distribution: {str(e)}")
                
            except Exception as e:
                print(f"Error with beans dataset: {str(e)}")
                
                # Try with MNIST as a fallback
                print("\nTrying with 'mnist' dataset from Hugging Face...")
                try:
                    mnist_dataset = load_dataset("mnist", split="train[:100]")  # Just take first 100 for speed
                    print(f"MNIST dataset loaded with {len(mnist_dataset)} samples")
                    print(f"MNIST column names: {mnist_dataset.column_names}")
                    
                    # Create DataVis with proper image and label keys
                    vis = DataVis(mnist_dataset, output_dir="figures/huggingface_mnist", 
                                 image_key="image", label_key="label")
                    
                    # Visualize samples
                    vis.plot_samples(n=25, nrow=5, name="huggingface_mnist_samples")
                    
                    # Plot class distribution
                    class_counts, _ = vis.plot_class_distribution(name="huggingface_mnist_class_distribution")
                    if class_counts:
                        print("MNIST class distribution:")
                        for class_id, count in class_counts.items():
                            print(f"  Digit {class_id}: {count}")
                    
                except Exception as e:
                    print(f"Fallback to MNIST also failed: {str(e)}")
        
        # If all fails or Hugging Face is not available, create a mock dataset
        if not huggingface_available or 'vis' not in locals():
            print("\nCreating a mock Hugging Face style dataset")
            # Create a simple dictionary-style dataset
            class MockHFDataset(Dataset):
                def __init__(self, size=100):
                    self.size = size
                
                def __len__(self):
                    return self.size
                
                def __getitem__(self, idx):
                    # Create a random image and label
                    img = torch.rand(3, 32, 32)
                    label = torch.randint(0, 10, (1,)).item()
                    return {"image": img, "label": label}
            
            mock_dataset = MockHFDataset(100)
            # Directly create DataVis with our mock dataset
            vis = DataVis(mock_dataset, output_dir="figures/mock_huggingface")
            vis.plot_samples(n=16, nrow=4, name="mock_huggingface_samples")
    
    except Exception as e:
        print(f"Error in Hugging Face example: {str(e)}")


# Run examples
if __name__ == "__main__":
    print("DataVis Module Examples with Real Datasets")
    print("-----------------------------------------")
    print(f"Figures will be saved to: {figures_dir.absolute()}")
    
    # Run each example
    example_numpy()
    example_torch()
    example_torchvision()
    example_dataloader()
    example_custom_bad_dataset()
    # example_huggingface()
    
    print(f"\nAll examples completed. Figures saved to: {figures_dir.absolute()}") 