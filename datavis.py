import numpy as np
import torch
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, Dataset, TensorDataset
from torchvision.utils import make_grid
from pathlib import Path
import random
from typing import Optional, Union, Tuple, List, Dict, Any, Callable
import math
import os

class DataVis:
    """
    A class for visualizing and processing image datasets with a robust API that handles
    various input types including DataLoader, Dataset, numpy arrays, torch tensors,
    and Hugging Face datasets.
    """
    
    def __init__(self, 
                 data: Any,
                 labels: Optional[Any] = None,
                 transform: Optional[Callable] = None,
                 channel_first: bool = True,
                 output_dir: str = "figures",
                 image_key: str = "image",
                 label_key: str = "label"):
        """
        Initialize the DataVis with image data and optional labels.
        
        Args:
            data: Input data in various formats (DataLoader, Dataset, numpy array, torch tensor, etc.)
            labels: Optional labels corresponding to the data
            transform: Optional transform to apply to images when retrieving
            channel_first: Whether the image data is in channel-first format (default: True)
            output_dir: Directory to save figures (default: "figures")
            image_key: Key for images in dictionary-style datasets (e.g., HuggingFace)
            label_key: Key for labels in dictionary-style datasets (e.g., HuggingFace)
        """
        self.transform = transform
        self.channel_first = channel_first
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True, parents=True)
        self.image_key = image_key
        self.label_key = label_key
        self._load_data(data, labels)
    
    def _load_data(self, data, labels):
        """Load data from various input formats into a standardized internal representation."""
        self.dataset = None
        self.dataloader = None
        self.images = None
        self.labels = None
        self._original_data = data
        self.is_huggingface = False
        
        # Handle different input types
        if isinstance(data, DataLoader):
            self.dataloader = data
            # Extract dataset from dataloader
            self.dataset = data.dataset
        # Special handling for Hugging Face datasets
        elif str(type(data)).find('datasets.arrow_dataset.Dataset') >= 0:
            # This is a Hugging Face dataset
            self.is_huggingface = True
            self.dataset = data
            try:
                # Try to access the first item to see available keys
                first_item = data[0]
                print(f"Detected Hugging Face dataset with keys: {list(first_item.keys())}")
                if self.image_key not in first_item:
                    print(f"Warning: Image key '{self.image_key}' not found in dataset. Available keys: {list(first_item.keys())}")
            except Exception as e:
                print(f"Warning: Could not access first item in dataset: {str(e)}")
        elif isinstance(data, Dataset):
            self.dataset = data
            
            # Check if it's likely a dictionary-style dataset (like Hugging Face)
            try:
                # Try to access the first item
                first_item = data[0]
                if isinstance(first_item, dict) and self.image_key in first_item:
                    self.is_huggingface = True
                    print(f"Detected dictionary-style dataset with keys: {list(first_item.keys())}")
            except:
                pass
                
        elif isinstance(data, torch.Tensor):
            # Torch tensor images
            self.images = data
            if labels is not None:
                self.labels = labels if isinstance(labels, torch.Tensor) else torch.tensor(labels)
            # Create dataset for easy iteration
            if self.labels is not None:
                self.dataset = TensorDataset(self.images, self.labels)
            else:
                # Create dataset with dummy labels
                self.dataset = TensorDataset(self.images, torch.zeros(len(self.images)))
        elif isinstance(data, np.ndarray):
            # Numpy array images
            self.images = torch.tensor(data)
            if labels is not None:
                self.labels = torch.tensor(labels) if isinstance(labels, np.ndarray) else labels
            # Create dataset for easy iteration
            if self.labels is not None:
                self.dataset = TensorDataset(self.images, self.labels)
            else:
                # Create dataset with dummy labels
                self.dataset = TensorDataset(self.images, torch.zeros(len(self.images)))
        elif isinstance(data, str) or isinstance(data, Path):
            # Assume it's a path to numpy files
            path = Path(data)
            if path.is_file() and path.suffix == '.npy':
                # Single numpy file
                self.images = torch.tensor(np.load(path))
                if labels is not None:
                    labels_path = labels if isinstance(labels, (str, Path)) else path.with_name(f"{path.stem}_labels.npy")
                    if Path(labels_path).exists():
                        self.labels = torch.tensor(np.load(labels_path))
                # Create dataset
                if self.labels is not None:
                    self.dataset = TensorDataset(self.images, self.labels)
                else:
                    # Create dataset with dummy labels
                    self.dataset = TensorDataset(self.images, torch.zeros(len(self.images)))
            else:
                raise ValueError(f"Unsupported file type or path doesn't exist: {path}")
        else:
            raise ValueError(f"Unsupported data type: {type(data)}")
        
        # If we have a dataset but haven't extracted images yet, create a sample cache
        if self.dataset is not None and self.images is None:
            self._cache_sample()
    
    def _cache_sample(self, n=100):
        """Cache a sample of images and labels for quick access."""
        self._sample_images = []
        self._sample_labels = []
        self._sample_indices = []
        
        for i in range(min(n, len(self.dataset))):
            try:
                if self.is_huggingface:
                    # Handle Hugging Face dataset with dictionary output
                    sample = self.dataset[i]
                    img = sample[self.image_key]
                    label = sample.get(self.label_key, None)
                    idx = i
                else:
                    # Try dataset with index (idx, img, label)
                    sample = self.dataset[i]
                    if len(sample) >= 3:
                        idx, img, label = sample[0], sample[1], sample[2]
                    elif len(sample) == 2:
                        img, label = sample[0], sample[1]
                        idx = i
                    else:
                        img = sample[0] if isinstance(sample, tuple) else sample
                        label = torch.tensor(0)
                        idx = i
            except:
                # Fallback to standard (img, label) tuple
                try:
                    img, label = self.dataset[i]
                    idx = i
                except:
                    # Final fallback: just an image
                    img = self.dataset[i]
                    label = torch.tensor(0)
                    idx = i
            
            # Standardize label format
            if label is not None:
                # Ensure label is tensor if possible
                if not isinstance(label, torch.Tensor) and not isinstance(label, (int, float, str)):
                    try:
                        # Try to convert numpy or other types to tensor
                        label = torch.tensor(label)
                    except:
                        # If conversion fails, keep as is
                        pass
            
            img_tensor = self._ensure_tensor(img)
            self._sample_images.append(img_tensor)
            self._sample_labels.append(label)
            self._sample_indices.append(idx)
    
    def _ensure_tensor(self, img):
        """Convert image to tensor if it's not already one."""
        if not isinstance(img, torch.Tensor):
            try:
                img = torch.tensor(img)
            except:
                # Handle PIL images or other formats
                try:
                    import numpy as np
                    img = torch.tensor(np.array(img))
                except:
                    raise ValueError(f"Could not convert image of type {type(img)} to tensor")
        return img
    
    def _prepare_for_display(self, img):
        """Prepare image tensor for display."""
        if self.transform:
            img = self.transform(img)
        
        # Ensure 3D tensor (C, H, W)
        if len(img.shape) == 2:
            img = img.unsqueeze(0)  # Add channel dimension
        
        # Handle grayscale images (expand to 3 channels for display)
        if img.shape[0] == 1:
            img = img.repeat(3, 1, 1)
        
        # If more than 3 channels, take first 3
        if img.shape[0] > 3:
            img = img[:3]
        
        # Normalize values to [0, 1] range for displaying
        min_val = img.min().item()
        max_val = img.max().item()
        
        # First check if image is already in [0, 1] range
        if 0 <= min_val and max_val <= 1:
            # Image is already properly scaled
            pass
        # MNIST normalization detection (mean=0.1307, std=0.3081)
        elif -0.5 < min_val < 0 and max_val < 0.9:
            # This looks like a normalized MNIST image
            # De-normalize: img = img * std + mean
            img = img * 0.3081 + 0.1307
        # ImageNet normalization detection
        elif -3 < min_val < -1 and 1 < max_val < 3:
            # This looks like ImageNet normalization - rescale to [0,1] range
            img = (img - min_val) / (max_val - min_val)
        # Generic normalization with negative values
        elif min_val < 0:
            # Rescale from [min, max] to [0, 1]
            img = (img - min_val) / (max_val - min_val + 1e-8)
        # Values above 1 but not negative (e.g. 0-255 range)
        elif max_val > 1:
            if max_val <= 255:
                # Standard 0-255 range
                img = img / 255.0
            else:
                # Arbitrary large values
                img = img / max_val
        
        # Final safety clip to ensure valid display range
        img = torch.clamp(img, 0.0, 1.0)
        
        return img
    
    def _save_figure(self, name, dpi=150, display=True):
        """Save a figure with a consistent naming pattern and optionally display it."""
        filename = f"{name}.png"
        path = self.output_dir / filename
        plt.savefig(path, dpi=dpi, bbox_inches='tight')
        
        if display:
            plt.show()
        else:
            plt.close()
            
        print(f"Figure saved to: {path}")
        return str(path)
    
    def get_sample(self, idx=None):
        """
        Get a sample image and label by index or random if not specified.
        
        Args:
            idx: Optional index to retrieve (random if None)
            
        Returns:
            tuple: (image, label, idx)
        """
        if idx is None:
            idx = random.randint(0, len(self.dataset) - 1)
        
        if self.images is not None:
            img = self.images[idx]
            label = self.labels[idx] if self.labels is not None else None
        else:
            try:
                if self.is_huggingface:
                    # Handle Hugging Face dataset with dictionary output
                    sample = self.dataset[idx]
                    img = sample[self.image_key]
                    label = sample.get(self.label_key, None)
                else:
                    # Try standard Dataset interface
                    if len(self.dataset[idx]) == 2:
                        img, label = self.dataset[idx]
                    else:
                        img = self.dataset[idx]
                        label = None
            except Exception as e:
                # Fallback for non-standard datasets
                try:
                    # Some datasets might have a custom __getitem__ method
                    sample = self.dataset[idx]
                    if isinstance(sample, dict) and self.image_key in sample:
                        # Handle dictionary output (common in Hugging Face datasets)
                        img = sample[self.image_key]
                        label = sample.get(self.label_key, None)
                    elif isinstance(sample, tuple) and len(sample) >= 2:
                        img, label = sample[0], sample[1]
                    else:
                        img = sample
                        label = None
                except:
                    # Last resort: access cached sample if available
                    if hasattr(self, '_sample_images') and idx < len(self._sample_images):
                        img = self._sample_images[idx]
                        label = self._sample_labels[idx] if idx < len(self._sample_labels) else None
                    else:
                        raise ValueError(f"Could not retrieve sample at index {idx}")
        
        img = self._prepare_for_display(img)
        return img, label, idx
    
    def plot_sample(self, idx=None, figsize=(5, 5), title=None, save=True, display=True, name=None):
        """
        Plot a single sample by index or random if not specified.
        
        Args:
            idx: Optional index to retrieve (random if None)
            figsize: Figure size (width, height) in inches
            title: Optional title for the plot
            save: Whether to save the figure (default: True)
            display: Whether to display the figure (default: True)
            name: Optional filename for saving (default: auto-generated)
        """
        img, label, idx = self.get_sample(idx)
        
        plt.figure(figsize=figsize)
        plt.imshow(img.permute(1, 2, 0).cpu().numpy())
        
        # Create title with label and index if available
        if title is None:
            if label is not None:
                # Handle different label types
                if hasattr(label, 'item'):
                    label_value = label.item()
                else:
                    label_value = label
                title = f"Sample {idx}, Label: {label_value}"
            else:
                title = f"Sample {idx}"
        
        plt.title(title)
        plt.axis('off')
        plt.tight_layout()
        
        # Generate name if not provided
        if name is None:
            name = f"sample_{idx}"
        
        # Save and/or display figure
        if save:
            self._save_figure(name, display=display)
        elif display:
            plt.show()
    
    def plot_samples(self, n=16, nrow=4, figsize=(10, 10), random_samples=True, 
                     with_labels=True, save=True, display=True, name=None):
        """
        Plot multiple samples in a grid.
        
        Args:
            n: Number of samples to plot
            nrow: Number of images per row
            figsize: Figure size (width, height) in inches
            random_samples: Whether to select random samples or first n
            with_labels: Whether to display labels on the images
            save: Whether to save the figure (default: True)
            display: Whether to display the figure (default: True)
            name: Optional filename for saving (default: auto-generated)
        """
        n = min(n, len(self.dataset))
        indices = random.sample(range(len(self.dataset)), n) if random_samples else list(range(n))
        
        images = []
        labels = []
        
        for idx in indices:
            img, label, _ = self.get_sample(idx)
            images.append(img)
            labels.append(label)
        
        # Create grid
        grid = make_grid(images, nrow=nrow, padding=2)
        
        plt.figure(figsize=figsize)
        plt.imshow(grid.permute(1, 2, 0).cpu().numpy())
        
        # Add labels
        if with_labels and labels[0] is not None:
            for i, (idx, label) in enumerate(zip(indices, labels)):
                row = i // nrow
                col = i % nrow
                
                # Calculate position for text
                x = col * (grid.shape[2] // nrow) + (grid.shape[2] // nrow) // 2
                y = row * (grid.shape[1] // (n // nrow + (n % nrow > 0))) + 10
                
                # Get label value, handling different types
                if hasattr(label, 'item'):
                    label_value = label.item()
                else:
                    label_value = label
                
                # Add label text
                plt.text(x, y, f"{idx}: {label_value}", 
                         color='white', fontsize=9, ha='center',
                         bbox=dict(facecolor='black', alpha=0.5, pad=1))
        
        plt.axis('off')
        plt.tight_layout()
        
        # Generate name if not provided
        if name is None:
            name = f"samples_grid_{n}"
        
        # Save and/or display figure
        if save:
            self._save_figure(name, display=display)
        elif display:
            plt.show()
    
    def filter_noise(self, threshold=10, freq_cutoff=14):
        """
        Filter out noisy images using FFT analysis.
        
        Args:
            threshold: Threshold for FFT sum to determine noise
            freq_cutoff: Frequency cutoff for the FFT analysis
            
        Returns:
            DataVis: New DataVis instance with filtered data
        """
        filtered_images = []
        filtered_labels = []
        
        # Use tqdm for progress bar if available
        try:
            from tqdm import tqdm
            iterator = tqdm(range(len(self.dataset)), desc="Filtering noise")
        except ImportError:
            print("Filtering noise... (install tqdm for progress bar)")
            iterator = range(len(self.dataset))
        
        for i in iterator:
            img, label, _ = self.get_sample(i)
            
            # Take first channel for FFT analysis
            fft = torch.fft.fft(img[0])
            
            # Check if high frequency components exceed threshold
            if fft[freq_cutoff:].abs().sum() < threshold:
                filtered_images.append(img)
                if label is not None:
                    filtered_labels.append(label)
        
        # Make sure we have at least one image
        if not filtered_images:
            print("Warning: No images passed the filter. Returning original dataset.")
            return self
        
        # Stack images and labels
        images_tensor = torch.stack(filtered_images)
        
        # Try to handle labels - carefully stack them or convert if needed
        if filtered_labels:
            try:
                labels = torch.stack(filtered_labels)
            except:
                # If stacking fails, try converting labels to tensors
                try:
                    labels = torch.tensor(filtered_labels)
                except:
                    print("Warning: Could not convert labels to tensor. Proceeding without labels.")
                    labels = None
        else:
            labels = None
        
        print(f"Filtered out {len(self.dataset) - len(filtered_images)} images. {len(filtered_images)} remaining.")
        
        # Create new DataVis with same output directory
        new_vis = DataVis(images_tensor, labels, output_dir=self.output_dir)
        return new_vis
    
    def filter_blank(self, min_threshold=0.05, max_threshold=0.95):
        """
        Filter out blank or near-blank images.
        
        Args:
            min_threshold: Minimum pixel value threshold
            max_threshold: Maximum pixel value threshold
            
        Returns:
            DataVis: New DataVis instance with filtered data
        """
        filtered_images = []
        filtered_labels = []
        
        # Use tqdm for progress bar if available
        try:
            from tqdm import tqdm
            iterator = tqdm(range(len(self.dataset)), desc="Filtering blank images")
        except ImportError:
            print("Filtering blank images... (install tqdm for progress bar)")
            iterator = range(len(self.dataset))
        
        for i in iterator:
            img, label, _ = self.get_sample(i)
            
            # Check if image is not blank (has enough contrast)
            if img.min() < min_threshold and img.max() > max_threshold:
                filtered_images.append(img)
                if label is not None:
                    filtered_labels.append(label)
        
        # Make sure we have at least one image
        if not filtered_images:
            print("Warning: No images passed the filter. Returning original dataset.")
            return self
        
        # Stack images and labels
        images_tensor = torch.stack(filtered_images)
        
        # Try to handle labels - carefully stack them or convert if needed
        if filtered_labels:
            try:
                labels = torch.stack(filtered_labels)
            except:
                # If stacking fails, try converting labels to tensors
                try:
                    labels = torch.tensor(filtered_labels)
                except:
                    print("Warning: Could not convert labels to tensor. Proceeding without labels.")
                    labels = None
        else:
            labels = None
        
        print(f"Filtered out {len(self.dataset) - len(filtered_images)} images. {len(filtered_images)} remaining.")
        
        # Create new DataVis with same output directory
        new_vis = DataVis(images_tensor, labels, output_dir=self.output_dir)
        return new_vis
    
    def filter_custom(self, filter_function):
        """
        Filter dataset using a custom filter function.
        
        Args:
            filter_function: Function that takes (img, label, idx) and returns True to keep, False to filter out
            
        Returns:
            DataVis: New DataVis instance with filtered data
        """
        filtered_images = []
        filtered_labels = []
        
        # Use tqdm for progress bar if available
        try:
            from tqdm import tqdm
            iterator = tqdm(range(len(self.dataset)), desc="Applying custom filter")
        except ImportError:
            print("Applying custom filter... (install tqdm for progress bar)")
            iterator = range(len(self.dataset))
        
        for i in iterator:
            img, label, idx = self.get_sample(i)
            
            # Apply the custom filter function
            try:
                keep = filter_function(img, label, idx)
            except Exception as e:
                print(f"Warning: Filter function failed for item {i}: {str(e)}. Skipping item.")
                continue
                
            if keep:
                filtered_images.append(img)
                if label is not None:
                    filtered_labels.append(label)
        
        # Make sure we have at least one image
        if not filtered_images:
            print("Warning: No images passed the filter. Returning original dataset.")
            return self
        
        # Stack images and labels
        images_tensor = torch.stack(filtered_images)
        
        # Try to handle labels - carefully stack them or convert if needed
        if filtered_labels:
            try:
                labels = torch.stack(filtered_labels)
            except:
                # If stacking fails, try converting labels to tensors
                try:
                    labels = torch.tensor(filtered_labels)
                except:
                    print("Warning: Could not convert labels to tensor. Proceeding without labels.")
                    labels = None
        else:
            labels = None
        
        print(f"Filtered out {len(self.dataset) - len(filtered_images)} images. {len(filtered_images)} remaining.")
        
        # Create new DataVis with same output directory
        new_vis = DataVis(images_tensor, labels, output_dir=self.output_dir)
        return new_vis
    
    def plot_fft_histogram(self, n_samples=100, figsize=(15, 10), save=True, display=True, name=None):
        """
        Plot histogram of FFT magnitudes for a sample of images.
        
        Args:
            n_samples: Number of samples to analyze
            figsize: Figure size (width, height) in inches
            save: Whether to save the figure (default: True)
            display: Whether to display the figure (default: True)
            name: Optional filename for saving (default: auto-generated)
            
        Returns:
            tuple: (ffts, figure_path)
        """
        # Sample images
        indices = random.sample(range(len(self.dataset)), min(n_samples, len(self.dataset)))
        ffts = []
        
        for idx in indices:
            img, _, _ = self.get_sample(idx)
            # Take first channel for FFT
            fft = torch.fft.fft(img[0][0])
            ffts.append(fft)
        
        ffts = torch.stack(ffts)
        
        # Determine dimensions
        num_cols = ffts.shape[1]  # Number of frequency components
        num_rows = 4              # Number of rows to display
        cols_per_row = math.ceil(num_cols / num_rows)
        
        plt.figure(figsize=figsize)
        
        for i in range(num_cols):
            row = i // cols_per_row
            col = i % cols_per_row
            
            plt.subplot(num_rows, cols_per_row, row * cols_per_row + col + 1)
            
            # Plot histogram of FFT magnitudes for this frequency component
            hists = torch.histc(ffts[:, i].abs(), bins=20, min=0, max=10)
            plt.plot(hists[1:])
            plt.title(f"Freq {i}")
            plt.xticks([])
        
        plt.tight_layout(pad=1.0)
        
        # Generate name if not provided
        if name is None:
            name = f"fft_histogram_{n_samples}"
        
        # Save and/or display figure
        path = None
        if save:
            path = self._save_figure(name, display=display)
        elif display:
            plt.show()
        
        return ffts, path
    
    def plot_class_distribution(self, figsize=(10, 6), save=True, display=True, name=None):
        """
        Plot distribution of classes in the dataset.
        
        Args:
            figsize: Figure size (width, height) in inches
            save: Whether to save the figure (default: True)
            display: Whether to display the figure (default: True)
            name: Optional filename for saving (default: auto-generated)
            
        Returns:
            tuple: (class_counts, figure_path) or (None, None) if no labels are available
        """
        if self.labels is None:
            print("No labels available to plot class distribution")
            return None, None
        
        # Count classes
        if isinstance(self.labels, torch.Tensor):
            labels = self.labels.cpu().numpy()
        else:
            # Try to extract all labels from dataset
            labels = []
            for i in range(len(self.dataset)):
                _, label, _ = self.get_sample(i)
                if label is not None:
                    # Handle different label types
                    if hasattr(label, 'item'):
                        labels.append(label.item())
                    else:
                        labels.append(label)
            labels = np.array(labels)
        
        # Check if we managed to collect any labels
        if len(labels) == 0:
            print("No valid labels found in the dataset")
            return None, None
        
        # Count occurrences of each label
        unique_labels, counts = np.unique(labels, return_counts=True)
        
        plt.figure(figsize=figsize)
        plt.bar(unique_labels, counts)
        plt.xlabel('Class')
        plt.ylabel('Count')
        plt.title('Class Distribution')
        plt.xticks(unique_labels)
        plt.grid(axis='y', alpha=0.3)
        
        # Generate name if not provided
        if name is None:
            name = "class_distribution"
        
        # Save and/or display figure
        path = None
        if save:
            path = self._save_figure(name, display=display)
        elif display:
            plt.show()
        
        # Return counts as dictionary
        # Safely handle non-integer label types
        class_counts = {}
        for i, (label, count) in enumerate(zip(unique_labels, counts)):
            try:
                # Try to convert to int if possible
                key = int(label)
            except (ValueError, TypeError):
                # If not convertible to int, use the original label or its string representation
                key = str(label)
            class_counts[key] = count
            
        return class_counts, path
    
    def plot_pixel_stats(self, stat='mean', n_samples=100, figsize=(10, 6), save=True, display=True, name=None):
        """
        Plot statistics of pixel values across images.
        
        Args:
            stat: Statistic to plot ('mean', 'min', 'max', 'std')
            n_samples: Number of samples to analyze
            figsize: Figure size (width, height) in inches
            save: Whether to save the figure (default: True)
            display: Whether to display the figure (default: True)
            name: Optional filename for saving (default: auto-generated)
            
        Returns:
            tuple: (values, figure_path)
        """
        # Sample images
        indices = random.sample(range(len(self.dataset)), min(n_samples, len(self.dataset)))
        values = []
        
        for idx in indices:
            img, _, _ = self.get_sample(idx)
            img_np = img.cpu().numpy()
            
            # Calculate statistic
            if stat == 'mean':
                value = np.mean(img_np)
            elif stat == 'min':
                value = np.min(img_np)
            elif stat == 'max':
                value = np.max(img_np)
            elif stat == 'std':
                value = np.std(img_np)
            else:
                raise ValueError(f"Unsupported statistic: {stat}")
            
            values.append(value)
        
        plt.figure(figsize=figsize)
        plt.hist(values, bins=30)
        plt.xlabel(f'Pixel {stat}')
        plt.ylabel('Count')
        plt.title(f'Distribution of pixel {stat} across {n_samples} images')
        plt.grid(alpha=0.3)
        
        # Generate name if not provided
        if name is None:
            name = f"pixel_{stat}_distribution"
        
        # Save and/or display figure
        path = None
        if save:
            path = self._save_figure(name, display=display)
        elif display:
            plt.show()
        
        return values, path
    
    def save(self, path, format="numpy"):
        """
        Save the dataset to disk.
        
        Args:
            path: Path to save the data
            format: Format to save in ('numpy', 'torch', etc.)
            
        Returns:
            dict: Paths to saved files
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        # Get all images and labels
        if self.images is None:
            # Need to extract all images from dataset
            all_images = []
            all_labels = []
            
            for i in range(len(self.dataset)):
                img, label, _ = self.get_sample(i)
                all_images.append(img)
                if label is not None:
                    all_labels.append(label)
            
            images = torch.stack(all_images)
            labels = torch.stack(all_labels) if all_labels else None
        else:
            images = self.images
            labels = self.labels
        
        saved_paths = {}
        
        if format.lower() == "numpy":
            # Save as numpy arrays
            images_path = path.with_suffix('.npy')
            np.save(images_path, images.cpu().numpy())
            saved_paths['images'] = str(images_path)
            
            if labels is not None:
                labels_path = path.with_name(f"{path.stem}_labels.npy")
                np.save(labels_path, labels.cpu().numpy())
                saved_paths['labels'] = str(labels_path)
                
        elif format.lower() == "torch":
            # Save as torch tensors
            images_path = path.with_suffix('.pt')
            torch.save(images, images_path)
            saved_paths['images'] = str(images_path)
            
            if labels is not None:
                labels_path = path.with_name(f"{path.stem}_labels.pt")
                torch.save(labels, labels_path)
                saved_paths['labels'] = str(labels_path)
                
        else:
            raise ValueError(f"Unsupported save format: {format}")
        
        print(f"Dataset saved to: {', '.join(saved_paths.values())}")
        return saved_paths
    
    def __len__(self):
        """Return the length of the dataset."""
        if self.dataset is not None:
            return len(self.dataset)
        elif self.images is not None:
            return len(self.images)
        else:
            return 0
    
    @staticmethod
    def load(path, labels_path=None, output_dir="figures"):
        """
        Load a dataset from disk.
        
        Args:
            path: Path to the dataset file
            labels_path: Optional path to labels file
            output_dir: Directory to save figures (default: "figures")
            
        Returns:
            DataVis: New DataVis instance with loaded data
        """
        path = Path(path)
        
        if path.suffix == '.npy':
            # Load numpy arrays
            images = np.load(path)
            labels = None
            
            if labels_path:
                labels_path = Path(labels_path)
                if labels_path.exists():
                    labels = np.load(labels_path)
            else:
                # Try to find labels with standard naming
                potential_labels = path.with_name(f"{path.stem}_labels.npy")
                if potential_labels.exists():
                    labels = np.load(potential_labels)
            
            return DataVis(images, labels, output_dir=output_dir)
        
        elif path.suffix == '.pt':
            # Load torch tensors
            images = torch.load(path)
            labels = None
            
            if labels_path:
                labels_path = Path(labels_path)
                if labels_path.exists():
                    labels = torch.load(labels_path)
            else:
                # Try to find labels with standard naming
                potential_labels = path.with_name(f"{path.stem}_labels.pt")
                if potential_labels.exists():
                    labels = torch.load(potential_labels)
            
            return DataVis(images, labels, output_dir=output_dir)
        
        else:
            raise ValueError(f"Unsupported file format: {path.suffix}")
    