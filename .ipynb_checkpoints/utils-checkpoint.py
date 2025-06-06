#!/home/damoncht/.conda/envs/ml/bin/python
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, Subset, TensorDataset, ConcatDataset
from tqdm import tqdm
import psutil
from scipy import stats
from PIL import Image
from scipy.stats import entropy
from multiprocessing import Pool, cpu_count


# Detect if CUDA is available and set the device accordingly
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#device = "cpu"

def compute_detection_statistic(images, ref_distribution=None):
    """
    Compute the KL divergence detection statistic for each image.

    Parameters:
        images (torch.Tensor): Input images, shape (batch_size, C, H, W).
        ref_distribution (np.ndarray): Reference distribution Q for KL divergence, 
                                        shape (H, W).

    Returns:
        np.ndarray: KL divergence statistics for each image.
    """
    images_np = images.cpu().numpy()
    detection_stats = np.mean(np.abs(images_np.transpose(0, 2, 3, 1)), axis=(1, 2, 3))
    return detection_stats
    """
    if ref_distribution is None:
        raise ValueError("A reference distribution must be provided.")

    # Move tensor to numpy
    images_np = images.cpu().numpy()

    # Initialize an array to store KL divergence for each image
    kl_div_stats = np.zeros(images_np.shape[0])
    #print(images_np.shape)
    for i in range(images_np.shape[0]):
        # Average over channels to get a single 2D array for each image
        image = np.mean(images_np[i], axis=0)
        #print(image.shape, ref_distribution.shape)
        # Compute KL divergence
        kl_div_stats[i] = compute_kl_divergence(image, ref_distribution)

    return kl_div_stats
    """

def compute_threshold_from_pfa(noise_det, pfa=0.01):
    """
    Compute the detection threshold based on the desired false alarm rate (Pfa)
    using the noise detection statistics.

    Parameters:
        noise_det (list): List of detection statistics from noise data.
        pfa (float): Desired false alarm rate (default is 1%).

    Returns:
        float: Detection threshold for the specified pfa.
    """
    #print(noise_det)
    # Compute the threshold corresponding to Pfa (percentile)
    threshold = np.percentile(noise_det, 100 * (1 - pfa))
    return threshold


def compute_kl_divergence(image: np.ndarray, Q: np.ndarray) -> float:
    # ref_distribution = np.ones(size)
    """
    Compute the KL divergence between an image and a reference distribution Q.

    Parameters:
        image (np.ndarray): The input image array.
        Q (np.ndarray): The reference distribution array.
        
    Returns:
        float: The computed KL divergence.
    """
    # Normalize the image to create a probability distribution P
    P = image / np.sum(image)
    
    # Normalize Q to ensure it is a valid probability distribution
    Q = Q / np.sum(Q)
    
    # Ensure both distributions are of the same shape
    if P.shape != Q.shape:
        raise ValueError("Input image and reference distribution Q must have the same shape.")
    
    # Compute the KL divergence
    kl_divergence = entropy(P.ravel(), Q.ravel())
    
    return kl_divergence


def load_signal_dataset_val(data, noise_levels_to_sample):
    """
    Load the full dataset once at the beginning, including noisy and pure noise images.
    Store indices for each noise level for easy selection during training.

    Parameters:
        noise_levels_to_sample (list): List of noise levels to include in the dataset.
    
    Returns:
        dataset (TensorDataset): The full dataset containing all samples.
        noise_level_indices (dict): A dictionary storing indices for each noise level.
    """
    # Load noisy image data
    #loaded_data = np.load(file_paths['noisy'], allow_pickle=True)
    noisy_data_dict = data['noisy_image']
    clean_images = data['clean_image']
    signal_masks = data['signal_mask']
    
    # Initialize lists to hold all images, targets, masks, and labels
    images_list = []
    target_images_list = []
    masks_list = []
    noise_labels_list = []
    noise_level_indices = {level: [] for level in noise_levels_to_sample}

    #current_index = 0  # Start from index 0 for all levels
    # Process noisy image data by noise level
    for level in noise_levels_to_sample:
        if str(level) in noisy_data_dict:
            noisy_images = noisy_data_dict[str(level)]
            images_list.append(noisy_images)  # Append noisy images
            target_images_list.append(clean_images)  # Append corresponding clean images
            masks_list.append(signal_masks)  # Append corresponding masks
            noise_labels_list.extend([level] * noisy_images.shape[0])  # Extend labels
            num_samples = len(noisy_images)
            
    # Concatenate data along the first axis
    images = np.concatenate(images_list, axis=0)
    target_images = np.concatenate(target_images_list, axis=0)
    masks = np.concatenate(masks_list, axis=0)
    noise_labels = np.array(noise_labels_list)

    # Convert to torch tensors
    images_tensor = torch.tensor(images, dtype=torch.float32).permute(0, 3, 1, 2)
    target_images_tensor = torch.tensor(target_images, dtype=torch.float32).permute(0, 3, 1, 2)
    masks_tensor = torch.tensor(masks, dtype=torch.bool).permute(0, 3, 1, 2)
    noise_tensor = torch.tensor(noise_labels, dtype=torch.float64)

    # Create the final datasets
    dataset = TensorDataset(images_tensor, target_images_tensor, masks_tensor, noise_tensor)

    return dataset

def update_train_val_loaders(full_dataset, noise_level_indices, noise_levels_to_sample, batch_size, val_size=0.2):
    """
    Update the train and validation data loaders based on the selected noise levels.

    Parameters:
        full_dataset (TensorDataset): The full dataset containing all samples.
        noise_level_indices (dict): A dictionary of indices for each noise level.
        noise_levels_to_sample (list): List of noise levels to sample for this epoch.
        batch_size (int): Batch size for DataLoader.
        val_size (float): Proportion of data to use for validation.

    Returns:
        tuple: Updated train and validation DataLoaders.
    """

    # Create a list of subsets for each noise level
    val_indices = []
    train_indices = []

    for level in noise_levels_to_sample:
        level_indices = noise_level_indices[level]
        num_samples = len(level_indices)
        val_size_count = int(val_size * num_samples)
        
        # The first `val_size_count` samples are the validation set
        val_indices.extend(level_indices[:val_size_count])
        
        # The remaining samples are the training set, shuffled
        train_indices.extend(level_indices[val_size_count:])
    
    # Shuffle the training indices
    np.random.shuffle(train_indices)

    # Create DataLoaders for training and validation
    train_loader = DataLoader(Subset(full_dataset, train_indices), batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(Subset(full_dataset, val_indices), batch_size=batch_size, shuffle=False)

    return train_loader, val_loader



def update_noise_levels(epoch, total_epochs, initial_levels, max_levels, growth_rate=0.2):
    """
    Update the noise levels to sample based on the current epoch.
    Gradually add higher noise levels as the training progresses.

    Parameters:
        epoch (int): Current epoch.
        total_epochs (int): Total number of epochs.
        initial_levels (list): Initial list of noise levels to sample.
        max_levels (list): Maximum list of noise levels to sample.
        growth_rate (float): The rate at which noise levels are added (default is 0.2).

    Returns:
        list: Updated list of noise levels to sample.
    """
    # Compute the number of levels to include based on current progress
    progress = epoch / total_epochs
    num_levels = int(len(initial_levels) + (len(max_levels) - len(initial_levels)) * progress)
    
    # Ensure that we are not exceeding the maximum noise levels
    updated_levels = max_levels[:num_levels]
    
    return updated_levels


def update_noise_levels(pdet):
    """
    Update the noise levels to sample based on the current epoch.
    Gradually add higher noise levels as the training progresses.

    Parameters:
        epoch (int): Current epoch.
        total_epochs (int): Total number of epochs.
        initial_levels (list): Initial list of noise levels to sample.
        max_levels (list): Maximum list of noise levels to sample.
        growth_rate (float): The rate at which noise levels are added (default is 0.2).

    Returns:
        list: Updated list of noise levels to sample.
    """
    
    if pdet > 0.95:
        return True
    else: 
        return False

    
def update_train_val_loaders(full_dataset, noise_level_indices, noise_levels_to_sample, batch_size, val_size=0.2):
    """
    Update the train and validation data loaders based on the selected noise levels.

    Parameters:
        full_dataset (TensorDataset): The full dataset containing all samples.
        noise_level_indices (dict): A dictionary of indices for each noise level.
        noise_levels_to_sample (list): List of noise levels to sample for this epoch.
        batch_size (int): Batch size for DataLoader.
        val_size (float): Proportion of data to use for validation.

    Returns:
        tuple: Updated train and validation DataLoaders.
    """
    selected_indices = []
    for level in noise_levels_to_sample:
        selected_indices.extend(noise_level_indices[level])

    # Create a list of subsets for each noise level
    val_indices = []
    train_indices = []

    for level in noise_levels_to_sample:
        level_indices = noise_level_indices[level]
        num_samples = len(level_indices)
        val_size_count = int(val_size * num_samples)
        
        # The first `val_size_count` samples are the validation set
        val_indices.extend(level_indices[:val_size_count])
        
        # The remaining samples are the training set, shuffled
        train_indices.extend(level_indices[val_size_count:])
    
    # Shuffle the training indices
    np.random.shuffle(train_indices)

    # Create DataLoaders for training and validation
    train_loader = DataLoader(Subset(full_dataset, train_indices), batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(Subset(full_dataset, val_indices), batch_size=batch_size, shuffle=False)

    return train_loader, val_loader

def load_noise_dataset(images):
    """
    Load the full dataset once at the beginning, including noisy and pure noise images.
    Store indices for each noise level for easy selection during training.

    Parameters:
        size (tuple): The size of the images (height, width).
        noise_levels_to_sample (list): List of noise levels to include in the dataset.
        file_paths (dict): Dictionary with paths for noisy and pure noise datasets.
        pure_noise (bool): Whether to include pure noise data in the dataset.

    Returns:
        dataset (TensorDataset): The full dataset containing all samples.
        noise_level_indices (dict): A dictionary storing indices for each noise level.
    """

    # Process pure noise data
    targets = np.zeros_like(images)  # No target for pure noise
    masks = np.zeros_like(images, dtype=bool) # No mask
    labels = [np.inf] * images.shape[0]  # Extend labels

    # Convert to torch tensors
    images_tensor = torch.tensor(images, dtype=torch.float32).permute(0, 3, 1, 2)
    targets_tensor = torch.tensor(targets, dtype=torch.float32).permute(0, 3, 1, 2)
    masks_tensor = torch.tensor(masks, dtype=torch.bool).permute(0, 3, 1, 2)
    labels_tensor = torch.tensor(labels, dtype=torch.float64)

    #print(images_tensor.shape)
    #print(targets_tensor.shape)
    #print(masks_tensor.shape)
    #print(labels_tensor.shape)
    # Create the full dataset
    full_dataset = TensorDataset(images_tensor, targets_tensor, masks_tensor, labels_tensor)
    
    return full_dataset


def load_signal_datasetv2(images, targets, masks, labels):
    """
    Load the full dataset once at the beginning, including noisy and pure noise images.
    Store indices for each noise level for easy selection during training.

    Parameters:
        size (tuple): The size of the images (height, width).
        noise_levels_to_sample (list): List of noise levels to include in the dataset.
        file_paths (dict): Dictionary with paths for noisy and pure noise datasets.
        pure_noise (bool): Whether to include pure noise data in the dataset.

    Returns:
        dataset (TensorDataset): The full dataset containing all samples.
        noise_level_indices (dict): A dictionary storing indices for each noise level.
    """

    # Convert to torch tensors
    images_tensor = torch.tensor(images, dtype=torch.float32).permute(0, 3, 1, 2)
    targets_tensor = torch.tensor(targets, dtype=torch.float32).permute(0, 3, 1, 2)
    masks_tensor = torch.tensor(masks, dtype=torch.bool).permute(0, 3, 1, 2)
    labels_tensor = torch.tensor(labels, dtype=torch.float64)
    
    full_dataset = TensorDataset(images_tensor, targets_tensor, masks_tensor, labels_tensor)
    
    return full_dataset


def load_signal_dataset(data, noise_levels_to_sample):
    """
    Load the full dataset once at the beginning, including noisy and pure noise images.
    Store indices for each noise level for easy selection during training.

    Parameters:
        noise_levels_to_sample (list): List of noise levels to include in the dataset.
    
    Returns:
        dataset (TensorDataset): The full dataset containing all samples.
        noise_level_indices (dict): A dictionary storing indices for each noise level.
    """
    # Load noisy image data
    #loaded_data = np.load(file_paths['noisy'], allow_pickle=True)
    noisy_data_dict = data['noisy_image']
    clean_images = data['clean_image']
    signal_masks = data['signal_mask']
    
    # Initialize lists to hold all images, targets, masks, and labels
    images_list = []
    target_images_list = []
    masks_list = []
    noise_labels_list = []
    noise_level_indices = {level: [] for level in noise_levels_to_sample}

    #current_index = 0  # Start from index 0 for all levels
    # Process noisy image data by noise level
    for level in noise_levels_to_sample:
        if str(level) in noisy_data_dict.item():
            noisy_images = noisy_data_dict.item()[str(level)]
            images_list.append(noisy_images)  # Append noisy images
            target_images_list.append(clean_images)  # Append corresponding clean images
            masks_list.append(signal_masks)  # Append corresponding masks
            noise_labels_list.extend([level] * noisy_images.shape[0])  # Extend labels
            #num_samples = len(noisy_images)
            
    # Concatenate data along the first axis
    images = np.concatenate(images_list, axis=0)
    target_images = np.concatenate(target_images_list, axis=0)
    masks = np.concatenate(masks_list, axis=0)
    noise_labels = np.array(noise_labels_list)

    # Convert to torch tensors
    images_tensor = torch.tensor(images, dtype=torch.float32).permute(0, 3, 1, 2)
    target_images_tensor = torch.tensor(target_images, dtype=torch.float32).permute(0, 3, 1, 2)
    masks_tensor = torch.tensor(masks, dtype=torch.bool).permute(0, 3, 1, 2)
    noise_tensor = torch.tensor(noise_labels, dtype=torch.float64)

    # Create the final datasets
    dataset = TensorDataset(images_tensor, target_images_tensor, masks_tensor, noise_tensor)

    return dataset


def make_data_loader(dataset, batch_size=8, shuffle=True):
    """
    Update the train and validation data loaders based on the selected noise levels.

    Parameters:
        full_dataset (TensorDataset): The full dataset containing all samples.
        noise_level_indices (dict): A dictionary of indices for each noise level.
        noise_levels_to_sample (list): List of noise levels to sample for this epoch.
        batch_size (int): Batch size for DataLoader.
        val_size (float): Proportion of data to use for validation.

    Returns:
        tuple: Updated train and validation DataLoaders.
    """
    
    # Create DataLoaders 
    data = ConcatDataset(dataset)
    data_loader = DataLoader(data, batch_size=batch_size, shuffle=shuffle)

    return data_loader
