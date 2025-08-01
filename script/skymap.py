#!/home/damoncht/.conda/envs/ml/bin/python
import numpy as np
from scipy.interpolate import interp1d
from tqdm import tqdm
import torch
from utils import *
from genData import *
from model.unet import Attention_UNet
import argparse
from multiprocessing import Pool

parser = argparse.ArgumentParser(description="Generate pdet skymap.")
#parser.add_argument('--depth', type=float, default=20, help='Depth for the training data (default: 20)')
parser.add_argument('--f0', type=int, default=500, help='Base frequency (default: 500)')
args = parser.parse_args()

def load_signal_datasetv(images, labels):
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
    labels_tensor = torch.tensor(labels, dtype=torch.float64)
    
    full_dataset = TensorDataset(images_tensor, labels_tensor)
    
    return full_dataset

# Set random seeds for reproducibility
np.random.seed(111)
torch.manual_seed(111)

# Parameters
det = 'H1L1'
#f0 = 500
args = parser.parse_args()
f0 = args.f0
size = (512, 64)
Tsft = 14400


if f0 == 20:
    max_train_levels = [16, 22, 27, 29, 32, 37, 40]

if f0 == 500:
    max_train_levels = [8, 12, 15, 18, 20, 22, 24]

if f0 == 1000:
    max_train_levels = [8, 12, 15, 18, 20, 22, 24]

seeds = range(400)  # Seeds 1 to 6
num_noise_realizations = 200

if f0 == 20:
    train_level = 32
if f0 == 500:
    train_level = 22
if f0 == 1000:
    train_level = 19
if f0 == 0:
    train_level = 22
    
version = version = f'H1L1_a1.0b1.0_{f0}Hz_D{train_level}-{train_level}_T10_Tsft14400_ndata7000_noise7000_latent64_batch8_lr0.0001_512x64_MSELoss_dropout0'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
output_file = f"sky2000pts_depth_at_p90_{f0}Hz.npz"

# Initialize model
model = Attention_UNet(in_channels=4, out_channels=4, latent_channels=64, dropout_prob=0).to(device)
best_val_model = torch.load(f"./trained_model/{f0}Hz/best_pdet_model_mean_sq_{version}.pth", weights_only=False)
    
model.load_state_dict(best_val_model)
model.eval()

# Function to compute detection statistic
def compute_detection_statistic(images, ref_distribution=None):
    detection_stats = np.mean(np.abs(images.transpose(0, 2, 3, 1)), axis=(1, 2, 3))
    return detection_stats

# Generate noise-only data for x_pfa
noise = np.empty((1000,) + size + (4,))
for i in range(noise.shape[0]):
    noise[i] = simNoise(sqrtSn=1, Tsft=Tsft, size=size, ndet=2, norm=False)
noise = normalize(noise)
noise_data = load_noise_dataset(noise)
noise_loader = make_data_loader([noise_data], batch_size=8, shuffle=False)

# Compute x_pfa
noise_predictions = []
with torch.no_grad():
    for batch in tqdm(noise_loader, desc="Processing noise for x_pfa"):
        images = batch[0].to(device)
        outputs = model(images)
        noise_predictions.append(outputs.cpu().numpy())
noise_predictions = np.concatenate(noise_predictions, axis=0)
x_pfa = np.percentile(compute_detection_statistic(noise_predictions), (1 - 0.71/100) * 100)

# Process each signal
depth_at_pdet_90 = []
for seed in seeds:
    # Load data
    filename = f'/scratch/kriles_root/kriles0/damoncht/unet_f/data/validation/skymap_{f0}Hz_H1L1_{size[0]}x{size[1]}_{Tsft}s_4c_traindata_n1000_seed{seed}.npz'
    targets = np.load(filename, allow_pickle=True)['clean_image']  # Shape: (1000, 512, 64, 4)
    #targets = normalize(targets)

    for signal_idx in tqdm(range(5), desc=f"Processing signals for seed {seed}"):
        signal = targets[signal_idx*num_noise_realizations:(signal_idx+1)*num_noise_realizations]  # Shape: (1, 512, 64, 4)
        pdet_values = []

        for Sn in max_train_levels:
            # Generate 200 noise realizations
            noisy_signals = np.empty((num_noise_realizations,) + size + (4,))
            for i in range(num_noise_realizations):
                noise = simNoise(sqrtSn=Sn, Tsft=Tsft, size=size, ndet=2, norm=False)
                noisy_signals[i] = normalize(noise + signal[i])
  
            dataset = load_signal_datasetv(noisy_signals, [Sn] * num_noise_realizations)
            loader = make_data_loader([dataset], batch_size=8, shuffle=False)

            # Evaluate model
            predictions = []
            with torch.no_grad():
                for batch in loader:
                    images = batch[0].to(device)
                    outputs = model(images)
                    predictions.append(outputs.cpu().numpy())
            predictions = np.concatenate(predictions, axis=0)

            # Compute detection statistic and pdet
            detection_stats = compute_detection_statistic(predictions)
            pdet = np.sum(detection_stats > x_pfa) / detection_stats.size
            pdet_values.append(pdet)
        # Interpolate to find depth at pdet = 0.9
        pdet_values = np.array(pdet_values)
        depth = np.array(max_train_levels)
        
        interp_func = interp1d(pdet_values, depth, kind='linear', fill_value="extrapolate")
        depth_value = interp_func(0.9)
        depth_at_pdet_90.append(depth_value)

# Save results
depth_at_pdet_90 = np.array(depth_at_pdet_90)
np.savez(output_file, depth_at_pdet_90=depth_at_pdet_90)

print(f"Depths at pdet = 0.9 saved to {output_file}")
print(f"Number of valid depths: {np.sum(~np.isnan(depth_at_pdet_90))}")
print(f"Mean depth at pdet = 0.9: {np.nanmean(depth_at_pdet_90):.2f}")
