#!/home/damoncht/.conda/envs/ml/bin/python
import numpy as np
from tqdm import tqdm
import torch
from tools import *
from genData import *
from model.unet import Attention_UNet
import argparse

parser = argparse.ArgumentParser(description="Generate skymap pdet.")
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
np.random.seed(1)
torch.manual_seed(1)

# Parameters
det = 'H1L1'
args = parser.parse_args()
f0 = args.f0
size = (512, 64)
Tsft = 14400

if f0 == 20:
    train_level = 32
    rho = 5.9
if f0 == 500:
    train_level = 22
    rho = 9.5
if f0 == 1000:
    train_level = 19
    rho = 12
if f0 == 0:
    train_level = 22
    rho = 10


version = f'H1L1_a1b1_{f0}Hz_D{train_level}-{train_level}_T10_f512xTsft14400_ndata7000_noise7000_latent64_batch8_lr0.0001_512x64_MSELoss_dropout0'

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Initialize model
model = Attention_UNet(in_channels=4, out_channels=4, latent_channels=64, dropout_prob=0).to(device)
best_val_model = torch.load(f"/scratch/kriles_root/kriles0/damoncht/unet_f/trained_model/{f0}Hz/best_pdet_model_mean_sq_{version}.pth", weights_only=False)
    
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

R2_H1 = np.loadtxt('./result/H1_geofactor.txt')
R2_L1 = np.loadtxt('./result/L1_geofactor.txt')

R2 = (R2_H1 + R2_L1)/2


# Function to load dataset
def load_dataset(images, batch_size=8, shuffle=True):
    images_tensor = torch.tensor(images, dtype=torch.float32).permute(0, 3, 1, 2)
    dataset = TensorDataset(images_tensor)
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    return data_loader


def getDepthFromSNR(r2, Tobs=921600, snr=8):
    return (4./25. * Tobs * r2 / snr**2)**0.5

model.eval()

stat = np.zeros(400*1000)
for i in tqdm(range(400), total=400):
    # Load data
    filename = f'/scratch/kriles_root/kriles0/damoncht/unet_f/data/validation/skymap_{f0}Hz_H1L1_{size[0]}x{size[1]}_{Tsft}s_4c_traindata_n1000_idx{i}.npz'
    targets = np.load(filename, allow_pickle=True)['clean_image']  # Shape: (1000, 512, 64, 4)
    
    pdet_values = []
                    
    noisy_signals = np.empty((1000,) + size + (4,))
    for j in range(1000):
        Sn = getDepthFromSNR(R2[i*1000 + j], Tobs=921600, snr=rho) # snr =6 for 20Hz, 15 for 1000Hz
                    
        noise = simNoise(sqrtSn=Sn, Tsft=Tsft, size=size, ndet=2, norm=False)
        noisy_signals[j] = normalize(noise + targets[j])

    loader = load_dataset(noisy_signals, batch_size=8, shuffle=False)

    # Evaluate model
    predictions = []
    with torch.no_grad():
        for batch in loader:
            images = batch[0].to(device)
            outputs = model(images)
            predictions.append(outputs.cpu().numpy())
    predictions = np.concatenate(predictions)

    # Compute detection statistic and pdet
    detection_stats = compute_detection_statistic(predictions)
    stat[i*1000:(i+1)*1000] = detection_stats
                    
detected = ( stat > x_pfa)
        
np.savez(f'./result/data/snr{rho}_skymap_{f0}Hz', stat=stat, x_pfa=x_pfa, detected = detected)

        
