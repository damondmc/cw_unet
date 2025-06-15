#!/home/damoncht/.conda/envs/ml/bin/python
from utils import *
from genData import *
from model.unet_leaky_norm_tanho_classification import UNet
from torch.optim.lr_scheduler import ReduceLROnPlateau
import time 
import argparse
import torch
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm

parser = argparse.ArgumentParser(description="Generate mock CW signal dataset.")
parser.add_argument('--f0', type=int, default=500, help='Base frequency (default: 500)')
parser.add_argument('--det', type=str, default='H1L1', help='Detector name (default: H1L1)')
parser.add_argument('--Tsft', type=int, default=7200, help='SFT duration in seconds (default: 7200)')
parser.add_argument('--obsTime', type=int, default=921600, help='Observation time in seconds (default: 921600)')
parser.add_argument('--freq_size', type=int, default=256, help='Frequency band size (default: 256)')
parser.add_argument('--num_cpus', type=int, default=16, help='Number of CPUs to use (default: 20)')
args = parser.parse_args()

# Combined loss function with FAR and pdet terms, adjusted for sigmoid output
def combined_loss(class_output, class_target, far_threshold, far_target=0.01):
    """
    Compute a loss that maximizes pdet while constraining FAR to 1%.
    
    Args:
        class_output: Model output probabilities [batch_size], already sigmoid-activated
        class_target: Ground truth labels (0 for noise, 1 for signal) [batch_size]
        far_threshold: Threshold for FAR computation
        far_target: Target FAR (default: 0.01 for 1%)
    
    Returns:
        tuple: (total_loss, bce_loss, far_penalty, pdet_loss)
    """
    class_output = class_output.squeeze()
    class_target = class_target.squeeze()
    
    bce_loss = F.binary_cross_entropy(class_output, class_target)
    probs = class_output  # No sigmoid needed, as output is already probabilities
    noise_mask = (class_target == 0).float()
    signal_mask = (class_target == 1).float()
    
    noise_above_threshold = torch.sigmoid((probs - far_threshold) * 50) * noise_mask
    far = noise_above_threshold.sum() / (noise_mask.sum() + 1e-8)
    signal_above_threshold = torch.sigmoid((probs - far_threshold) * 50) * signal_mask
    pdet = signal_above_threshold.sum() / (signal_mask.sum() + 1e-8)
    
    far_penalty = torch.abs(far - far_target)
    pdet_loss = -pdet
    total_loss = bce_loss + 1.0 * far_penalty + 1.0 * pdet_loss
    
    return total_loss, bce_loss, far_penalty, pdet_loss

def compute_threshold_from_pfa(noise_outputs, far=0.01):
    """
    Compute the threshold for a given false alarm rate (FAR) using noise class outputs.
    
    Args:
        noise_outputs (np.ndarray): Array of class_output values (probabilities in [0,1]) for noise samples.
        far (float): Desired false alarm rate (default: 0.01 for 1% FAR).
    
    Returns:
        float: Threshold such that `far` proportion of noise samples exceed it.
    """
    if len(noise_outputs) == 0:
        print("Warning: No noise outputs provided, returning default threshold 0.5")
        return 0.5
    
    noise_outputs = np.asarray(noise_outputs).flatten()
    if len(noise_outputs) < 100:
        print(f"Warning: Too few noise samples ({len(noise_outputs)}) for reliable threshold")
        return 0.5
    
    sorted_outputs = np.sort(noise_outputs)[::-1]
    index = int(len(sorted_outputs) * far)
    
    if index == 0:
        print("Index = 0")
        return sorted_outputs[0] if len(sorted_outputs) > 0 else 0.5
    elif index >= len(sorted_outputs):
        print("Index >= len(sorted_outputs)")
        return sorted_outputs[-1] if len(sorted_outputs) > 0 else 0.5
    
    threshold = sorted_outputs[index]
    return threshold

t0 = time.time()
print("Start")

# Set random seed for reproducibility
np.random.seed(100000)
torch.manual_seed(100000)

# Use arguments from argparse
f0 = args.f0
det = args.det
Tsft = args.Tsft
obsTime = args.obsTime
freq_size = args.freq_size
num_cpus = args.num_cpus

homedir = '/scratch/kriles_root/kriles0/damoncht/unet_f/'
tmpdir = homedir + 'tmp/'
size = (freq_size, obsTime // Tsft)
n_data = 1
n_step = 3
threshold = 50

max_train_levels = [20]
max_val_levels = [0, 9, 12, 15, 18, 20, 22, 35]

label = 'UNETwCLASS_{}Hz_D{}-{}_T{}_Tsft{}_ndata{}_step{}_th{}_onlyEntropyLoss'.format(f0, int(max_train_levels[0]), int(max_train_levels[-1]), int(obsTime//86400), Tsft, n_data, n_step, threshold)
version = '{}_{}_{}x{}_MSELoss_dropout0'.format(det, label, size[0], size[1])

print(f"Nominal frequency: {f0}")
print(f"Detector: {det}")
print(f"SFT duration (Tsft): {Tsft} seconds")
print(f"Observation time (obsTime): {obsTime} seconds")
print(f"Home directory: {homedir}")
print(f"Frequency band size: {freq_size}")
print(f"Spectrogram size: {size}")
print(f"Data generation label: {label}")
print(f"Save file label: {version}")

# Initialize dictionaries to store `pdet` by noise level
train_pdet = {noise_level: [] for noise_level in max_train_levels}
val_pdet = {noise_level: [] for noise_level in max_val_levels}

batch_size = 8

filename = '/scratch/kriles_root/kriles0/damoncht/unet_f/data/pure_noise/H1L1_purenoise_n1000_seed300.npz'
data = np.load(filename, allow_pickle=True)['dataset']
noise = normalize(data)
noise_dataset = load_noise_dataset(noise)

filename = '/scratch/kriles_root/kriles0/damoncht/unet_dyn_fastnoise/data/validation/{0}Hz_H1L1_D0-35_256x128_7200s_4c_traindata_n200_seed0.npz'.format(f0)
data = np.load(filename, allow_pickle=True)
signal_dataset = load_signal_dataset(data, max_val_levels)
del data
val_loader = make_data_loader([signal_dataset, noise_dataset], batch_size=batch_size)

# Initialize the model
size_filter_in = 16
dropout_prob = 0.0
model = UNet(input_channels=4, output_channels=4, size_filter_in=size_filter_in, dropout_prob=dropout_prob).to(device)
criterion = torch.nn.MSELoss(reduction='none')

# Initialize the optimizer
lr = 1e-4
optimizer = torch.optim.Adam(model.parameters(), lr=lr)
scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=100)

# Initialize variables to store the top models and their losses
best_val_loss = float('inf')
best_val_model = None
best_pdet = 0.0
best_pdet_model = None

train_losses = []
val_losses = []
train_bce_losses = []
train_far_penalties = []
train_pdet_losses = []
val_bce_losses = []
val_far_penalties = []
val_pdet_losses = []
far_threshold = 0.5  # Initial FAR threshold
val_far_threshold = 0.5

num_epochs = 1000

print(model)

for epoch in tqdm(range(num_epochs)):   
    if epoch % n_step == 0: 
        seed = (epoch // n_step) % 300
        if threshold != 50:
            filename = '/scratch/kriles_root/kriles0/damoncht/unet_dyn_fastnoise/data/{0}Hz/{0}Hz_H1L1_D0-35_256x128_7200s_4c_traindata_n1000_seed{1}_th{2}_norm.npz'.format(f0, seed, threshold)
        else:
            filename = '/scratch/kriles_root/kriles0/damoncht/unet_dyn_fastnoise/data/{0}Hz/{0}Hz_H1L1_D0-35_256x128_7200s_4c_traindata_n1000_seed{1}_norm.npz'.format(f0, seed)
        print("Using {}".format(filename))
        data = np.load(filename, allow_pickle=True)
        signal_dataset = load_signal_dataset(data, max_train_levels)        
        del data 

        print('Loading pure noise ...')
        filename = '/scratch/kriles_root/kriles0/damoncht/unet_f/data/pure_noise/H1L1_purenoise_n1000_seed{0}.npz'.format(seed)
        data = np.load(filename, allow_pickle=True)['dataset']
        noise = normalize(data)
        noise_dataset = load_noise_dataset(noise)
        train_loader = make_data_loader([signal_dataset, noise_dataset], batch_size=batch_size)        
        del noise, noise_dataset

    print("Training:")
    # Initialize for within-epoch threshold updates
    model.eval()
    noise_class_outputs = []
    with torch.no_grad():
        for inputs, _, _, labels in train_loader:
            inputs = inputs.to(device)
            _, class_output = model(inputs)
            noise_mask = (labels == float('inf')).cpu().numpy()
            if noise_mask.any():
                noise_class_outputs.extend(class_output.squeeze().detach().cpu().numpy()[noise_mask].tolist())
    
    noise_class_outputs = np.array(noise_class_outputs)
    far_threshold = compute_threshold_from_pfa(noise_class_outputs)
    print(f"Epoch {epoch + 1}, Initial Training FAR Threshold: {far_threshold:.4f}")
    
    # Training phase with within-epoch threshold updates
    model.train()
    running_train_loss = 0.0
    running_train_bce = 0.0
    running_train_far = 0.0
    running_train_pdet = 0.0
    train_det = []
    train_label = []
    batch_idx = 0
    noise_class_outputs = []  # Accumulate noise outputs for threshold updates
    N = len(train_loader) // 10  # Update threshold every 10% of the epoch
    #prev_threshold = far_threshold  # For smoothing
    far_threshold_tensor = torch.tensor(far_threshold, device=device, dtype=torch.float32)
        
    for inputs, targets, mask, labels in train_loader:
        inputs, targets, mask, labels = inputs.to(device), targets.to(device), mask.to(device), labels.to(device)
        class_target = (labels != float('inf')).float()
        optimizer.zero_grad()
        
        # Forward pass
        _, class_output = model(inputs)
        
        # Collect noise outputs for threshold update
        #noise_mask = (labels == float('inf')).cpu().numpy()
        #if noise_mask.any():
        #    noise_class_outputs.extend(class_output.squeeze().detach().cpu().numpy()[noise_mask].tolist())
        
        # # Update threshold every N batches
        # if (batch_idx + 1) % N == 0 and len(noise_class_outputs) >= 100:
        #     noise_class_outputs_np = np.array(noise_class_outputs)
        #     new_threshold = compute_threshold_from_pfa(noise_class_outputs_np)
        #     # Smooth threshold updates
        #     far_threshold = 0.9 * prev_threshold + 0.1 * new_threshold
        #     prev_threshold = far_threshold
        #     noise_class_outputs = []  # Reset
        #     print(f"Batch {batch_idx + 1}, Updated FAR Threshold: {far_threshold:.4f}")
        
        # far_threshold_tensor = torch.tensor(far_threshold, device=device, dtype=torch.float32)
        total_loss, bce_loss, far_penalty, pdet_loss = combined_loss(class_output, class_target, far_threshold_tensor, far_target=0.01)
        
#         # Monitor batch FAR and pdet
#         probs = class_output.squeeze()
#         noise_mask_t = (class_target == 0).float()
#         far = (probs > far_threshold_tensor).float() * noise_mask_t
#         far = far.sum() / (noise_mask_t.sum() + 1e-8)
#         pdet = (probs > far_threshold_tensor).float() * (class_target == 1).float()
#         pdet = pdet.sum() / ((class_target == 1).sum() + 1e-8)
#         print(f"Batch {batch_idx + 1}, FAR: {far.item():.4f}, Pdet: {pdet.item():.4f}, BCE: {bce_loss.item():.4f}, FAR Penalty: {far_penalty.item():.4f}, Pdet Loss: {pdet_loss.item():.4f}")
        
        total_loss.backward()
        optimizer.step()
        running_train_loss += total_loss.item()
        running_train_bce += bce_loss.item()
        running_train_far += far_penalty.item()
        running_train_pdet += pdet_loss.item()
        
        detection_stats = class_output.squeeze().detach()
        train_det.append(detection_stats.cpu().numpy())
        train_label.append(labels.cpu().numpy())
        batch_idx += 1

    train_loss = running_train_loss / len(train_loader)
    train_bce = running_train_bce / len(train_loader)
    train_far = running_train_far / len(train_loader)
    train_pdet_loss = running_train_pdet / len(train_loader)
    train_losses.append(train_loss)
    train_bce_losses.append(train_bce)
    train_far_penalties.append(train_far)
    train_pdet_losses.append(train_pdet_loss)

    train_det = np.concatenate(train_det, axis=0)
    train_label = np.concatenate(train_label, axis=0)
    noise_det = train_det[train_label == float('inf')]
    pdet = (noise_det < far_threshold).sum() / noise_det.size if noise_det.size > 0 else np.nan
    print("D=inf, pdet={}%".format(pdet*100))
    
    for noise_level in train_pdet.keys():
        signal_det = train_det[train_label == float(noise_level)]
        if signal_det.size != 0:
            pdet = (signal_det > far_threshold).sum() / signal_det.size 
            train_pdet[noise_level].append(pdet)
            print("D={}, pdet={}%".format(noise_level, pdet*100))
        else:
            pdet = np.nan
            train_pdet[noise_level].append(pdet)
            
    print("Validation:")
    model.eval()
    val_noise_class_outputs = []
    with torch.no_grad():
        for inputs, _, _, labels in val_loader:
            inputs = inputs.to(device)
            _, class_output = model(inputs)
            noise_mask = (labels == float('inf')).cpu().numpy()
            if noise_mask.any():
                val_noise_class_outputs.extend(class_output.squeeze().detach().cpu().numpy()[noise_mask].tolist())

    val_noise_class_outputs = np.array(val_noise_class_outputs)
    val_far_threshold = compute_threshold_from_pfa(val_noise_class_outputs)
    far_threshold_tensor = torch.tensor(far_threshold, device=device, dtype=torch.float32)
    print(f"Epoch {epoch + 1}, Validation FAR Threshold: {val_far_threshold:.4f}")
        
    val_det = []
    val_label = []
    running_val_loss = 0.0
    running_val_bce = 0.0
    running_val_far = 0.0
    running_val_pdet = 0.0
    with torch.no_grad():
        for inputs, targets, mask, labels in val_loader: 
            inputs, targets, mask, labels = inputs.to(device), targets.to(device), mask.to(device), labels.to(device)
            class_target = (labels != float('inf')).float()
        
            _, class_output = model(inputs)
            #far_threshold_tensor = torch.tensor(far_threshold, device=device, dtype=torch.float32)
            total_loss, bce_loss, far_penalty, pdet_loss = combined_loss(class_output, class_target, far_threshold_tensor, far_target=0.01)
            running_val_loss += total_loss.item()
            running_val_bce += bce_loss.item()
            running_val_far += far_penalty.item()
            running_val_pdet += pdet_loss.item()
            detection_stats = class_output.squeeze().detach()
            val_det.append(detection_stats.cpu().numpy())
            val_label.append(labels.cpu().numpy())
                        
    val_loss = running_val_loss / len(val_loader)
    val_bce = running_val_bce / len(val_loader)
    val_far = running_val_far / len(val_loader)
    val_pdet_loss = running_val_pdet / len(val_loader)
    val_losses.append(val_loss)
    val_bce_losses.append(val_bce)
    val_far_penalties.append(val_far)
    val_pdet_losses.append(val_pdet_loss)

    val_det = np.concatenate(val_det, axis=0)
    val_label = np.concatenate(val_label, axis=0)
    noise_det = val_det[val_label == float('inf')]
    pdet = (noise_det < far_threshold).sum() / noise_det.size if noise_det.size > 0 else np.nan
    print("D=inf, pdet={}%".format(pdet*100))
    
    print('Validation:')
    for noise_level in val_pdet.keys():
        signal_det = val_det[val_label == float(noise_level)]
        if signal_det.size != 0:
            pdet = (signal_det > far_threshold).sum() / signal_det.size 
            val_pdet[noise_level].append(pdet)
            print("D={}, pdet={}%".format(noise_level, pdet*100))
        else:
            pdet = np.nan
            val_pdet[noise_level].append(pdet) 
            
    scheduler.step(val_pdet[max_val_levels[-1]][-1] if val_pdet[max_val_levels[-1]][-1] is not np.nan else 0.0)

    if epoch % 3 == 0:
        print(f"Epoch {epoch + 1}/{num_epochs}, Training Loss: {train_loss:.3e}, Validation Loss: {val_loss:.3e}")

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        best_val_model = model.state_dict()
        print(f'Replace best val model at epoch {epoch + 1}')
    if pdet > best_pdet:
        best_pdet = pdet
        best_pdet_model = model.state_dict()
        print(f'Replace best pdet model at epoch {epoch + 1}')

    current_lr = optimizer.param_groups[0]['lr']
    print(f"Learning rate: {current_lr}")
    print(f"Time used = {time.time() - t0:.2f} seconds")
    if current_lr < 1e-4 / 2**3:
        break

train_losses = np.array(train_losses)
val_losses = np.array(val_losses)
train_bce_losses = np.array(train_bce_losses)
train_far_penalties = np.array(train_far_penalties)
train_pdet_losses = np.array(train_pdet_losses)
val_bce_losses = np.array(val_bce_losses)
val_far_penalties = np.array(val_far_penalties)
val_pdet_losses = np.array(val_pdet_losses)

torch.save(best_val_model, homedir+"trained_model/{0}Hz/best_val_model_{1}.pth".format(f0, version))
torch.save(best_pdet_model, homedir+"trained_model/{0}Hz/best_pdet_model_{1}.pth".format(f0, version))

losses = {
    "train_losses": train_losses,
    "val_losses": val_losses,
    "train_bce_losses": train_bce_losses,
    "train_far_penalties": train_far_penalties,
    "train_pdet_losses": train_pdet_losses,
    "val_bce_losses": val_bce_losses,
    "val_far_penalties": val_far_penalties,
    "val_pdet_losses": val_pdet_losses,
    "top_val_losses": best_val_loss,
    "train_pdet": train_pdet,
    "val_pdet": val_pdet,
    "top_val_pdet": best_pdet,
}
np.save(homedir+'trained_model/{0}Hz/losses_{1}.npy'.format(f0, version), losses)

print("Done.")
