#!/home/damoncht/.conda/envs/ml/bin/python
from utils import *
from genData import *
from model.unet_leaky_norm_tanho_classification import UNet
from torch.optim.lr_scheduler import ReduceLROnPlateau
import time 
import argparse

parser = argparse.ArgumentParser(description="Generate mock CW signal dataset.")
parser.add_argument('--f0', type=int, default=500, help='Base frequency (default: 500)')
parser.add_argument('--det', type=str, default='H1L1', help='Detector name (default: H1L1)')
parser.add_argument('--Tsft', type=int, default=7200, help='SFT duration in seconds (default: 7200)')
parser.add_argument('--obsTime', type=int, default=921600, help='Observation time in seconds (default: 921600)')
parser.add_argument('--freq_size', type=int, default=256, help='Frequency band size (default: 256)')
parser.add_argument('--num_cpus', type=int, default=16, help='Number of CPUs to use (default: 20)')
args = parser.parse_args()



# Combined loss function with threshold-based labels
def combined_loss(denoised, target, class_output, class_target, mask, threshold, alpha=1, beta=1, gamma=1):
    # MSE for signal
    mse_signal = F.mse_loss(denoised * mask, target * mask, reduction='none')
    mse_signal = mse_signal.mean()
    
    # MSE for noise
    background_zeros = torch.zeros_like(denoised).to(denoised.device)
    mse_noise = F.mse_loss(denoised * (~mask), background_zeros * (~mask), reduction='none')
    mse_noise = mse_noise.mean()
    
    # Binary cross-entropy using threshold-based labels
    thresholded_labels = (class_output.squeeze() > threshold).float()
    class_loss = F.binary_cross_entropy(class_output.squeeze(), thresholded_labels)
    
    # Combined loss
    total_loss = alpha * mse_signal + beta * mse_noise + gamma * class_loss
    return total_loss


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
    
    # Sort noise outputs in descending order
    sorted_outputs = np.sort(noise_outputs)[::-1]
    
    # Compute index for 1% FAR (top 1% of noise outputs)
    index = int(len(sorted_outputs) * far)
    
    # Handle edge cases
    if index == 0:
        print("Index = 0")
        # If too few samples, return the maximum output or a default
        return sorted_outputs[0] if len(sorted_outputs) > 0 else 0.5
    elif index >= len(sorted_outputs):
        print("Index >= len(sorted_outputs)")
        # If index exceeds array length, return a small value
        return sorted_outputs[-1] if len(sorted_outputs) > 0 else 0.5
    
    # Return the threshold at the 1% FAR
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
n_data = 1  # modify the load data method in the loop to allow n > 1
n_step = 2
threshold = 50

# Initial noise levels and total possible noise levels 
max_train_levels = [20]
max_val_levels = [0, 9, 12, 15, 18, 20, 22, 35]

#max_val_levels = [0, 15,20]

label = 'UNETwCLASS_{}Hz_D{}-{}_T{}_Tsft{}_ndata{}_step{}_th{}'.format(f0, int(max_train_levels[0]), int(max_train_levels[-1]), int(obsTime//86400), Tsft, n_data, n_step, threshold)
version = '{}_{}_{}x{}_MSELoss_dropout0'.format(det, label, size[0], size[1])


print(f"Nominal frequency: {f0}")
print(f"Detector: {det}")
print(f"SFT duration (Tsft): {Tsft} seconds")
print(f"Observation time (obsTime): {obsTime} seconds")
print(f"Home directory: {homedir}")
print(f"Frequency band size: {freq_size}")
print(f"Spectrogram size: {size}")
#print(f"Number of training samples per noise level: {nTrain}")
print(f"Data generation label: {label}")
print(f"Save file label: {version}")


# Initialize dictionaries to store `pdet` by noise level
train_pdet = {noise_level: [] for noise_level in max_train_levels}
val_pdet = {noise_level: [] for noise_level in max_val_levels}

batch_size = 8

filename = '/scratch/kriles_root/kriles0/damoncht/unet_f/data/pure_noise/H1L1_purenoise_n1000_seed300.npz'
data = np.load(filename, allow_pickle=True)['dataset']
#noise = np.stack(noise)
noise = normalize(data)
noise_dataset = load_noise_dataset(noise)

#### gengerate validation data
# Set random seed for reproducibility

filename = '/scratch/kriles_root/kriles0/damoncht/unet_dyn_fastnoise/data/validation/500Hz_H1L1_D0-35_256x128_7200s_4c_traindata_n200_seed0.npz'
data = np.load(filename, allow_pickle=True)

signal_dataset = load_signal_dataset(data, max_val_levels)
del data
val_loader = make_data_loader(signal_dataset, noise_dataset, batch_size=batch_size)
#del signal_dataset, noise_dataset

# Initialize the model
size_filter_in = 16
dropout_prob = 0.0 # 0.1 
# 16, 0.3 version 1
model = UNet(input_channels=4, output_channels=4, size_filter_in=size_filter_in, dropout_prob=dropout_prob).to(device)
criterion = torch.nn.MSELoss(reduction='none')  # Default loss function
#criterion = torch.nn.L1Loss(reduction='none')  # Default loss function

# Initialize the optimizer
lr=1e-4
optimizer = torch.optim.Adam(model.parameters(), lr=lr)
scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=100)

# Initialize variables to store the top models and their losses
best_val_loss = float('inf')
best_val_model = None

best_pdet = 0.0
best_pdet_model = None

train_losses = []
val_losses = []

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
        data = np.load(filename, allow_pickle=True)['dataset'][:500]
        noise = normalize(data)
        noise_dataset = load_noise_dataset(noise)

        train_loader = make_data_loader(signal_dataset, noise_dataset, batch_size=batch_size)

        
        
    print("Training:")
    # First Pass: Collect class_output for noise samples
    model.eval()  # No gradients needed
    noise_class_outputs = []
    with torch.no_grad():
        for inputs, _, _, labels in train_loader:
            inputs = inputs.to(device)
            _, class_output = model(inputs)
            noise_mask = (labels == np.float('inf')).cpu().numpy()
            if noise_mask.any():
                noise_class_outputs.append(class_output.squeeze().cpu().numpy()[noise_mask])

    # Compute 1% FAR threshold
    inoise_class_outputs = np.concatenate(noise_class_outputs, axis=0)
    far_threshold = compute_threshold_from_pfa(noise_class_outputs)  # Assumes 1% FAR
    print(f"Epoch {epoch + 1}, Training FAR Threshold: {far_threshold:.4f}")
          
        
    # Training phase
    model.train()
    running_train_loss = 0.0
    train_det = []
    train_label = []
    for inputs, targets, mask, labels in train_loader:
        inputs, targets, mask = inputs.to(device), targets.to(device), mask.to(device)
        class_target = (labels != np.float('inf')).float().to(device)  # For pdet
        optimizer.zero_grad()
        
        # Forward pass
        denoised, class_output = model(inputs)
        loss = combined_loss(denoised, targets, class_output, class_target, mask, far_threshold)
        
        # Backward pass and optimization step
        loss.backward()
        optimizer.step()

        # Accumulate training loss
        running_train_loss += loss.item()
        
        detection_stats = class_output.squeeze().detach()
        train_det.append(detection_stats.cpu().numpy())
        train_label.append(labels)
        
        detection_stats = compute_detection_statistic(outputs.detach())

        # Count signals and those passing the threshold
        detection_stats = class_output.squeeze().detach()
        train_det.append(detection_stats.cpu().numpy())
        train_label.append(labels)

    # Calculate average training loss
    train_loss = running_train_loss / len(train_loader)
    train_losses.append(train_loss)

    
    # Compute pdet using validation FAR threshold
    train_det = np.concatenate(train_det, axis=0)
    train_label = np.concatenate(train_label, axis=0)
    noise_det = train_det[train_label == np.float('inf')]
    
    pdet = (noise_det < far_threshold).sum() / noise_det.size 
    print("D={}, pdet={}%".format('inf', pdet*100))
    
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
    # First Pass (Validation): Collect class_output for noise samples
    model.eval()
    val_noise_class_outputs = []
    with torch.no_grad():
        for inputs, _, _, labels in val_loader:
            inputs = inputs.to(device)
            _, class_output = model(inputs)
            noise_mask = (labels == np.float('inf')).cpu().numpy()
            if noise_mask.any():
                val_noise_class_outputs.append(class_output.squeeze().cpu().numpy()[noise_mask])

    # Compute validation 1% FAR threshold
    val_noise_class_outputs = np.concatenate(val_noise_class_outputs, axis=0)
    val_far_threshold = compute_threshold_from_pfa(val_noise_class_outputs)
    print(f"Epoch {epoch + 1}, Validation FAR Threshold: {val_far_threshold:.4f}")
        
    running_val_loss = 0.0
    val_det = []
    val_label = []
    with torch.no_grad():
        for inputs, targets, mask, labels in val_loader: 
            inputs, targets, mask = inputs.to(device), targets.to(device), mask.to(device)
            class_target = (labels != np.float('inf')).float().to(device)

            denoised, class_output = model(inputs)
            val_loss = combined_loss(denoised, targets, class_output, class_target, mask, far_threshold)  # Use training threshold for loss

            running_val_loss += val_loss.item()
            
            detection_stats = class_output.squeeze().detach()
            val_det.append(detection_stats.cpu().numpy())
            val_label.append(labels)
                        
    # Calculate average validation loss
    val_loss = running_val_loss / len(val_loader)
    val_losses.append(val_loss)
        
    val_det = np.concatenate(val_det, axis=0)
    val_label = np.concatenate(val_label, axis=0)
    noise_det = val_det[val_label == np.float('inf')]
    pdet = (noise_det < far_threshold).sum() / noise_det.size 
    print("D={}, pdet={}%".format('inf', pdet*100))
    
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
            
    #scheduler.step(val_loss)
    scheduler.step(val_pdet[max_val_levels[-1]][-1])
    # Print epoch loss every 5 epochs
    if epoch % 3 == 0:
        print(f"Epoch {epoch + 1}/{num_epochs}, Training Loss: {train_loss:.3e}, Validation Loss: {val_loss:.3e}")

    # Track the best model with the lowest training loss
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        best_val_model = model.state_dict()
        print('Replace best val model at epoch {0}'.format(epoch))        
    if pdet > best_pdet:
        best_pdet = pdet
        best_pdet_model = model.state_dict()
        print('Replace best pdet model at epoch {0}'.format(epoch))

       
    # Print the learning rate
    current_lr = optimizer.param_groups[0]['lr']
    print("the learning rate:", current_lr)
    print("Time used = {}".format(time.time()-t0))
    if current_lr < 1e-4 / 2 **3:
        break

train_losses =  np.array(train_losses)
val_losses = np.array(val_losses)

# Save the top validation models and the best training model
torch.save(best_val_model, homedir+"trained_model/{0}Hz/best_val_model_{1}.pth".format(f0, version))
torch.save(best_pdet_model, homedir+"trained_model/{0}Hz/best_pdet_model_{1}.pth".format(f0, version))


# Save all losses in a single file
losses = {
    "train_losses": train_losses,
    "val_losses": val_losses,
    "top_val_losses": best_val_loss,
    "train_pdet": train_pdet,
    "val_pdet": val_pdet,
    "top_val_pdet": best_pdet,
}
np.save(homedir+'trained_model/{0}Hz/losses_{1}.npy'.format(f0, version), losses)

print("Done.")
