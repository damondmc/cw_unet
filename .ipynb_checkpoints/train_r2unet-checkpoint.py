#!/home/damoncht/.conda/envs/ml/bin/python
from utils import *
from genData import *
from model.unet import R2_UNet
from torch.optim.lr_scheduler import ReduceLROnPlateau
import torch.nn.functional as F
import time 
import argparse

parser = argparse.ArgumentParser(description="Generate mock CW signal dataset.")
parser.add_argument('--f0', type=int, default=500, help='Base frequency (default: 500)')
parser.add_argument('--det', type=str, default='H1L1', help='Detector name (default: H1L1)')
parser.add_argument('--Tsft', type=int, default=7200, help='SFT duration in seconds (default: 7200)')
parser.add_argument('--obsTime', type=int, default=921600, help='Observation time in seconds (default: 921600)')
parser.add_argument('--freq_size', type=int, default=256, help='Frequency band size (default: 256)')
parser.add_argument('--num_cpus', type=int, default=16, help='Number of CPUs to use (default: 20)')
parser.add_argument('--n_step', type=int, default=3, help='Steps for each dataset to be trained.')
parser.add_argument('--n_data', type=int, default=3, help='Number of dataset to be used for each loop.')
parser.add_argument('--alpha', type=float, default=1, help='Weight for signal MSE in loss function.')
parser.add_argument('--beta', type=float, default=1, help='Weight for noise MSE in loss function.')

args = parser.parse_args()


# Combined loss function with threshold-based labels
def combined_loss(denoised, target, mask, alpha=1, beta=1):
    # MSE for signal
    mse_signal = F.mse_loss(denoised * mask, target * mask, reduction='none')
    mse_signal = mse_signal.mean()
    
    # MSE for noise
    background_zeros = torch.zeros_like(denoised).to(denoised.device)
    mse_noise = F.mse_loss(denoised * (~mask), background_zeros * (~mask), reduction='none')
    mse_noise = mse_noise.mean()

    # Combined loss
    total_loss = alpha * mse_signal + beta * mse_noise 
    return total_loss, mse_signal, mse_noise


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
alpha = args.alpha
beta = args.beta

homedir = '/scratch/kriles_root/kriles0/damoncht/unet_f/'
tmpdir = homedir + 'tmp/'
size = (freq_size, obsTime // Tsft)
n_data = args.n_data  # modify the load data method in the loop to allow n > 1
n_step = args.n_step

threshold = 50

# Initial noise levels and total possible noise levels 
max_train_levels = [20]
max_val_levels = [5, 9, 12, 15, 18, 20, 35]

label = 'dynnoise_fast_UNET_weight_alpha{}beta{}_{}Hz_D{}-{}_T{}_Tsft{}_ndata{}_step{}_ndata{}_th{}'.format(alpha, beta, f0, int(max_train_levels[0]), int(max_train_levels[0]), int(obsTime//86400), Tsft, n_data, n_step, n_data*1000, threshold)
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
print(f"Alpha (signal): {alpha}")
print(f"Beta (noise): {beta}")



# Initialize dictionaries to store `pdet` by noise level
train_pdet = {noise_level: [] for noise_level in max_train_levels}
val_pdet = {noise_level: [] for noise_level in max_val_levels}
batch_size = 8

# filename = '/scratch/kriles_root/kriles0/damoncht/unet_f/data/pure_noise/H1L1_purenoise_{}x{}_n1000_seed300.npz'.format(size[0], size[1])
# noise = np.load(filename, allow_pickle=True)['dataset']
# noise = normalize(noise)
# noise_dataset = load_noise_dataset(noise)

noise = np.empty((1000,) + size + (4,))
for i in range(noise.shape[0]):
    noise[i] = simNoise(sqrtSn=1, Tsft=Tsft, size=size, ndet=2, norm=False)
noise = normalize(noise)
noise_dataset = load_noise_dataset(noise)

filename = '/scratch/kriles_root/kriles0/damoncht/unet_f/data/validation/{0}Hz_H1L1_D0-{4}_{1}x{2}_{3}s_4c_traindata_n{5}_norm.npz'.format(f0, size[0], size[1], Tsft, 35, 400)
print("Using {}".format(filename))
data = np.load(filename, allow_pickle=True)
signal_dataset = load_signal_dataset(data, max_val_levels)    

val_loader = make_data_loader([signal_dataset, noise_dataset], batch_size=batch_size)

# Set random seed for reproducibility
# filename = '/scratch/kriles_root/kriles0/damoncht/unet_f/data/pure_noise/H1L1_purenoise_{}x{}_n1000_seed299.npz'.format(size[0], size[1])
# noise = np.load(filename, allow_pickle=True)['dataset'][:400]

# filename = '/scratch/kriles_root/kriles0/damoncht/unet_f/data/validation/{0}Hz_H1L1_D0-{4}_{1}x{2}_{3}s_4c_traindata_n{5}_seed0.npz'.format(f0, size[0], size[1], Tsft, 35, 400)
# targets = np.load(filename, allow_pickle=True)['clean_image']
# masks = np.load(filename, allow_pickle=True)['signal_mask']
# targets = normalize(targets)


# data = []
# mask_data = []
# target_data = []
# label_data = []
# for Sn in max_val_levels:
#     noise = np.empty((targets.shape[0],) + size + (4,))
#     for i in range(noise.shape[0]):
#         noise[i] = simNoise(sqrtSn=Sn, Tsft=Tsft, size=size, ndet=2, norm=False)

#     data.append(normalize(noise + targets))
#     mask_data.append(masks)
#     target_data.append(targets)
#     labels = [Sn] * noise.shape[0]  # Extend labels
#     label_data.append(labels)
    
# data = np.concatenate(data)
# mask_data = np.concatenate(mask_data)
# target_data = np.concatenate(target_data)
# label_data = np.concatenate(label_data)
# data = load_signal_datasetv2(data, target_data, mask_data, label_data)
    
    
# noise = np.empty((1000,) + size + (4,))
# for i in range(noise.shape[0]):
#     noise[i] = simNoise(sqrtSn=1, Tsft=Tsft, size=size, ndet=2, norm=False)
# noise = normalize(noise)
# noise_dataset = load_noise_dataset(noise)

# val_loader = make_data_loader([data, noise_dataset], batch_size=batch_size)

# # load clean signal data 
# target_datasets = []
# mask_datasets = []
# # Load n datasets based on different seeds
# for i in range(n_data):  # Iterate over n datasets
#     filename = '/scratch/kriles_root/kriles0/damoncht/unet_f/data/{0}Hz/{0}Hz_H1L1_D0-{4}_{2}x{3}_{5}s_4c_traindata_n1000_seed{1}.npz'.format(f0, i, size[0], size[1], 20, Tsft)
#     print("Using {}".format(filename))
#     targets = np.load(filename, allow_pickle=True)['clean_image']
#     masks = np.load(filename, allow_pickle=True)['signal_mask']
#     target_datasets.append(targets)
#     mask_datasets.append(masks)
    
# target_datasets  = np.concatenate(target_datasets)
# mask_datasets = np.concatenate(mask_datasets)


# Initialize the model
latent_channels = 16
dropout_prob = 0.1 # 0.1 
# 16, 0.3 version 1
model = R2_UNet(in_channels=4, out_channels=4, latent_channels=latent_channels, dropout_prob=dropout_prob).to(device)
criterion = torch.nn.MSELoss(reduction='none')  # Default loss function

# Initialize the optimizer
lr=1e-4
optimizer = torch.optim.Adam(model.parameters(), lr=lr)
scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=1000)

# Initialize variables to store the top models and their losses
best_val_loss = float('inf')
best_val_model = None

best_pdet = 0.0
best_pdet_model = None

train_losses = []
train_mse_signal = []
train_mse_noise = []

val_losses = []
val_mse_signal = []
val_mse_noise = []

num_epochs = 1000
print(model)

for epoch in tqdm(range(num_epochs)):   
    if epoch % n_step == 0:
        # generate gaussian noise
#         noise = np.empty((target_datasets.shape[0],) + size + (4,))
#         for i in range(target_datasets.shape[0]):
#             noise[i] = simNoise(sqrtSn=max_train_levels[0], Tsft=Tsft, size=size, ndet=2, norm=False)
        
#         # add noise into clean signal 
#         data = normalize(target_datasets + noise)
#         label_data = [max_train_levels[0]] * data.shape[0]  # Extend labels
#         signal_dataset = load_signal_datasetv2(data, target_datasets, mask_datasets, label_data)
        
        # load pure noise training data
        seed = (epoch // n_step) % 300  # Generate unique seed for each dataset
        dataset = []
        
        #filename = '/scratch/kriles_root/kriles0/damoncht/unet_f/data/pure_noise/H1L1_purenoise_{0}x{1}_n1000_seed{2}.npz'.format(size[0], size[1], seed)
        #print('Using pure noise {} ...'.format(filename))
        #noise = np.load(filename, allow_pickle=True)['dataset']
        #noise = normalize(noise)
        #noise_data = load_noise_dataset(noise)
        
        noise = np.empty((3000,) + size + (4,))
        for i in range(noise.shape[0]):
            noise[i] = simNoise(sqrtSn=1, Tsft=Tsft, size=size, ndet=2, norm=False)
        noise = normalize(noise)
        noise_dataset = load_noise_dataset(noise)
        dataset.append(noise_dataset)
        
        # for i in range(n_data):
        #     seed = ((epoch // n_step) * n_data + i) % 300
        #     filename = '/scratch/kriles_root/kriles0/damoncht/unet_f/data/pure_noise/H1L1_purenoise_{}x{}_n1000_seed{}.npz'.format(size[0], size[1], seed)
        #     print("Using {}".format(filename))
        #     noise = np.load(filename, allow_pickle=True)['dataset']
        #     noise = normalize(noise)
        #     noise_dataset = load_noise_dataset(noise)
        #     dataset.append(noise_dataset)

        for i in range(n_data):
            seed = ((epoch // n_step) * n_data + i) % 300
            filename = '/scratch/kriles_root/kriles0/damoncht/unet_f/data/{0}Hz/{0}Hz_H1L1_D0-{1}_{2}x{3}_{4}s_4c_traindata_n1000_seed{5}_norm.npz'.format(f0, max_train_levels[0], size[0], size[1], Tsft, seed)
            print("Using {}".format(filename))
            data = np.load(filename, allow_pickle=True)
            signal_dataset = load_signal_dataset(data, max_val_levels)  
            dataset.append(signal_dataset)

        train_loader = make_data_loader(dataset, batch_size=batch_size)

    # Training phase
    running_train_loss = 0.0
    running_train_mse_signal = 0.0
    running_train_mse_noise = 0.0
    train_det = []
    train_label = []
    
    model.train()
    for inputs, targets, mask, labels in train_loader:
        inputs, targets, mask = inputs.to(device), targets.to(device), mask.to(device)
        
        optimizer.zero_grad()
        
        # Forward pass
        denoised = model(inputs)
        total_loss, mse_signal, mse_noise = combined_loss(denoised, targets, mask, alpha, beta)
        
        # Backward pass and optimization step
        total_loss.backward()
        optimizer.step()

        # Accumulate training loss
        running_train_loss += total_loss.item()
        running_train_mse_signal += mse_signal.item()
        running_train_mse_noise += mse_noise.item()
        
        # Count signals and those passing the threshold
        detection_stats = compute_detection_statistic(denoised.detach())
        train_det.append(detection_stats)
        train_label.append(labels)

    # Calculate average training loss
    train_loss = running_train_loss / len(train_loader)
    train_mse_signal_epoch = running_train_mse_signal / len(train_loader)
    train_mse_noise_epoch = running_train_mse_noise / len(train_loader)
    train_losses.append(train_loss)
    train_mse_signal.append(train_mse_signal_epoch)
    train_mse_noise.append(train_mse_noise_epoch)

    
    # Validation phase
    running_val_loss = 0.0
    running_val_mse_signal = 0.0
    running_val_mse_noise = 0.0
    model.eval()
    with torch.no_grad():
        val_det = []
        val_label = []
        
        for inputs, targets, mask, labels in val_loader: 
            inputs, targets, mask = inputs.to(device), targets.to(device), mask.to(device)

            denoised = model(inputs)
            total_loss, mse_signal, mse_noise = combined_loss(denoised, targets, mask, alpha, beta)  # Use training threshold for loss

            #running_val_loss += val_loss.item()
            running_val_loss += total_loss.item()
            running_val_mse_signal += mse_signal.item()
            running_val_mse_noise += mse_noise.item()
            
            detection_stats = compute_detection_statistic(denoised.detach())
            val_det.append(detection_stats)
            val_label.append(labels)
            
    val_loss = running_val_loss / len(val_loader)
    val_mse_signal_epoch = running_val_mse_signal / len(val_loader)
    val_mse_noise_epoch = running_val_mse_noise / len(val_loader)
    val_losses.append(val_loss)
    val_mse_signal.append(val_mse_signal_epoch)
    val_mse_noise.append(val_mse_noise_epoch)
    
    # Compute Pdet as the fraction of signals passing the threshold
    train_det = np.concatenate(train_det, axis=0)
    train_label = np.concatenate(train_label, axis=0)    
    
    noise_det = train_det[train_label==np.float('inf')]
    pfa = compute_threshold_from_pfa(noise_det)
    
    # Compute training pdet by noise level
    for noise_level in train_pdet.keys():
        signal_det = train_det[train_label == float(noise_level)]
        if signal_det.size != 0:
            pdet = (signal_det > pfa).sum() / signal_det.size 
            train_pdet[noise_level].append(pdet)
            print("D={}, pdet={}%".format(noise_level, pdet*100))
        else:
            pdet = np.nan
            train_pdet[noise_level].append(pdet)
        
    # Compute validation pdet by noise level
    val_det = np.concatenate(val_det, axis=0)
    val_label = np.concatenate(val_label, axis=0)    
    
    noise_det = val_det[val_label==np.float('inf')]
    pfa = compute_threshold_from_pfa(noise_det)
    
    # Compute valing pdet by noise level
    print('Validation:')
    print('pfa th = ', pfa)
    for noise_level in val_pdet.keys():
        signal_det = val_det[val_label == float(noise_level)]
        if signal_det.size != 0:
            pdet = (signal_det > pfa).sum() / signal_det.size 
            val_pdet[noise_level].append(pdet)
            print("D={}, pdet={}%".format(noise_level, pdet*100))
        else:
            pdet = np.nan
            val_pdet[noise_level].append(pdet) 
            
    #scheduler.step(val_loss)
    scheduler.step(val_pdet[max_val_levels[-1]][-1])
    # Print epoch loss every 5 epochs
    if epoch % 3 == 0:
        print(f"Epoch {epoch + 1}/{num_epochs}, Training Loss: {train_loss:.3e}, Validation Loss: {val_loss:.3e}.")

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

train_losses = np.array(train_losses)
train_mse_signal = np.array(train_mse_signal)
train_mse_noise = np.array(train_mse_noise)

val_losses = np.array(val_losses)
val_mse_signal = np.array(val_mse_signal)
val_mse_noise = np.array(val_mse_noise)

# Save the top validation models and the best training model
torch.save(best_val_model, homedir+"trained_model/{0}Hz/best_val_model_{1}.pth".format(f0, version))
torch.save(best_pdet_model, homedir+"trained_model/{0}Hz/best_pdet_model_{1}.pth".format(f0, version))

# Save all losses in a single file
losses = {
    "train_losses": train_losses,
    "train_mse_signal": train_mse_signal,
    "train_mse_noise": train_mse_noise,
    "val_losses": val_losses,
    "val_mse_signal": val_mse_signal,
    "val_mse_noise": val_mse_noise,
    "top_val_losses": best_val_loss,
    "train_pdet": train_pdet,
    "val_pdet": val_pdet,
    "top_val_pdet": best_pdet
}
np.save(homedir+'trained_model/{0}Hz/losses_{1}.npy'.format(f0, version), losses)

print("Done.")
