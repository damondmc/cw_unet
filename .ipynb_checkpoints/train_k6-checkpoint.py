#!/home/damoncht/.conda/envs/ml/bin/python
from utils import *
from genData import *
from model.unet_k import Attention_UNet
from torch.optim.lr_scheduler import ReduceLROnPlateau
import torch.nn.functional as F
import time 
import argparse
from multiprocessing import Pool

parser = argparse.ArgumentParser(description="Generate mock CW signal dataset.")
parser.add_argument('--f0', type=int, default=500, help='Base frequency (default: 500)')
parser.add_argument('--det', type=str, default='H1L1', help='Detector name (default: H1L1)')
parser.add_argument('--Tsft', type=int, default=7200, help='SFT duration in seconds (default: 7200)')
parser.add_argument('--obsTime', type=int, default=921600, help='Observation time in seconds (default: 921600)')
parser.add_argument('--freq_size', type=int, default=256, help='Frequency band size (default: 256)')
parser.add_argument('--num_cpus', type=int, default=16, help='Number of CPUs to use (default: 20)')
parser.add_argument('--n_step', type=int, default=3, help='Steps for each dataset to be trained.')
parser.add_argument('--n_data', type=int, default=3, help='Number of dataset to be used for each loop.')
parser.add_argument('--n_noise', type=int, default=1, help='Number of noise dataset to be used for each loop.')
parser.add_argument('--alpha', type=float, default=1, help='Weight for signal MSE in loss function.')
parser.add_argument('--beta', type=float, default=1, help='Weight for noise MSE in loss function.')
parser.add_argument('--latent_channels', type=int, default=16, help='Number of latent_channels for our model.')
parser.add_argument('--batch_size', type=int, default=8, help='Batch size.')
parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate.')
args = parser.parse_args()

def simulate_noise_batch(Sn, Tsft, size, ns, epoch, num_cpus, base=0):
    """
    Simulate noise in parallel using a process pool.
    
    Args:
        max_train_levels: Maximum training levels (list or tuple)
        Tsft: Time shift parameter
        size: Shape of the noise array
        ns: Number of samples
        epoch: Current epoch number
        num_cpus: Number of CPU processes to use
    
    Returns:
        numpy.ndarray: Array of simulated noise with shape (ns, *size, 4)
    """
    with Pool(processes=num_cpus) as pool:
        _noise = pool.starmap(simNoise,
                            [(Sn, Tsft, size, 2, False, base+epoch*ns + i)
                             for i in range(ns)])
    
    noise = np.empty((ns,) + size + (4,))
    for i, _n in enumerate(_noise):
        noise[i] = _n
        
    return noise

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

num_epochs = 800
noise_train = 1000 * args.n_noise
print("Number of pure nosie = {}".format(noise_train))

# Set random seed for reproducibility
np.random.seed(0)
torch.manual_seed(0)

# Use arguments from argparse
f0 = args.f0
det = args.det
Tsft = args.Tsft
obsTime = args.obsTime
freq_size = args.freq_size
num_cpus = args.num_cpus
alpha = args.alpha
beta = args.beta
latent_channels = args.latent_channels
batch_size = args.batch_size
lr = args.lr

homedir = '/scratch/kriles_root/kriles0/damoncht/unet_f'
size = (freq_size, obsTime // Tsft)
n_data = args.n_data  # modify the load data method in the loop to allow n > 1
n_step = args.n_step

threshold = 50
# Initial noise levels and total possible noise levels 
max_train_levels = [20]
max_val_levels = [18, 19, 19.5, 20, 35, 35.6]

label = 'newmask_k_UNET_a{}b{}_{}Hz_D{}-{}_T{}_Tsft{}_step{}_ndata{}_noise{}_latent{}_th{}_batch{}_lr{}'.format(alpha, beta, f0, int(max_train_levels[0]), int(max_train_levels[0]), int(obsTime//86400), Tsft, n_step, n_data*1000, noise_train, latent_channels, threshold, batch_size, lr)
#f"UNET_a{alpha}b{beta}_{f0}Hz_D{int(max_train_levels[0])}-{int(max_train_levels[0])}_T{int(obsTime//86400)}_Tsft{Tsft}_step{n_step}_ndata{n_data*1000}_noise{noise_train}_latent{latent_channels}_th{threshold}"i
version = '{}_{}_{}x{}_MSELoss_dropout0'.format(det, label, size[0], size[1])

print(f"Batch size: {batch_size}")
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
print(f"Learning rate: {lr}")

# Initialize dictionaries to store `pdet` by noise level
train_pdet = {noise_level: [] for noise_level in max_train_levels}
val_pdet = {noise_level: [] for noise_level in max_val_levels}

filename = '{6}/data/validation/{0}Hz_{1}_{2}x{3}_{4}s_4c_traindata_n{5}_seed0.npz'.format(f0, det, size[0], size[1], Tsft, 1000, homedir)
targets = np.load(filename, allow_pickle=True)['clean_image'][:500]
#masks = np.load(filename, allow_pickle=True)['signal_mask'][:500]
masks = new_mask(normalize(targets))
data = []
mask_data = []
target_data = []
label_data = []
for Sn in max_val_levels:
    noise = np.empty((targets.shape[0],) + size + (4,))
    for i in range(noise.shape[0]):
        noise[i] = simNoise(sqrtSn=Sn, Tsft=Tsft, size=size, ndet=2, norm=False)

    data.append(normalize(noise + targets))
    mask_data.append(masks)
    target_data.append(normalize(targets))
    labels = [Sn] * noise.shape[0]  # Extend labels
    label_data.append(labels)
        
data = np.concatenate(data)
mask_data = np.concatenate(mask_data)
target_data = np.concatenate(target_data)
label_data = np.concatenate(label_data)
data = load_signal_datasetv2(data, target_data, mask_data, label_data)
    
noise = np.empty((500,) + size + (4,))
for i in range(noise.shape[0]):
    noise[i] = simNoise(sqrtSn=1, Tsft=Tsft, size=size, ndet=2, norm=False)
noise = normalize(noise)
noise_data = load_noise_dataset(noise)

val_loader = make_data_loader([data, noise_data], batch_size=batch_size)

# load clean signal data 
target_datasets = []
mask_datasets = []
# Load n datasets based on different seeds
for i in range(n_data):  # Iterate over n datasets
    filename = '{6}/data/{0}Hz/{0}Hz_{1}_{2}x{3}_{4}s_4c_traindata_n1000_seed{5}.npz'.format(f0, det, size[0], size[1], Tsft, i, homedir)
    print("Using {}".format(filename))
    targets = np.load(filename, allow_pickle=True)['clean_image']
    #masks = np.load(filename, allow_pickle=True)['signal_mask']
    masks = new_mask(normalize(targets))
    target_datasets.append(targets)
    mask_datasets.append(masks)
    
target_datasets  = np.concatenate(target_datasets)
mask_datasets = np.concatenate(mask_datasets)

ns = target_datasets.shape[0]

# Initialize the model
dropout_prob = 0.0 # 0.1 
model = Attention_UNet(in_channels=4, out_channels=4, latent_channels=latent_channels, dropout_prob=dropout_prob).to(device)
criterion = torch.nn.MSELoss(reduction='none')  # Default loss function

# Initialize the optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=lr)
scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=100)

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

print(model)

for epoch in tqdm(range(num_epochs)):   
    if epoch % n_step == 0:
        # generate noise 
        noise = simulate_noise_batch(max_train_levels[0], Tsft, size, ns, epoch, num_cpus, 1)

        # add noise into clean signal 
        data = normalize(target_datasets + noise)
        label_data = [max_train_levels[0]] * data.shape[0]  # Extend labels
        data = load_signal_datasetv2(data, normalize(target_datasets), mask_datasets, label_data)
        
        # generate pure noise training data       # original 500 noise 
        noise = simulate_noise_batch(1, Tsft, size, noise_train, epoch, num_cpus, 1+num_epochs*ns)
        noise = normalize(noise)
        noise_data = load_noise_dataset(noise)

        train_loader = make_data_loader([data, noise_data], batch_size=batch_size)

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
torch.save(best_val_model, homedir+"/trained_model/{0}Hz/best_val_model_{1}.pth".format(f0, version))
torch.save(best_pdet_model, homedir+"/trained_model/{0}Hz/best_pdet_model_{1}.pth".format(f0, version))

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
np.save(homedir+'/trained_model/{0}Hz/losses_{1}.npy'.format(f0, version), losses)

print("Done.")
