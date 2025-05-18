#!/home/damoncht/.conda/envs/ml/bin/python
from utils import *
from genData import *
from model.unet_leaky_norm_tanho import UNet
from torch.optim.lr_scheduler import ReduceLROnPlateau
import time 

t0 = time.time()
print("Start")

f0 = 500
det = 'H1L1'
Tsft = 7200  
obsTime = 921600     # 921600 ~ 10 days
homedir = '/scratch/kriles_root/kriles0/damoncht/unet_f/'
tmpdir = homedir + 'tmp/'
freq_size = 256  
size = (freq_size, obsTime // Tsft)
num_cpus = 10
n_data = 3

# Initial noise levels and total possible noise levels 
max_train_levels = [25]
#max_val_levels = [0, 15, 17, 18, 19, 20, 35]
max_val_levels = [0, 15, 20, 35]

label = 'v2D{}{}day{}Tsft{}'.format(int(max_train_levels[0]), int(max_train_levels[-1]), int(obsTime//86400), Tsft)
version = '{}_{}_{}x{}_MSELoss'.format(det, label, size[0], size[1])


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

with Pool(processes=num_cpus) as pool:
    noise = pool.starmap(simNoise, [
        (1, Tsft, size, 2, True)
        for _ in range(400)
    ])

noise = np.stack(noise)
noise_dataset = load_noise_dataset(noise)

#### gengerate validation data
# Set random seed for reproducibility
np.random.seed(100000)

f1min = -1e-10 
f1max = 0
# Generate parameters based on the chosen method
params = genSampleParam(max_val_levels, f1min, f1max, 100)

_d1, _d2 = generate_mock_cw_signals(label=label, f0=f0, params=params, h0=1, num_cpus=num_cpus, obsTime=obsTime, Tsft=Tsft, homedir=tmpdir)
signal_dataset = crop_signal_img(_d1, _d2, freq_size=freq_size, threshold = 50)
signal_dataset = load_signal_dataset_val(signal_dataset, max_val_levels)
del _d1, _d2

val_loader = make_data_loader(signal_dataset, noise_dataset, batch_size=batch_size)
del signal_dataset, noise_dataset

        
# Initialize the model
size_filter_in = 16
dropout_prob = 0.0 # 0.1 
# 16, 0.3 version 1
model = UNet(input_channels=4, output_channels=4, size_filter_in=size_filter_in, dropout_prob=dropout_prob).to(device)
criterion = torch.nn.MSELoss(reduction='none')  # Default loss function
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

num_epochs = 150
print(model)

for epoch in tqdm(range(num_epochs)):   
    print('Noisy image generation...')

    if epoch % 5 ==0:
        clean_image_data = []
        mask_data = []

        for i in range(1):
            filename = '../unet_dyn_fastnoise/data/H1L1_D0-35_256x128_7200s_4c_traindata_n1000_seed{}.npz'.format(1*epoch+i)
            print("Loading {}".format(filename))
            data = np.load(filename, allow_pickle=True)
            clean_image_data.append(data['clean_image'])
            mask_data.append(data['signal_mask'])

        del data

        clean_image_data = np.vstack(clean_image_data)
        mask_data = np.vstack(mask_data)

   
    # make new training data for every 5 steps
    with Pool(processes=num_cpus) as pool:
        noise = pool.starmap(simNoise, [
            (max_train_levels[0], Tsft, size, 2, False)
            for _ in range(clean_image_data.shape[0])
        ])

    noise = np.stack(noise)  
    
    # Process pure noise data
    images = noise + clean_image_data
    
    # make new training data for every 5 steps
    with Pool(processes=num_cpus) as pool:
        images = pool.starmap(normalize, [
            (d,)
            for d in images
        ])

    images = np.stack(images)
   
    signal_dataset = load_signal_dataset(images, clean_image_data, mask_data, 25)
    
    print('Noise generation...')
    # make new training data for every 5 steps
    with Pool(processes=num_cpus) as pool:
        noise = pool.starmap(simNoise, [
            (1, Tsft, size, 2, True)
            for _ in range(1000)
        ])

    noise = np.stack(noise)    
    print("Using {} noise images".format(noise.shape[0]))
    noise_dataset = load_noise_dataset(noise)

    train_loader = make_data_loader(signal_dataset, noise_dataset, batch_size=batch_size)

    # Training phase
    model.train()
    running_train_loss = 0.0
    train_det = []
    train_label = []
    for inputs, targets, mask, labels in train_loader:
        inputs, targets, mask = inputs.to(device), targets.to(device), mask.to(device)
        optimizer.zero_grad()
        
        # Forward pass
        outputs = model(inputs)
        signal_loss = criterion(outputs, targets)
        signal_loss *= mask

        background_zeros = torch.zeros_like(inputs).to(device)
        background_loss = criterion(outputs, background_zeros)
        background_loss *= (~mask)
        loss = signal_loss.mean() + background_loss.mean()
        
        # Backward pass and optimization step
        loss.backward()
        optimizer.step()

        # Accumulate training loss
        running_train_loss += loss.item()
        
        detection_stats = compute_detection_statistic(outputs.detach())

        # Count signals and those passing the threshold
        train_det.append(detection_stats)
        train_label.append(labels)

    # Calculate average training loss
    train_loss = running_train_loss / len(train_loader)
    train_losses.append(train_loss)

    
    # Validation phase
    model.eval()
    running_val_loss = 0.0
    with torch.no_grad():
        val_det = []
        val_label = []
        
        for inputs, targets, mask, labels in val_loader: 
            inputs, targets, mask = inputs.to(device), targets.to(device), mask.to(device)

            outputs = model(inputs)
            background_zeros = torch.zeros_like(inputs).to(device)
            val_signal_loss = criterion(outputs, targets)
            val_signal_loss *= mask
            
            val_background_loss = criterion(outputs, background_zeros)
            val_background_loss *= (~mask)
            val_loss = val_signal_loss.mean() + val_background_loss.mean()

            # Accumulate validation loss
            running_val_loss += val_loss.item()
            
            # Compute detection statistics
            detection_stats = compute_detection_statistic(outputs)

            # Count signals and those passing the threshold
            val_det.append(detection_stats)
            val_label.append(labels)
            
            
    # Calculate average validation loss
    val_loss = running_val_loss / len(val_loader)
    val_losses.append(val_loss)
    
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
        print(f"Epoch {epoch + 1}/{num_epochs}, Training Loss: {train_loss:.3e}, Validation Loss: {val_loss:.3e}")

    # Track the best model with the lowest training loss
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        best_val_model = model.state_dict()
        
    if pdet > best_pdet:
        best_pdet = pdet
        best_pdet_model = model.state_dict()

       
    # Print the learning rate
    current_lr = optimizer.param_groups[0]['lr']
    print("the learning rate:", current_lr)
    print("Time used = {}".format(time.time()-t0))
    if current_lr < 1e-4 / 2 **3:
        break

train_losses =  np.array(train_losses)
val_losses = np.array(val_losses)

# Save the top validation models and the best training model
torch.save(best_val_model, homedir+"trained_model/best_val_model_{}.pth".format(version))
torch.save(best_pdet_model, homedir+"trained_model/best_pdet_model_{}.pth".format(version))

# Save all losses in a single file
losses = {
    "train_losses": train_losses,
    "val_losses": val_losses,
    "top_val_losses": best_val_loss,
    "train_pdet": train_pdet,
    "val_pdet": val_pdet,
}
np.save(homedir+'trained_model/losses_{}.npy'.format(version), losses)

print("Done.")
