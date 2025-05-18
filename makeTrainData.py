#!/home/damoncht/.conda/envs/ml/bin/python
from tqdm import tqdm
from utils import *
from genData import *
import json


t0 = time.time()
print("Start")

f0 = 500
det = 'H1L1'
Tsft = 7200 
obsTime = 921600    # 921600 ~ 10 days
homedir = '/scratch/kriles_root/kriles0/damoncht/unet_dyn_fastnoise/' 
freq_size = 256
size = (freq_size, obsTime // Tsft)
num_cpus = 20

nSample = 200000
neach = 1000

print(f"Detector: {det}")
print(f"SFT duration (Tsft): {Tsft} seconds")
print(f"Observation time (obsTime): {obsTime} seconds")
print(f"Home directory: {homedir}")
print(f"Frequency band size: {freq_size}")
print(f"Spectrogram size: {size}")

hnoise = [0, 20, 35]

label = 'train4cD10N'
version = '{}_D{}-{}_{}x{}_{}s_4c'.format(det, hnoise[0], hnoise[-1], size[0], size[1], Tsft)

# Set random seed for reproducibility
np.random.seed(7)

f1min = -1e-10 
f1max = 0
# Generate parameters based on the chosen method
params = genSampleParam(hnoise, f1min, f1max, nSample)

batch_size = 8
n = int(nSample//neach)
for seed in range(94, 100):
    p = params[len(hnoise)*seed*neach:len(hnoise)*(seed+1)*neach]
    _d1, _d2 = generate_mock_cw_signals(label=label, f0=f0, params=p, h0=1, num_cpus=num_cpus, obsTime=obsTime, Tsft=Tsft, homedir=homedir+'tmp/')
    signal_dataset = crop_signal_img(_d1, _d2, freq_size=freq_size, threshold = 50)
    
    np.savez("data/{}_traindata_n{}_seed{}.npz".format(version, neach, seed), **signal_dataset, f0=f0, params=p)

print("Time used ={} s".format(time.time()-t0))
print("Dataset saved successfully.")
