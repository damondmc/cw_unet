#!/home/damoncht/.conda/envs/ml/bin/python
from tqdm import tqdm
from utils import *
from genData import *
import json
import argparse

parser = argparse.ArgumentParser(description="Generate mock CW signal dataset.")
parser.add_argument('--f0', type=int, default=500, help='Base frequency (default: 500)')
parser.add_argument('--det', type=str, default='H1L1', help='Detector name (default: H1L1)')
parser.add_argument('--Tsft', type=int, default=7200, help='SFT duration in seconds (default: 7200)')
parser.add_argument('--obsTime', type=int, default=921600, help='Observation time in seconds (default: 921600)')
parser.add_argument('--freq_size', type=int, default=256, help='Frequency band size (default: 256)')
parser.add_argument('--num_cpus', type=int, default=20, help='Number of CPUs to use (default: 20)')
parser.add_argument('--validation', action='store_true', help='Enable validation mode (default: False)')
args = parser.parse_args()

t0 = time.time()
print("Start")

# Set random seed for reproducibility
np.random.seed(7)

if args.validation:
    np.random.seed(23) 

#np.random.seed(2323) 
# # for purenoise
# np.random.seed(8)

# Use arguments from argparse
f0 = args.f0
det = args.det
Tsft = args.Tsft
obsTime = args.obsTime
freq_size = args.freq_size
num_cpus = args.num_cpus
size = (freq_size, obsTime // Tsft)
    
homedir = '/scratch/kriles_root/kriles0/damoncht/unet_f/' 

nSample = 33000
neach = 1000

f1min = -1e-10 
f1max = 0

print(f"Detector: {det}")
print(f"SFT duration (Tsft): {Tsft} seconds")
print(f"Observation time (obsTime): {obsTime} seconds")
print(f"Home directory: {homedir}")
print(f"Frequency band size: {freq_size}")
print(f"Spectrogram size: {size}")

#hnoise = [0, 20]
# hnoise = [0, 3, 6, 9, 12,
#              15, 18, 20, 22, 
#              24, 26, 30, 40, 
#              60, 80, 100, 120]

#hnoise = [0, 9, 12, 15, 18, 20, 22, 35]
hnoise = [0, 20]


label = '{}train4cD10N'.format(f0)
version = '{}_D{}-{}_{}x{}_{}s_4c'.format(det, hnoise[0], hnoise[-1], size[0], size[1], Tsft)


# Generate parameters based on the chosen method
params = genSampleParam(hnoise, f1min, f1max, nSample)

batch_size = 8
n = int(nSample//neach)
# 320
for seed in range(n):
    p = params[len(hnoise)*seed*neach:len(hnoise)*(seed+1)*neach]
    _d1, _d2 = generate_mock_cw_signals(label=label, f0=f0, params=p, h0=1, num_cpus=num_cpus, obsTime=obsTime, Tsft=Tsft, homedir=homedir+'tmp/')
    signal_dataset = crop_signal_img(_d1, _d2, freq_size=freq_size, threshold = 50)
    #signal_dataset = crop_noise_img(_d1, _d2, freq_size=freq_size)
    
    if args.validation:
        np.savez("data/validation/{}Hz_{}_traindata_n{}_seed{}.npz".format(f0, version, neach, seed), **signal_dataset, f0=f0, params=p)
        print("Saved to data/validation/{}Hz_{}_traindata_n{}_seed{}.npz".format(f0, version, neach, seed))    
    else:
        np.savez("data/{0}Hz/{0}Hz_{1}_traindata_n{2}_seed{3}.npz".format(f0, version, neach, seed), **signal_dataset, f0=f0, params=p)
        print("Saved to data/{0}Hz/{0}Hz_{1}_traindata_n{2}_seed{3}.npz".format(f0, version, neach, seed))   

print("Time used ={} s".format(time.time()-t0))
print("Dataset saved successfully.")
