#!/home/damoncht/.conda/envs/ml/bin/python
from tqdm import tqdm
from utils import *
from genData import *


t0 = time.time()
f0 = 500
det = 'H1L1'
Tsft = 7200 
obsTime = 921600    # 921600 ~ 10 days
homedir = '/scratch/kriles_root/kriles0/damoncht/unet_f/' 
freq_size = 256
size = (freq_size, obsTime // Tsft)
num_cpus = 8
nSample = 4
neach = 1000

hnoise = [0, 20]

label = 'train4cD10N'
version = '{}_D{}-{}_{}x{}_{}s_4c'.format(det, hnoise[0], hnoise[-1], size[0], size[1], Tsft)

# Set random seed for reproducibility
np.random.seed(7)

f1min = -1e-10 
f1max = 0
# Generate parameters based on the chosen method
params = genSampleParam(hnoise, f1min, f1max, nSample)
p1 = params[0].copy()
p2 = params[0].copy()
p2[-1] = 20
print(params)
params = [p1,p2,p1,p2,p1,p2,p1,p2]
#params[1][-1] = 20
#params[3][-1] = 20

print(params)
_d1, _d2 = generate_mock_cw_signals(label=label, f0=f0, params=params, h0=1, num_cpus=num_cpus, obsTime=obsTime, Tsft=Tsft, homedir=homedir+'tmp/')
#signal_dataset = crop_signal_img(_d1, _d2, freq_size=freq_size, threshold = 50)


np.savez("./{}_traindata_n{}_seed{}v3.npz".format(version, neach, 'x'), d1=_d1, d2=_d2, f0=f0, params=params)

print("Time used ={} s".format(time.time()-t0))
print("Dataset saved successfully.")
