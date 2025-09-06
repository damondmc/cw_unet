#!/home/damoncht/.conda/envs/ml/bin/python
import numpy as np
import pyfstat
from pyfstat.utils import get_sft_as_arrays
import logging
import argparse
from tqdm import tqdm 
from multiprocessing import Pool, cpu_count
import time
import subprocess
from collections import defaultdict

def new_mask(data):
    _data = np.abs(data)
    mu = _data.mean(axis=(1, 2, 3), keepdims=True)
    std = _data.std(axis=(1, 2, 3), keepdims=True)
    new_mask = _data >  (mu + std ) 
    return new_mask

def normalize(data, channel=False, sym=True):
    """
    Normalizes each image using the global min and max across all channels.
    Supports single images (m, n, c) or batches (num, m, n, c).
    
    Parameters:
        data (numpy.ndarray): Input image(s) with shape (m, n, c) or (num, m, n, c).
    
    Returns:
        numpy.ndarray: Normalized image(s) with the same shape, scaled to [-1, 1].
    """
    # Ensure input is a 3D or 4D array
    if data.ndim not in (3, 4):
        raise ValueError("Input must be a 3D array (m, n, c) or 4D array (num, m, n, c)")
    
    # Compute min and max across all channels
    if data.ndim == 3:
        if channel:
            # Single image: min/max over all pixels and channels
            xmin = np.min(data, axis=(0, 1), keepdims=True)
            xmax = np.max(data, axis=(0, 1), keepdims=True)
        else:
            xmin = np.min(data, keepdims=True)
            xmax = np.max(data, keepdims=True)
    else:
        if channel:
            # Single image: min/max over all pixels and channels
            xmin = np.min(data, axis=(1, 2), keepdims=True)
            xmax = np.max(data, axis=(1, 2), keepdims=True)
        else:
            xmin = np.min(data, axis=(1, 2, 3), keepdims=True)
            xmax = np.max(data, axis=(1, 2, 3), keepdims=True)
     
    # Avoid division by zero by setting denominator to 1 where max equals min
    denom = xmax - xmin
    # Normalize to [0, 1] per image, then scale to [-1, 1]
    normalized_data = (data - xmin) / denom
    
    if sym:
        normalized_data = normalized_data * 2 - 1
    
    return normalized_data

def simNoise(sqrtSn=1, Tsft=7200, size=(256,128), ndet=2, norm=True, seed=None):
    """
    Generates Gaussian noise in the frequency domain with given parameters.

    Parameters:
        Tsft (float): Total time duration for Fourier transform.
        size (tuple): Shape of each dataset in the form (num_bins, num_segments).
        ndet (int): Number of detector.

    Returns:
        np.ndarray: Generated noise data with shape (num_segments, num_bins, 2*ndet).
    """
    
    if seed is not None:
        np.random.seed(seed)
    
    #num_bins, num_segments = size
    # sigma in time domain = 1 and therefor in complex domain (real + imag), the variance is half of the original one 
    sigma = 0.5 * sqrtSn * np.sqrt(Tsft)  # Variance for real and imaginary parts 

    dataset = np.empty(size + (2*ndet,))
    for i in range(ndet):  # Two time series
        dataset[...,2*i] = np.random.normal(0, sigma, size)
        dataset[...,2*i+1] = np.random.normal(0, sigma, size)
        
    if norm:
        return normalize(dataset)
    else:
        return dataset

def genSampleParam(f0min, f0max, f1min, f1max, nSample):
    """
    Generate parameters for a given number of random samples.

    Parameters:
    - f0min, f0max (float): Frequency range for parameter sampling.
    - f1min, f1max (float): Frequency derivative range for parameter sampling.
    - nSample (int): Number of random samples.

    Returns:
    - params (list): List of parameter combinations for signal generation.
    """
    print("Sample generation.")
    f0_arr = np.random.uniform(f0min, f0max, nSample)
    f1_arr = np.random.uniform(f1min, f1max, nSample)
    phi_arr = np.random.uniform(0, 2*np.pi, nSample)
    psi_arr = np.random.uniform(0, np.pi, nSample)
    cosi_arr = np.random.uniform(-1, 1, nSample)
    
    alpha_arr = np.random.uniform(0, 2*np.pi, nSample)
    sinDelta = np.random.uniform(-1 , 1, nSample)
    delta_arr = np.arcsin(sinDelta)
    
    params = np.column_stack((f0_arr, f1_arr, phi_arr, psi_arr, cosi_arr, alpha_arr, delta_arr))
    return params

import numpy as np

def genSampleParam_skymap(f0min, f0max, f1min, f1max, nSky, nSample):
    """
    Generate parameters for nSample * nSky combinations, with each sky location repeated nSample times
    consecutively, paired with nSample sets of f0, f1, phi, psi, cosi.

    Parameters:
    - f0min, f0max (float): Frequency range for parameter sampling.
    - f1min, f1max (float): Frequency derivative range for parameter sampling.
    - nSky (int): Number of unique sky locations.
    - nSample (int): Number of parameter sets for f0, f1, phi, psi, cosi.

    Returns:
    - params (np.ndarray): Array of shape (nSample * nSky, 7) containing parameter combinations
                          [f0, f1, phi, psi, cosi, alpha, delta], ordered by sky location.
    """
    print("Sample generation.")
    
    total_samples = nSky * nSample
    
    # Generate nSample sets of non-sky parameters
    f0_arr = np.random.uniform(f0min, f0max, nSample)
    f1_arr = np.random.uniform(f1min, f1max, nSample)
    phi_arr = np.random.uniform(0, 2 * np.pi, nSample)
    psi_arr = np.random.uniform(0, np.pi, nSample)
    cosi_arr = np.random.uniform(-1, 1, nSample)
    
    # Tile non-sky parameters to match each sky location
    f0_all = np.tile(f0_arr, nSky)
    f1_all = np.tile(f1_arr, nSky)
    phi_all = np.tile(phi_arr, nSky)
    psi_all = np.tile(psi_arr, nSky)
    cosi_all = np.tile(cosi_arr, nSky)
    
    # Generate nSky unique sky locations
    alpha_arr = np.random.uniform(0, 2 * np.pi, nSky)
    sinDelta = np.random.uniform(-1, 1, nSky)
    delta_arr = np.arcsin(sinDelta)
    
    # Repeat each sky location nSample times consecutively
    alpha_all = np.repeat(alpha_arr, nSample)
    delta_all = np.repeat(delta_arr, nSample)
    
    # Combine into parameter array
    params = np.column_stack((f0_all, f1_all, phi_all, psi_all, cosi_all, alpha_all, delta_all))
    return params


def simSignal(label, homedir, jobID, f0, f1, ra, dec, cosi, psi, phi, obsTime, startTime, det='H1', bandWidth=0.2, Tsft=7200, h0=1, Sn=0, WindowType="tukey", WindowParam=0.01):
    """
    Simulate a CW signal and return its spectrogram.

    Parameters:
    - jobID (int): Unique ID for the signal generation task.
    - h0 (float): Amplitude of the gravitational wave signal.
    - hnoise (float): Noise level of the detector.
    - f0 (float): Frequency of the signal.
    - f1 (float): Frequency derivative of the signal.
    - ra, dec (float): Right ascension and declination of the signal source.
    - cosi, psi, phi (float): Inclination, polarization angle, and initial phase.
    - obsTime (int): Observation time in seconds.
    - startTime (int): Start time for the observation.
    - det (str): Detector name (e.g., 'H1', 'L1').
    - bandWidth (float): Frequency band width.
    - Tsft (int): Time segment length for the SFT.
    - WindowType (str): Window type for SFT.
    - WindowParam (float): Window parameter for SFT.

    Returns:
    - list: [frequency array, timestamp array, Fourier data array].
    """
    simlabel = '{}sim{}{}'.format(label, jobID, det)

    writer_kwargs = {
        "label": simlabel,
        "outdir": homedir+simlabel,
        "tstart": startTime,
        "duration": obsTime,
        "detectors": det,
        "sqrtSX": Sn,
        "Tsft": Tsft,
        "SFTWindowType": WindowType,
        "SFTWindowParam": WindowParam,
    }
    signal_parameters = {
        "F0": f0,
        "F1": f1,
        "Alpha": ra,
        "Delta": dec,
        "h0": h0,
        "cosi": cosi,
        "psi": psi,
        "phi": phi,
        "tref": writer_kwargs["tstart"]+obsTime/2,
        "Band": bandWidth,
    }

    writer = pyfstat.Writer(**writer_kwargs, **signal_parameters)
    writer.make_data()
    frequency, timestamps, fourier_data = get_sft_as_arrays(writer.sftfilepath)

    #print("{} done.".format(jobID))
    command = 'rm -r {}'.format(homedir+simlabel)
    
    result = subprocess.run(command, shell=True, capture_output=True, text=True)
    return [frequency, timestamps, fourier_data]

# Assuming genGridParam, genSampleParam, and simSignal are pre-defined functions
def generate_mock_cw_signals(label, params, obsTime=921600, num_cpus=8, band=0.426, Tsft=7200, h0=1, Sn=0, startTime=1368970000,
                             nSample=800, homedir = '/scratch/kriles_root/kriles0/damoncht/unet_f/'):
    """
    Generate mock continuous wave signals based on input arguments.

    Parameters:
    args (argparse.Namespace): Parsed command-line arguments.
    """
    # Configure logger
    logging.getLogger('pyfstat').setLevel(logging.CRITICAL)

    # Simulate signals for H1 detector
    data1 = simulate_signals(label, homedir, params, obsTime, startTime, band, Tsft, num_cpus, 'H1', h0, Sn)
    
    # Simulate signals for L1 detector
    data2 = simulate_signals(label, homedir, params, obsTime, startTime, band, Tsft, num_cpus, 'L1', h0, Sn)

    return data1, data2

def simulate_signals(label, homedir, params, obsTime, startTime, band, Tsft, num_cpus, detector, h0, Sn):
    """
    Simulate signals using multiprocessing.

    Parameters:
    detector (str): 'H1' or 'L1'.

    Returns:
    dict: Simulated data.
    """
    t0 = time.time()
    with Pool(processes=num_cpus) as pool:
        results = pool.starmap(simSignal, [
            (label, homedir, i, f0, f1, ra, dec, cosi, psi, phi, obsTime, startTime, detector, band, Tsft, h0, Sn)
            for i, (f0, f1, phi, psi, cosi, ra, dec) in enumerate(params)
        ])

    data = defaultdict(list)
    for i, _data in enumerate(results):
        h0_value = params[i][6]  # h0 is the 7th element of params
        data[Sn].append({
            'frequency': _data[0],
            'timestamps': _data[1],
            'fourier_data': _data[2]
        })

    size1, size2 = _data[-1][detector].shape
    data['params'] = params
    return {str(key): value for key, value in data.items()}

# Prepare clean images and masks
def prepare_images_and_masks(signal_data1, signal_data2, num_per_h0, size, threshold):
    mask = np.zeros((num_per_h0, size[0], size[1], 4), dtype=bool)
    clean_image_set = np.zeros((num_per_h0, size[0], size[1], 4))

    for i in tqdm(range(num_per_h0)):
        # Process H1 data
        h1_data = signal_data1[i]['fourier_data']['H1']
        real_h1, imag_h1 = h1_data.real, h1_data.imag
        mask[i, :, :, 0] = np.abs(real_h1) >= threshold
        mask[i, :, :, 1] = np.abs(imag_h1) >= threshold
        clean_image_set[i, :, :, 0] = real_h1
        clean_image_set[i, :, :, 1] = imag_h1

        # Process L1 data
        l1_data = signal_data2[i]['fourier_data']['L1']
        real_l1, imag_l1 = l1_data.real, l1_data.imag
        mask[i, :, :, 2] = np.abs(real_l1) >= threshold
        mask[i, :, :, 3] = np.abs(imag_l1) >= threshold
        clean_image_set[i, :, :, 2] = real_l1
        clean_image_set[i, :, :, 3] = imag_l1
    return clean_image_set, mask

# Generate augmented image set
def generate_augmented_set(random_offsets, clean_image_set, mask, freq_size, num_per_h0, size):
    new_clean_image_set = np.zeros((random_offsets.size, freq_size, size[1], 4))
    new_mask = np.zeros((random_offsets.size, freq_size, size[1], 4), dtype=bool)

    for i in tqdm(range(random_offsets.size)):
        # Find all (y, x, c) coordinates where arr is True
        true_indices = np.argwhere(mask[i])  # Shape (N, 3), where N is the number of True values
        # Find closest freq-bin to top and bottom across all channels
        bottom_freq = np.min(true_indices[:, 0])  # Smallest y (closest to top)
        top_freq = np.max(true_indices[:, 0])  # Largest y (closest to bottom)
        f_band = top_freq - bottom_freq
        offset = freq_size - f_band
        start_idx = bottom_freq-int(random_offsets[i]*offset)

        
        new_clean_image_set[i] = clean_image_set[i, start_idx:start_idx + freq_size, :, :]
        new_mask[i] = mask[i, start_idx:start_idx + freq_size, :, :]

    return new_clean_image_set, new_mask

def crop_signal_img(data_h1, data_l1, freq_size=256, threshold=100):
    h0_values = np.delete(np.unique(np.array(data_h1['params'])[:, -1]), 0)
    num_per_h0 = len(data_h1['0'])
    size = data_h1['0'][0]['fourier_data']['H1'].shape
    # Prepare images and masks
    clean_image_set, mask = prepare_images_and_masks(data_h1['0'], data_l1['0'], num_per_h0, size, threshold)

    random_offsets = np.random.uniform(0, 1, num_per_h0)
    
    # Generate augmented clean images and masks
    new_clean_image_set, new_mask = generate_augmented_set(random_offsets, clean_image_set, mask, freq_size, num_per_h0, size)

    # Save dataset
    dataset = {
        'signal_mask': new_mask,
        'clean_image': new_clean_image_set
    }
    
    return dataset