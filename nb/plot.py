import matplotlib.pyplot as plt
import matplotlib
import matplotlib.mlab as mlab
from matplotlib.ticker import MultipleLocator
import matplotlib.colors as colors
from scipy import stats
from PIL import Image
import numpy as np
from scipy.optimize import curve_fit
import warnings

# Matplotlib configuration
plt.rcParams.update({
    'text.usetex': True,
    'axes.linewidth': 1,
    'axes.grid': False,
    'axes.labelweight': 'normal',
    'font.family': 'DejaVu Sans',
    'font.size': 25,
    'mathtext.fontset': 'cm'
})

def binomialError(y, n):
    err =  np.sqrt(y*(1.-y)/n)
    err[err==0] = 1./n
    return err

# Linear interpolation for decreasing y
def interpolate_x(y_target, x, y):
    return np.interp(y_target, y[::-1], x[::-1])

# Estimate second derivative locally
def estimate_second_derivative(x, y, idx):
    h = np.diff(x)
    h_avg = np.mean(h) if not np.allclose(h, h[0]) else h[0]
    # Use points around the interpolation interval (idx-1, idx)
    if idx >= 1 and idx < len(y) - 1:
        # Central difference at x[idx-1] or x[idx]
        deriv2 = (y[idx+1] - 2*y[idx] + y[idx-1]) / (h_avg**2)
        return abs(deriv2)
    elif idx == 0 and len(y) >= 3:
        # Forward difference at x[0]
        deriv2 = (y[2] - 2*y[1] + y[0]) / (h_avg**2)
        return abs(deriv2)
    elif idx == len(y) - 1 and len(y) >= 3:
        # Backward difference at x[-1]
        deriv2 = (y[-1] - 2*y[-2] + y[-3]) / (h_avg**2)
        return abs(deriv2)
    return 1.0  # Fallback if insufficient points

# Error estimation
def interpolation_error(y_target, x, y, dy):
    idx = np.searchsorted(y[::-1], y_target, side='right')
    if idx == 0 or idx == len(y):
        raise ValueError("y_target is outside the range of y-values")
    # Adjust indices for decreasing y
    idx = len(y) - idx  # Convert to original array index
    x0, x1 = x[idx-1], x[idx]
    y0, y1 = y[idx-1], y[idx]
    dy0, dy1 = dy[idx-1], dy[idx]
    h = x1 - x0
    
    # Interpolated x
    x_interp = np.interp(y_target, y[::-1], x[::-1])
    
    # Interpolation error
    second_derivative = estimate_second_derivative(x, y, idx)
    y_interp_error = (1/8) * h**2 * second_derivative
    
    # Slope of the interpolated line
    slope = (y1 - y0) / (x1 - x0)
    
    # Propagate dy uncertainty to x
    t = (y_target - y0) / (y1 - y0)
    y_uncertainty = np.sqrt((1-t)**2 * dy0**2 + t**2 * dy1**2)
    
    # Total x-error
    x_error = np.sqrt((y_interp_error / abs(slope))**2 + (y_uncertainty / abs(slope))**2)
    
    return x_interp, x_error

def plot_spectrograms(title, timestamps, frequency, fourier_data_list, subtitles):
    """
    Plots three spectrograms (power: real^2 + imag^2) in a single figure with shared x-axis and a single colorbar,
    handling zeroed columns (from duty factor) by replacing them with NaN for plotting. NaN columns are rendered
    with a sharp white color for maximum visibility.

    Parameters:
    - title (str): Overall title of the figure.
    - timestamps (ndarray): Array of timestamps corresponding to SFTs (in seconds).
    - frequency (ndarray): Array of frequency values corresponding to the spectrogram's y-axis.
    - fourier_data_list (list of ndarray): List of three 3D arrays (noisy input, denoised output, pure signal).
      Shape of each: (frequency_bins, time_bins, 2) where last dimension is [real, imag].
    - subtitles (list of str): List of three subtitles for each spectrogram.

    Returns:
    - fig (Figure): Matplotlib figure object for the plot.
    - axs (Axes): Array of axes objects for further customization.
    """
    # Validate inputs
    if len(fourier_data_list) != 3 or len(subtitles) != 3:
        raise ValueError("fourier_data_list and subtitles must contain exactly 3 elements")
    expected_shape = fourier_data_list[0].shape
    if not all(data.shape == expected_shape for data in fourier_data_list):
        raise ValueError(f"All arrays in fourier_data_list must have shape {expected_shape}")
    if expected_shape[-1] != 2:
        raise ValueError(f"Last dimension of fourier_data_list arrays must be 2 (real, imag), got {expected_shape[-1]}")
    if len(timestamps) != expected_shape[1]:
        raise ValueError(f"Length of timestamps ({len(timestamps)}) must match time_bins ({expected_shape[1]})")
    if len(frequency) != expected_shape[0]:
        raise ValueError(f"Length of frequency ({len(frequency)}) must match frequency_bins ({expected_shape[0]})")

    # Debugging: Print shape and number of zero columns
    print(f"Input shape: {expected_shape}")
    power_data = fourier_data_list[0][:, :, 0]**2 + fourier_data_list[0][:, :, 1]**2
    zero_columns = np.all(power_data == 0, axis=0)
    print(f"Number of zero columns in data: {np.sum(zero_columns)}")

    # Convert timestamps to days for plotting
    time_in_days = (timestamps - timestamps[0]) / 86400  # Convert seconds to days (86400 s/day)

    # Create figure with three subplots, sharing x-axis
    fig, axs = plt.subplots(3, 1, figsize=(6, 10), sharex=True,
                            gridspec_kw={'height_ratios': [1, 1, 1], 'hspace': 0.05})

    # Copy data to avoid modifying originals
    data, predictions, targets = [np.copy(d) for d in fourier_data_list]

    # Identify zeroed columns in data (all values in the column are zero)
    power_data = data[:, :, 0]**2 + data[:, :, 1]**2
    zero_columns = np.all(power_data == 0, axis=0)

    # Propagate zeroed columns to predictions and targets
    predictions[:, zero_columns, :] = 0
    targets[:, zero_columns, :] = 0

    # Process each dataset to compute power
    processed_powers = []
    for i, data in enumerate([data, predictions, targets]):
        power = data[:, :, 0]**2 + data[:, :, 1]**2  # Power: real^2 + imag^2
        # Replace zeros with NaN for plotting
        power[power == 0] = np.nan
        masked_power = np.ma.masked_invalid(power)  # Mask NaN values
        processed_powers.append(masked_power)
        # Debugging: Print number of NaN columns
        nan_columns = np.any(np.isnan(power), axis=0)
        print(f"Number of NaN columns in dataset {i} ({subtitles[i]}): {np.sum(nan_columns)}")

    # Compute vmin and vmax based on non-masked (non-NaN) data
    non_empty_powers = [power for power in processed_powers if power.compressed().size > 0]
    if not non_empty_powers:
        raise ValueError("All power data is zero or NaN, cannot compute vmin/vmax")
    vmin = min(np.min(power.compressed()) for power in non_empty_powers)
    vmax = max(np.max(power.compressed()) for power in non_empty_powers)
    print(f"vmin: {vmin}, vmax: {vmax}")
    vmin, vmax = 0, 1
    norm = colors.Normalize(vmin=vmin, vmax=vmax)

    # Create colormap and set NaN color to white for sharp visibility
    cmap = plt.get_cmap('magma').copy()
    cmap.set_bad(color='grey')  # White for NaN regions

    # Plot each spectrogram
    for i, (ax, masked_power, subtitle) in enumerate(zip(axs, processed_powers, subtitles)):
        c = ax.pcolormesh(
            time_in_days,
            frequency,
            masked_power,
            cmap=cmap,
            norm=norm,
            shading='auto'
        )

        if i == 1:
            ax.set_ylabel(r'Frequency (Hz)', fontsize=18, color='white')
        
        ax.set_title(subtitle, fontsize=22, loc='right', y=0, color='white')
        if i == len(axs) - 1:
            ax.set_xlabel(r'Time (days)', fontsize=18, color='white')
        ax.tick_params(axis='both', which='major', labelsize=16, colors='white')
        ax.set_yticks([])
        ax.set_yticklabels([])

    # Add a single colorbar below the subplots
    cbar = fig.colorbar(c, ax=axs, orientation='horizontal', pad=0.08, shrink=0.8)
    cbar.set_label(r'Power', fontsize=18)
    cbar.ax.tick_params(labelsize=16)
    return fig, axs

def plot_combined_spectrograms(title, timestamps_list, frequency_list, fourier_data_lists, subtitles, main_titles):
    """
    Plots three spectrograms for each of two frequencies in a 3x2 grid (three rows: noisy input, denoised output, pure signal;
    two columns: different frequencies) with separate colorbars for each column. Zeroed columns are replaced with NaN for plotting,
    rendered with a sharp grey color for visibility. Each subplot has its own x and y labels.

    Parameters:
    - title (str): Overall title of the figure.
    - timestamps_list (list of ndarray): List of two timestamp arrays (in seconds) for each frequency set.
    - frequency_list (list of ndarray): List of two frequency arrays for each frequency set.
    - fourier_data_lists (list of lists): List of two lists, each containing three 3D arrays (noisy input, denoised output, pure signal).
      Shape of each array: (frequency_bins, time_bins, 2) where last dimension is [real, imag].
    - subtitles (list of str): List of three subtitles for each spectrogram (Noisy Signal Input, Denoised Output, Pure Signal).
    - main_titles (list of str): List of two titles for each frequency column (e.g., 'Spectrograms at 20 Hz', 'Spectrograms at 500 Hz').

    Returns:
    - fig (Figure): Matplotlib figure object for the plot.
    - axs (Axes): Array of axes objects for further customization.
    """
    # Validate inputs
    if len(fourier_data_lists) != 2 or len(timestamps_list) != 2 or len(frequency_list) != 2 or len(main_titles) != 2:
        raise ValueError("fourier_data_lists, timestamps_list, frequency_list, and main_titles must contain exactly 2 elements")
    if len(subtitles) != 3:
        raise ValueError("subtitles must contain exactly 3 elements")
    
    for i, (fourier_data_list, timestamps, frequency) in enumerate(zip(fourier_data_lists, timestamps_list, frequency_list)):
        if len(fourier_data_list) != 3:
            raise ValueError(f"fourier_data_lists[{i}] must contain exactly 3 arrays")
        expected_shape = fourier_data_list[0].shape
        if not all(data.shape == expected_shape for data in fourier_data_list):
            raise ValueError(f"All arrays in fourier_data_lists[{i}] must have shape {expected_shape}")
        if expected_shape[-1] != 2:
            raise ValueError(f"Last dimension of fourier_data_lists[{i}] arrays must be 2 (real, imag), got {expected_shape[-1]}")
        if len(timestamps) != expected_shape[1]:
            raise ValueError(f"Length of timestamps[{i}] ({len(timestamps)}) must match time_bins ({expected_shape[1]})")
        if len(frequency) != expected_shape[0]:
            raise ValueError(f"Length of frequency[{i}] ({len(frequency)}) must match frequency_bins ({expected_shape[0]})")

    # Debugging: Print shape and number of zero columns
    for i, fourier_data_list in enumerate(fourier_data_lists):
        print(f"Input shape for {main_titles[i]}: {fourier_data_list[0].shape}")
        power_data = fourier_data_list[0][:, :, 0]**2 + fourier_data_list[0][:, :, 1]**2
        zero_columns = np.all(power_data == 0, axis=0)
        print(f"Number of zero columns in data ({main_titles[i]}): {np.sum(zero_columns)}")

    # Create figure with 3x2 subplots
    fig, axs = plt.subplots(3, 2, figsize=(12, 10), sharex='col',
                            gridspec_kw={'width_ratios': [1, 1], 'height_ratios': [1, 1, 1], 'wspace': 0.15, 'hspace': 0.15})

    # Process each frequency set
    processed_powers = []
    norms = []
    for i, (fourier_data_list, timestamps) in enumerate(zip(fourier_data_lists, timestamps_list)):
        # Convert timestamps to days
        time_in_days = (timestamps - timestamps[0]) / 86400  # Convert seconds to days (86400 s/day)

        # Copy data to avoid modifying originals
        data, predictions, targets = [np.copy(d) for d in fourier_data_list]

        # Identify zeroed columns in data
        power_data = data[:, :, 0]**2 + data[:, :, 1]**2
        zero_columns = np.all(power_data == 0, axis=0)

        # Propagate zeroed columns to predictions and targets
        predictions[:, zero_columns, :] = 0
        targets[:, zero_columns, :] = 0

        # Process each dataset to compute power
        column_powers = []
        for j, data in enumerate([data, predictions, targets]):
            power = data[:, :, 0]**2 + data[:, :, 1]**2  # Power: real^2 + imag^2
            power[power == 0] = np.nan
            masked_power = np.ma.masked_invalid(power)
            column_powers.append(masked_power)
            nan_columns = np.any(np.isnan(power), axis=0)
            print(f"Number of NaN columns in dataset {i},{j} ({main_titles[i]} - {subtitles[j]}): {np.sum(nan_columns)}")
        processed_powers.append(column_powers)

        # Compute vmin and vmax for this column
        non_empty_powers = [power for power in column_powers if power.compressed().size > 0]
        if not non_empty_powers:
            raise ValueError(f"All power data for {main_titles[i]} is zero or NaN, cannot compute vmin/vmax")
        #vmin = min(np.min(power.compressed()) for power in non_empty_powers)
        #vmax = max(np.max(power.compressed()) for power in non_empty_powers)
        #print(f"{main_titles[i]} - vmin: {vmin}, vmax: {vmax}")
        vmin, vmax = 0, 1  # Override as in original code
        #norms.append(colors.Normalize(vmin=vmin, vmax=vmax))

    # Create colormap
    cmap = plt.get_cmap('magma').copy()
    cmap.set_bad(color='grey')
    norm = norm = colors.Normalize(vmin=vmin, vmax=vmax)
    # Plot spectrograms
    for i, (fourier_data_list, frequency, timestamps, main_title) in enumerate(zip(fourier_data_lists, frequency_list, timestamps_list, main_titles)):
        time_in_days = (timestamps - timestamps[0]) / 86400
        for j, (masked_power, subtitle) in enumerate(zip(processed_powers[i], subtitles)):
            ax = axs[j, i]
            c = ax.pcolormesh(
                time_in_days,
                frequency,
                masked_power,
                cmap=cmap,
                norm=norm,
                shading='auto'
            )
                
            if i==0 and j == 1:
                ax.set_ylabel(r'Frequency (Hz)', fontsize=18, color='k' , labelpad=2)
            if j == 2:
                ax.set_xlabel(r'Time (day)', fontsize=18, color='k' , labelpad=1)
            ax.set_title(f"{subtitle}", fontsize=22, loc='right', y=0, color='white')
            ax.tick_params(axis='both', which='major', labelsize=16, colors='k')

    # Add ONE colorbar spanning all 6 axes
    cbar = fig.colorbar(
        c,
        ax=axs.ravel().tolist(),   # flatten (3,2) grid to list
        orientation='horizontal',
        pad=0.08,
        shrink=0.5
    )

    cbar.set_label(r'Power', fontsize=18)
    cbar.ax.tick_params(labelsize=16)

    fig.suptitle(title, fontsize=22, color='k', y=0.92)
    return fig, axs