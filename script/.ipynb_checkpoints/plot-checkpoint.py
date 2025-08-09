import matplotlib.pyplot as plt
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

# def plot_spectrograms(title, timestamps, frequency, fourier_data1):
#     """
#     Plots a single spectrogram.

#     Parameters:
#     - title (str): Title of the spectrogram plot.
#     - timestamps (ndarray): Array of timestamps corresponding to SFTs.
#     - frequency (ndarray): Array of frequency values corresponding to the spectrogram's y-axis.
#     - fourier_data1 (ndarray): 3D array representing the spectrogram data. 
#       Shape: (frequency_bins, time_bins, 2) where last dimension represents real and imaginary parts.

#     Returns:
#     - fig (Figure): Matplotlib figure object for the plot.
#     - axs (Axes): List of axes objects for further customization.
#     """
#     # Create the plot
#     fig, axs = plt.subplots(1, 1, figsize=(16, 10))  # Single subplot
#     axs = [axs]  # Ensure axs is a list for consistent handling

#     # Set labels for axes
#     for ax in axs:
#         ax.set(xlabel="SFT index", ylabel="Frequency index")
    
#     # Set the title of the plot
#     axs[0].set_title(title)
    
#     # Plot the spectrogram
#     c1 = axs[0].pcolormesh(
#         timestamps,
#         frequency,
#         fourier_data1[:, :, 0] + fourier_data1[:, :, 1]**2,  # Sum of squares of components
#         cmap="viridis",
#         norm=colors.CenteredNorm(),  # Center color normalization
#         shading='auto'
#     )
#     fig.colorbar(c1, ax=axs[0], orientation="horizontal", label="Power")  # Add colorbar for the plot
    
#     return fig, axs
