import matplotlib.pyplot as plt
from matplotlib import colors
from scipy import stats
from PIL import Image
# Matplotlib configuration for consistent, readable plots
plt.rcParams["font.family"] = "serif"
plt.rcParams["font.size"] = 20
import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
import warnings


def plot_spectrograms(title, timestamps, frequency, fourier_data1):
    """
    Plots a single spectrogram.

    Parameters:
    - title (str): Title of the spectrogram plot.
    - timestamps (ndarray): Array of timestamps corresponding to SFTs.
    - frequency (ndarray): Array of frequency values corresponding to the spectrogram's y-axis.
    - fourier_data1 (ndarray): 3D array representing the spectrogram data. 
      Shape: (frequency_bins, time_bins, 2) where last dimension represents real and imaginary parts.

    Returns:
    - fig (Figure): Matplotlib figure object for the plot.
    - axs (Axes): List of axes objects for further customization.
    """
    # Create the plot
    fig, axs = plt.subplots(1, 1, figsize=(16, 10))  # Single subplot
    axs = [axs]  # Ensure axs is a list for consistent handling

    # Set labels for axes
    for ax in axs:
        ax.set(xlabel="SFT index", ylabel="Frequency index")
    
    # Set the title of the plot
    axs[0].set_title(title)
    
    # Plot the spectrogram
    c1 = axs[0].pcolormesh(
        timestamps,
        frequency,
        fourier_data1[:, :, 0] + fourier_data1[:, :, 1]**2,  # Sum of squares of components
        cmap="viridis",
        norm=colors.CenteredNorm(),  # Center color normalization
        shading='auto'
    )
    fig.colorbar(c1, ax=axs[0], orientation="horizontal", label="Power")  # Add colorbar for the plot
    
    return fig, axs
