import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from .common_plot_styles import apply_common_plot_styles

apply_common_plot_styles()  # Apply styles globally


def plot_latent_space_1D(data, tag, var, name, nbins=100, x_max=None, ax=None):
    """
    Plot one latent space distribution (LS1 to LS4)

    Overlay the weighted and non-weighted distributions.

    Args:
        df:    DataFrame containing the data
        var:   Column name of the variable to plot (LS1 ... LS4)
        name:  Label for the x-axis
        tag:   Dataset tag to include in the plot (default is 'EB_test')
        bins:  Number of bins for the histogram.
        x_max: Maximum value for the x-axis. If None or Auto, it will be set to the maximum value in the data.
        ax:    Matplotlib axis to use for the plot. If None, a new figure and axis will be created and returned.
    """
import numpy as np
import matplotlib.pyplot as plt

def plot_latent_space_1D(data, tag, var, name, nbins=100, x_max=None, ax=None):
    """
    Plot one latent space distribution (LS1 to LS4)

    Overlay the weighted and non-weighted distributions.

    Args:
        data:  Dictionary of DataFrames containing the data
        tag:   Dataset tag to select DataFrame
        var:   Column name of the variable to plot (LS1 ... LS4)
        name:  Label for the x-axis
        nbins: Number of bins for the histogram
        x_max: Maximum value for the x-axis. If None or "Auto", it will be set to the max value in the data.
        ax:    Matplotlib axis to use for the plot. If None, a new figure and axis will be created and returned.
    """
    df = data[tag]
    
    # Get x-axis range
    x_min = 0  # Start x-axis at zero
    x_max = df[var].max() if x_max in [None, "Auto"] else x_max

    # Set fixed bin edges for proper alignment
    bins = np.linspace(x_min, x_max, nbins + 1)  # nbins+1 ensures correct bin edges

    # Create figure if no axis is provided
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 8), dpi=150)
    else:
        fig = None  # If ax is provided, do not create a new figure

    # Plot unweighted histogram
    ax.hist(df[var], 
            bins=bins, 
            density=True,
            histtype='step', 
            linewidth=1.5,
            color='dodgerblue', 
            label='Unweighted')

    # Plot weighted histogram if weights exist
    if 'weights' in df.columns:
        weights = df['weights']
        weights_norm = weights / df['weights'].sum()
        ax.hist(df[var], 
                bins=bins, 
                weights=weights_norm, 
                histtype='step', 
                linewidth=1.5,
                color='midnightblue', 
                label='Weighted events')

    # Set labels
    ax.set_xlabel(name)
    ax.set_ylabel('Normalized units')

    # Create custom legend with lines (Line2D)
    legend_elements = [
        Line2D([0], [0], color='dodgerblue', lw=1.5, label='Unweighted'),
        Line2D([0], [0], color='midnightblue', lw=1.5, label='Weighted events')
    ]

    # Add legend
    ax.legend(handles=legend_elements, loc='upper right', bbox_to_anchor=(1, 1), borderpad=2)

    plt.tight_layout()
    return fig, ax
