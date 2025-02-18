import matplotlib.pyplot as plt
import numpy as np
from common_plot_styles import apply_common_plot_styles

# Generated automatically by Co-Pilot, not tested 
def plot_ad_score_contour(dataframes, dataset_tag, column_name, variable_name, x_max=None, score_limit=10000, ylog_scale=False, ax=None):
    """
    Plot a contour plot of AD scores against a specified kinematic variable with filled colors and isolines.

    Args:
        dataframes: Dictionary of dataframes containing the data.
        dataset_tag: String identifying which dataset to plot.
        column_name: Name of the kinematic variable to plot on the x-axis (e.g., 'j0pt').
        variable_name: Label for the x-axis.
        score_limit: Upper limit for AD scores (scores above this will be clipped).
        ax: Optional Matplotlib axis to plot on. If None, a new figure and axis will be created.
        x_max: Optional maximum value for the x-axis.

    Returns:
        fig: The figure object.
        ax: The axis object.
    """
    # Extract the relevant DataFrame
    df = dataframes[dataset_tag]

    # Filter the DataFrame to remove rows with NaN values in the specified columns
    df = df[[column_name, 'HLT_AD_scores']].dropna()

    # Clip AD scores to the specified limit
    df['HLT_AD_scores'] = df['HLT_AD_scores'].clip(upper=score_limit)

    # Check the range of the data
    print(f"Data range for {column_name}: {df[column_name].min()} to {df[column_name].max()}")
    print(f"Data range for AD Scores: {df['HLT_AD_scores'].min()} to {df['HLT_AD_scores'].max()}")

    # Create a grid of values for the contour plot
    x = df[column_name]
    y = df['HLT_AD_scores']

    # Create a 2D histogram to estimate the density of points
    histogram, xedges, yedges = np.histogram2d(x, y, bins=[20, 20])

    # Create the x and y coordinates for the contour plot
    xcenters = 0.5 * (xedges[:-1] + xedges[1:])
    ycenters = 0.5 * (yedges[:-1] + yedges[1:])
    
    # Create a meshgrid for contour plotting
    X, Y = np.meshgrid(xcenters, ycenters)

    # Normalize the histogram to create a density
    Z = histogram.T  # Transpose to match the x and y axes
    Z = Z / Z.max()  # Normalize to [0, 1]

    # Set up the contour plot
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 8))  # Make a square plot
    else:
        fig = ax.figure  # Get the figure from the provided axis

    # Draw filled contours
    contourf = ax.contourf(X, Y, Z, levels=np.linspace(0, 1, 100), cmap='viridis')

    # Define the levels for isolines
    levels = [0.2, 0.4, 0.6, 0.8]

    # Draw contour lines
    contour_lines = ax.contour(X, Y, Z, levels=levels, colors='white', linewidths=1.5)

    # Label the contour lines
    clabels = ax.clabel(contour_lines, inline=True, fontsize=10, colors='white', fmt='%.1f', inline_spacing=2)
    for label in clabels:
        label.set_rotation(0)  # horizontal labels

    # Apply common plot styles
    apply_common_plot_styles(ax)

    # Set labels and title
    object_name = variable_name.replace(" [GeV]", "")
    ax.set_title(f'AD Scores vs. {object_name} ({dataset_tag})', fontsize=16, pad=20)
    ax.set_xlabel(variable_name, fontsize=14, labelpad=20)
    ax.set_ylabel('AD Score', fontsize=14)
    if ylog_scale:  
        plt.yscale('log')

    # Set x-axis limit if x_max is provided
    if x_max is not None:
        ax.set_xlim(0, x_max)

    # Adjust layout to add more space around the plot
    plt.tight_layout()  # Use default padding

    # Return the figure and axis
    return fig, ax