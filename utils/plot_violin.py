import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from .common_plot_styles import apply_common_plot_styles

def plot_AD_scores_violin(dataframes, dataset_tags=None, score_limit=1000, ylog=False):
    """
    Create a violin plot comparing AD score distributions across specified datasets.
    
    Args:
        dataframes: Dictionary of dataframes.
        dataset_tags: List of dataset tags to include in the plot. If None, include all datasets.
        score_limit: Upper limit for y-axis (scores above this will be clipped).
        ylog: Boolean indicating whether to use a logarithmic scale for the y-axis.
    """
    
    # Define a color palette
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', 
              '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf',
              '#aec7e8', '#ffbb78', '#98df8a', '#ff9896']
    
    # Prepare data for violin plot
    original_scores = []  # Store original scores for statistics
    clipped_scores = []  # Store clipped scores for visualization
    labels = []  # Stores the dataset tags in the order they are added
    over_score_limit_percentages = []  # Stores the percentage of events above the score limit for each dataset

    # Ensure EB_test is first if it exists
    if 'EB_test' in dataframes:
        scores = dataframes['EB_test']['HLT_AD_scores']
        original_scores.append(scores)
        clipped_scores.append(np.clip(scores, 0, score_limit))
        labels.append('EB_test')
        over_score_limit_percentages.append((scores > score_limit).mean() * 100)
    
    # Filter datasets based on dataset_tags
    if dataset_tags is None:
        dataset_tags = dataframes.keys()
    
    # Add datasets to the plot
    for dataset_tag in dataset_tags:
        if dataset_tag in dataframes and dataset_tag != 'EB_test':
            scores = dataframes[dataset_tag]['HLT_AD_scores']
            original_scores.append(scores)
            clipped_scores.append(np.clip(scores, 0, score_limit))
            labels.append(dataset_tag)
            over_score_limit_percentages.append((scores > score_limit).mean() * 100)
    
    # Create figure and axis
    fig, ax = plt.subplots(figsize=(16, 8), dpi=150)
    violin_parts = ax.violinplot(clipped_scores, showextrema=True, showmedians=True)
    
    # Customize violin appearance with different colors
    for i, (pc, color) in enumerate(zip(violin_parts['bodies'], colors)):
        pc.set_facecolor(color)
        pc.set_alpha(0.7)
        
        # Also color the lines (median, quartiles, whiskers)
        if i < len(original_scores):  # Make sure we don't exceed the number of datasets
            violin_parts['cbars'].set_color(colors[i])    # Whisker color
            violin_parts['cmins'].set_color(colors[i])    # Min line color
            violin_parts['cmaxes'].set_color(colors[i])   # Max line color
            violin_parts['cmedians'].set_color('white')   # Keep median white for visibility
    
    # Customize plot
    if ylog:
        ax.set_yscale('log')
    ax.set_ylabel('HLT AD Scores')
    ax.set_title('Distribution of HLT AD Scores Across Datasets', y=1.05, fontsize=18)
    
    # Set x-ticks with dataset labels
    ax.set_xticks(range(1, len(labels) + 1))
    ax.set_xticklabels(labels, rotation=45, ha='right')
    
    # Add grid for y-axis only
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Add percentage annotations above each violin
    for i, (dataset_tag, percentage) in enumerate(zip(labels, over_score_limit_percentages), 1):
        ax.text(i, score_limit * 1.05, 
                f'{percentage:.2f}% > {score_limit}',
                ha='center', va='bottom',
                fontsize=10,
                color=colors[i-1])  # Match text color to violin color
    
    # Adjust y-limits to add space at top and bottom
    if ylog:
        ax.set_ylim(0.01, score_limit * 10) 
    else:
        ax.set_ylim(-score_limit * 0.02, score_limit * 1.15)  # Start slightly below 0
    
    # Adjust layout to prevent label cutoff
    plt.tight_layout()
    
    plt.show()
    
    return fig, ax  # Return the figure and axis objects
    """
    # Print statistics for each dataset
    print("\nAD Score Statistics:")
    for dataset_tag in labels:
        scores = dataframes[dataset_tag]['HLT_AD_scores']
        print(f"\n{dataset_tag}:")
        print(f"Mean: {scores.mean():.4f}")
        print(f"Median: {np.median(scores):.4f}")
        print(f"Std: {scores.std():.4f}")
        print(f"% above {score_limit}: {(scores > score_limit).mean()*100:.2f}%")
    """