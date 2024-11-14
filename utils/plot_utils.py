import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

# Set plotting style at module level
plt.rcParams.update({
    # Font sizes
    'font.size': 18,
    'axes.labelsize': 18,
    'axes.titlesize': 18,
    'xtick.labelsize': 16,
    'ytick.labelsize': 16,
    'legend.fontsize': 16,
    'legend.frameon': False,  # No box around legend
    'axes.grid': False, 
    # Tick settings
    'xtick.direction': 'in',
    'ytick.direction': 'in',
    'xtick.major.size': 10,
    'ytick.major.size': 10,
    'xtick.minor.size': 5,
    'ytick.minor.size': 5,
    'xtick.major.width': 1,
    'ytick.major.width': 1,
    'xtick.top': True,
    'ytick.right': True,
    'xtick.minor.visible': True,
    'ytick.minor.visible': True
})




def plot_event_2D(df, event_idx):
    """
    Plot physics objects as 2D vectors in the transverse plane for a given event.
    All vectors start from (0,0) and point outwards.
    The plot is perfectly square to represent the circular detector.
    """
    fig = plt.figure(figsize=(12, 12), dpi=150)
    ax = fig.add_subplot(111)
    
    # Dictionary to store different object types and their plotting styles
    obj_styles = {
        'j': {'color': 'royalblue', 'label': 'Jets', 'width': 0.003},
        'e': {'color': 'red', 'label': 'Electrons', 'width': 0.003},
        'mu': {'color': 'green', 'label': 'Muons', 'width': 0.003},
        'ph': {'color': 'orange', 'label': 'Photons', 'width': 0.003},
        'MET': {'color': 'black', 'label': 'MET', 'width': 0.005}
    }
    
    max_pt = 0
    legend_elements = []  # Store legend entries
    
    # Plot each object type
    for prefix, style in obj_styles.items():
        # Get relevant columns for this object type
        if prefix == 'MET':
            pt_col = 'METpt'
            phi_col = 'METphi'
            cols = [pt_col] if pt_col in df.columns else []
        else:
            cols = sorted([col for col in df.columns if col.startswith(prefix) and col.endswith('pt')])
        
        # Store object info for this type
        object_info = []
        
        for i, pt_col in enumerate(cols):
            phi_col = pt_col.replace('pt', 'phi')
            
            if phi_col not in df.columns:
                continue
                
            pt = df.loc[event_idx, pt_col]
            phi = df.loc[event_idx, phi_col]
            
            # Skip if pt is 0 or 0.001 (for MET)
            if pt <= 0.001:
                continue
                
            object_info.append((pt, phi))
            max_pt = max(max_pt, pt)
            
            # Calculate x and y components
            x = pt * np.cos(phi)
            y = pt * np.sin(phi)
            
            # Plot vector from origin
            ax.quiver(0, 0, x, y, angles='xy', scale_units='xy', scale=1,
                     color=style['color'], width=style['width'])
        
        # Add main legend entry if objects of this type exist
        if object_info:
            legend_elements.append(plt.Line2D([0], [0], color=style['color'], 
                                           label=style['label']))
            # Add individual pT and phi values
            for pt, phi in sorted(object_info, reverse=True):  # Sort by pT
                legend_elements.append(plt.Line2D([0], [0], color=style['color'],
                                               linestyle='', 
                                               label=f'    $p_T$ = {pt:.1f} GeV, $\phi$ = {phi:.3f}'))
    
    # Rest of the plot settings
    ax.set_aspect('equal', adjustable='box')
    limit = max_pt * 1.2
    ax.set_xlim(-limit, limit)
    ax.set_ylim(-limit, limit)
    
    # Add grid and axes through (0,0)
    ax.grid(True, linestyle='--', alpha=0.3)
    ax.axhline(y=0, color='k', linestyle='-', alpha=0.2)
    ax.axvline(x=0, color='k', linestyle='-', alpha=0.2)
    
    # Add circular guidelines with round numbers
    max_radius = int(np.ceil(max_pt))  # Round up to nearest integer
    # If max_pt would create more than 10 circles with 10 GeV steps, use 20 GeV steps instead
    step = 20 if max_radius > 100 else 10
    circles = np.arange(step, max_radius + step, step)  # Create array of round numbers
    
    for radius in circles:
        circle = plt.Circle((0, 0), radius, fill=False, linestyle='--', 
                          alpha=0.2, color='gray')
        ax.add_artist(circle)
        ax.text(radius*np.cos(np.pi/4), radius*np.sin(np.pi/4), 
                f'{radius:.0f} GeV', fontsize=8, alpha=0.5)
    
    ax.set_xlabel(r'$p_T \,\cos(\phi)$ [GeV]')
    ax.set_ylabel(r'$p_T \,\sin(\phi)$ [GeV]')
    ax.set_title(f'Event {event_idx}: Transverse Plane View')
    
    # Add legend with lines instead of rectangles
    ax.legend(handles=legend_elements, bbox_to_anchor=(1.05, 1), loc='upper left')
    
    plt.tight_layout()
    
    return fig, ax

# For the AD score plotting functions below, printing statistics after calling the plot function
def print_statistics(dataset_tag, variable_values, variable_name, has_weights=True, weight_values=None):
    """
    Print statistics for a given variable
    """
    print(f"\n{variable_name}: Statistics for {dataset_tag}:")

    # Create weights if not provided
    if not has_weights or weight_values is None:
        weight_values = np.ones(len(variable_values))

    weighted_mean = np.average(variable_values, weights=weight_values)
    weighted_std = np.sqrt(np.average((variable_values - weighted_mean)**2, weights=weight_values))
    
    # Print results with unweighted values only if using weights
    if has_weights:
        print(f"Mean: {weighted_mean:.4f}  (unweighted: {variable_values.mean():.4f})")
        print(f"Std: {weighted_std:.4f}  (unweighted: {variable_values.std():.4f})")
    else:
        print(f"Mean: {weighted_mean:.4f}")
        print(f"Std: {weighted_std:.4f}")
    print(f"Min: {variable_values.min():.4f}")
    print(f"Max: {variable_values.max():.4f}")



def plot_AD_scores(dataframes, dataset_tag, nbins= 50, score_bin_overflow=1000, use_weights=True, ylog_scale=False):
    """
    Plot AD scores distribution for a given dataset
    
    Args:
        dataframes: Dictionary of dataframes
        dataset_tag: String identifying which dataset to plot (e.g., 'EB_test')
        use_weights: Boolean, whether to use event weights (default: True)
    """
    plt.figure(figsize=(8, 8))
    
    # Get data
    HLT_AD_scores = dataframes[dataset_tag]['HLT_AD_scores']
    weights_array = np.ones_like(HLT_AD_scores) if not use_weights else dataframes[dataset_tag]['weights']
    total_weights = np.sum(weights_array)
    weights_array_normalized = weights_array / total_weights

    # Create bins with overflow
    #bins = np.linspace(0, score_bin_overflow, nbins)  # Changed to nbins instead of nbins + 1
    #bins = np.append(bins, np.inf)  # Use infinity for the last bin edge

    bin_width = score_bin_overflow / (nbins - 1)
    bins = np.linspace(0, score_bin_overflow + bin_width, nbins + 1)
    
    # Plot histogram
    plt.hist(HLT_AD_scores, 
             bins=bins, 
             histtype='step', 
             label=dataset_tag,
             weights=weights_array_normalized,
             linewidth=1.5)
    
    # Customize plot
    plt.xlabel('HLT AD Score')
    plt.ylabel('Normalized Counts')
    if ylog_scale:  
        plt.yscale('log')
    unweight_text = ' (non-weighted)' if not use_weights else ''
    plt.suptitle(f'Distribution of HLT AD Scores in {dataset_tag} {unweight_text}')

    # Modify tick labels
    ax = plt.gca()
    ax.set_xlim(0, score_bin_overflow * 1.1)  # Set x-axis limit slightly beyond overflow threshold
    
    # Get current ticks and create new labels
    xticks = ax.get_xticks()
    xticks = xticks[xticks <= score_bin_overflow * 1.1]  # Only keep ticks up to our limit
    ax.set_xticks(xticks)
    
    # Create labels with overflow bin
    xticks_labels = [f'{int(x)}' if x < score_bin_overflow else f'>{score_bin_overflow}' 
                     for x in xticks]
    ax.set_xticklabels(xticks_labels)
    
    plt.show()

    # Print statistics
    print_statistics(dataset_tag, HLT_AD_scores, 'HLT AD Scores', use_weights, weights_array)

    # Get events above overflow bin
    df_above = dataframes[dataset_tag][dataframes[dataset_tag]['HLT_AD_scores'] > score_bin_overflow]
    n_above = len(df_above)  # Raw number of events
    weights_above = df_above['weights'].sum()  # Sum of weights for events above

    # Print results
    print(f"\nPercentage of raw events above HLT_AD_score of {score_bin_overflow}: {(n_above/len(HLT_AD_scores))*100:.2f}% (Check:{n_above}/{len(HLT_AD_scores)})")
    if use_weights:
        print(f"Percentage of weighted events above HLT_AD_score of {score_bin_overflow}: {(weights_above/total_weights)*100:.4f}% (Check:{weights_above:.2f}/{total_weights:.2f})")
    

###########  AD score for multiple datasets ###########

def overlay_AD_scores(dataframes, dataset_tags, nbins=50, score_bin_overflow=1000, use_weights=True, ylog_scale=False):
    """
    Plot overlaid AD scores distributions for multiple datasets
    
    Args:
        dataframes: Dictionary of dataframes
        dataset_tags: List of strings identifying which datasets to plot
        use_weights: Boolean, whether to use event weights (default: True)
    """
    # Define color palette (same as violin plot)
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', 
              '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
    
    plt.figure(figsize=(14, 8))  # Wider to accommodate legend

    datasets = []
    
    # Process EB_test first if it's in the dataset_tags
    if 'EB_test' in dataset_tags:
        datasets.append('EB_test')
    
    # Then add all other datasets
    for dataset_tag in dataset_tags:
        if dataset_tag != 'EB_test':
            datasets.append(dataset_tag)

    
    
    # Create bins with overflow
    bin_width = score_bin_overflow / (nbins - 1)
    bins = np.linspace(0, score_bin_overflow + bin_width, nbins + 1)
    
    # Plot histograms for each dataset
    for idx, tag in enumerate(datasets):
        if tag == 'jjJZ1':
            continue
        scores = dataframes[tag]['HLT_AD_scores']
        weights = np.ones_like(scores) if not use_weights else dataframes[tag]['weights']
        weights_normalized = weights / np.sum(weights)
        
        plt.hist(scores, 
                bins=bins, 
                histtype='step', 
                label=tag,
                weights=weights_normalized,
                linewidth=3 if tag=='EB_test' else 1.5,  # Thicker line for EB_test
                color=colors[idx])
    
    # Customize plot
    plt.xlabel('HLT AD Score')
    plt.ylabel('Normalized Counts')
    if ylog_scale:
        plt.yscale('log')
    unweight_text = ' (non-weighted)' if not use_weights else ''
    plt.title(f'Distribution of HLT AD Scores{unweight_text}')
    
    # Modify tick labels
    ax = plt.gca()
    ax.set_xlim(0, score_bin_overflow * 1.1)
    xticks = ax.get_xticks()
    xticks = xticks[xticks <= score_bin_overflow * 1.1]
    ax.set_xticks(xticks)
    xticks_labels = [f'{int(x)}' if x < score_bin_overflow else f'>{score_bin_overflow}' 
                     for x in xticks]
    ax.set_xticklabels(xticks_labels)
    
    # Create custom legend with lines
    custom_lines = [Line2D([0], [0], color=colors[i], lw=3 if datasets[i]=='EB_test' else 1.5) 
                   for i in range(len(datasets))]
    
    # Add legend with custom lines
    plt.legend(custom_lines, datasets,
              loc='upper right',
              frameon=False)
    
    # Adjust layout to prevent legend cutoff
    plt.tight_layout()
    
    plt.show()



def plot_AD_scores_4_parts(dataframes, dataset_tag, nbins=50, use_weights=True):
    """
    Plot AD scores distribution in 4 ranges
    """
    # Create figure with 4 subplots
    fig, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4, figsize=(24, 6))
    
    # Get data and set up weights
    HLT_AD_scores = dataframes[dataset_tag]['HLT_AD_scores']
    if use_weights:
        weights_array = dataframes[dataset_tag]['weights']
    else:
        weights_array = np.ones_like(HLT_AD_scores)
    weights_array_normalized = weights_array / np.sum(weights_array)

    total_weights = np.sum(weights_array)
    
    # Plot for different ranges
    ###########################
    #     0-10
    ###########################
    bins1 = np.linspace(0, 10, nbins)
    ax1.hist(HLT_AD_scores, bins=bins1, histtype='step', weights=weights_array_normalized, linewidth=1.5)
    ax1.set_xlabel('HLT AD Score')
    ax1.set_ylabel('Normalized Counts')
    ax1.set_title('0 < AD Score < 10')

    # Print the weighted number of events in this range on the top right of the plot (rounded up to next integer)
    n_events_range1 = np.sum(weights_array[(HLT_AD_scores >= 0) & (HLT_AD_scores <= 10)])
    percentage_range1 = n_events_range1 / total_weights
    ax1.text(0.95, 0.95, f'{int(np.ceil(n_events_range1)):,} events ({percentage_range1*100:.2f}%) ', transform=ax1.transAxes, ha='right', va='top', fontsize=14)

    print(f'Test: n_events_range1 = {n_events_range1} | raw events = {np.sum((HLT_AD_scores >= 0) & (HLT_AD_scores <= 10))}')

    ##########################
    #     10-100
    ##########################
    bins2 = np.logspace(np.log10(10), np.log10(100), nbins)
    ax2.hist(HLT_AD_scores, bins=bins2, histtype='step', weights=weights_array_normalized, linewidth=1.5)
    ax2.set_xlabel('HLT AD Score')
    ax2.set_xscale('log')
    ax2.set_title('10 < AD Score < 100')

    # Print the weighted number of events in this range on the top right of the plot (rounded up to next integer)
    n_events_range2 = np.sum(weights_array[(HLT_AD_scores >= 10) & (HLT_AD_scores <= 100)])
    percentage_range2 = n_events_range2 / total_weights
    ax2.text(0.95, 0.95, f'{int(np.ceil(n_events_range2)):,} events ({percentage_range2*100:.3f}%) ', transform=ax2.transAxes, ha='right', va='top', fontsize=14)

    print(f'Test: n_events_range2 = {n_events_range2} | raw events = {np.sum((HLT_AD_scores >= 10) & (HLT_AD_scores <= 100))}')

    ###########################
    #     100-1000
    ###########################
    bins3 = np.linspace(100, 1000, nbins)
    ax3.hist(HLT_AD_scores, bins=bins3, histtype='step', weights=weights_array_normalized, linewidth=1.5)
    ax3.set_xlabel('HLT AD Score')
    ax3.set_title('100 < AD Score < 1000')

    # Print the weighted number of events in this range on the top right of the plot (rounded up to next integer)
    n_events_range3 = np.sum(weights_array[(HLT_AD_scores >= 100) & (HLT_AD_scores <= 1000)])
    percentage_range3 = n_events_range3 / total_weights 
    ax3.text(0.95, 0.95, f'{int(np.ceil(n_events_range3)):,} events ({percentage_range3*100:.3f}%) ', transform=ax3.transAxes, ha='right', va='top', fontsize=14)

    print(f'Test: n_events_range3 = {n_events_range3} | raw events = {np.sum((HLT_AD_scores >= 100) & (HLT_AD_scores <= 1000))}')
    
    ###########################     
    #     >1000
    ###########################
    bins4 = np.linspace(1000, max(HLT_AD_scores), nbins)
    ax4.hist(HLT_AD_scores, bins=bins4, histtype='step', weights=weights_array_normalized, linewidth=1.5)
    ax4.set_xlabel('HLT AD Score')
    ax4.set_title('AD Score > 1000')

    # Print the weighted number of events in this range on the top right of the plot (rounded up to next integer)
    n_events_range4 = np.sum(weights_array[(HLT_AD_scores > 1000)])
    percentage_range4 = n_events_range4 / total_weights
    ax4.text(0.95, 0.95, f'{int(np.ceil(n_events_range4)):,} events ({percentage_range4*100:.4f}%) ', transform=ax4.transAxes, ha='right', va='top', fontsize=14)      

    print(f'Test: n_events_range4 = {n_events_range4} | raw events = {np.sum(HLT_AD_scores > 1000)}')

    # Format x-axis for last plot with scientific notation
    ax4.ticklabel_format(axis='x', style='sci', scilimits=(0,0), useMathText=True)
    ax4.xaxis.offsetText.set_position((1, -0.1))  # Position the multiplier
    
    # Mention if non-weighted
    if not use_weights:
        non_weighted_text = ' (non-weighted)'
    else:
        non_weighted_text = ''          

    # Add main title
    plt.suptitle(f'Distribution of HLT AD Scores in {dataset_tag} {non_weighted_text}', y=1.05, fontsize=20)
    
    # Adjust layout
    plt.tight_layout()

    # Show plot
    plt.show()
    
    # Print statistics
    print_statistics(dataset_tag, HLT_AD_scores, 'HLT AD Scores', use_weights, weights_array)


def plot_AD_scores_violin(dataframes, score_limit=1000, ylog=False):
    """
    Create a violin plot comparing AD score distributions across all datasets
    
    Args:
        dataframes: Dictionary of dataframes
        score_limit: Upper limit for y-axis (scores above this will be clipped)
    """
    
    # Define a color palette
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', 
              '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
    
    # Prepare data for violin plot, ensuring EB_test is first
    original_scores = []  # Store original scores for statistics
    clipped_scores = []  # Store clipped scores for visualization
    labels = [] # stores the dataset tags in the order they are added
    over_score_limit_percentages = [] # stores the percentage of events above the score limit for each dataset

        # First add EB_test if it exists
    if 'EB_test' in dataframes:
        scores = dataframes['EB_test']['HLT_AD_scores']
        original_scores.append(scores)
        clipped_scores.append(np.clip(scores, 0, score_limit))
        labels.append('EB_test')
        over_score_limit_percentages.append((scores > score_limit).mean() * 100)
    
    # Then add all other datasets
    for dataset_tag in dataframes.keys():
        if dataset_tag != 'EB_test':
            scores = dataframes[dataset_tag]['HLT_AD_scores']
            original_scores.append(scores)
            clipped_scores.append(np.clip(scores, 0, score_limit))
            labels.append(dataset_tag)
            over_score_limit_percentages.append((scores > score_limit).mean() * 100)
    
    # Create violin plot
    plt.figure(figsize=(12, 8), dpi=150)
    violin_parts = plt.violinplot(clipped_scores, showextrema=True, showmedians=True)
    
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
        plt.yscale('log')
    plt.ylabel('HLT AD Scores')
    plt.title('Distribution of HLT AD Scores Across Datasets', y=1.05, fontsize=18)
    
    # Set x-ticks with dataset labels
    plt.xticks(range(1, len(labels) + 1), labels, rotation=45)
    
    # Add grid for y-axis only
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Add percentage annotations above each violin
    for i, (dataset_tag, percentage) in enumerate(zip(labels, over_score_limit_percentages), 1):
        plt.text(i, score_limit * 1.05, 
                f'{percentage:.2f}% > {score_limit}',
                ha='center', va='bottom',
                fontsize=10,
                color=colors[i-1])  # Match text color to violin color
    
    # Adjust y-limits to add space at top and bottom
    plt.ylim(-score_limit * 0.02, score_limit * 1.15)  # Start slightly below 0
    
    # Adjust layout to prevent label cutoff
    plt.tight_layout()
    
    plt.show()

    # Create statistics DataFrame 
    stats_dict = {}
    for scores, dataset_tag in zip(original_scores, labels):
        stats_dict[dataset_tag] = {
            'Mean': np.mean(scores),
            'Median': np.median(scores),
            'Std Dev': np.std(scores),
            'Min': np.min(scores),
            'Max': np.max(scores),
            f'% above {score_limit}': (dataframes[dataset_tag]['HLT_AD_scores'] > score_limit).mean() * 100  # Need original unclipped data for this
        }
    
    # Convert to DataFrame and format
    stats_df = pd.DataFrame(stats_dict).round(2)
    
    print("\nAD Score Statistics:")
    print(stats_df.to_string())


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

def plot_event_weights(dataframes, dataset_tag, weight_limit = 5000, nbins=100):
    """
    Plot event weights distribution for EB_[test | val | train] datasets 
    """
    
    evt_weights = dataframes[dataset_tag]['weights']

    plt.figure(figsize=(8, 8))
    plt.hist(evt_weights, bins=nbins, range=(0, weight_limit), histtype='step', linewidth=1.5)
    plt.xlabel('Event Weights')
    plt.ylabel('Counts')
    plt.title(f'Distribution of Event Weights in {dataset_tag}')
    plt.show()

    
    print("\nWeight Statistics:")
    print(f"Mean weight: {evt_weights.mean():.1f}")
    print(f"Median weight: {evt_weights.median():.1f}")
    print(f"Min weight: {evt_weights.min():.1f}")
    print(f"Max weight: {evt_weights.max():.1f}")

    print(f"\nNumber of events: {len(evt_weights)}")
    print(f"Number of events with weight > {weight_limit}: {(evt_weights > weight_limit).sum()}")
    print(f"Percentage of events with weight > {weight_limit}: {(evt_weights > weight_limit).sum() / len(evt_weights) * 100:.2f}%")



# DEPRECATED: TWO RANGES ARE ON DIFFERENT SUBPLOTS. PUT ON SAME PLOT, SEE NEXT FUNCTION 
def DEPRECATED_overlay_variable_by_AD_range(dataframes, dataset_tags, column_name, variable_name, 
                               range_AD_normal, range_AD_anomalous, nbins=50):
    """
    Plot overlaid distributions for a variable, split by AD score ranges: low (normal events) or high (anomalous events)
    
    Args:
        dataframes: Dictionary of dataframes
        dataset_tags: List of dataset tags (EB_test should be first)
        column_name: Name of column to plot
        variable_name: Label for x-axis
        range_AD_normal: Tuple of (min, max) for normal AD scores
        range_AD_anomalous: Tuple of (min, max) for anomalous AD scores
        nbins: Number of bins for histogram
    """
    # Define color palette (same as other plots)
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', 
              '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
    
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 16))
    
    # Process each AD score range
    for ax, (ad_range, title) in zip([ax1, ax2], 
                                    [(range_AD_normal, "Low AD scores"), 
                                     (range_AD_anomalous, "High AD scores")]):
        
        # Plot for each dataset
        for idx, tag in enumerate(dataset_tags):
            df = dataframes[tag]
            
            # Select events in AD score range
            mask = (df['HLT_AD_scores'] >= ad_range[0]) & (df['HLT_AD_scores'] <= ad_range[1])
            selected_data = df[mask][column_name]
            selected_weights = df[mask]['weights']
            
            # Normalize weights
            weights_normalized = selected_weights / selected_weights.sum()
            
            # Plot histogram
            ax.hist(selected_data, 
                   bins=nbins,
                   weights=weights_normalized,
                   histtype='step',
                   label=tag,
                   linewidth=4 if tag=='EB_test' else 2,
                   color=colors[idx])
        
        # Customize subplot
        ax.set_xlabel(variable_name)
        ax.set_ylabel('Normalized Events')
        
        # Add legend with lines
        from matplotlib.lines import Line2D
        custom_lines = [Line2D([0], [0], color=colors[i], 
                             lw=4 if tag=='EB_test' else 2) 
                       for i, tag in enumerate(dataset_tags)]
        
        ax.legend(custom_lines, dataset_tags,
                 title=title,
                 loc='upper right',
                 frameon=False)
        
        # Scientific notation for y-axis if needed
        ax.ticklabel_format(axis='y', style='sci', scilimits=(-2,3))
    
    plt.tight_layout()
    plt.show()








def overlay_kin_variable_2_AD_ranges(dataframes, dataset_tags, column_name, variable_name, 
                               range_AD_normal, range_AD_anomalous, 
                               remove_zero_entries=False, nbins=50, x_max=None, y_max_factor=None, ylog=False):
    """
    Plot overlaid distributions for a variable, comparing low/high AD score ranges
    
    Args:
        dataframes: Dictionary of dataframes
        dataset_tags: List of dataset tags (EB_test should be first)
        column_name: Name of column to plot
        variable_name: Label for x-axis
        range_AD_normal: Tuple of (min, max) for normal AD scores
        range_AD_anomalous: Tuple of (min, max) for anomalous AD scores
        remove_zero_entries: If True, exclude events where variable is > zero (for pT) or within epsilon (for eta/phi)
        nbins: Number of bins for histogram
        x_max: Maximum value for x-axis (if None, use data range)
        y_max_factor: Scaling coefficient for the maximum value for y-axis (beautifying plot with no cutting legend)
        ylog: If True, use logarithmic scale for y-axis
    """
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', 
              '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
    
    # Adjust figure size based on variable type
    if 'phi' in variable_name.lower() or 'eta' in variable_name.lower():
        plt.figure(figsize=(8, 8))  # Square for angular variables
    else:
        plt.figure(figsize=(10, 8))  # Rectangle for other variables
    
    # Order datasets
    datasets = []
    if 'EB_test' in dataset_tags:
        datasets.append('EB_test')
    for dataset_tag in dataset_tags:
        if dataset_tag != 'EB_test':
            datasets.append(dataset_tag)
            
    # Set x axis limits and bins
    is_eta = 'eta' in variable_name.lower()
    is_phi = 'phi' in variable_name.lower()

    if is_eta:
        bins = np.linspace(-3.8, 3.8, nbins+1)
    elif is_phi:
        bins = np.linspace(-4.9, 4.9, nbins+1)
    else:  # pt variable
        if x_max is not None:
            bins = np.linspace(0, x_max, nbins+1)
        else:
            # Find maximum value across all datasets
            xmax_val = 0
            for tag in datasets:
                df = dataframes[tag]
                xmax_val = max(xmax_val, df[column_name].max())
            bins = np.linspace(0, xmax_val, nbins+1)
    
    """
    ############## not working yet 
    # Set y axis limits 
    if y_max_factor is not None:
        # Find the maximum value across all datasets
        ymax_val = 0    
        for tag in datasets:
            df = dataframes[tag]
            ymax_val = max(ymax_val, df[column_name].max())
            print(f"ymax_val for {tag}: {ymax_val}")  
        plt.ylim(0, y_max_factor * ymax_val)
    #############
    """
    
    if ylog:
        plt.yscale('log')


    # Then handle zero removal if requested
    if remove_zero_entries:
        for tag in datasets:
            if is_eta or is_phi:
                dataframes[tag] = dataframes[tag][dataframes[tag][column_name] != 0]
            else:
                dataframes[tag] = dataframes[tag][dataframes[tag][column_name] > 0]
    
    # Create two separate legend handles
    normal_lines = []
    normal_labels = []
    anomalous_lines = []
    anomalous_labels = []
    
    # Plot for each dataset
    for idx, tag in enumerate(datasets):
        df = dataframes[tag]
        
        # Process low AD scores (dashed lines)
        mask_low = (df['HLT_AD_scores'] >= range_AD_normal[0]) & (df['HLT_AD_scores'] <= range_AD_normal[1])
        if remove_zero_entries:
            if is_eta or is_phi:
                epsilon = 1e-6  # Small threshold for angular variables
                mask_low &= (np.abs(df[column_name]) > epsilon)
            else:  # pt variables
                mask_low &= (df[column_name] > 0)
        data_low = df[mask_low][column_name]
        weights_low = df[mask_low]['weights']
        weights_low_norm = weights_low / weights_low.sum() if len(weights_low) > 0 else weights_low
        
        line_low = plt.hist(data_low, 
                          bins=bins,
                          weights=weights_low_norm,
                          histtype='step',
                          linewidth=2.5 if tag=='EB_test' else 1.5,
                          linestyle='--',
                          color=colors[idx])
        normal_lines.append(line_low[2][0])
        normal_labels.append(tag)
        
        # Process high AD scores (solid lines)
        mask_high = (df['HLT_AD_scores'] >= range_AD_anomalous[0]) & (df['HLT_AD_scores'] <= range_AD_anomalous[1])
        if remove_zero_entries:           
            if is_eta or is_phi:
                epsilon = 1e-6  # Small threshold for angular variables
                mask_low &= (np.abs(df[column_name]) > epsilon)
            else:  # pt variables
                mask_low &= (df[column_name] > 0)
        data_high = df[mask_high][column_name]
        weights_high = df[mask_high]['weights']
        weights_high_norm = weights_high / weights_high.sum() if len(weights_high) > 0 else weights_high
        
        line_high = plt.hist(data_high, 
                           bins=bins,
                           weights=weights_high_norm,
                           histtype='step',
                           linewidth=2.5 if tag=='EB_test' else 1.5,
                           linestyle='-',
                           color=colors[idx])
        anomalous_lines.append(line_high[2][0])
        anomalous_labels.append(tag)
    
    # Title plot & axes
    object_name = variable_name.replace(" [GeV]", "")
    plt.title(f'Distribution of {object_name} for low vs high AD scores', fontsize=20)
    plt.xlabel(variable_name)
    plt.ylabel('Normalized Events')
    
    if x_max is not None:
        plt.xlim(0, x_max)
    
    # Create custom legend with lines
    normal_lines = [Line2D([0], [0], color=colors[i], 
                          linestyle='--',
                          lw=2.5 if datasets[i]=='EB_test' else 1.5) 
                   for i in range(len(datasets))]
    
    anomalous_lines = [Line2D([0], [0], color=colors[i], 
                             linestyle='-',
                             lw=2.5 if datasets[i]=='EB_test' else 1.5) 
                      for i in range(len(datasets))]
    
    # Add first legend at the top right
    first_legend = plt.legend(normal_lines, normal_labels,
                            title=f'Events with AD scores: {range_AD_normal[0]} - {range_AD_normal[1]}',
                            loc='upper right',
                            frameon=False,
                            bbox_to_anchor=(0.95, 0.95))  # Position at (1,1)
    plt.gca().add_artist(first_legend)
    
    # Add second legend below the first one
    plt.legend(anomalous_lines, anomalous_labels,
              title=f'Events with AD scores > {range_AD_anomalous[0]}',
              loc='upper right',
              frameon=False,
              bbox_to_anchor=(0.95, 0.7))  # Position at (1,0.7) - adjust this value as needed
    


    plt.ticklabel_format(axis='y', style='sci', scilimits=(-2,3))
    plt.tight_layout()
    plt.show()

def overlay_kin_variable_2_AD_ranges_errors(dataframes, dataset_tags, column_name, variable_name, 
                               range_AD_normal, range_AD_anomalous, 
                               remove_zero_entries=False, # not used
                               nbins=50, 
                               x_max=None, 
                               y_max_factor=None, ylog=False, 
                               ax=None):
    """
    Plot overlaid distributions for a variable, comparing low/high AD score ranges + stat errors
    
    Args:
        dataframes: Dictionary of dataframes
        dataset_tags: List of dataset tags (EB_test should be first)
        column_name: Name of column to plot
        variable_name: Label for x-axis
        range_AD_normal: Tuple of (min, max) for normal AD scores
        range_AD_anomalous: Tuple of (min, max) for anomalous AD scores
        remove_zero_entries: If True, exclude events where variable is > zero (for pT) or within epsilon (for eta/phi)
        nbins: Number of bins for histogram
        x_max: Maximum value for x-axis (if None, use data range)
        y_max_factor: Scaling coefficient for the maximum value for y-axis (beautifying plot with no cutting legend)
        ylog: If True, use logarithmic scale for y-axis
    """

    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', 
              '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
    
    # Adjust figure size based on variable type
    if ax is None:
        fig, ax = plt.subplots(figsize=(12, 8))  # Create a new figure if ax is not provided
    else:
        # Change the figure size based on the variable name
        if 'phi' in variable_name.lower() or 'eta' in variable_name.lower():
            ax.figure.set_size_inches(8, 8)
        #else:
        #    ax.figure.set_size_inches(12, 8)
    
    # Order datasets
    datasets = []
    if 'EB_test' in dataset_tags:
        datasets.append('EB_test')
    for dataset_tag in dataset_tags:
        if dataset_tag != 'EB_test':
            datasets.append(dataset_tag)
            
    # Set x axis limits and bins
    is_eta = 'eta' in variable_name.lower()
    is_phi = 'phi' in variable_name.lower()

    if is_eta:
        bins = np.linspace(-3.8, 3.8, nbins+1)
    elif is_phi:
        bins = np.linspace(-4.5, 4.5, nbins+1)
    else:  # pt variable
        if x_max is not None:
            bins = np.linspace(0, x_max, nbins+1)
        else:
            xmax_val = 0
            for tag in datasets:
                df = dataframes[tag]
                xmax_val = max(xmax_val, df[column_name].max())
            bins = np.linspace(0, xmax_val, nbins+1)
    
    if ylog:
        ax.yscale('log')

    """
    # NOT GOOD: SUPPRESS DATA! And handled by mask later anyway
    # Handle zero removal if requested
    if remove_zero_entries:
        for tag in datasets:
            if is_eta or is_phi:
                dataframes[tag] = dataframes[tag][dataframes[tag][column_name] != 0]
            else:
                dataframes[tag] = dataframes[tag][dataframes[tag][column_name] > 0]
    """

    # Create two separate legend handles
    normal_lines = []
    normal_labels = []
    anomalous_lines = []
    anomalous_labels = []
    
    # Plot for each dataset
    for idx, tag in enumerate(datasets):
        df = dataframes[tag]
        color = colors[idx]

        #=================================================
        # Process low AD scores (dashed lines)
        #=================================================
        mask_low = (df['HLT_AD_scores'] >= range_AD_normal[0]) & (df['HLT_AD_scores'] <= range_AD_normal[1])
        if remove_zero_entries:
            if is_eta or is_phi:
                epsilon = 1e-6
                mask_low &= (np.abs(df[column_name]) > epsilon)
            else:
                mask_low &= (df[column_name] > 0)
        data_low = df[mask_low][column_name]
        weights_low = df[mask_low]['weights']
        
        #weights_low_norm = weights_low / weights_low.sum() if len(weights_low) > 0 else weights_low
        
        
        # Calculate histogram and errors for low AD scores
        #hist_low, bin_edges = np.histogram(data_low, bins=bins, weights=weights_low_norm)
        #hist_low_err = np.sqrt(np.histogram(data_low, bins=bins, weights=weights_low_norm**2)[0])
        #bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

        # Compute histogram and errors using raw weights
        hist_low, bin_edges = np.histogram(data_low, bins=bins, weights=weights_low)
        hist_low_err = np.sqrt(np.histogram(data_low, bins=bins, weights=weights_low**2)[0])

        # Normalize the histogram and errors
        norm_factor_low = hist_low.sum()
        if norm_factor_low > 0:
            hist_low /= norm_factor_low
            hist_low_err /= norm_factor_low
        
        # Plot histogram with step style
        line_low = ax.hist(data_low, bins=bins, weights=weights_low / norm_factor_low,
                          histtype='step', linestyle='--',
                          color=color, linewidth=1.5 if tag=='EB_test' else 1,
                          label=f'{tag} ({range_AD_normal[0]}-{range_AD_normal[1]})')
        
        # Add error band
        ax.fill_between(bin_edges[:-1], hist_low - hist_low_err, hist_low + hist_low_err,
                        alpha=0.15, color=color, step='post')
        
        normal_lines.append(line_low)
        normal_labels.append(tag)
        
        #=================================================
        # Process high AD scores (solid lines)
        #=================================================
        mask_high = (df['HLT_AD_scores'] >= range_AD_anomalous[0]) & (df['HLT_AD_scores'] <= range_AD_anomalous[1])
        if remove_zero_entries:           
            if is_eta or is_phi:
                epsilon = 1e-6
                mask_high &= (np.abs(df[column_name]) > epsilon)
            else:
                mask_high &= (df[column_name] > 0)
        data_high = df[mask_high][column_name]
        weights_high = df[mask_high]['weights']
        #weights_high_norm = weights_high / weights_high.sum() if len(weights_high) > 0 else weights_high
        
        # Calculate histogram and errors for high AD scores
        #hist_high, _ = np.histogram(data_high, bins=bins, weights=weights_high_norm)
        #hist_high_err = np.sqrt(np.histogram(data_high, bins=bins, weights=weights_high_norm**2)[0])
        
        # Compute histogram and errors using raw weights
        hist_high, bin_edges = np.histogram(data_high, bins=bins, weights=weights_high)
        hist_high_err = np.sqrt(np.histogram(data_high, bins=bins, weights=weights_high**2)[0])

        # Normalize the histogram and errors
        norm_factor_high = hist_high.sum()
        if norm_factor_high > 0:
            hist_high /= norm_factor_high
            hist_high_err /= norm_factor_high

        # Plot histogram with step style
        line_high = ax.hist(data_high, bins=bins, weights=weights_high / norm_factor_high,
                           histtype='step', linestyle='-',
                           color=color, linewidth=1.5 if tag=='EB_test' else 1,
                           label=f'{tag} (>{range_AD_anomalous[0]})')
        
        # Add error band
        ax.fill_between(bin_edges[:-1], hist_high - hist_high_err, hist_high + hist_high_err,
                        alpha=0.15, color=color, step='post')
        
        anomalous_lines.append(line_high)
        anomalous_labels.append(tag)
    
    # Title plot & axes
    object_name = variable_name.replace(" [GeV]", "")
    ax.set_title(f'Distribution of {object_name} for low vs high AD scores', fontsize=22)
    ax.set_xlabel(variable_name)
    ax.set_ylabel('Normalized Events')
    
    if x_max is not None:
        ax.set_xlim(0, x_max)
    
    # Create custom legend with lines
    normal_lines = [Line2D([0], [0], color=colors[i], 
                          linestyle='--',
                          lw=1.5 if datasets[i]=='EB_test' else 1) 
                   for i in range(len(datasets))]
    
    anomalous_lines = [Line2D([0], [0], color=colors[i], 
                             linestyle='-',
                             lw=1.5 if datasets[i]=='EB_test' else 1) 
                      for i in range(len(datasets))]
    
    # Add first legend at the top right
    first_legend = ax.legend(normal_lines, normal_labels,
                            title=f'Events with AD scores: {range_AD_normal[0]} - {range_AD_normal[1]}',
                            loc='upper right',
                            frameon=False,
                            bbox_to_anchor=(0.95, 0.95))
    ax.add_artist(first_legend)
    
    # Add second legend below the first one
    ax.legend(anomalous_lines, anomalous_labels,
              title=f'Events with AD scores > {range_AD_anomalous[0]}',
              loc='upper right',
              frameon=False,
              bbox_to_anchor=(0.95, 0.7))

    ax.ticklabel_format(axis='y', style='sci', scilimits=(-2,3))
    fig = ax.figure  # Get the figure from the axes
    plt.tight_layout()  # Call tight_layout on the figure    ax.show()
    return fig, ax  # Return the figure and axis