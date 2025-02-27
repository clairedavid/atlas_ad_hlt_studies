
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from .common_plot_styles import apply_common_plot_styles


def print_event(df, event_idx):
    """
    Print all physics objects' values for a given event in a nicely formatted table.
    
    Args:
        df: DataFrame containing the event data
        event_idx: Index of the event to analyze
    """
    # Get all object columns (pt, eta, phi)
    pt_cols = sorted([col for col in df.columns if col.endswith('pt')])
    
    print(f"\nEvent {event_idx} Details:")
    print("-" * 65)
    print(f"{'Object':<15} {'pT [GeV]':>10} {'η':>10} {'φ':>10}")
    print("-" * 65)
    
    # Print each object's values
    for pt_col in pt_cols:
        base_name = pt_col[:-2]  # Remove 'pt' from column name
        eta_col = base_name + 'eta'
        phi_col = base_name + 'phi'
        
        pt = df.loc[event_idx, pt_col]
        
        # Skip objects with zero or negligible pT (except MET)
        #if pt <= 0.001 and not pt_col.startswith('MET'):
        #    continue
            
        # Get eta (if exists)
        eta = df.loc[event_idx, eta_col] if eta_col in df.columns else float('nan')
        # Get phi
        phi = df.loc[event_idx, phi_col] if phi_col in df.columns else float('nan')
        
        print(f"{base_name:<15} {pt:>10.1f} {eta:>10.2f} {phi:>10.3f}")
    
    print("-" * 65)
    
    # Print event weight if it exists
    if 'weights' in df.columns:
        print(f"\nEvent weight: {df.loc[event_idx, 'weights']:.3f}")
    
    # Print AD score if it exists
    if 'HLT_AD_scores' in df.columns:
        print(f"HLT AD score: {df.loc[event_idx, 'HLT_AD_scores']:.1f}")




def print_tables_pt_ranges(dataframes, dataset_tag):
    """
    Print four tables showing pT ranges statistics (percentages) for a given dataset
    """
    df = dataframes[dataset_tag]
    total_weighted_events = df['weights'].sum()
    
    # Split columns into categories
    jet_cols = sorted([col for col in df.columns if col.startswith('j') and col.endswith('pt')])
    lepton_cols = sorted([col for col in df.columns if (col.startswith('e') or col.startswith('mu')) and col.endswith('pt')])
    photon_cols = sorted([col for col in df.columns if col.startswith('ph') and col.endswith('pt')])

    def print_subtable(columns, title, is_met=False):
        if not columns:
            return
            
        print(f"\n{title} pT ranges for {dataset_tag} (percentages)")
        print("-" * 100)
        
        # Header
        print(f"{'Range':<17}", end='')
        for col in columns:
            print(f"{col:<14}", end='')
        print("\n" + "-" * 100)
        
        if is_met:
            # Special ranges for MET
            ranges = ['MET < 0', 'MET = 0', 'MET = 0.001', '0.001 < MET < 50', 'MET ≥ 50']
            conditions = [
                lambda x: x < 0,
                lambda x: x == 0,
                lambda x: x == 0.001,
                lambda x: (x > 0.001) & (x < 50),
                lambda x: x >= 50
            ]
        else:
            # Standard ranges for other variables
            ranges = ['pt = 0', '0 < pt < 50', 'pt ≥ 50']
            conditions = [
                lambda x: x == 0,
                lambda x: (x > 0) & (x < 50),
                lambda x: x >= 50
            ]
        
        for range_text, condition in zip(ranges, conditions):
            print(f"{range_text:<12}", end='')
            for col in columns:
                weighted_sum = df[condition(df[col])]['weights'].sum()
                percentage = (weighted_sum / total_weighted_events) * 100
                print(f"{percentage:>8.1f}%     ", end='')
            print()
            
        print("-" * 100)
    
    # Print the three tables
    print_subtable(jet_cols, "Jets")
    print_subtable(lepton_cols, "Leptons")
    print_subtable(photon_cols, "Photons")
    print_subtable(['METpt'], "MET", is_met=True)



def get_manual_MET(df, event_idx, debug=False):
    """
    Compute MET manually by vector-summing all physics objects' pT vectors (with flipped signs)
    for a given event. Skips duplicate objects (same pT and phi) among leptons and photons.
    
    Args:
        df: DataFrame containing the event data
        event_idx: Index of the event to analyze
        debug: If True, print detailed table of all objects
    """
    # Get all pT and phi columns except MET
    pt_cols = [col for col in df.columns if col.endswith('pt') and not col.startswith('MET')]
    
    if debug:
        print(f"\nDetailed object list for Event {event_idx}:")
        print("-" * 70)
        print(f"{'Object':<15} {'pT [GeV]':>10} {'φ':>10} {'Used':>10}")
        print("-" * 70)
    
    # Initialize sum of components
    sum_px = 0
    sum_py = 0
    
    # Keep track of (pt, phi) pairs for leptons and photons
    used_values = set()
    
    # Sum all object vectors
    for pt_col in pt_cols:
        phi_col = pt_col.replace('pt', 'phi')
        if phi_col not in df.columns:
            continue
            
        pt = df.loc[event_idx, pt_col]
        phi = df.loc[event_idx, phi_col]
        
        # Skip objects with zero or negligible pT
        if pt <= 0.001:
            if debug:
                print(f"{pt_col[:-2]:<15} {pt:>10.1f} {phi:>10.3f} {'skip-zero':>10}")
            continue
        
        # For leptons and photons, check if we've seen these values before
        is_duplicate = False
        if any(pt_col.startswith(x) for x in ['e', 'mu', 'ph']):
            values = (round(pt, 3), round(phi, 3))  # Round to avoid floating point issues
            if values in used_values:
                is_duplicate = True
            else:
                used_values.add(values)
        
        if debug:
            status = 'duplicate' if is_duplicate else 'used'
            print(f"{pt_col[:-2]:<15} {pt:>10.1f} {phi:>10.3f} {status:>10}")
            
        # Skip if duplicate
        if is_duplicate:
            continue
            
        # Add vector components (with flipped signs)
        sum_px -= pt * np.cos(phi)
        sum_py -= pt * np.sin(phi)
    
    if debug:
        print("-" * 70)
    
    # Compute manual MET (already has correct sign from above)
    manual_met_pt = np.sqrt(sum_px**2 + sum_py**2)
    manual_met_phi = np.arctan2(sum_py, sum_px)
    
    # Get measured MET values
    measured_met_pt = df.loc[event_idx, 'METpt']
    measured_met_phi = df.loc[event_idx, 'METphi']
    
    # Print comparison
    print(f"\nMET Comparison for Event {event_idx}:\n")
    print(f"{'':4}Measured MET: pT = {measured_met_pt:.1f} GeV,\tφ = {measured_met_phi:.3f}")
    print(f"{'':4}Manual MET:   pT = {manual_met_pt:.1f} GeV,\tφ = {manual_met_phi:.3f}")
    print("\n\n")
    
    return manual_met_pt, manual_met_phi


# Took 6 min with EB_test but converged 
def inspect_duplicates(dataframes, dataset_tag):
    """
    Print a table inspecting duplicates for electrons, muons, and photons.
    """
    df = dataframes[dataset_tag]
    
    def count_objects(prefix):
        """Count valid objects where pt > 0."""
        cols = [col for col in df.columns if col.startswith(prefix) and col.endswith('pt')]
        return sum((df[col] > 0).sum() for col in cols)
    
    def count_unique_objects(prefix):
        """Count unique objects by rounding (pt, eta, phi) to 3 decimals."""
        cols = [(f'{prefix}{i}pt', f'{prefix}{i}eta', f'{prefix}{i}phi') 
                for i in range(3) if f'{prefix}{i}pt' in df.columns]  # Assuming max 3 objects
        unique_tuples = set()
        for _, row in df.iterrows():
            for pt_col, eta_col, phi_col in cols:
                if row[pt_col] > 0:
                    unique_tuples.add((round(row[pt_col], 3), round(row[eta_col], 3), round(row[phi_col], 3)))
        return len(unique_tuples)
    
    def count_duplicates(raw, unique):
        """Count duplicates as difference between raw and unique."""
        return raw - unique
    
    # Compute statistics
    raw_counts = {obj: count_objects(obj) for obj in ['e', 'mu', 'ph']}
    unique_counts = {obj: count_unique_objects(obj) for obj in ['e', 'mu', 'ph']}
    duplicates = {obj: count_duplicates(raw_counts[obj], unique_counts[obj]) for obj in ['e', 'mu', 'ph']}
    reduction = {obj: (duplicates[obj] / raw_counts[obj]) * 100 if raw_counts[obj] > 0 else 0 for obj in ['e', 'mu', 'ph']}
    
    # Create DataFrame for table display
    table = pd.DataFrame({
        "Electrons": [raw_counts['e'], unique_counts['e'], duplicates['e'], f"{reduction['e']:.1f}%"],
        "Muons": [raw_counts['mu'], unique_counts['mu'], duplicates['mu'], f"{reduction['mu']:.1f}%"],
        "Photons": [raw_counts['ph'], unique_counts['ph'], duplicates['ph'], f"{reduction['ph']:.1f}%"]
    }, index=["Raw Data", "Unique", "Duplicates", "Reduction (%)"])
    
    print(f"\nDuplicate Inspection Table for {dataset_tag}")
    print("-" * 50)
    print(table)
    print("-" * 50)


# Not tested 
def inspect_duplicates_spedup(dataframes, dataset_tag):
    """
    Print a table inspecting duplicates for electrons, muons, and photons.

    This update replaces row iteration with a more efficient pandas approach. It flattens the (pt, eta, phi) columns into a single DataFrame and uses .drop_duplicates(),
    """
    df = dataframes[dataset_tag]
    
    def count_objects(prefix):
        """Count valid objects where pt > 0."""
        cols = [col for col in df.columns if col.startswith(prefix) and col.endswith('pt')]
        return (df[cols] > 0).sum().sum()
    
    def count_unique_objects(prefix):
        """Count unique objects by rounding (pt, eta, phi) to 3 decimals."""
        cols = [(f'{prefix}{i}pt', f'{prefix}{i}eta', f'{prefix}{i}phi') 
                for i in range(3) if f'{prefix}{i}pt' in df.columns]  # Assuming max 3 objects
        
        stacked_df = pd.DataFrame({
            'pt': df[[pt for pt, eta, phi in cols]].values.flatten(),
            'eta': df[[eta for pt, eta, phi in cols]].values.flatten(),
            'phi': df[[phi for pt, eta, phi in cols]].values.flatten()
        })
        
        stacked_df = stacked_df[stacked_df['pt'] > 0]
        stacked_df = stacked_df.round(3)
        
        return len(stacked_df.drop_duplicates())
    
    def count_duplicates(raw, unique):
        """Count duplicates as difference between raw and unique."""
        return raw - unique
    
    # Compute statistics
    raw_counts = {obj: count_objects(obj) for obj in ['e', 'mu', 'ph']}
    unique_counts = {obj: count_unique_objects(obj) for obj in ['e', 'mu', 'ph']}
    duplicates = {obj: count_duplicates(raw_counts[obj], unique_counts[obj]) for obj in ['e', 'mu', 'ph']}
    reduction = {obj: (duplicates[obj] / raw_counts[obj]) * 100 if raw_counts[obj] > 0 else 0 for obj in ['e', 'mu', 'ph']}
    
    # Create DataFrame for table display
    table = pd.DataFrame({
        "Electrons": [raw_counts['e'], unique_counts['e'], duplicates['e'], f"{reduction['e']:.1f}%"],
        "Muons": [raw_counts['mu'], unique_counts['mu'], duplicates['mu'], f"{reduction['mu']:.1f}%"],
        "Photons": [raw_counts['ph'], unique_counts['ph'], duplicates['ph'], f"{reduction['ph']:.1f}%"]
    }, index=["Raw Data", "Unique", "Duplicates", "Reduction (%)"])
    
    print(f"\nDuplicate Inspection Table for {dataset_tag}")
    print("-" * 50)
    print(table)
    print("-" * 50)

# Adding jets
def inspect_duplicates_all_objects_fast(dataframes, dataset_tag):

    """
    Print a table inspecting duplicates for jets, electrons, muons, and photons.
    """

    df = dataframes[dataset_tag]
    
    def count_objects(prefix, max_objects):
        """Count valid objects where pt > 0."""
        cols = [f'{prefix}{i}pt' for i in range(max_objects) if f'{prefix}{i}pt' in df.columns]
        return (df[cols] > 0).sum().sum()
    
    def count_unique_objects(prefix, max_objects):
        """Count unique objects by rounding (pt, eta, phi) to 3 decimals."""
        cols = [(f'{prefix}{i}pt', f'{prefix}{i}eta', f'{prefix}{i}phi') 
                for i in range(max_objects) if f'{prefix}{i}pt' in df.columns]
        
        stacked_df = pd.DataFrame({
            'pt': df[[pt for pt, eta, phi in cols]].values.flatten(),
            'eta': df[[eta for pt, eta, phi in cols]].values.flatten(),
            'phi': df[[phi for pt, eta, phi in cols]].values.flatten()
        })
        
        stacked_df = stacked_df[stacked_df['pt'] > 0]
        stacked_df = stacked_df.round(3)
        
        return len(stacked_df.drop_duplicates())
    
    def count_duplicates(raw, unique):
        """Count duplicates as difference between raw and unique."""
        return raw - unique
    
    # Max objects per category
    max_objects = {'j': 6, 'e': 3, 'mu': 3, 'ph': 3}
    
    # Compute statistics
    raw_counts = {obj: count_objects(obj, max_objects[obj]) for obj in ['j', 'e', 'mu', 'ph']}
    unique_counts = {obj: count_unique_objects(obj, max_objects[obj]) for obj in ['j', 'e', 'mu', 'ph']}
    duplicates = {obj: count_duplicates(raw_counts[obj], unique_counts[obj]) for obj in ['j', 'e', 'mu', 'ph']}
    reduction = {obj: (duplicates[obj] / raw_counts[obj]) * 100 if raw_counts[obj] > 0 else 0 for obj in ['j', 'e', 'mu', 'ph']}
    
    # Set display options to enlarge columns
    pd.set_option('display.max_colwidth', None)
    pd.set_option("display.width", 1000)  # Increase table width
    pd.set_option("display.colheader_justify", "center")  # Center column headers

    # Create DataFrame for table display
    table = pd.DataFrame({
        "Jets": [raw_counts['j'], unique_counts['j'], duplicates['j'], f"{reduction['j']:.1f}%"],
        "Electrons": [raw_counts['e'], unique_counts['e'], duplicates['e'], f"{reduction['e']:.1f}%"],
        "Muons": [raw_counts['mu'], unique_counts['mu'], duplicates['mu'], f"{reduction['mu']:.1f}%"],
        "Photons": [raw_counts['ph'], unique_counts['ph'], duplicates['ph'], f"{reduction['ph']:.1f}%"]
    }, index=["Raw Data", "Unique", "Duplicates", "Reduction (%)"])
    
    # Set display options to enlarge columns
    # NOT WORKING
    """
    pd.set_option('display.max_colwidth', None)
    pd.set_option("display.width", 1000)  # Increase table width
    pd.set_option("display.colheader_justify", "center")  # Center column headers

    # Print the table with the desired width
    print(f"\nDuplicate Inspection Table for {dataset_tag}")
    print("-" * 60)
    print(table.to_string())
    print("-" * 60)
    """

    # Manually format the table as a string
    table_str = (
        f"\nDuplicate Inspection Table for {dataset_tag}\n"
        f"{'-' * 100}\n"
        f"{'':<20}{'Jets':>20}{'Electrons':>20}{'Muons':>20}{'Photons':>20}\n"
        f"{'-' * 100}\n"
        f"{'Number (raw)':<20}{raw_counts['j']:>20}{raw_counts['e']:>20}{raw_counts['mu']:>20}{raw_counts['ph']:>20}\n"
        f"{'Number (unique)':<20}{unique_counts['j']:>20}{unique_counts['e']:>20}{unique_counts['mu']:>20}{unique_counts['ph']:>20}\n"
        f"{'Number of duplicates':<20}{duplicates['j']:>20}{duplicates['e']:>20}{duplicates['mu']:>20}{duplicates['ph']:>20}\n"
        f"{'Reduction (%)':<20}{reduction['j']:>20.1f}{reduction['e']:>20.1f}{reduction['mu']:>20.1f}{reduction['ph']:>20.1f}\n"
        f"{'-' * 100}"
    )
    
    # Print the table
    print(table_str)

def tables_latent_variables(dataframes, dataset_tag, use_scientific_notation=False):
    """
    Print a table of descriptive statistics for the four latent variables LS1, LS2, LS3, and LS4.

    Args:
        dataframes: Dictionary of dataframes
        dataset_tag: Tag of the dataset to analyze
        use_scientific_notation: Boolean to use scientific notation for the numbers
    """
    df = dataframes[dataset_tag]
    latent_vars = ['LS1', 'LS2', 'LS3', 'LS4']
    
    # Get descriptive statistics for each latent variable
    stats = {var: df[var].describe() for var in latent_vars}
    
    # Create a DataFrame for the statistics
    stats_df = pd.DataFrame(stats)
    
    # Convert 'count' to an integer
    stats_df.loc['count'] = stats_df.loc['count'].astype(int)

    # Apply number formatting with right alignment
    def format_value(x, row):
        if row == 'count':
            return f"{int(x):,}".rjust(15)  # Right-align integer with thousands separator
        return (f"{x:.4e}" if use_scientific_notation else f"{x:.4f}").rjust(15)  # Right-align float

    formatted_stats = stats_df.apply(lambda col: [format_value(val, idx) for idx, val in col.items()], axis=0)
    formatted_stats_df = pd.DataFrame(formatted_stats, index=stats_df.index)

    # Print the table
    print(f"\nDescriptive Statistics for Latent Variables in {dataset_tag}")
    print("-" * 100)
    print(formatted_stats_df.to_string(index=True, justify="center"))
    print("-" * 100)


# Plot a distribution of the weights of the events for a dataset tag
def plot_event_weights(data, dataset_tag, ax=None):
    """
    Plot a distribution of the weights of the events for a dataset tag.

    Args:
        dataframe: DataFrame containing the data
        dataset_tag: Tag of the dataset to analyze
    """

    df = data[dataset_tag]

    # Get the weights column
    weights = df['weights']

    # Create figure if no axis is provided
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 8), dpi=150)
    else:
        fig = None  # If ax is provided, do not create a new figure

    # Plot the histogram of the weights
    ax.hist(weights, bins=100, color='dodgerblue', alpha=0.7)

    # Set the labels and title
    ax.set_xlabel('Event Weight')
    ax.set_ylabel('Frequency')
    ax.set_title(f'Event Weight Distribution for {dataset_tag}')

    # Display the plot
    plt.show()

    return fig, ax
