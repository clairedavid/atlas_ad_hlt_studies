
import numpy as np
import pandas as pd

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
            ranges = ['MET = 0', 'MET = 0.001', '0.001 < MET < 50', 'MET ≥ 50']
            conditions = [
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