import matplotlib.pyplot as plt
import numpy as np
from .common_plot_styles import apply_common_plot_styles


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
    ax.legend(handles=legend_elements, bbox_to_anchor=(1.02, 1), loc='upper left')
    
    plt.tight_layout()
    plt.subplots_adjust(right=0.75)  # Adjust the right margin

    
    return fig, ax