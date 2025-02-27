import matplotlib.pyplot as plt

def apply_common_plot_styles():
    """
    Apply common plotting styles to the given Matplotlib axis.
    """
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


