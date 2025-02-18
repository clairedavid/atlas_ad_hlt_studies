import matplotlib.pyplot as plt

def apply_common_plot_styles(ax):
    """
    Apply common plotting styles to the given Matplotlib axis.

    Args:
        ax: Matplotlib axis object to which styles will be applied.
    """
    # Set grid
    ax.grid(True, linestyle='--', alpha=0.7)

    # Set font sizes
    for label in (ax.get_xticklabels() + ax.get_yticklabels()):
        label.set_fontsize(12)

    # Set title font size
    ax.title.set_fontsize(16)

    # Set labels font size
    ax.xaxis.label.set_size(14)
    ax.yaxis.label.set_size(14)

    # Set ticks
    ax.tick_params(axis='both', which='major', labelsize=12)

    # Set tight layout
    plt.tight_layout()