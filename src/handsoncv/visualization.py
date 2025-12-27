
import matplotlib.colors as colors
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
# import wandb
import numpy as np

"""
The following functions expand upon the notebooks provided for the Nvidia course Building AI Agents with Multimodal Models https://learn.nvidia.com/courses/course-detail?course_id=course-v1:DLI+C-FX-17+V1
"""

def hex_to_RGB(hex_str):
    """ #FFFFFF -> [255,255,255]"""
    #Pass 16 to the integer function for change of base
    return [int(hex_str[i:i+2], 16) for i in range(1,6,2)]

def get_color_gradient(c1, c2, c3, n1, n2):
    """
    Given two hex colors, returns a color gradient
    with n colors.
    """
    c1_rgb = np.array(hex_to_RGB(c1))/255
    c2_rgb = np.array(hex_to_RGB(c2))/255
    c3_rgb = np.array(hex_to_RGB(c3))/255
    mix_pcts_c1_c2 = [x/(n1-1) for x in range(n1)]
    mix_pcts_c2_c3 = [x/(n2-1) for x in range(n2)]
    rgb_c1_c2 = [((1-mix)*c1_rgb + (mix*c2_rgb)) for mix in mix_pcts_c1_c2]
    rgb_c2_c3 = [((1-mix)*c2_rgb + (mix*c3_rgb)) for mix in mix_pcts_c2_c3]
    rgb_colors = rgb_c1_c2 + rgb_c2_c3
    return ["#" + "".join([format(int(round(val*255)), "02x") for val in item]) for item in rgb_colors]

cmap = colors.ListedColormap(get_color_gradient("#000000", "#76b900", "#f1ffd9", 64, 128))

def log_similarity_heatmap(logits): #, epoch, prefix="val"):
    """Logs heatmap of the CILP similarity matrix of RGB and LiDAR embeddings to wandb."""
    fig, ax = plt.subplots(figsize=(4, 4))
    
    im = ax.imshow(
        logits.detach().cpu().numpy(),
        cmap="inferno",
        vmin=logits.min().item(),
        vmax=logits.max().item()
    )

    # Divider for the existing axes
    divider = make_axes_locatable(ax)
    # Append an axes at the top of the plot. 
    # 'size' is the thickness of the bar, 'pad' is the gap between plot and bar.
    cax = divider.append_axes("top", size="5%", pad=0.3)
    # Create the colorbar in the new 'cax' axes
    cbar = fig.colorbar(im, cax=cax, orientation="horizontal")
    
    cax.xaxis.set_ticks_position('top') # Move the ticks to the top of the colorbar 
    cax.xaxis.set_label_position('top')
    ax.set_xticks([x - 0.5 for x in range(1, logits.shape[1])], minor=True) # Add gridlines 
    ax.set_yticks([y - 0.5 for y in range(1, logits.shape[0])], minor=True)
    ax.grid(which="minor", color="black", linestyle='-', linewidth=0.5)
    ax.tick_params(which="minor", bottom=False, left=False)

    ax.set_xlabel("LiDAR")
    ax.set_ylabel("RGB")
    
    # ax.set_xticks([])
    # ax.set_yticks([])

    return fig