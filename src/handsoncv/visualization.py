
import matplotlib.colors as colors
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.patches as patches
import pandas as pd
import numpy as np
import os

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

def log_similarity_heatmap(logits): 
    """
    Logs heatmap of the CILP similarity matrix of RGB and LiDAR embeddings to wandb.
    
    Parameters
    ----------
    logits : torch.Tensor
        Similarity matrix of shape (N, N) containing pairwise RGB–LiDAR logits.
    """
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

def plot_table_task4_metrics(df, output_path):
    """
    Plot a ranked table comparing model variants across Task 4 metrics (please do not modify this function, as several visualization components are hard-coded).

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame containing model variants and metric values.
    output_path : str or pathlib.Path
        Path where the generated figure will be saved.
    """
    # Setup data and metrics logic
    df_variants = df.drop('Difference', errors='ignore').copy()
    df_diff = df.loc[['Difference']].copy()
    lower_is_better = ['val_loss', 'params', 'total_time_sec', 'sec_per_epoch', 'gpu_mem_mb']
    
    # Calculate ranks for variants only
    ranks = pd.DataFrame(index=df_variants.index)
    for col in df_variants.columns:
        if col in lower_is_better:
            ranks[col] = df_variants[col].rank(ascending=True)
        else:
            ranks[col] = df_variants[col].rank(ascending=False)

    # Sort variants by mean rank (Best on top)
    df_variants['mean_rank'] = ranks.mean(axis=1)
    df_variants = df_variants.sort_values('mean_rank')
    ranks = ranks.loc[df_variants.index]
    
    # Combine back: Sorted variants first, then Difference at the bottom
    df_final = pd.concat([df_variants.drop(columns=['mean_rank']), df_diff])
    models_to_plot = df_final.index.tolist()
    
    # Design parameters
    sage_green = (59/255, 217/255, 135/255, 1.0) #(17/255, 69/255, 30/255, 0.4) 
    sage_green_interior = (59/255, 217/255, 135/255, .7)
    light_blue = (173/255, 216/255, 230/255, .6)
    light_blue_interior = (173/255, 216/255, 230/255, .4)
    metrics = ['val_loss', 'accuracy', 'params', 'gpu_mem_mb']
    x_positions = [0, 0.4, .85, 1.35] 
    
    # Create Figure
    fig, ax = plt.subplots(figsize=(8, len(models_to_plot) * 1.4))
    ax.set_xlim(-0.4, x_positions[-1] + 0.8)
    # Adjust ylim to fit the extra row (Difference at y=0, variants at y=1,2, header above)
    ax.set_ylim(-0.8, len(models_to_plot) + 2)
    ax.axis('off')

    # Header
    header_y = len(models_to_plot) - 0.4
    for i, col in enumerate(metrics):
        name = col.replace('_', ' ').title()
        ax.text(x_positions[i], header_y, name.lower(), ha='center', fontsize=13)
    
    # Horizontal and vertical lines creation   
    v_line_x = x_positions[0] - 0.2
    h_line_y = len(models_to_plot) - 0.5
    bottom_y = -0.5 # Extends past the Difference row
    right_x = x_positions[-1] + 0.3

    # Horizontal and vertical line chosen for table styling
    ax.plot([v_line_x, right_x], [h_line_y, h_line_y], color='#777676', linewidth=1, zorder=3)
    ax.plot([v_line_x, v_line_x], [bottom_y, h_line_y], color='#777676', linewidth=1, zorder=3)
    
    leg_y = len(models_to_plot) + .6 # Height for legend
    leg_x_end = right_x
    leg_x_start = 0.55
    
    # Outer Rounded Legend Box
    leg_outer = patches.FancyBboxPatch(
        (leg_x_start, leg_y - 0.25), leg_x_end - leg_x_start, 1.1,
        boxstyle="round,pad=0.02,rounding_size=0.1",
        linewidth=1.5, edgecolor='#CCCCCC', facecolor='none', zorder=4
    )
    ax.add_patch(leg_outer)
    
    # Legend Text
    ax.text(leg_x_start + 0.05, leg_y + 0.5, "Ranking", fontsize=15, color='#666666', va='center')
    ax.text(leg_x_start + 0.05, leg_y + 0.1, "column-wise", fontsize=11, color='#888888', va='center')
    ax.text(leg_x_start + 0.2, leg_y - 0.5, "↓", fontsize=20, color='#888888', ha='center')

    # Rank 1 Box in Legend
    r1_x = leg_x_start + 0.6
    ax.add_patch(patches.FancyBboxPatch(
        (r1_x - 0.1, leg_y - 0.15), 0.2, 0.8,
        boxstyle="round,pad=0.02,rounding_size=0.1",
        linewidth=1.5, edgecolor=sage_green, facecolor=sage_green_interior, zorder=5
    ))
    ax.text(r1_x, leg_y + .15, "1st", ha='center', fontsize=13, color='black', zorder=6)

    # Rank 2 Box in Legend
    r2_x = leg_x_start + 0.9
    ax.add_patch(patches.FancyBboxPatch(
        (r2_x - 0.1, leg_y - 0.15), 0.2, 0.8,
        boxstyle="round,pad=0.02,rounding_size=0.1",
        linewidth=1.5, edgecolor=light_blue, facecolor=light_blue_interior, zorder=5
    ))
    ax.text(r2_x, leg_y + .15, "2nd", ha='center', fontsize=13, color='black', zorder=6)

    # Draw Rows (Ranked)
    # We plot from top to bottom. Row 0 (bottom) will be Difference.
    for row_idx, model_id in enumerate(reversed(models_to_plot)):
        y_pos = row_idx  
        
        # Row Label
        color = 'black' if model_id != 'Difference' else "#777676"
        fontsize = 11 if model_id != 'Difference'  else 12
        ax.text(v_line_x - 0.05, y_pos, model_id.lower(), ha='right', va='center', fontsize=13, color=color, weight='normal')

        for col_idx, col_name in enumerate(metrics):
            val = df_final.loc[model_id, col_name]
            x_pos = x_positions[col_idx]
            
            # Variant rows (with colored fancy boxes wrapping numbers and indicating by size and color the best results) 
            if model_id != 'Difference':
                rank = ranks.loc[model_id, col_name]
                if rank == 1:
                    color, edgecolor = sage_green_interior, sage_green
                    box_w = .27
                    box_h = .8
                    if col_name in ("params", "gpu_mem_mb"):
                        box_w = .38
                        box_h = .8
                else:
                    color, edgecolor = light_blue_interior, light_blue
                    box_w = .25
                    box_h = .78
                    if col_name in ("params", "gpu_mem_mb"):
                        box_w = .39
                        box_h = .77

                rect = patches.FancyBboxPatch(
                    (x_pos - box_w/2, y_pos - box_h/2),
                    box_w, box_h,
                    # boxstyle="round4,pad=0.01,rounding_size=0.04", 
                    # boxstyle="Round,pad=0.01,rounding_size=0.1", #rounding_size
                    boxstyle="round,pad=0.01,rounding_size=0.15",
                    linewidth=2,
                    facecolor=color,
                    edgecolor=edgecolor,
                    mutation_scale=.7, # Controls corner tightness (lower = tighter/squarer)
                    # mutation_aspect=0.9,
                    zorder=1
                )
                ax.add_patch(rect)
            
            # Text Formatting with zero-stripping 
            if col_name == 'accuracy': 
                text_val = f"{val/100:.3f}"
            elif col_name == 'val_loss': 
                text_val = f"{val:.4f}"
                # Remove leading zero (0.04 -> .04)
            if text_val.startswith("0."):
                text_val = text_val[1:]
            elif text_val.startswith("-0."):
                text_val = "-" + text_val[2:]
            else: 
                text_val = f"{int(val):,}" if abs(val) > 1000 else f"{val:.3f}"
            
            # For Difference row, maybe use a sign (+/-)
            if model_id == 'Difference' and val > 0:
                text_val = "+" + text_val
            # elif model_id == 'Difference' and val < 0:
            #     text_val = "-" + text_val
            
            color = 'black' if model_id != 'Difference' else "#777676"
            fontsize = 13 if model_id != 'Difference' else 12
            ax.text(x_pos, y_pos, text_val, ha='center', va='center', fontsize=fontsize, color=color, zorder=2)

    plt.savefig(output_path, dpi=300, bbox_inches='tight', transparent=True)
    plt.show()