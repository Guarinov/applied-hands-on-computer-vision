
import os 
import torch
import torchvision.transforms as transforms
import matplotlib.colors as colors
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import seaborn as sns
import pandas as pd
import numpy as np

from mpl_toolkits.axes_grid1 import make_axes_locatable

"""
The following functions expand upon the notebooks provided for the Nvidia course 
Building AI Agents with Multimodal Models https://learn.nvidia.com/courses/course-detail?course_id=course-v1:DLI+C-FX-17+V1
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

    return fig

def plot_class_distribution(labels, counts, output_dir):
    """
    Creates a bar chart of class distributions per split with internal vertical labels 
    and saves it to the specified directory.
    
    Args:
        labels (list): String labels for each bar (e.g., ['train cube', ...]).
        counts (list): Numerical values representing the count for each label.
        output_dir (str): The full path including filename where the image will be saved (e.g., 'results/task1_label_split_distribution.png').
    """
    # Define colors
    colors = [
        (114/255, 144/255, 184/255, 1.0), # Dark Blue
        (154/255, 181/255, 217/255, 1.0), # Light Blue
        (184/255, 51/255, 51/255, 1.0),   # Dark Red
        (217/255, 91/255, 91/255, 1.0)    # Light Red
    ]

    fig, ax = plt.subplots(figsize=(6, 5))

    # Create the bars
    bars = ax.bar(labels, counts, color=colors, width=0.8)

    # Title and Layout Style
    ax.set_title("class distribution per split", fontsize=20, pad=20, loc='right')
    ax.set_ylabel("number of samples", fontsize=16)

    # Styling axes to match reference style
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_linewidth(2)
    ax.spines['bottom'].set_linewidth(2)

    # Grid lines (horizontal only, dashed)
    ax.yaxis.grid(True, linestyle='--', color='gray', alpha=0.6)
    ax.set_axisbelow(True) 

    # Remove standard X-ticks to place labels inside the bars instead
    ax.set_xticks([])
    ax.tick_params(axis='y', labelsize=16)

    # Add internal vertical labels
    for i, bar in enumerate(bars):
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width()/2., 
            height * 0.05,                     
            labels[i], 
            ha='center', 
            va='bottom', 
            rotation=90, 
            fontsize=24, 
            color='black',                     
            fontweight='normal'
        )

    plt.tight_layout()
    
    # Save
    plt.savefig(output_dir, dpi=300, bbox_inches='tight', transparent=True) 
    plt.show()

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
        (leg_x_start, leg_y - 0.25), 
        leg_x_end - leg_x_start, 1.1,
        boxstyle="round,pad=0.02,rounding_size=0.1",
        linewidth=1.5, 
        edgecolor='#CCCCCC', 
        facecolor='none', 
        zorder=4
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
        (r2_x - 0.1, leg_y - 0.15), 0.2, 0.75,
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
            
            color = 'black' if model_id != 'Difference' else "#777676"
            fontsize = 13 if model_id != 'Difference' else 12
            ax.text(x_pos, y_pos, text_val, ha='center', va='center', fontsize=fontsize, color=color, zorder=2)

    plt.savefig(output_path, dpi=300, bbox_inches='tight', transparent=True)
    plt.show()

def plot_table_task3_metrics(df, output_path):
    """
    Plot a ranked table comparing model variants across Task 3 metrics (please do not modify this function, as several visualization components are hard-coded).

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame containing model variants and metric values.
    output_path : str or pathlib.Path
        Path where the generated figure will be saved.
    """
    # Data Preparation
    if 'Architecture' in df.columns:
        df = df.set_index('Architecture')
    
    df_variants = df.copy()
    lower_is_better = ['val_loss', 'params', 'sec_per_epoch', 'gpu_mem_mb']
    
    # Calculate Ranks
    ranks = pd.DataFrame(index=df_variants.index)
    for col in df_variants.columns:
        if col in lower_is_better:
            ranks[col] = df_variants[col].rank(ascending=True, method='min')
        else:
            ranks[col] = df_variants[col].rank(ascending=False, method='min')

    # Sort variants by mean rank (Best on top)
    df_variants['mean_rank'] = ranks.mean(axis=1)
    df_variants = df_variants.sort_values('mean_rank')
    ranks = ranks.loc[df_variants.index]
    
    df_final = df_variants.drop(columns=['mean_rank'])
    models_to_plot = df_final.index.tolist()
    
    # Design Parameters
    sage_green = (59/255, 217/255, 135/255, 1.0) #(17/255, 69/255, 30/255, 0.4) 
    sage_green_interior = (59/255, 217/255, 135/255, .7)
    light_blue = (173/255, 216/255, 230/255, .6)
    light_blue_interior = (173/255, 216/255, 230/255, .4)
    dark_blue = (126/255, 141/255, 194/255, 1.0)
    dark_blue_interior = (126/255, 141/255, 194/255, .4)
    unranked_grey = '#BBBBBB' # Color for 3rd rank and below
    
    metrics = ['val_loss', 'accuracy', 'params', 'sec_per_epoch', 'gpu_mem_mb']
    x_positions = [0, 0.45, 1.0, 1.6, 2.15] 
    
    # Create Figure
    fig, ax = plt.subplots(figsize=(8, len(models_to_plot) * 1.4))
    ax.set_xlim(-0.7, x_positions[-1] + 0.8)
    ax.set_ylim(-0.5, len(models_to_plot) + 2)
    ax.axis('off')

    # Header
    header_y = len(models_to_plot) - 0.4
    for i, col in enumerate(metrics):
        name = col.replace('_', ' ').replace('sec per epoch', 'sec/ep').lower()
        ax.text(x_positions[i], header_y, name, ha='center', fontsize=13) # fontweight='bold', color='#444444')
    
    # Lines
    v_line_x = x_positions[0] - 0.25
    h_line_y = len(models_to_plot) - 0.5
    bottom_y = -0.6
    right_x = x_positions[-1] + 0.3
    ax.plot([v_line_x, right_x], [h_line_y, h_line_y], color='#777676', linewidth=1, zorder=3)
    ax.plot([v_line_x, v_line_x], [bottom_y, h_line_y], color='#777676', linewidth=1, zorder=3)
    
    # Legend Contour
    leg_y = len(models_to_plot) + 0.4
    leg_x_start = .8
    ax.add_patch(patches.FancyBboxPatch(
        (leg_x_start, leg_y - 0.2), 
        1.7, 
        0.8,
        boxstyle="round,pad=0.02,rounding_size=0.1",
        linewidth=1, 
        edgecolor='#CCCCCC', 
        facecolor='none', 
        zorder=4
    ))
    
    ax.text(leg_x_start + 0.05, leg_y + 0.25, "Ranking", fontsize=15, color='#666666')
    ax.text(leg_x_start + 0.05, leg_y + 0.05, "column-wise", fontsize=11, color='#888888', va='center')
    ax.text(leg_x_start + 0.2, leg_y - 0.4, "↓", fontsize=20, color='#888888', ha='center')
    
    # Legend 1st
    ax.add_patch(patches.FancyBboxPatch((leg_x_start + 0.68, leg_y - 0.1), 0.25, 0.6, 
                 boxstyle="round,pad=0.02,rounding_size=0.1", edgecolor=sage_green, facecolor=sage_green_interior))
    ax.text(leg_x_start + 0.8, leg_y + 0.1, "1st", ha='center', fontsize=13)
    
    # Legend 2nd
    ax.add_patch(patches.FancyBboxPatch((leg_x_start + 1.03, leg_y - 0.1), 0.25, .55, 
                 boxstyle="round,pad=0.02,rounding_size=0.11", edgecolor=light_blue, facecolor=light_blue_interior))
    ax.text(leg_x_start + 1.15, leg_y + 0.1, "2nd", ha='center', fontsize=13)
    
    # Legend 3rd
    ax.add_patch(patches.FancyBboxPatch((leg_x_start + 1.38, leg_y - 0.1), 0.25, .50, 
                 boxstyle="round,pad=0.02,rounding_size=0.1", edgecolor=dark_blue, facecolor=dark_blue_interior))
    ax.text(leg_x_start + 1.5, leg_y + 0.1, "3rd", ha='center', fontsize=13)


    # Draw Rows
    for row_idx, model_id in enumerate(reversed(models_to_plot)):
        y_pos = row_idx  
        ax.text(v_line_x - 0.06, y_pos, model_id.lower(), ha='right', va='center', fontsize=13, color='black')

        for col_idx, col_name in enumerate(metrics):
            val = df_final.loc[model_id, col_name]
            rank = ranks.loc[model_id, col_name]
            x_pos = x_positions[col_idx]
            
            # Coloring Logic based on Rank per column
            draw_patch = False
            text_color = unranked_grey
            
            if rank == 1:
                color, edgecolor = sage_green_interior, sage_green
                draw_patch = True
                text_color = 'black'
                box_w = 0.52 if col_name in ["params", "gpu_mem_mb"] else 0.31
                box_h = 0.8
                r_style="round,pad=0.01,rounding_size=0.15"
            elif rank == 2:
                color, edgecolor = light_blue_interior, light_blue
                draw_patch = True
                text_color = 'black'
                box_w = 0.49 if col_name in ["params", "gpu_mem_mb"] else 0.28
                box_h = 0.7
                r_style="round,pad=0.01,rounding_size=0.17"
            elif rank == 3:
                color, edgecolor = dark_blue_interior, dark_blue
                draw_patch = True
                text_color = 'black'
                box_w = 0.47 if col_name in ["params", "gpu_mem_mb"] else 0.26
                box_h = 0.6
                r_style="round,pad=0.01,rounding_size=0.19"
                
            if draw_patch:
                rect = patches.FancyBboxPatch(
                    (x_pos - box_w/2, y_pos - box_h/2), 
                    box_w, box_h,
                    boxstyle=r_style,
                    linewidth=2,
                    facecolor=color,
                    edgecolor=edgecolor,
                    mutation_scale=.7, 
                    zorder=1
                )
                ax.add_patch(rect)
            
            # Formatting
            if col_name == 'accuracy':
                text_val = f"{val/100:.3f}" if val > 1 else f"{val:.3f}"
            elif col_name == 'val_loss':
                text_val = f"{val:.4f}"
            elif col_name == 'params':
                text_val = f"{int(val):,}"
            else:
                text_val = f"{val:.3f}"
            
            # Zero-stripping 
            if text_val.startswith("0."):
                text_val = text_val[1:]
            
            ax.text(x_pos, y_pos, text_val, ha='center', va='center', fontsize=13, color=text_color, zorder=2)

    plt.savefig(output_path, dpi=300, bbox_inches='tight', transparent=True)
    plt.show()

def show_tensor_image(image):
    reverse_transforms = transforms.Compose([
        transforms.Lambda(lambda t: (t + 1) / 2),
        transforms.Lambda(lambda t: torch.minimum(torch.tensor([1]), t)),
        transforms.Lambda(lambda t: torch.maximum(torch.tensor([0]), t)),
        transforms.ToPILImage(),
    ])
    plt.imshow(reverse_transforms(image[0].detach().cpu()))

def plot_bonus_tsk_confidence_distribution(model, loader, device):
    model.eval()
    correct_confidences = []
    incorrect_confidences = []

    with torch.no_grad():
        for data, target in loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            
            # Get actual probabilities
            probs = torch.exp(output) 
            conf, pred = torch.max(probs, dim=1)
            
            correct_mask = pred.eq(target)
            correct_confidences.extend(conf[correct_mask].cpu().numpy())
            incorrect_confidences.extend(conf[~correct_mask].cpu().numpy())

    # Styling Block
    sns.set_style("white") # White background
    plt.rcParams['figure.dpi'] = 100
    
    fig, ax = plt.subplots(figsize=(5, 4))
    
    # Define colors (Light Blue and Soft Pink/Purple)
    color_correct = "#A0C4FF" 
    color_incorrect = "#FFC2D1"

    # Plot Correct Predictions
    sns.histplot(correct_confidences, bins=30, kde=True, color=color_correct, 
                 label='Correct', stat="density", alpha=0.8, edgecolor=None)
    
    # Plot Incorrect Predictions
    sns.histplot(incorrect_confidences, bins=30, kde=False, color=color_incorrect, 
                 label='Incorrect', stat="density", alpha=0.8, edgecolor=None)

    # Styling the KDE line of Correct Predictions (making it thicker and darker)
    for line in ax.lines:
        line.set_linewidth(2)
        line.set_color("#333333") # Dark charcoal gray
        line.set_alpha(0.8)

    # Specify fontsizes of Title and Axes
    ax.set_xlabel("Confidence", fontsize=16, labelpad=10)
    ax.set_ylabel("Density", fontsize=16, labelpad=10)
    
    # Thick axes lines and removing top/right spines
    sns.despine() # Removes top and right border
    ax.spines['left'].set_linewidth(2)
    ax.spines['bottom'].set_linewidth(2)
    
    # Adjust tick thickness and size
    ax.tick_params(direction='out', length=6, width=2, labelsize=14)

    plt.legend(frameon=True, fontsize=12)
    plt.tight_layout()
    plt.show()

    return correct_confidences, incorrect_confidences

def find_stability_limit(coverages, accuracies):
    """Finds the 'elbow' point furthest from the line connecting start and end."""
    x = np.array(coverages)
    y = np.array(accuracies)
    
    # Define the start and end points of the curve
    p1 = np.array([x[0], y[0]])
    p2 = np.array([x[-1], y[-1]])
    
    # Calculate the perpendicular distance from each point to the line p1-p2
    # Formula: distance = |(y2-y1)x0 - (x2-x1)y0 + x2y1 - y2x1| / sqrt((y2-y1)^2 + (x2-x1)^2)
    numerator = np.abs((p2[1]-p1[1])*x - (p2[0]-p1[0])*y + p2[0]*p1[1] - p2[1]*p1[0])
    denominator = np.sqrt((p2[1]-p1[1])**2 + (p2[0]-p1[0])**2)
    
    distances = numerator / (denominator + 1e-10) # 1e-10 prevents divide by zero
    
    # The 'knee' or 'elbow' is the point with the maximum distance
    return np.argmax(distances)

def plot_bonus_tsk_acc_coverage_curve(coverages, accuracies):
    # Find point where accuracy starts dropping 
    knee_idx = find_stability_limit(coverages, accuracies)
    knee_acc = accuracies[knee_idx]
    knee_cov = coverages[knee_idx]

    # Styling Block
    sns.set_style("white")
    plt.rcParams['figure.dpi'] = 100
    
    fig, ax = plt.subplots(figsize=(5, 4))
    
    # Plot the line 
    ax.plot(coverages, accuracies, color="#16867B", linewidth=3, label='Model Performance')
    
    # Draw the charcoal "cliff" line
    ax.axhline(y=knee_acc, color="#333333", linestyle='--', linewidth=2, alpha=0.7)
    
    # Mark the intersection point (the 'elbow')
    ax.scatter(knee_cov, knee_acc, color="#333333", s=50, zorder=5)

    # Adjust fontsize of Title and Axes
    ax.set_xlabel("Coverage (Fraction of data accepted)", fontsize=15, labelpad=10)
    ax.set_ylabel("Acc. on Accepted Samples", fontsize=15, labelpad=10)
    
    # Remove top and right spines
    sns.despine()
    
    # Make the left and bottom axes thicker
    ax.spines['left'].set_linewidth(2.5)
    ax.spines['bottom'].set_linewidth(2.5)
    
    # Adjust tick thickness and size to match
    ax.tick_params(direction='out', length=6, width=2.5, labelsize=13)
    
    # Add a subtle grid only for the Y axis to help read accuracy levels
    # ax.grid(axis='y', linestyle='--', alpha=0.3)

    plt.tight_layout()
    plt.show()
    
    return knee_acc, knee_cov