import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

"""
The following functions are based on the notebooks provided for the Nvidia course Building AI Agents with Multimodal Models https://learn.nvidia.com/courses/course-detail?course_id=course-v1:DLI+C-FX-17+V1
"""


def get_lidar_projections(lidar_depth, azimuth, zenith):
    """
    Calculate X, Y, Z projections based on Nvidia/Omniverse LiDAR math.
    """
    # Create the projections using broadcasting
    # Azimuth: rows ([:, None]), Zenith: columns ([None, :])
    x = lidar_depth * np.sin(-azimuth[:, None]) * np.cos(-zenith[None, :])
    y = lidar_depth * np.cos(-azimuth[:, None]) * np.cos(-zenith[None, :])
    z = lidar_depth * np.sin(-zenith[None, :])
    
    # Create mask (a) for valid returns (max range 50.0 for this specific dataset; oterhwsie adjust accordingly)
    # This matches the 'a' in the Nvidia get_torch_xyza function
    mask = (lidar_depth < 50.0).astype(np.float32)
    
    return x, y, z, mask

def create_lidar_viz(npy_path, azimuth, zenith, output_path):
    """
    Converts a raw .npy LiDAR depth map into a colormapped .png for FiftyOne visualization.
    """
    data = np.load(npy_path)
    x, y, z, mask = get_lidar_projections(data, azimuth, zenith)
    
    # Filter points using the mask 'mask == 1' (matching Nvidia's 3D scatter logic)
    x_scatter = x[mask == 1]
    z_scatter = z[mask == 1] # Z is the vertical height
    # y_scatter = y[mask == 1]
    
    # (Z-normalization)
    if len(z_scatter) > 0:
        c_min, c_max = np.min(z_scatter), np.max(z_scatter)
        if c_max > c_min:
            colors = (z_scatter - c_min) / (c_max - c_min)
        else:
            colors = z_scatter
    else:
        colors = z_scatter

    fig = plt.figure(figsize=(2, 2), dpi=128, facecolor='black')
    ax = fig.add_subplot(111, facecolor='black')
    
    if len(x_scatter) > 0:
        ax.scatter(x_scatter, z_scatter, c=colors, cmap='inferno', s=5, edgecolors='none')

    # Adjust axes limit
    ax.set_xlim(-6, 6) 
    ax.set_ylim(-6, 6) 
    
    ax.set_axis_off()
    plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
    
    plt.savefig(output_path, bbox_inches='tight', pad_inches=0, facecolor='black')
    plt.close(fig)
    
    return output_path
        
    # return x_scatter, z_scatter #, output_path