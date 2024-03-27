"""Tool to convert panorama images to point clouds.

Usage:
    python pan_to_pcd.py -i <input_dir> -o <output_dir> -t <threshold> -d <dmax>
"""

import argparse
import functools
import multiprocessing
import os
from telnetlib import DM

import cv2
import numpy as np
import open3d as o3d
import tqdm


def pan_to_xyz(pan_img, threshold=4.5, dmax=10.0):
    """Convert a panorama image to 3D coordinates."""

    height, width = pan_img.shape
    lon = np.linspace(0, 2 * np.pi, width)
    lat = np.linspace(np.pi / 2, -np.pi / 2, height)

    lon, lat = np.meshgrid(lon, lat)
    depth = pan_img / 65535.0 * dmax  # Normalize depth values

    mask = depth <= threshold  # Apply depth threshold
    
    # Calculate x, y, z coordinates
    z = depth * np.sin(lat)
    x = -depth * np.cos(lat) * np.sin(lon)
    y = depth * np.cos(lat) * np.cos(lon)
    x, y, z = x.flatten(), y.flatten(), z.flatten()
    mask = mask.flatten()
    
    pcd = np.concatenate([x[:, None], y[:, None], z[:, None]], axis=-1)
    return pcd[mask]


def process_single(filename, threshold, dmax):
    """Process a single panorama file to generate a point cloud."""
    outname = SAVE_DIR + filename.replace(".png", f".{int(100 * threshold)}.ply")
    pan_img = cv2.imread(PAN_DIR + filename, cv2.IMREAD_ANYDEPTH)
    xyz = pan_to_xyz(pan_img, threshold=threshold, dmax=dmax)
    
    # Create and downsample the point cloud
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz)
    downpcd = pcd.voxel_down_sample(voxel_size=0.075)
    o3d.io.write_point_cloud(outname, downpcd)


def process(threshold, dmax, n_jobs=16):
    """Main process function to handle multiple files."""
    files = [f for f in os.listdir(PAN_DIR) if f.endswith(".png")]
    print("Number of images to convert:", len(files))
    
    func = functools.partial(process_single, threshold=threshold, dmax=dmax)
    with multiprocessing.Pool(n_jobs) as pool:
        list(tqdm.tqdm(pool.imap(func, files), total=len(files)))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert panorama images to point clouds.")
    parser.add_argument("-i", "--input", type=str, required=True, help="Input directory containing panorama images.")
    parser.add_argument("-o", "--output", type=str, required=True, help="Output directory for point cloud files.")
    parser.add_argument("-t", "--threshold", type=float, default=4.5, help="Depth threshold (m) for filtering points.")
    parser.add_argument("-d", "--dmax", type=float, default=10.0, help="Maximum depth (m) for point normalization.")
    parser.add_argument("-j", "--n_jobs", type=int, default=16, help="Number of parallel jobs to run.")
   
    args = parser.parse_args()
    PAN_DIR = args.input
    SAVE_DIR = args.output

    # Start processing
    os.makedirs(SAVE_DIR, exist_ok=True)
    process(args.threshold, args.dmax, args.n_jobs)