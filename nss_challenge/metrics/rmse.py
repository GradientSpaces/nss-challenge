"""Compute RMSE between predicted and ground truth pose graphs."""


import os
import numpy as np

from ..utils.pointcloud import PointCloudCache, get_correspondences
from .common import get_edge_transforms, get_node_transforms, look_up_transforms


def _get_node_name_by_id(node_id, nodes):
    """Find the node name given a node ID."""
    for node in nodes:
        if node['id'] == node_id:
            return node['name']
    return None


def _compute_rmse(src_path, tgt_path, trans):
    """
    Compute the RMSE between corresponding points of source and target point clouds
    after applying a transformation to the source.
    """
    correspondences = get_correspondences(src_path, tgt_path, trans)
    if correspondences.size == 0:
        return float('inf')  # Return infinity if no correspondences found
    
    point_cloud_cache = PointCloudCache()
    src_points = point_cloud_cache.load(src_path)
    tgt_points = point_cloud_cache.load(tgt_path)
    src_points = src_points[correspondences[:, 0], :]
    tgt_points = tgt_points[correspondences[:, 1], :]
    distances = np.linalg.norm(src_points - tgt_points, axis=1)
    rmse = np.sqrt(np.mean(np.square(distances)))
    return rmse


def compute_pairwise_rmse(gt_graph, pred_graph, base_dir):
    """Compute the RMSE for each pair of fragments in the pose graph.
    
    Args
    ----
        gt_graph (dict): Ground truth pose graph.
        pred_graph (dict): Predicted pose graph.
        base_dir (str): Base directory where point clouds are stored.

    Returns
    -------
        rmses (list[float]): List of RMSEs for each pair of fragments in the pose graph.
    """
    rmses = []

    pred_nodes = pred_graph['nodes']
    pred_edges = pred_graph.get('edges', None)
    if pred_edges is None:
        pred_transforms = get_node_transforms(pred_nodes)
    else:
        pred_transforms = get_edge_transforms(pred_edges)

    for gt_edge in gt_graph['edges']:
        src_node_name = _get_node_name_by_id(gt_edge['source'], gt_graph['nodes'])
        tgt_node_name = _get_node_name_by_id(gt_edge['target'], gt_graph['nodes'])
        if pred_edges is not None:
            pred_tsfm = look_up_transforms(gt_edge['source'], gt_edge['target'], pred_transforms)
        else:
            pred_tsfm = look_up_transforms(gt_edge['source'], gt_edge['target'], pred_transforms, compute_pairwise=True)
            
        src_path = os.path.join(base_dir, src_node_name)
        tgt_path = os.path.join(base_dir, tgt_node_name)

        # Use predicted transformation
        trans = np.array(pred_tsfm)
        rmse = _compute_rmse(src_path, tgt_path, trans)
        rmses.append(rmse)

    overall_rmse = np.mean(rmses)
    return {"Pairwise RMSE": overall_rmse}
