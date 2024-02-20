"""Compute RMSE between predicted and ground truth pose graphs."""


import os
import numpy as np

from ..utils.open3d import get_correspondences, load_ply


def _get_node_name_by_id(node_id, nodes):
    """Find the node name given a node ID."""
    for node in nodes:
        if node['id'] == node_id:
            return node['name']
    return None


def _compute_rmse(src_pcd, tgt_pcd, trans, search_voxel_size=0.2):
    """
    Compute the RMSE between corresponding points of source and target point clouds
    after applying a transformation to the source.
    """
    correspondences = get_correspondences(src_pcd, tgt_pcd, trans, search_voxel_size)
    if correspondences.size == 0:
        return float('inf')  # Return infinity if no correspondences found

    src_points = np.asarray(src_pcd.points)[correspondences[:, 0], :]
    tgt_points = np.asarray(tgt_pcd.points)[correspondences[:, 1], :]
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
    for gt_edge, pred_edge in zip(gt_graph['edges'], pred_graph['edges']):
        src_node_name = _get_node_name_by_id(gt_edge['source'], gt_graph['nodes'])
        tgt_node_name = _get_node_name_by_id(gt_edge['target'], gt_graph['nodes'])
        src_pcd = load_ply(os.path.join(base_dir, src_node_name))
        tgt_pcd = load_ply(os.path.join(base_dir, tgt_node_name))

        # Use predicted transformation
        trans = np.array(pred_edge['tsfm'])
        rmse = _compute_rmse(src_pcd, tgt_pcd, trans)
        rmses.append(rmse)

    overall_rmse = np.sqrt(np.mean([rmse**2 for rmse in rmses]))
    return {"Pairwise RMSE": rmses, "Overall RMSE": overall_rmse}
