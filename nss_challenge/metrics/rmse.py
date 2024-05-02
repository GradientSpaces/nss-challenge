"""Compute RMSE between predicted and ground truth pose graphs."""


import os
import numpy as np

from ..utils.pointcloud import PointCloudCache, get_correspondences, transform_points
from .common import (
    has_transform_on_all_edges,
    get_edge_transforms,
    get_node_transforms,
    look_up_transforms
)


def _get_node_name_by_id(node_id, nodes):
    """Find the node name given a node ID."""
    for node in nodes:
        if node['id'] == node_id:
            return node['name']
    return None


def _compute_rmse(src_path, tgt_path, trans, gt_trans):
    """
    Compute the RMSE between corresponding points of source and target point clouds
    after applying a transformation to the source.
    """
    correspondences = get_correspondences(src_path, tgt_path, gt_trans)
    if correspondences.size == 0:
        return float('inf')  # Return infinity if no correspondences found
    
    point_cloud_cache = PointCloudCache()
    src_points = point_cloud_cache.load(src_path)
    tgt_points = point_cloud_cache.load(tgt_path)
    
    src_points = transform_points(src_points, trans)
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
        dict: Dict that includes the average RMSE for each pair of fragments.
    """
    rmses = []

    pred_nodes = pred_graph['nodes']
    pred_edges = pred_graph.get('edges', None)

    compute_pairwise = pred_edges is None or not has_transform_on_all_edges(pred_edges)

    if compute_pairwise:
        pred_transforms = get_node_transforms(pred_nodes)
    else:
        pred_transforms = get_edge_transforms(pred_edges)

    for gt_edge in gt_graph['edges']:
        src_node_name = _get_node_name_by_id(gt_edge['source'], gt_graph['nodes'])
        tgt_node_name = _get_node_name_by_id(gt_edge['target'], gt_graph['nodes'])
        src_path = os.path.join(base_dir, src_node_name)
        tgt_path = os.path.join(base_dir, tgt_node_name)
        
        pred_trans = look_up_transforms(
            source=gt_edge['source'],
            target=gt_edge['target'],
            pred_transforms=pred_transforms,
            compute_pairwise=compute_pairwise
        )
        pred_trans = np.array(pred_trans)
        gt_trans = np.array(gt_edge['tsfm'])
        # breakpoint()

        rmse = _compute_rmse(src_path, tgt_path, pred_trans, gt_trans)
        rmses.append(rmse)

    overall_rmse = np.mean(rmses)
    return {"Pairwise RMSE": overall_rmse}


def compute_global_rmse(gt_graph, pred_graph, base_dir):
    """Compute the RMSE for the entire pose graph by merging all fragments.

    Args
    ----
        gt_graph (dict): Ground truth pose graph.
        pred_graph (dict): Predicted pose graph.
        base_dir (str): Base directory where point clouds are stored.

    Returns
    -------
        dict: Dict that includes the RMSE for the entire pose graph.
    """
    point_cloud_cache = PointCloudCache()
    gt_nodes = gt_graph['nodes']
    pred_nodes = pred_graph['nodes']
    gt_transforms = get_node_transforms(gt_nodes)
    pred_transforms = get_node_transforms(pred_nodes)

    
    points_gt = []
    points_pred = []

    # Find the anchor node and its transformation
    anchor_id = None
    for node in pred_nodes:
        if 'anchor' in node and node['anchor'] == True:
            anchor_id = node['id']
            break
    if anchor_id is None:
        # Use the first node in prediction as the anchor if it's not found
        anchor_id = pred_nodes[0]['id']

    anchor_gt = np.linalg.inv(gt_transforms.get(anchor_id, np.eye(4)))
    anchor_pred = np.linalg.inv(pred_transforms.get(anchor_id, np.eye(4)))

    for gt_node in gt_nodes:
        node_name = gt_node['name']
        path = os.path.join(base_dir, node_name)
        gt_trans = anchor_gt @ np.array(gt_node['tsfm'])
        pred_trans = anchor_pred @ np.array(pred_transforms.get(gt_node['id'], np.eye(4)))

        points_gt.append(
            transform_points(point_cloud_cache.load(path), gt_trans)
        )
        points_pred.append(
            transform_points(point_cloud_cache.load(path), pred_trans)
        )
    
    points_gt = np.concatenate(points_gt, axis=0)
    points_pred = np.concatenate(points_pred, axis=0)
    distances = np.linalg.norm(points_gt - points_pred, axis=1)
    rmse = np.sqrt(np.mean(np.square(distances)))
    return {"Global RMSE": rmse}
