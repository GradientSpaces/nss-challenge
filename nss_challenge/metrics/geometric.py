"""Geometric registration error metrics."""


import numpy as np

from ..utils.eval import get_rot_trans_error


def _get_node_transforms(nodes):
    """Precompute the transformation for each node in the pose graph for quick lookup."""
    transforms = {}
    for node in nodes:
        name = node['name']
        pose = np.array(node['pose'])
        transforms[name] = pose
    return transforms


def evaluate_geometric_error(gt_graph, pred_graph, translation_threshold, rotation_threshold):
    """Evaluates geometric registration errors.
    
    Args
    ----
        edges_gt (list[dict]): List of ground truth edges with 'source', 'target', and 'tsfm'.
        edges_pred (list[dict]): List of predicted edges with 'source', 'target', and 'tsfm'.
        translation_threshold (float): Threshold for translation error to consider alignment correct.
        rotation_threshold (float): Threshold for rotation error (in degrees) to consider alignment correct.

    Returns
    -------
        metrics (dict): Dictionary with pairwise RMSE, recall, average translation error, and average rotation error.
    """
    _translation_error = 0
    _rotation_error = 0
    _success = 0

    gt_edges = gt_graph['edges']
    pred_nodes = pred_graph['nodes']
    pred_edges = pred_graph.get('edges', None)

    if pred_edges is None:
        pred_transforms = _get_node_transforms(pred_nodes)
        print(f"Using predicted node transformations for {gt_graph['name']}.")

    for idx, gt_edge in enumerate(gt_edges):
        gt_tsfm = np.array(gt_edge['tsfm'])
        if pred_edges is not None:
            pred_tsfm = np.array(pred_edges[idx]['tsfm'])
        else:
            # Use precomputed transformations for predicted nodes
            src_node_name = gt_graph['nodes'][gt_edge['source']]['name']
            tgt_node_name = gt_graph['nodes'][gt_edge['target']]['name']
            src_tsfm = pred_transforms.get(src_node_name, np.eye(4))
            tgt_tsfm = pred_transforms.get(tgt_node_name, np.eye(4))
            pred_tsfm = np.matmul(np.linalg.inv(src_tsfm), tgt_tsfm)
        rotation_error, translation_error = get_rot_trans_error(gt_tsfm, pred_tsfm)

        if translation_error <= translation_threshold and rotation_error <= rotation_threshold:
            _success += 1
            _translation_error += translation_error.item()
            _rotation_error += rotation_error.item()
    
    num_pairs = len(gt_edges)
    recall = _success / num_pairs if num_pairs > 0 else 0
    avg_translation_error = _translation_error / _success if _success > 0 else float('inf')
    avg_rotation_error = _rotation_error / _success if _success > 0 else float('inf')
    
    metrics = {
        'Registration Recall': recall,
        'Average Translation Error': avg_translation_error,
        'Average Rotation Error': avg_rotation_error
    }
    return metrics