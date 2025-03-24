"""Geometric registration error metrics."""


import numpy as np

from ..utils.eval import get_rot_trans_error
from ..utils.logging import get_logger
from .common import (
    has_transform_on_all_edges,
    get_edge_transforms,
    get_node_transforms, 
    look_up_transforms, 
    filter_outlier_nodes
)


logger = get_logger("Metrics")


def evaluate_geometric_error(gt_graph, pred_graph, translation_threshold, rotation_threshold):
    """Evaluates geometric registration errors.
    
    Args
    ----
        edges_gt (list[dict]): List of ground truth edges with 'source_id', 'target_id', and 'relative_transform'.
        edges_pred (list[dict]): List of predicted edges with 'source_id', 'target_id', and 'relative_transform'.
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
    _, pred_nodes = filter_outlier_nodes(gt_graph['nodes'], pred_graph['nodes']) 
    pred_edges = pred_graph.get('edges', None)

    compute_pairwise = pred_edges is None or not has_transform_on_all_edges(pred_edges)

    if compute_pairwise:
        pred_trans = get_node_transforms(pred_nodes)
        logger.info("No valid edge prediction found in the graph %s, using predicted node transformations.", gt_graph['name'])
    else:
        pred_trans = get_edge_transforms(pred_edges)

    for gt_edge in gt_edges:
        gt_tsfm = np.array(gt_edge.get('relative_transform', gt_edge.get('tsfm')))
        src_node_id = gt_edge.get('source_id', gt_edge.get('source'))
        tgt_node_id = gt_edge.get('target_id', gt_edge.get('target'))
        pred_tsfm = look_up_transforms(src_node_id, tgt_node_id, pred_trans, compute_pairwise=compute_pairwise)
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
        'Registration Recall': recall * 100,
        'Average Translation Error': avg_translation_error,
        'Average Rotation Error': avg_rotation_error
    }
    return metrics