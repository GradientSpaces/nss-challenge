"""Helper functions for the metrics evaluation."""

import numpy as np
from itertools import compress

from ..utils.logging import get_logger


logger = get_logger("Metrics")


def has_transform_on_all_edges(edges):
    """Check if all edges have non-empty tsfm."""
    return all(edge.get('relative_transform', edge.get('tsfm')) is not None for edge in edges)


def get_node_transforms(nodes):
    """Precompute the transformation for each node in the pose graph for quick lookup."""
    transforms = {}
    for node in nodes:
        node_id = node['id']
        pose = np.array(node.get('global_transform', node.get('tsfm')))
        transforms[node_id] = pose
    return transforms


def filter_outlier_nodes(gt_nodes, pred_nodes=None):
    """Filter nodes of the graph if ground truth indicates outlier (transformation is the zero matrix)"""
    
    # non-outleir mask 
    mask = [not np.all(np.asarray(node.get('global_transform', node.get('tsfm'))) == 0) 
            for node in gt_nodes]
    
    filtered_gt = list(compress(gt_nodes, mask))
    if pred_nodes is None:
        return filtered_gt
    
    filtered_pred = list(compress(pred_nodes, mask))
    return filtered_gt, filtered_pred
    
def filter_edges(graph, same_stage=True):
    """Filter edges of the graph based on whether they are same-stage or cross-stage."""
    if "edges" not in graph:
        return graph
    filtered_edges = [edge for edge in graph["edges"] if edge["same_stage"] == same_stage]
    return {**graph, "edges": filtered_edges}

def get_edge_transforms(edges):
    """Precompute the transformation for each edge in the pose graph for quick lookup."""
    transforms = {}
    for edge in edges:
        src_node_id = edge.get('source_id', edge.get('source'))
        tgt_node_id = edge.get('target_id', edge.get('target'))
        edge_tsfm = np.array(edge.get('relative_transform', edge.get('tsfm')))
        k = (src_node_id, tgt_node_id)
        if k in transforms:
            logger.warning("Duplicate edge found in prediction: %s -> %s", src_node_id, tgt_node_id)
        transforms[k] = edge_tsfm
    return transforms


def look_up_transforms(source, target, pred_transforms, compute_pairwise=False):
    """Look up the transformation for the given source and target nodes."""
    if compute_pairwise:
        # Look up and compute the pairwise transformation.
        src_tsfm = pred_transforms.get(source, np.eye(4))
        tgt_tsfm = pred_transforms.get(target, np.eye(4))
        pred_tsfm = np.matmul(np.linalg.inv(tgt_tsfm), src_tsfm)
    else:
        # Look up the predicted transformation.
        k = (source, target)
        pred_tsfm = pred_transforms.get(k, np.eye(4))
    return pred_tsfm
