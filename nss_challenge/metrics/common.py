"""Helper functions for the metrics evaluation."""

import numpy as np

from ..utils.logging import get_logger


logger = get_logger("Metrics")


def has_transform_on_all_edges(edges):
    """Check if all edges has non-empty tsfm."""
    return all(edge['tsfm'] is not None for edge in edges)


def get_node_transforms(nodes):
    """Precompute the transformation for each node in the pose graph for quick lookup."""
    transforms = {}
    for node in nodes:
        node_id = node['id']
        pose = np.array(node['tsfm'])
        transforms[node_id] = pose
    return transforms


def get_edge_transforms(edges):
    """Precompute the transformation for each edge in the pose graph for quick lookup."""
    transforms = {}
    for edge in edges:
        src_node_id = edge['source']
        tgt_node_id = edge['target']
        edge_tsfm = np.array(edge['tsfm'])
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
