"""Compute F1 Score for the detection of outlier nodes between predicted and ground truth pose graphs."""


import numpy as np
from sklearn.metrics import f1_score


def get_outlier_nodes(nodes): 
    """Return True if the transformation for a node is the zero's matrix."""
    outliers = np.zeros(len(nodes))
    for i, node in enumerate(nodes):
        pose = np.array(node.get('global_transform', node.get('tsfm')))
        outliers[i] = np.all(pose == 0.0)
    return outliers


def compute_outlier_f1(gt_graph, pred_graph):
    """Compute the RMSE for the entire pose graph by merging all fragments.

    Args
    ----
        gt_graph (dict): Ground truth pose graph.
        pred_graph (dict): Predicted pose graph.

    Returns
    -------
        dict: Dict that includes the F1 Score for the entire pose graph.
    """
    outliers_true = get_outlier_nodes(gt_graph['nodes'])
    outliers_pred = get_outlier_nodes(pred_graph['nodes']) 
    
    f1 = f1_score(outliers_true, outliers_pred)
    return {"Outlier F1": f1 * 100}