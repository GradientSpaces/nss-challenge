"""NSS Challenge Evaluator for Pose Graphs.

Overview
--------
This script evaluates the performance of multiway registration methods for the
NSS Challenge. The pose graph of each scene is defined with nodes and edges 
where nodes represent individual point clouds and their poses, while edges
define the relationships (e.g., transformations and overlaps) between pairs of
point clouds. The evaluation is performed using the following metrics:

  - Global RMSE                     Measures the Root Mean Squared Error (RMSE) across all fragments in the global coordinate system.
  - Pairwise RMSE                   Measures the RMSE for each pair of fragments in the scene averaged across all pairs.
  - Recall                          The percentage of correctly aligned point pairs.
  - Precision                       
    - Average Translation Error     The average translation error of the correctly aligned point pairs.
    - Average Rotation Error        The average rotation error of the correctly aligned point pairs.

    
Format
------
We use JSON files defining the pose graphs for global and pairwise point cloud
registration evaluations. Each JSON file contains a list of pose graphs, each
representing a specific building scene.

- list[dict]:                       List of pose graphs.

  - name (str):                     Name for the building scene, formatted as "BldgX_SceneN".

  - nodes (list[dict]):             List of nodes representing point clouds within the scene.
    - id (int):                     Identifier (ID) for the node within its scene.
    - name (str):                   Name of the point cloud file, formatted as "BldgX_StageY_SpotZ.ply".
    - tsfm (list[list[float]]):     4x4 transformation matrix of the pose of the point cloud in global coordinates.
    - building (str):               Building name, matching "X" in the node name.
    - stage (str):                  Temporal stage, matching "Y" in the node name.
    - spot (str):                   Spot number,  matching "Z" in the node name.
    - points (int):                 Number of points in the point cloud.

  - edges (list[dict], optional):   List of edges representing pairwise relationships between nodes. Each edge is a dictionary:
    - source (int):                 Node ID of the source point cloud.
    - target (int):                 Node ID of the target point cloud.
    - tsfm (list[list[float]]):     4x4 transformation matrix of the relative pose from the source to the target.
    - overlap (float):              Overlap ratio between the source and target, ranging from 0.0 to 1.0.
    - temporal_change (float):      Temporal change ratio indicating the amount of temporal change between the source and target, ranging from 0.0 to 1.0.
    - same_stage (bool):            Indicates whether the source and target come from the same temporal stage.

Notes
-----
- For global pose evaluation, only the transformation in the `nodes` are  considered.
- For pairwise pose evaluation, metrics are computed over the defined edges. If `edges`
  are missing or partially missing in the prediction files, the missing parts will be 
  computed using the nodes' global poses.

For more details, please refer to the challenge website:
https://nothing-stands-still.com/challenge
"""

from argparse import ArgumentParser
import json

import numpy as np
import open3d as o3d


def load_json(path):
    with open(path, "r") as f:
        return json.load(f)
    

def evaluate_pairwise(edges_gt, edges_pred, translation_threshold, rotation_threshold_rad):
    """Evaluates pairwise registration metrics.
    
    Args
    ----
        edges_gt (list[dict]): List of ground truth edges with 'source', 'target', and 'tsfm'.
        edges_pred (list[dict]): List of predicted edges with 'source', 'target', and 'tsfm'.
        translation_threshold (float): Threshold for translation error to consider alignment correct.
        rotation_threshold_rad (float): Threshold for rotation error (in radians) to consider alignment correct.

    Returns
    -------
        metrics (dict): Dictionary with pairwise RMSE, recall, average translation error, and average rotation error.
    """
    total_translation_error = 0
    total_rotation_error = 0
    correctly_aligned_count = 0
    rmse_sum = 0
    
    for gt_edge, pred_edge in zip(edges_gt, edges_pred):
        gt_tsfm = np.array(gt_edge['tsfm'])
        pred_tsfm = np.array(pred_edge['tsfm'])
        translation_error, rotation_error = calculate_error(gt_tsfm, pred_tsfm)
        rmse_sum += translation_error**2 + rotation_error**2
        
        if translation_error <= translation_threshold and rotation_error <= rotation_threshold_rad:
            correctly_aligned_count += 1
            total_translation_error += translation_error
            total_rotation_error += rotation_error
    
    num_pairs = len(edges_gt)
    pairwise_rmse = np.sqrt(rmse_sum / num_pairs)
    recall = correctly_aligned_count / num_pairs if num_pairs > 0 else 0
    avg_translation_error = total_translation_error / correctly_aligned_count if correctly_aligned_count > 0 else 0
    avg_rotation_error = total_rotation_error / correctly_aligned_count if correctly_aligned_count > 0 else 0
    
    metrics = {
        'Pairwise RMSE': pairwise_rmse,
        'Registration Recall': recall,
        'Average Translation Error': avg_translation_error,
        'Average Rotation Error': avg_rotation_error
    }
    
    return metrics
        

def evaluate(prediction_path, ground_truth_path):
    ground_truth = load_json(ground_truth_path)
    prediction = load_json(prediction_path)
    assert len(ground_truth) == len(prediction), "Length of ground truth and submission are not equal."


if __name__ == "__main__":
    parser = ArgumentParser(
        "Evaluator for pose graphs in the NSS Challenge."
    )
    parser.add_argument(
        "prediction", 
        type=str,
        required=True,
        help="Path to prediction json file."
    )
    parser.add_argument(
        "target",
        type=str,
        required=True,
        help="Path to target json file."
    )

    args = parser.parse_args()
    evaluate(args.prediction, args.target)




