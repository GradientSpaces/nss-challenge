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
import os

import numpy as np

from .metrics.rmse import compute_pairwise_rmse
from .metrics.geometric import evaluate_geometric_error


def load_json(path):
    """Load a JSON file."""
    with open(path, "r", encoding="utf8") as f:
        return json.load(f)


def evaluate(args):
    """Evaluate the performance of the predicted pose graph."""
    ground_truth = load_json(args.prediction)
    prediction = load_json(args.target)
    assert len(ground_truth) == len(prediction), "Length of ground truth and submission are not equal."

    # Compute pairwise metrics for each scene
    metrics_per_scene = {}
    for gt_graph, pred_graph in zip(ground_truth, prediction):
        assert gt_graph["name"] == pred_graph["name"], f"Scene names do not match for {gt_graph['name']}."
        metrics = evaluate_geometric_error(
            gt_graph, pred_graph, args.translation_threshold, args.rotation_threshold
        )

        if args.point_cloud_dir is not None:
            if os.path.exists(args.point_cloud_dir):
                rmses = compute_pairwise_rmse(
                    gt_graph, pred_graph, base_dir=args.point_cloud_dir
                )
                metrics.update(rmses)
            else:
                print(f"Point cloud directory not found: {args.point_cloud_dir}")
        else:
            print("Point cloud directory not provided. Skipping pairwise RMSE evaluation.")

        metrics_per_scene[gt_graph["name"]] = metrics
    
    # Average the metrics over all scenes
    metrics_overall = {}
    for metric in metrics_per_scene[ground_truth[0]["name"]]:
        metrics_overall[metric] = np.mean(
            [metrics_per_scene[scene][metric] for scene in ground_truth]
        )
    return metrics_overall
        

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
    parser.add_argument(
        "--translation_threshold",
        type=float,
        default=0.1,
        help="Threshold (in meters) for translation error to consider successfully aligned."
    )
    parser.add_argument(
        "--rotation_threshold",
        type=float,
        default=10,
        help="Threshold (in degrees) for rotation error to consider successfully aligned."
    )

    args = parser.parse_args()
    metrics = evaluate(args)

    print("Evaluation results:")
    for metric, value in metrics.items():
        print(f"{metric}: {value:.4f}")
    
