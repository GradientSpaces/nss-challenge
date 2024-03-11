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
- In the submission, only the `id`, `tsfm` fields in the nodes and `source`, `target`, `tsfm` fields in edges are considered.
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

from .metrics.rmse import compute_pairwise_rmse, compute_global_rmse
from .metrics.geometric import evaluate_geometric_error
from .utils.logging import get_logger, format_table


logger = get_logger("Evaluator")

METRICS = [
    {"name": "Global RMSE", "unit": "m"},
    {"name": "Pairwise RMSE", "unit": "m"},
    {"name": "Registration Recall", "unit": "%"},
    {"name": "Average Translation Error", "unit": "m"},
    {"name": "Average Rotation Error", "unit": "deg"},
]

SUBSET = ["All", "Same-Stage", "Cross-Stage"]


def load_json(path):
    """Load a JSON file."""
    with open(path, "r", encoding="utf8") as f:
        return json.load(f)


def evaluate_graph(gt_graph, pred_graph, translation_threshold, rotation_threshold, point_cloud_dir=None):
    """Evaluate the performance of the predicted pose graph."""
    metrics = evaluate_geometric_error(
        gt_graph, pred_graph, translation_threshold, rotation_threshold
    )
    if point_cloud_dir is not None:
        if os.path.exists(point_cloud_dir):
            pairwise_rmse = compute_pairwise_rmse(
                gt_graph, pred_graph, base_dir=point_cloud_dir
            )
            global_rmse = compute_global_rmse(
                gt_graph, pred_graph, base_dir=point_cloud_dir
            )
            metrics.update(pairwise_rmse)
            metrics.update(global_rmse)
    return metrics


def filter_edges(graph, same_stage=True):
    """Filter edges of the graph based on whether they are same-stage or cross-stage."""
    if "edges" not in graph:
        return graph
    filtered_edges = [edge for edge in graph["edges"] if edge["same_stage"] == same_stage]
    return {**graph, "edges": filtered_edges}


def evaluate_scene(gt_graph, pred_graph, translation_threshold, rotation_threshold, point_cloud_dir=None):
    """Evaluate the performance of the predicted pose graph for a single scene."""
    logger.info("Evaluating scene: %s", gt_graph["name"])

    # 1. Evaluate the full pose graph
    metrics_all = evaluate_graph(gt_graph, pred_graph, translation_threshold, rotation_threshold, point_cloud_dir)

    # 2. Evaluate the pose graph with only same-stage edges
    gt_same_stage = filter_edges(gt_graph, same_stage=True)
    pred_same_stage = filter_edges(pred_graph, same_stage=True)
    metrics_same_stage = evaluate_graph(gt_same_stage, pred_same_stage, translation_threshold, rotation_threshold, point_cloud_dir)

    # 3. Evaluate the pose graph with only cross-stage edges
    gt_cross_stage = filter_edges(gt_graph, same_stage=False)
    pred_cross_stage = filter_edges(pred_graph, same_stage=False)
    metrics_cross_stage = evaluate_graph(gt_cross_stage, pred_cross_stage, translation_threshold, rotation_threshold, point_cloud_dir)

    return {
        "All": metrics_all,
        "Same-Stage": metrics_same_stage,
        "Cross-Stage": metrics_cross_stage,
    }


def evaluate(args):
    """Evaluate the performance of the predicted pose graph."""
    # Load ground truth and prediction
    logger.info("Loading ground truth file: %s", args.target)
    ground_truth = load_json(args.target)
    logger.info("Loading prediction file: %s", args.prediction)
    prediction = load_json(args.prediction)
    if len(ground_truth) != len(prediction):
        logger.error("Number of scenes of ground truth and submission are not equal.")
        logger.info("Ground truth: %d scenes, Prediction: %d scenes", len(ground_truth), len(prediction))
        return
    
    if args.point_cloud_dir is not None:
        if not os.path.exists(args.point_cloud_dir):
            logger.error("Point cloud directory doesn't exist: %s", args.point_cloud_dir)
            return
    else:
        logger.info("Point cloud directory not provided. Skipping the RMSE evaluation.")

    # Compute metrics for each scene
    metrics = {}
    for gt_graph, pred_graph in zip(ground_truth, prediction):
        metrics_per_scene = evaluate_scene(
            gt_graph, pred_graph, args.translation_threshold, args.rotation_threshold, args.point_cloud_dir
        )
        metrics[gt_graph["name"]] = metrics_per_scene
    return metrics


def log_result(metrics):
    """Log the evaluation results."""

    # Average over all scenes
    metrics_overall = {}
    scenes = list(metrics.keys())
    for m in METRICS:
        for subset in SUBSET:
            name, unit = m["name"], m["unit"]
            if name in metrics[scenes[0]]["All"]:
                metrics_overall[f"{name} [{unit}]/{subset}"] = np.mean(
                    [metrics[scene_name][subset][name] for scene_name in scenes]
                )

    table = format_table(metrics_overall, 'Overall')
    logger.info("Results:\n\n%s", table)

    # Per-scene results
    log_str = ""
    for scene_name, metrics_per_scene in metrics.items():
        for m in METRICS:
            for subset in SUBSET:
                name, unit = m["name"], m["unit"]
                if name in metrics[scenes[0]]["All"]:
                    metrics_overall[f"{name} [{unit}]/{subset}"] = metrics_per_scene[subset][name]
        table = format_table(metrics_overall, scene_name)
        log_str += "\n\n" + table
    logger.info("Results for each scene:%s", log_str)



if __name__ == "__main__":
    parser = ArgumentParser(
        description="Evaluator for pose graphs in the NSS Challenge.",
        usage="python -m nss_challenge.evaluate_registration <target> <prediction> [--translation_threshold] [--rotation_threshold] [--point_cloud_dir]",
        epilog="For more details, please refer to the challenge website: https://nothing-stands-still.com/challenge",
    )
    parser.add_argument(
        "target",
        type=str,
        help="Path to target json file."
    )
    parser.add_argument(
        "prediction", 
        type=str,
        help="Path to prediction json file."
    )
    parser.add_argument(
        "--translation_threshold",
        "-t",
        type=float,
        default=0.1,
        help="Threshold (in meters) for translation error to consider successfully aligned."
    )
    parser.add_argument(
        "--rotation_threshold",
        "-r",
        type=float,
        default=10,
        help="Threshold (in degrees) for rotation error to consider successfully aligned."
    )
    parser.add_argument(
        "--point_cloud_dir",
        "-p",
        type=str,
        default=None,
        help="Directory containing point clouds for RMSE evaluation."
    )

    args = parser.parse_args()
    metrics = evaluate(args)
    log_result(metrics)
    logger.info("Evaluation complete.")
