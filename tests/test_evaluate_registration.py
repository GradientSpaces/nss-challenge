"""Test case for evaluate_registration module."""

import os
import unittest

import numpy as np

from nss_challenge.evaluate_registration import evaluate


class Args:
    """Helper class to convert dictionary to object."""
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)


class TestEvaluateRegistration(unittest.TestCase):

    def setUp(self):
        self.args = Args(
            prediction=os.path.join(
                os.path.dirname(__file__), "testdata", "mock_target_2025.json"
            ),
            target=os.path.join(
                os.path.dirname(__file__), "testdata", "mock_target_2025.json"
            ),
            translation_threshold=0.1,
            rotation_threshold=10,
            point_cloud_dir=os.path.join(
                os.path.dirname(__file__), "testdata", "pointclouds"
            ),
        )

    def test_evaluate(self):
        metrics = evaluate(self.args)
        self.assertIsInstance(metrics, dict)
        
        metrics_scene = metrics["Bldg0_Scene1"]
        self.assertEqual(len(metrics_scene), 3)

        metric = metrics_scene["All"]
        self.assertTrue(
            np.allclose(metric["Pairwise RMSE"], 0.0, atol=1e-1)
        )
        self.assertTrue(
            np.allclose(metric["Outlier F1"], 100.0, atol=1e-1)
        )
        self.assertTrue(
            np.allclose(metric["Registration Recall"], 100.0, atol=1e-1)
        )
        self.assertTrue(
            np.allclose(metric["Average Rotation Error"], 0.0, atol=1e-3)
        )
        self.assertTrue(
            np.allclose(metric["Average Translation Error"], 0.0, atol=1e-3)
        )