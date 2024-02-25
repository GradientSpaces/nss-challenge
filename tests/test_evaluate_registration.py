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
                os.path.dirname(__file__), "testdata", "mock_target.json"
            ),
            target=os.path.join(
                os.path.dirname(__file__), "testdata", "mock_target.json"
            ),
            translation_threshold=0.1,
            rotation_threshold=10,
        )

    def test_evaluate(self):
        metrics = evaluate(self.args)
        self.assertIsInstance(metrics, dict)
        self.assertEqual(len(metrics), 4)
        self.assertTrue("pairwise_rmse" in metrics)
        self.assertTrue("recall" in metrics)
        self.assertTrue("avg_translation_error" in metrics)
        self.assertTrue("avg_rotation_error" in metrics)
        print(metrics)
        # self.assertTrue(
        #     np.allclose(metrics["pairwise_rmse"], 0.0, atol=1e-3)
        # )
        # self.assertTrue(
        #     np.allclose(metrics["recall"], 1.0, atol=1e-3)
        # )
        # self.assertTrue(
        #     np.allclose(metrics["avg_translation_error"], 0.0, atol=1e-3)
        # )
        # self.assertTrue(
        #     np.allclose(metrics["avg_rotation_error"], 0.0, atol=1e-3)
        # )