#!/usr/bin/env python3
"""Tests for the MotionForge three-view Diffusion Policy observation bridge."""

from __future__ import annotations

import re
import unittest
from pathlib import Path

import motionforge_diffusion_policy_bridge_client as bridge
import numpy as np
import torch

WORKSPACE_ROOT = Path(__file__).resolve().parents[2]
CHECKPOINT = (
    WORKSPACE_ROOT
    / "ckpts"
    / "MotionforgeGroup"
    / "DiffusionPolicy"
    / "FC-80000-3views"
)
CM_CHECKPOINT = (
    WORKSPACE_ROOT
    / "ckpts"
    / "MotionforgeGroup"
    / "DiffusionPolicy"
    / "CM-80000-3views"
)


class ThreeViewObservationTest(unittest.TestCase):
    """Verify that runtime observations reproduce the checkpoint training contract."""

    @classmethod
    def setUpClass(cls) -> None:
        """Load the local checkpoint configuration and deterministic resize once."""
        bridge.validate_checkpoint_files(CHECKPOINT)
        cls.config = bridge.load_diffusion_config(CHECKPOINT, "cpu")
        cls.image_transform = bridge.load_training_image_transform(CHECKPOINT)

    def test_checkpoint_contract_matches_training_configuration(self) -> None:
        """Require checkpoint policy and dataset metadata to remain synchronized."""
        bridge.validate_diffusion_contract(CHECKPOINT, self.config)

    def test_cm_checkpoint_contract_and_training_transform(self) -> None:
        """Accept the CM checkpoint only with its serialized three-view resize."""
        bridge.validate_checkpoint_files(CM_CHECKPOINT)
        config = bridge.load_diffusion_config(CM_CHECKPOINT, "cpu")
        bridge.validate_diffusion_contract(CM_CHECKPOINT, config)
        image_transform = bridge.load_training_image_transform(CM_CHECKPOINT)
        message = bridge.synthetic_message()
        observation = bridge.build_diffusion_observation(
            message,
            config.input_features,
            image_transform,
        )

        self.assertEqual(set(observation), {bridge.STATE_KEY, *bridge.IMAGE_KEYS})
        for key in bridge.IMAGE_KEYS:
            self.assertEqual(tuple(observation[key].shape), (3, 240, 320))
            self.assertEqual(observation[key].dtype, torch.float32)

    def test_all_views_apply_the_training_transform_before_normalization(self) -> None:
        """Apply the serialized training transform identically to all three views."""
        message = bridge.synthetic_message()
        for key, shape in bridge.IMAGE_NATIVE_HWC_SHAPES.items():
            message[key] = np.arange(np.prod(shape), dtype=np.uint8).reshape(shape)

        observation = bridge.build_diffusion_observation(
            message,
            self.config.input_features,
            self.image_transform,
        )

        self.assertEqual(set(observation), {bridge.STATE_KEY, *bridge.IMAGE_KEYS})
        for key in bridge.IMAGE_KEYS:
            source = torch.from_numpy(np.ascontiguousarray(message[key].transpose(2, 0, 1)))
            expected = self.image_transform(source).to(dtype=torch.float32).div_(255.0)
            self.assertEqual(tuple(observation[key].shape), (3, 240, 320))
            self.assertEqual(observation[key].dtype, torch.float32)
            torch.testing.assert_close(observation[key], expected, rtol=0, atol=0)

    def test_missing_wrist_view_is_rejected(self) -> None:
        """Reject observations that omit the checkpoint-required wrist view."""
        message = bridge.synthetic_message()
        del message["observation.images.wrist"]

        with self.assertRaisesRegex(KeyError, re.escape("observation.images.wrist")):
            bridge.build_diffusion_observation(
                message,
                self.config.input_features,
                self.image_transform,
            )

    def test_wrong_wrist_native_shape_is_rejected(self) -> None:
        """Reject wrist data that does not match MotionForge's native resolution."""
        message = bridge.synthetic_message()
        message["observation.images.wrist"] = np.zeros((240, 320, 3), dtype=np.uint8)

        with self.assertRaisesRegex(ValueError, "expected native HWC"):
            bridge.build_diffusion_observation(
                message,
                self.config.input_features,
                self.image_transform,
            )

    def test_non_uint8_wrist_view_is_rejected(self) -> None:
        """Reject wrist data that bypasses the training dataset's uint8 contract."""
        message = bridge.synthetic_message()
        message["observation.images.wrist"] = np.zeros((160, 160, 3), dtype=np.float32)

        with self.assertRaisesRegex(TypeError, "expected uint8"):
            bridge.build_diffusion_observation(
                message,
                self.config.input_features,
                self.image_transform,
            )


if __name__ == "__main__":
    unittest.main()
