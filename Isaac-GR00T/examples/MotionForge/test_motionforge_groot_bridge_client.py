#!/usr/bin/env python3
"""Lightweight lifecycle tests for the MotionForge GR00T bridge."""

from __future__ import annotations

import unittest
from types import SimpleNamespace
from unittest import mock

import motionforge_groot_bridge_client as bridge
import numpy as np


class LifecycleResetTest(unittest.TestCase):
    """Verify server RESET owns GR00T history, policy state, and RNG seed."""

    def test_reset_clears_history_and_forwards_seed(self) -> None:
        class Policy:
            def __init__(self) -> None:
                self.reset_calls = 0

            def reset(self) -> None:
                self.reset_calls += 1

        inference = object.__new__(bridge.GR00TInference)
        inference.policy = Policy()
        inference.history = bridge.ObservationHistory(maxlen=2)
        inference.history.append({"stale": True})

        with mock.patch.object(bridge, "seed_policy_rng") as seed_rng:
            inference.reset(SimpleNamespace(seed=123))

        seed_rng.assert_called_once_with(123)
        self.assertEqual(len(inference.history.frames), 0)
        self.assertEqual(inference.policy.reset_calls, 1)

    def test_reset_rejects_seed_outside_protocol_range(self) -> None:
        inference = object.__new__(bridge.GR00TInference)
        with self.assertRaisesRegex(ValueError, "RESET seed"):
            inference.reset(SimpleNamespace(seed=2**63 - 1))


class ActionChunkTest(unittest.TestCase):
    """Verify GR00T action parts become the canonical 10D source chunk."""

    def test_combines_rot6d_pose_and_gripper(self) -> None:
        pose = np.zeros((1, 16, 9), dtype=np.float32)
        gripper = np.ones((1, 16, 1), dtype=np.float32)

        chunk = bridge.motionforge_action_chunk(
            action={"action.eef_pose_rot6d": pose, "action.gripper": gripper},
            action_key=None,
        )

        self.assertEqual(chunk.shape, (16, 10))
        self.assertEqual(chunk.dtype, np.float32)
        np.testing.assert_array_equal(chunk[:, -1], np.ones(16, dtype=np.float32))

    def test_rejects_legacy_8d_wire_action(self) -> None:
        with self.assertRaisesRegex(ValueError, r"\[T, 10\]"):
            bridge.motionforge_action_chunk(
                action={"action": np.zeros((1, 16, 8), dtype=np.float32)},
                action_key=None,
            )


if __name__ == "__main__":
    unittest.main()
