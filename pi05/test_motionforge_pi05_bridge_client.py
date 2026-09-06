#!/usr/bin/env python3
"""Lightweight lifecycle tests for the MotionForge PI0.5 bridge."""

from __future__ import annotations

import unittest
from types import SimpleNamespace

import motionforge_pi05_bridge_client as bridge
import torch


class LifecycleResetTest(unittest.TestCase):
    """Verify RESET owns PI0.5 model state and flow-matching RNG state."""

    def test_reset_replays_server_seed(self) -> None:
        class Resettable:
            def __init__(self) -> None:
                self.reset_calls = 0

            def reset(self) -> None:
                self.reset_calls += 1

        inference = object.__new__(bridge.PI05Inference)
        inference.policy = Resettable()
        inference.preprocessor = Resettable()
        inference.postprocessor = Resettable()
        inference.noise_generator = torch.Generator(device="cpu")
        inference._current_policy_seed = 0

        reset = SimpleNamespace(seed=9876)
        inference.reset(reset)
        first = torch.randn(8, generator=inference.noise_generator)
        inference.reset(reset)
        second = torch.randn(8, generator=inference.noise_generator)

        self.assertEqual(inference.current_policy_seed, 9876)
        torch.testing.assert_close(first, second, rtol=0, atol=0)
        for component in (
            inference.policy,
            inference.preprocessor,
            inference.postprocessor,
        ):
            self.assertEqual(component.reset_calls, 2)

    def test_reset_rejects_seed_outside_protocol_range(self) -> None:
        inference = object.__new__(bridge.PI05Inference)
        with self.assertRaisesRegex(ValueError, "RESET seed"):
            inference.reset(SimpleNamespace(seed=-1))


if __name__ == "__main__":
    unittest.main()
