# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from gr00t.configs.data.embodiment_configs import register_modality_config
from gr00t.data.embodiment_tags import EmbodimentTag
from gr00t.data.types import (
    ActionConfig,
    ActionFormat,
    ActionRepresentation,
    ActionType,
    ModalityConfig,
)


fc003_panda_config = {
    "video": ModalityConfig(
        delta_indices=[0],
        modality_keys=["overview", "front", "wrist"],
    ),
    "state": ModalityConfig(
        delta_indices=[0],
        modality_keys=["joint_position", "gripper_width"],
    ),
    "action": ModalityConfig(
        delta_indices=list(range(16)),
        modality_keys=["eef_pose_rot6d", "gripper"],
        action_configs=[
            ActionConfig(
                rep=ActionRepresentation.ABSOLUTE,
                type=ActionType.EEF,
                format=ActionFormat.XYZ_ROT6D,
                state_key=None,
            ),
            ActionConfig(
                rep=ActionRepresentation.ABSOLUTE,
                type=ActionType.NON_EEF,
                format=ActionFormat.DEFAULT,
                state_key=None,
            ),
        ],
    ),
    "language": ModalityConfig(
        delta_indices=[0],
        modality_keys=["annotation.human.task_description"],
    ),
}


register_modality_config(fc003_panda_config, embodiment_tag=EmbodimentTag.NEW_EMBODIMENT)
