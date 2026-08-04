"""Input/output transforms for the 10-D MotionForge Franka conveyor action space."""

import dataclasses

import einops
import numpy as np

from openpi import transforms
from openpi.models import model as _model


def _parse_image(image) -> np.ndarray:
    image = np.asarray(image)
    if np.issubdtype(image.dtype, np.floating):
        image = (255 * image).clip(0, 255).astype(np.uint8)
    if image.shape[0] == 3:
        image = einops.rearrange(image, "c h w -> h w c")
    return image


@dataclasses.dataclass(frozen=True)
class FactoryConveyorInputs(transforms.DataTransformFn):
    model_type: _model.ModelType

    def __call__(self, data: dict) -> dict:
        overview = _parse_image(data["observation/overview"])
        wrist = _parse_image(data["observation/wrist"])
        if self.model_type == _model.ModelType.PI0_FAST:
            names = ("base_0_rgb", "base_1_rgb", "wrist_0_rgb")
            images = (overview, np.zeros_like(overview), wrist)
            masks = (np.True_, np.True_, np.True_)
        else:
            names = ("base_0_rgb", "left_wrist_0_rgb", "right_wrist_0_rgb")
            images = (overview, wrist, np.zeros_like(overview))
            masks = (np.True_, np.True_, np.False_)
        result = {
            "state": np.asarray(data["observation/state"], dtype=np.float32),
            "image": dict(zip(names, images, strict=True)),
            "image_mask": dict(zip(names, masks, strict=True)),
        }
        if "actions" in data:
            result["actions"] = np.asarray(data["actions"], dtype=np.float32)
        if "prompt" in data:
            result["prompt"] = data["prompt"]
        return result


@dataclasses.dataclass(frozen=True)
class FactoryConveyorOutputs(transforms.DataTransformFn):
    def __call__(self, data: dict) -> dict:
        return {"actions": np.asarray(data["actions"][:, :10])}
