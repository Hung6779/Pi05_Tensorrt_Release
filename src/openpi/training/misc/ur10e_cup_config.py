"""Reconstructed TrainConfig for NNDam/Pi0.5-UR10E-CUP (asset_id="local/ur10e-cup").

*** UNVERIFIED -- READ BEFORE RUNNING ON REAL HARDWARE ***
The original TrainConfig used to train this checkpoint was not found anywhere
recoverable: not in openpi's git history (any branch, case/separator-insensitive
search), not in the checkpoint's own HF repo history (2 generic "upload folder"
commits, no descriptive messages), and the HF repo has no model card. This file
was written from scratch, modeled on openpi's own examples/ur5/README.md
(UR5Inputs / UR5Outputs / LeRobotUR5DataConfig -- the closest published example
for a UR-series arm), with two assumptions that are NOT verified against the
real training data:

  1. Camera key names: assumes the raw observation dict uses "base_rgb" (side
     camera) and "wrist_rgb" (wrist camera) -- copied verbatim from the UR5
     example's repack transform. If your data pipeline used different raw key
     names, change the RepackTransform mapping below to match.

  2. Action convention: assumes DELTA joint actions (dims 0-5) + ABSOLUTE
     gripper (dim 6), same as the UR5 example. The DeltaActions/AbsoluteActions
     transform pair below converts model output back to absolute joint targets
     automatically inside policy.infer(). If this checkpoint actually outputs
     absolute joint targets directly, this transform will silently corrupt
     every action (misinterprets already-absolute values as deltas).

VERIFY BEFORE LIVE USE (no robot needed): start run_pi05_server.py with
--openpi-config pi05_ur10e_cup, then query it (e.g.
gr00t/eval/verify_gr00t_server_cka.py --port 8792, or
smolvla_policy_client.py --dry-run) and look at the printed action[0]:
  - Absolute joint angles for a real UR10e are typically within about +-3.14
    rad of each other and resemble a plausible arm pose.
  - A per-step DELTA (at ~20-50 Hz control rate) should be small, generally
    well under ~0.1 rad.
If action[0]'s first 6 values look like plausible absolute joint positions
(not small deltas), this config guessed wrong: delete the
`.push(inputs=[DeltaActions(...)], outputs=[AbsoluteActions(...)])` call
below and re-test -- the model is likely already trained on absolute actions.
"""

import dataclasses
import pathlib

import numpy as np
from typing_extensions import override

from openpi.models import model as _model
from openpi.models import pi0_config
from openpi.training import weight_loaders
from openpi.training.config import (
    AssetsConfig,
    DataConfig,
    DataConfigFactory,
    ModelTransformFactory,
    TrainConfig,
)
import openpi.transforms as _transforms


def _parse_image(image) -> np.ndarray:
    image = np.asarray(image)
    if np.issubdtype(image.dtype, np.floating):
        image = (255 * image).astype(np.uint8)
    if image.shape[0] == 3:
        image = np.transpose(image, (1, 2, 0))
    return image


@dataclasses.dataclass(frozen=True)
class UR10ECupInputs(_transforms.DataTransformFn):
    model_type: _model.ModelType = _model.ModelType.PI0

    def __call__(self, data: dict) -> dict:
        # ASSUMPTION 1 (see module docstring): raw key names copied from the
        # UR5 example. Change these three if your training pipeline used
        # different raw observation keys.
        state = np.concatenate([data["joints"], data["gripper"]])
        base_image = _parse_image(data["base_rgb"])
        wrist_image = _parse_image(data["wrist_rgb"])

        inputs = {
            "state": state,
            "image": {
                "base_0_rgb": base_image,
                "left_wrist_0_rgb": wrist_image,
                # UR10e has no second wrist camera -- zeroed + masked out, same
                # as the UR5 example.
                "right_wrist_0_rgb": np.zeros_like(base_image),
            },
            "image_mask": {
                "base_0_rgb": np.True_,
                "left_wrist_0_rgb": np.True_,
                "right_wrist_0_rgb": np.False_,
            },
        }

        if "actions" in data:
            inputs["actions"] = data["actions"]
        if "prompt" in data:
            inputs["prompt"] = data["prompt"]

        return inputs


@dataclasses.dataclass(frozen=True)
class UR10ECupOutputs(_transforms.DataTransformFn):
    def __call__(self, data: dict) -> dict:
        # 6 joint angles + gripper = 7 action dims (same convention as every
        # other server in this repo -- run_gr00t_server_n1d6.py etc).
        return {"actions": np.asarray(data["actions"][:, :7])}


@dataclasses.dataclass(frozen=True)
class LeRobotUR10ECupDataConfig(DataConfigFactory):
    @override
    def create(self, assets_dirs: pathlib.Path, model_config) -> DataConfig:
        repack_transform = _transforms.Group(
            inputs=[
                _transforms.RepackTransform(
                    {
                        "base_rgb": "image",
                        "wrist_rgb": "wrist_image",
                        "joints": "joints",
                        "gripper": "gripper",
                        "prompt": "prompt",
                    }
                )
            ]
        )

        data_transforms = _transforms.Group(
            inputs=[UR10ECupInputs(model_type=model_config.model_type)],
            outputs=[UR10ECupOutputs()],
        )

        # ASSUMPTION 2 (see module docstring): delta joints + absolute gripper.
        # DELETE this .push(...) call entirely if verification shows the
        # checkpoint outputs absolute joint targets directly.
        
        delta_action_mask = _transforms.make_bool_mask(6, -1)
        data_transforms = data_transforms.push(
            inputs=[_transforms.DeltaActions(delta_action_mask)],
            outputs=[_transforms.AbsoluteActions(delta_action_mask)],
        )
        

        model_transforms = ModelTransformFactory()(model_config)

        return dataclasses.replace(
            self.create_base_config(assets_dirs, model_config),
            repack_transforms=repack_transform,
            data_transforms=data_transforms,
            model_transforms=model_transforms,
        )


def get_ur10e_cup_configs():
    return [
        TrainConfig(
            name="pi05_ur10e_cup",
            #model=pi0_config.Pi0Config(pi05=True, action_horizon=10, discrete_state_input=False), cause error close/open gripper
            model=pi0_config.Pi0Config(pi05=True, pytorch_compile_mode=None),
            data=LeRobotUR10ECupDataConfig(
                repo_id="local/ur10e-cup",
                #assets=AssetsConfig(asset_id="local/ur10e-cup"),
                assets=AssetsConfig(assets_dir="/home/hung/hungvd27/Pi05_ONNX_Tensorrt/openpi/Pi0.5-UR10E-CUP/assets"),
                base_config=DataConfig(prompt_from_task=True),
            ),
            # Not actually used to load weights at inference time -- create_trained_policy()
            # loads straight from --model-path. This is only consulted if you ever
            # *train* with this config, so any valid placeholder is fine here.
            weight_loader=weight_loaders.CheckpointWeightLoader(
                #"gs://openpi-assets/checkpoints/pi05_base/params",
                "/home/hung/hungvd27/Pi05_ONNX_Tensorrt/openpi/Pi0.5-UR10E-CUP/params",
            ),
            num_train_steps=1,
        ),
    ]

