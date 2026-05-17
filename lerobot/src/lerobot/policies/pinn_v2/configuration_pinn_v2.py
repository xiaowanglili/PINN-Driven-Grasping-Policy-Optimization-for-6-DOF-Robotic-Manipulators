#!/usr/bin/env python

from dataclasses import dataclass, field

import numpy as np


@dataclass
class PINNv2Config:
    vision_backbone: str = "resnet18"
    pretrained_backbone_weights: str | None = "ResNet18_Weights.IMAGENET1K_V1"
    grasp_dim: int = 4
    
    lambda_kinematics: float = 0.01
    lambda_dynamics: float = 1e-5
    
    dh_params: list = field(default_factory=lambda: [
        [0.0, -np.pi/2, 0.071, 0.0],
        [0.0, np.pi/2, 0.0, 0.0],
        [0.0, -np.pi/2, 0.125, 0.0],
        [0.0, np.pi/2, 0.0, 0.0],
        [0.0, -np.pi/2, 0.095, 0.0],
        [0.0, 0.0, 0.065, 0.0],
    ])
    
    link_masses: list = field(default_factory=lambda: [0.1, 0.15, 0.12, 0.08, 0.05, 0.02])
    gravity: float = 9.81
    
    joint_limits_lower: list = field(default_factory=lambda: [
        -np.pi, -np.pi/2, -np.pi, -np.pi, -np.pi, -np.pi
    ])
    joint_limits_upper: list = field(default_factory=lambda: [
        np.pi, np.pi/2, np.pi, np.pi, np.pi, np.pi
    ])
    
    optimizer_lr: float = 1e-3
    optimizer_weight_decay: float = 1e-4
    
    def __post_init__(self):
        pass

    def get_optimizer_preset(self):
        from lerobot.optim.optimizers import AdamConfig
        return AdamConfig(
            lr=self.optimizer_lr,
            weight_decay=self.optimizer_weight_decay,
        )

    def get_scheduler_preset(self):
        return None

    def validate_features(self):
        pass

    @property
    def observation_delta_indices(self):
        return None

    @property
    def action_delta_indices(self):
        return None

    @property
    def reward_delta_indices(self):
        return None