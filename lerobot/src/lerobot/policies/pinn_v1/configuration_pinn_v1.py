from dataclasses import dataclass, field

import numpy as np

from lerobot.configs.policies import PreTrainedConfig
from lerobot.optim.optimizers import AdamConfig


@PreTrainedConfig.register_subclass("pinn_v1")
@dataclass
class PINNv1Config(PreTrainedConfig):
    
    vision_backbone: str = "resnet18"
    pretrained_backbone_weights: str | None = "ResNet18_Weights.IMAGENET1K_V1"
    grasp_dim: int = 4
    
    lambda_kinematics: float = 0.01  
    
    dh_params: list = field(default_factory=lambda: [
        [0.0, -np.pi/2, 0.071, 0.0],      
        [0.0, np.pi/2, 0.0, 0.0],       
        [0.0, -np.pi/2, 0.125, 0.0],     
        [0.0, np.pi/2, 0.0, 0.0],          
        [0.0, -np.pi/2, 0.095, 0.0],      
        [0.0, 0.0, 0.065, 0.0],          
    ])
    
    
    joint_limits_lower: list = field(default_factory=lambda: [
        -np.pi, -np.pi/2, -np.pi, -np.pi, -np.pi, -np.pi
    ])
    joint_limits_upper: list = field(default_factory=lambda: [
        np.pi, np.pi/2, np.pi, np.pi, np.pi, np.pi
    ])
    
    optimizer_lr: float = 1e-3
    optimizer_weight_decay: float = 1e-4
    
    def __post_init__(self):
        super().__post_init__()

    def get_optimizer_preset(self) -> AdamConfig:
        return AdamConfig(
            lr=self.optimizer_lr,
            weight_decay=self.optimizer_weight_decay,
        )

    def get_scheduler_preset(self) -> None:
        return None

    def validate_features(self) -> None:
        if not self.image_features:
            raise ValueError("必须提供至少一个图像输入")

    @property
    def observation_delta_indices(self) -> list | None:
        return None

    @property
    def action_delta_indices(self) -> list | None:
        return None

    @property
    def reward_delta_indices(self) -> list | None:
        return None