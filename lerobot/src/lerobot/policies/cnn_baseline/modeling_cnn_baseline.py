#!/usr/bin/env python

import torch
import torch.nn as nn
import torchvision
from torch import Tensor

from lerobot.configs.policies import PreTrainedConfig
from lerobot.policies.cnn_baseline.configuration_cnn_baseline import CNNBaselineConfig
from lerobot.policies.pretrained import PreTrainedPolicy


class CNNBaselinePolicy(PreTrainedPolicy):
    
    config_class = CNNBaselineConfig
    name = "cnn_baseline"

    def __init__(
        self,
        config: CNNBaselineConfig,
        dataset_stats: dict[str, dict[str, Tensor]] | None = None,
    ):
        super().__init__(config)
        self.config = config
        
        backbone = getattr(torchvision.models, config.vision_backbone)(
            weights=config.pretrained_backbone_weights
        )
        self.backbone = nn.Sequential(*list(backbone.children())[:-1])
        
        self.grasp_head = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, config.grasp_dim) 
        )
        
        self.criterion = nn.SmoothL1Loss(beta=1.0)
        
        self.reset()

    def get_optim_params(self) -> list:
        return [
            {
                "params": self.backbone.parameters(),
                "lr": self.config.optimizer_lr * 0.1,  
            },
            {
                "params": self.grasp_head.parameters(),
                "lr": self.config.optimizer_lr,
            }
        ]

    def reset(self):
        pass

    @torch.no_grad()
    def select_action(self, batch: dict[str, Tensor]) -> Tensor:
        self.eval()
        # 提取图像特征
        images = batch["observation.images"]
        if isinstance(images, dict):
            images = list(images.values())[0]
        
        features = self.backbone(images).flatten(1)
        pred_grasp = self.grasp_head(features)
        
        # 构造动作输出（抓取点坐标映射到动作空间）
        action = pred_grasp  # [B, 4]
        return action

    def forward(self, batch: dict[str, Tensor]) -> tuple[Tensor, dict]:
        images = batch["observation.images"]
        if isinstance(images, dict):
            images = list(images.values())[0]  # [B, C, H, W]
        
        features = self.backbone(images).flatten(1)  # [B, 512]
        
        pred_grasp = self.grasp_head(features)  # [B, 4]

        target = batch["action"][:, :self.config.grasp_dim]
        loss = self.criterion(pred_grasp, target)
        
        output_dict = {
            "loss": loss.item(),
            "pred_grasp": pred_grasp.detach(),
        }
        
        return loss, output_dict