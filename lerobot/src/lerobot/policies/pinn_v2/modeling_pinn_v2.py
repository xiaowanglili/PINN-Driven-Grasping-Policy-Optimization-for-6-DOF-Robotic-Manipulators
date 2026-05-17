#!/usr/bin/env python

import torch
import torch.nn as nn
from torch import Tensor

from lerobot.policies.pinn_v1.modeling_pinn_v1 import PINNv1Policy
from lerobot.policies.pinn_v2.configuration_pinn_v2 import PINNv2Config


class PINNv2Policy(PINNv1Policy):
    config_class = PINNv2Config
    name = "pinn_v2"

    def __init__(
        self,
        config: PINNv2Config,
        dataset_stats: dict[str, dict[str, Tensor]] | None = None,
    ):
        super().__init__(config, dataset_stats)
        self.config = config
        self.register_buffer("link_masses", torch.tensor(config.link_masses))

    def compute_dynamics_residual(self, joint_states, joint_velocities, joint_acc, joint_torques):
        batch_size = joint_states.shape[0]
        inertia_forces = torch.zeros_like(joint_acc)
        for i in range(6):
            inertia_forces[:, i] = self.link_masses[i] * joint_acc[:, i]
        
        gravity_torques = torch.zeros_like(joint_states)
        gravity_torques[:, 1] = self.link_masses[1] * self.config.gravity * torch.cos(joint_states[:, 1])
        
        residual = joint_torques - (inertia_forces + gravity_torques)
        return torch.mean(residual ** 2, dim=1)

    def forward(self, batch: dict[str, Tensor]) -> tuple[Tensor, dict]:
        images = batch["observation.images"]
        if isinstance(images, dict):
            images = list(images.values())[0]
        
        features = self.backbone(images).flatten(1)
        pred_grasp = self.grasp_head(features)
        
        target = batch["action"][:, :self.config.grasp_dim]
        loss_data = self.criterion(pred_grasp, target)
        
        loss_kinematics = torch.tensor(0.0, device=loss_data.device)
        loss_dynamics = torch.tensor(0.0, device=loss_data.device)
        
        if "observation.state" in batch:
            joint_angles = batch["observation.state"][:, :6]
            
            pred_3d = torch.cat([
                pred_grasp[:, :2],
                torch.ones_like(pred_grasp[:, :1]) * 0.1
            ], dim=1)
            kinematic_residual = self.compute_kinematics_residual(joint_angles, pred_3d)
            loss_kinematics = torch.mean(kinematic_residual ** 2)
            
            if all(k in batch for k in ["joint_velocities", "joint_accelerations", "joint_torques"]):
                joint_vel = batch["joint_velocities"][:, :6]
                joint_acc = batch["joint_accelerations"][:, :6]
                joint_torques = batch["joint_torques"][:, :6]
                
                dynamics_residual = self.compute_dynamics_residual(
                    joint_angles, joint_vel, joint_acc, joint_torques
                )
                loss_dynamics = torch.mean(dynamics_residual)
        
        total_loss = (
            loss_data + 
            self.config.lambda_kinematics * loss_kinematics +
            self.config.lambda_dynamics * loss_dynamics
        )
        
        output_dict = {
            "loss": total_loss.item(),
            "loss_data": loss_data.item(),
            "loss_kinematics": loss_kinematics.item(),
            "loss_dynamics": loss_dynamics.item(),
            "pred_grasp": pred_grasp.detach(),
        }
        
        return total_loss, output_dict