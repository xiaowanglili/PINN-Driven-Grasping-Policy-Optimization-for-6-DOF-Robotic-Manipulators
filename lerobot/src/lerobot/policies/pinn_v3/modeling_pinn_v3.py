#!/usr/bin/env python

import torch
import torch.nn as nn
from torch import Tensor

from lerobot.policies.pinn_v2.modeling_pinn_v2 import PINNv2Policy
from lerobot.policies.pinn_v3.configuration_pinn_v3 import PINNv3Config


class PINNv3Policy(PINNv2Policy):
    config_class = PINNv3Config
    name = "pinn_v3"

    def __init__(
        self,
        config: PINNv3Config,
        dataset_stats: dict[str, dict[str, Tensor]] | None = None,
    ):
        nn.Module.__init__(self)
        self.config = config
        
        import torchvision
        backbone = getattr(torchvision.models, config.vision_backbone)(
            weights=config.pretrained_backbone_weights
        )
        self.backbone = nn.Sequential(*list(backbone.children())[:-1])
        
        if config.use_temporal:
            self.temporal_encoder = nn.GRU(
                input_size=512,
                hidden_size=config.gru_hidden_size,
                num_layers=config.gru_num_layers,
                batch_first=True
            )
            self.hidden_state = None
            
            grasp_head_input_dim = config.gru_hidden_size
        else:
            grasp_head_input_dim = 512
        
        self.grasp_head = nn.Sequential(
            nn.Linear(grasp_head_input_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, config.grasp_dim)
        )
        
        self.criterion = nn.SmoothL1Loss(beta=1.0)
        
        dh_tensor = torch.tensor(config.dh_params, dtype=torch.float32)
        self.register_buffer("dh_a", dh_tensor[:, 0])
        self.register_buffer("dh_alpha", dh_tensor[:, 1])
        self.register_buffer("dh_d", dh_tensor[:, 2])
        self.register_buffer("dh_theta", dh_tensor[:, 3])
        self.register_buffer("link_masses", torch.tensor(config.link_masses))

    def reset(self):
        self.hidden_state = None

    def forward(self, batch: dict[str, Tensor]) -> tuple[Tensor, dict]:
        images = batch["observation.images"]
        if isinstance(images, dict):
            images = list(images.values())[0]
        
        if images.dim() == 5:
            B, T = images.shape[:2]
            images_flat = images.view(B * T, *images.shape[2:])
            visual_feat = self.backbone(images_flat).view(B, T, -1)
        else:
            visual_feat = self.backbone(images).unsqueeze(1)
            B, T = visual_feat.shape[:2]
        
        if self.config.use_temporal:
            if self.hidden_state is None or B != self.hidden_state.shape[1]:
                gru_out, self.hidden_state = self.temporal_encoder(visual_feat)
            else:
                gru_out, self.hidden_state = self.temporal_encoder(
                    visual_feat, self.hidden_state
                )
            
            features = gru_out[:, -1, :]
        else:
            features = visual_feat[:, -1, :]
        
        pred_grasp = self.grasp_head(features)
        
        target = batch["action"][:, :self.config.grasp_dim]
        loss_data = self.criterion(pred_grasp, target)
        
        loss_kinematics = torch.tensor(0.0, device=loss_data.device)
        loss_dynamics = torch.tensor(0.0, device=loss_data.device)
        
        if "observation.state" in batch:
            if batch["observation.state"].dim() == 3:
                joint_angles = batch["observation.state"][:, -1, :6]
            else:
                joint_angles = batch["observation.state"][:, :6]
            
            pred_3d = torch.cat([
                pred_grasp[:, :2],
                torch.ones_like(pred_grasp[:, :1]) * 0.1
            ], dim=1)
            
            kinematic_residual = self.compute_kinematics_residual(joint_angles, pred_3d)
            loss_kinematics = torch.mean(kinematic_residual ** 2)
            
            if all(k in batch for k in ["joint_velocities", "joint_accelerations", "joint_torques"]):
                joint_vel = batch["joint_velocities"][:, -1, :6] if batch["joint_velocities"].dim() == 3 else batch["joint_velocities"][:, :6]
                joint_acc = batch["joint_accelerations"][:, -1, :6] if batch["joint_accelerations"].dim() == 3 else batch["joint_accelerations"][:, :6]
                joint_torques = batch["joint_torques"][:, -1, :6] if batch["joint_torques"].dim() == 3 else batch["joint_torques"][:, :6]
                
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

    def compute_kinematics_residual(self, joint_angles, pred_grasp_3d):
        ee_pos = self.forward_kinematics(joint_angles)
        residual = torch.norm(ee_pos - pred_grasp_3d, dim=1)
        return residual

    def compute_dynamics_residual(self, joint_states, joint_velocities, joint_acc, joint_torques):
        inertia_forces = torch.zeros_like(joint_acc)
        for i in range(6):
            inertia_forces[:, i] = self.link_masses[i] * joint_acc[:, i]
        
        gravity_torques = torch.zeros_like(joint_states)
        gravity_torques[:, 1] = self.link_masses[1] * 9.81 * torch.cos(joint_states[:, 1])
        
        residual = joint_torques - (inertia_forces + gravity_torques)
        return torch.mean(residual ** 2, dim=1)

    def forward_kinematics(self, joint_angles):
        batch_size = joint_angles.shape[0]
        T = torch.eye(4, device=joint_angles.device).unsqueeze(0).repeat(batch_size, 1, 1)
        
        for i in range(6):
            theta = joint_angles[:, i] + self.dh_theta[i]
            ct = torch.cos(theta)
            st = torch.sin(theta)
            ca = torch.cos(self.dh_alpha[i])
            sa = torch.sin(self.dh_alpha[i])
            
            T_i = torch.stack([
                torch.stack([ct, -st, torch.zeros_like(ct), self.dh_a[i].expand(batch_size)], dim=1),
                torch.stack([st*ca, ct*ca, -sa.expand(batch_size), -sa*self.dh_d[i].expand(batch_size)], dim=1),
                torch.stack([st*sa, ct*sa, ca.expand(batch_size), ca*self.dh_d[i].expand(batch_size)], dim=1),
                torch.stack([torch.zeros_like(ct), torch.zeros_like(ct), torch.zeros_like(ct), torch.ones_like(ct)], dim=1)
            ], dim=1)
            
            T = torch.bmm(T, T_i)
        
        return T[:, :3, 3]