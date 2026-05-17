
import torch
import torch.nn as nn
import torchvision
from torch import Tensor

from lerobot.policies.cnn_baseline.modeling_cnn_baseline import CNNBaselinePolicy
from lerobot.policies.pinn_v1.configuration_pinn_v1 import PINNv1Config


class PINNv1Policy(CNNBaselinePolicy):
    
    config_class = PINNv1Config
    name = "pinn_v1"

    def __init__(
        self,
        config: PINNv1Config,
        dataset_stats: dict[str, dict[str, Tensor]] | None = None,
    ):
        super().__init__(config, dataset_stats)
        self.config = config
        
        dh_tensor = torch.tensor(config.dh_params, dtype=torch.float32)  # [6, 4]
        self.register_buffer("dh_a", dh_tensor[:, 0])
        self.register_buffer("dh_alpha", dh_tensor[:, 1])
        self.register_buffer("dh_d", dh_tensor[:, 2])
        self.register_buffer("dh_theta", dh_tensor[:, 3])
        
        self.register_buffer("joint_lower", torch.tensor(config.joint_limits_lower))
        self.register_buffer("joint_upper", torch.tensor(config.joint_limits_upper))

    def dh_transform(self, a, alpha, d, theta):
        ct = torch.cos(theta)
        st = torch.sin(theta)
        ca = torch.cos(alpha)
        sa = torch.sin(alpha)
        
        T = torch.stack([
            torch.stack([ct, -st, 0, a]),
            torch.stack([st*ca, ct*ca, -sa, -sa*d]),
            torch.stack([st*sa, ct*sa, ca, ca*d]),
            torch.stack([0, 0, 0, 1])
        ], dim=0)
        return T

    def forward_kinematics(self, joint_angles):
        batch_size = joint_angles.shape[0]
        T = torch.eye(4, device=joint_angles.device).unsqueeze(0).repeat(batch_size, 1, 1)
        
        for i in range(6):
            theta = joint_angles[:, i] + self.dh_theta[i]
            T_i = self.dh_transform(
                self.dh_a[i], 
                self.dh_alpha[i], 
                self.dh_d[i], 
                theta
            ).unsqueeze(0).repeat(batch_size, 1, 1)
            T = torch.bmm(T, T_i)
        
        end_effector_pos = T[:, :3, 3]  # [B, 3]
        return end_effector_pos

    def compute_kinematics_residual(self, joint_angles, pred_grasp_3d):
       
        ee_pos = self.forward_kinematics(joint_angles)  # [B, 3]
        

        residual = torch.norm(ee_pos - pred_grasp_3d, dim=1)  # [B]
        
        return residual

    def forward(self, batch: dict[str, Tensor]) -> tuple[Tensor, dict]:

        images = batch["observation.images"]
        if isinstance(images, dict):
            images = list(images.values())[0]
        
        features = self.backbone(images).flatten(1)
        pred_grasp = self.grasp_head(features)  # [B, 4]
        
        target = batch["action"][:, :self.config.grasp_dim]
        loss_data = self.criterion(pred_grasp, target)
        
        loss_kinematics = torch.tensor(0.0, device=loss_data.device)
        if "observation.state" in batch:
            joint_angles = batch["observation.state"][:, :6]
            

            pred_3d = torch.cat([
                pred_grasp[:, :2],  # x, y
                torch.ones_like(pred_grasp[:, :1]) * 0.1  
            ], dim=1)
            

            kinematic_residual = self.compute_kinematics_residual(joint_angles, pred_3d)
            loss_kinematics = torch.mean(kinematic_residual ** 2)
        

        total_loss = loss_data + self.config.lambda_kinematics * loss_kinematics
        
        output_dict = {
            "loss": total_loss.item(),
            "loss_data": loss_data.item(),
            "loss_kinematics": loss_kinematics.item(),
            "pred_grasp": pred_grasp.detach(),
        }
        
        return total_loss, output_dict