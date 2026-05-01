import torch
import torch.nn.functional as F
from einops import repeat, rearrange

from ..backbones import *
from ..type_emb import *
from ..final_layers import *
from dexlearn.dataset.grasp_types import GRASP_TYPES


class HierarchicalModel(torch.nn.Module):
    def __init__(self, cfg: dict):
        super().__init__()
        self.cfg = cfg
        self.backbone = eval(cfg.backbone.name)(cfg.backbone)

        # Grasp type classifier
        self.type_classifier = torch.nn.Sequential(
            torch.nn.Linear(cfg.backbone.out_feat_dim, 256), torch.nn.ReLU(), torch.nn.Linear(256, len(GRASP_TYPES))
        )

        # Grasp type embedding for diffusion conditioning
        self.grasp_type_emb = eval(cfg.grasp_type_emb.name)(cfg.grasp_type_emb)

        type_loss_weights = getattr(cfg, "type_loss_weights", None)
        if type_loss_weights is not None:
            self.register_buffer("type_loss_weights", torch.tensor(type_loss_weights, dtype=torch.float32), persistent=False)
        else:
            self.type_loss_weights = None

        # Diffusion head
        cfg.head.in_feat_dim = cfg.backbone.out_feat_dim + cfg.grasp_type_emb.out_feat_dim
        self.output_head = eval(cfg.head.name)(cfg.head)

    def forward(self, data: dict):
        result_dict = {}

        # Encode object pointcloud
        global_feature, local_feature = self.backbone(data)

        # Predict grasp type distribution
        type_logits = self.type_classifier(global_feature)
        batch_num, sample_num = data["grasp_type_id"].shape[0], data["right_hand_trans"].shape[1]
        assert sample_num == 1
        type_logits_expanded = repeat(type_logits, "b c -> (b t) c", t=sample_num)
        gt_type = repeat(data["grasp_type_id"], "b -> (b t)", t=sample_num)

        # Exclude type 0 from loss calculation
        if (gt_type == 0).any():
            raise ValueError("Training data contains grasp_type_id = 0, which should not be used for training")
        result_dict["loss_type"] = F.cross_entropy(type_logits_expanded, gt_type, weight=self.type_loss_weights)

        # Generate wrist poses conditioned on object feature and GT grasp type.
        global_feature_expanded = repeat(global_feature, "b c -> (b t) c", t=sample_num)
        grasp_type_id = repeat(data["grasp_type_id"], "b -> (b t)", t=sample_num)
        cond_feat = torch.cat([global_feature_expanded, self.grasp_type_emb(grasp_type_id)], dim=-1)
        diffusion_dict = self.output_head.forward(data, cond_feat)
        result_dict.update(diffusion_dict)

        return result_dict

    def sample(self, data: dict, sample_num: int = 1):
        # Encode object pointcloud
        global_feature, local_feature = self.backbone(data)

        # Determine grasp type: use sampled type if input is 0, otherwise use input
        input_type = data["grasp_type_id"]
        type_logits = self.type_classifier(global_feature)
        type_probs = F.softmax(type_logits, dim=-1)
        sampled_types = torch.multinomial(type_probs, num_samples=sample_num, replacement=True)

        # Use sampled type where input is 0, otherwise use input type
        grasp_type_ids = torch.where(input_type.unsqueeze(1) == 0, sampled_types, input_type.unsqueeze(1))

        # Flatten batch and sample dimensions
        batch_size = global_feature.shape[0]
        global_feature_expanded = repeat(global_feature, "b c -> (b s) c", s=sample_num)
        grasp_type_ids_flat = rearrange(grasp_type_ids, "b s -> (b s)")

        # Generate robot poses for all samples at once
        cond_feat = torch.cat([global_feature_expanded, self.grasp_type_emb(grasp_type_ids_flat)], dim=-1)
        robot_pose, log_prob = self.output_head.sample(cond_feat, grasp_type_ids_flat, 1)

        # Reshape back to (batch, sample_num, ...)
        robot_pose = rearrange(robot_pose, "(b s) t ... -> b (s t) ...", b=batch_size, s=sample_num)
        log_prob = rearrange(log_prob, "(b s) t -> b (s t)", b=batch_size, s=sample_num)
        type_probs = repeat(type_probs, "b c -> b s c", s=sample_num)

        return robot_pose, grasp_type_ids, type_probs, log_prob
