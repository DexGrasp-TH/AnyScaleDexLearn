import torch
import torch.nn.functional as F
from einops import repeat

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
        result_dict["loss_type"] = F.cross_entropy(type_logits_expanded, gt_type)

        # Generate wrist poses conditioned on object feature and GT grasp type
        global_feature_expanded = repeat(global_feature, "b c -> (b t) c", t=sample_num)
        cond_feat = torch.cat([global_feature_expanded, self.grasp_type_emb(data, global_feature)], dim=-1)
        diffusion_dict = self.output_head.forward(data, cond_feat)
        result_dict.update(diffusion_dict)

        return result_dict

    def sample(self, data: dict, sample_num: int = 1):
        # Encode object pointcloud
        global_feature, local_feature = self.backbone(data)

        # Sample grasp type from predicted distribution
        type_logits = self.type_classifier(global_feature)
        type_probs = F.softmax(type_logits, dim=-1)
        sampled_type = torch.multinomial(type_probs, num_samples=1).squeeze(-1)
        data["grasp_type_id"] = sampled_type

        # Generate wrist poses conditioned on sampled grasp type
        cond_feat = torch.cat([global_feature, self.grasp_type_emb(data, global_feature, True)], dim=-1)
        return self.output_head.sample(cond_feat, sampled_type, sample_num)
