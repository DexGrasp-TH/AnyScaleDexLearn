import torch
import torch.nn.functional as F
from einops import repeat, rearrange
from typing import Tuple

from ..backbones import *
from ..type_emb import *
from ..final_layers import *
from dexlearn.dataset.grasp_types import GRASP_TYPES

REAL_GRASP_TYPE_IDS = tuple(range(1, len(GRASP_TYPES)))
REAL_GRASP_TYPE_NUM = len(REAL_GRASP_TYPE_IDS)


def _build_type_head(input_dim: int, output_dim: int, hidden_dim: int = 256) -> torch.nn.Module:
    """Build the shared MLP used by grasp-type heads.

    Args:
        input_dim: Input feature dimension from the point-cloud backbone.
        output_dim: Number of logits produced by the head.
        hidden_dim: Hidden feature size of the intermediate MLP layer.

    Returns:
        Torch module that maps backbone features to type logits.
    """
    return torch.nn.Sequential(
        torch.nn.Linear(input_dim, hidden_dim),
        torch.nn.ReLU(),
        torch.nn.Linear(hidden_dim, output_dim),
    )


def _normalize_feasibility_loss_weights(type_loss_weights):
    """Normalize configured type-loss weights for the feasibility head.

    Args:
        type_loss_weights: Optional iterable of class weights. The legacy
            categorical path may provide weights for all ``GRASP_TYPES``,
            including ``0_any``.

    Returns:
        Tensor-ready Python list aligned with the 5 real grasp types, or
        ``None`` when no weights are configured.
    """
    if type_loss_weights is None:
        return None
    weight_list = list(type_loss_weights)
    if len(weight_list) == len(GRASP_TYPES):
        return weight_list[1:]
    if len(weight_list) == REAL_GRASP_TYPE_NUM:
        return weight_list
    raise ValueError(
        f"type_loss_weights must have length {len(GRASP_TYPES)} or {REAL_GRASP_TYPE_NUM}, "
        f"got {len(weight_list)}"
    )


class HierarchicalModel(torch.nn.Module):
    def __init__(self, cfg: dict):
        super().__init__()
        self.cfg = cfg
        self.backbone = eval(cfg.backbone.name)(cfg.backbone)

        # Grasp type classifier
        self.type_classifier = _build_type_head(cfg.backbone.out_feat_dim, len(GRASP_TYPES))

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


class HierarchicalFeasibilityModel(torch.nn.Module):
    """Hierarchical model with object-level multi-label type feasibility.

    The feasibility head predicts scores for the 5 real grasp types
    ``1..5``. During training, the diffusion head is still supervised by one
    concrete grasp type and one concrete pose sampled from the selected
    object.
    """

    def __init__(self, cfg: dict):
        super().__init__()
        self.cfg = cfg
        self.backbone = eval(cfg.backbone.name)(cfg.backbone)

        type_head_cfg = getattr(cfg, "type_head", None)
        hidden_dim = int(getattr(type_head_cfg, "hidden_dim", 256))
        self.type_classifier = _build_type_head(cfg.backbone.out_feat_dim, REAL_GRASP_TYPE_NUM, hidden_dim=hidden_dim)

        self.grasp_type_emb = eval(cfg.grasp_type_emb.name)(cfg.grasp_type_emb)

        type_loss_weights = _normalize_feasibility_loss_weights(getattr(cfg, "type_loss_weights", None))
        if type_loss_weights is not None:
            self.register_buffer(
                "type_loss_weights",
                torch.tensor(type_loss_weights, dtype=torch.float32),
                persistent=False,
            )
        else:
            self.type_loss_weights = None

        cfg.head.in_feat_dim = cfg.backbone.out_feat_dim + cfg.grasp_type_emb.out_feat_dim
        self.output_head = eval(cfg.head.name)(cfg.head)

    def _compute_feasibility_loss(self, type_logits: torch.Tensor, data: dict) -> torch.Tensor:
        """Compute masked BCE loss for object-level type feasibility.

        Args:
            type_logits: Predicted logits with shape ``(B, 5)`` for real types
                ``1..5``.
            data: Batch dictionary containing ``feasible_type_mask`` and
                ``tested_type_mask``.

        Returns:
            Scalar feasibility loss.
        """
        if "feasible_type_mask" not in data or "tested_type_mask" not in data:
            raise KeyError("HierarchicalFeasibilityModel requires feasible_type_mask and tested_type_mask in data")

        feasible_type_mask = data["feasible_type_mask"].to(dtype=type_logits.dtype)
        tested_type_mask = data["tested_type_mask"].to(dtype=type_logits.dtype)
        if feasible_type_mask.shape != type_logits.shape or tested_type_mask.shape != type_logits.shape:
            raise ValueError(
                "feasible/tested masks must match type_logits shape, got "
                f"{tuple(feasible_type_mask.shape)} and {tuple(tested_type_mask.shape)} vs {tuple(type_logits.shape)}"
            )

        loss_per_type = F.binary_cross_entropy_with_logits(type_logits, feasible_type_mask, reduction="none")
        if self.type_loss_weights is not None:
            loss_per_type = loss_per_type * self.type_loss_weights.view(1, -1)

        masked_loss = loss_per_type * tested_type_mask
        normalizer = tested_type_mask.sum().clamp_min(1.0)
        return masked_loss.sum() / normalizer

    def _compute_type_scores(self, global_feature: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Predict real-type logits and sigmoid feasibility scores.

        Args:
            global_feature: Backbone feature tensor with shape ``(B, C)``.

        Returns:
            Tuple ``(type_logits, type_scores)`` where each tensor has shape
            ``(B, 5)`` for real grasp types ``1..5``.
        """
        type_logits = self.type_classifier(global_feature)
        type_scores = torch.sigmoid(type_logits)
        return type_logits, type_scores

    def forward(self, data: dict):
        result_dict = {}

        global_feature, local_feature = self.backbone(data)
        type_logits, type_scores = self._compute_type_scores(global_feature)
        result_dict["loss_type"] = self._compute_feasibility_loss(type_logits, data)

        batch_num, sample_num = data["grasp_type_id"].shape[0], data["right_hand_trans"].shape[1]
        assert sample_num == 1
        grasp_type_id = repeat(data["grasp_type_id"], "b -> (b t)", t=sample_num)
        if ((grasp_type_id <= 0) | (grasp_type_id >= len(GRASP_TYPES))).any():
            raise ValueError("Training data for HierarchicalFeasibilityModel must use explicit grasp types in [1, 5]")

        global_feature_expanded = repeat(global_feature, "b c -> (b t) c", t=sample_num)
        cond_feat = torch.cat([global_feature_expanded, self.grasp_type_emb(grasp_type_id)], dim=-1)
        diffusion_dict = self.output_head.forward(data, cond_feat)
        result_dict.update(diffusion_dict)

        with torch.no_grad():
            result_dict["metric_type_score_mean"] = type_scores.mean()

        return result_dict

    def sample(self, data: dict, sample_num: int = 1):
        global_feature, local_feature = self.backbone(data)
        type_logits, type_scores = self._compute_type_scores(global_feature)
        input_type = data["grasp_type_id"]
        if input_type.ndim != 1:
            raise ValueError(f"Expected grasp_type_id to have shape (B,), got {tuple(input_type.shape)}")

        if torch.any(input_type == 0):
            if not torch.all(input_type == 0):
                raise ValueError("Mixed 0_any and explicit grasp types are not supported in one batch")
            pred_type_id = torch.argmax(type_scores, dim=-1) + 1
            return {
                "pred_grasp_type_prob": type_scores.unsqueeze(1),
                "pred_grasp_type_id": pred_type_id.unsqueeze(1),
            }

        if ((input_type <= 0) | (input_type >= len(GRASP_TYPES))).any():
            raise ValueError("Explicit sampling with HierarchicalFeasibilityModel requires grasp_type_id in [1, 5]")

        batch_size = global_feature.shape[0]
        grasp_type_ids = input_type.unsqueeze(1).expand(-1, sample_num)
        global_feature_expanded = repeat(global_feature, "b c -> (b s) c", s=sample_num)
        grasp_type_ids_flat = rearrange(grasp_type_ids, "b s -> (b s)")
        cond_feat = torch.cat([global_feature_expanded, self.grasp_type_emb(grasp_type_ids_flat)], dim=-1)
        robot_pose, log_prob = self.output_head.sample(cond_feat, grasp_type_ids_flat, 1)

        robot_pose = rearrange(robot_pose, "(b s) t ... -> b (s t) ...", b=batch_size, s=sample_num)
        log_prob = rearrange(log_prob, "(b s) t -> b (s t)", b=batch_size, s=sample_num)
        type_scores = repeat(type_scores, "b c -> b s c", s=sample_num)

        return robot_pose, grasp_type_ids, type_scores, log_prob


class HierarchicalTypeCEModel(HierarchicalModel):
    """Neutral name for the scene/record softmax-CE hierarchical model."""


class HierarchicalTypeObjectiveModel(torch.nn.Module):
    """Unified hierarchical human model with switchable type objectives."""

    def __init__(self, cfg: dict):
        super().__init__()
        self.cfg = cfg
        self.type_objective = str(getattr(cfg, "type_objective", "ce")).lower()
        if self.type_objective not in {"ce", "object_bce", "scene_ranking"}:
            raise ValueError(f"Unsupported type_objective={self.type_objective}")

        self.backbone = eval(cfg.backbone.name)(cfg.backbone)
        type_head_cfg = getattr(cfg, "type_head", None)
        hidden_dim = int(getattr(type_head_cfg, "hidden_dim", 256))
        self.type_classifier = _build_type_head(
            cfg.backbone.out_feat_dim,
            REAL_GRASP_TYPE_NUM,
            hidden_dim=hidden_dim,
        )

        self.grasp_type_emb = eval(cfg.grasp_type_emb.name)(cfg.grasp_type_emb)
        self.focal_gamma = getattr(cfg, "focal_gamma", None)
        self.ranking_loss = str(getattr(cfg, "ranking_loss", "logistic")).lower()
        self.ranking_margin = float(getattr(cfg, "ranking_margin", 1.0))

        type_loss_weights = getattr(cfg, "type_loss_weights", None)
        if type_loss_weights is not None:
            type_loss_weights = _normalize_feasibility_loss_weights(type_loss_weights)
        if type_loss_weights is not None:
            self.register_buffer(
                "type_loss_weights",
                torch.tensor(list(type_loss_weights), dtype=torch.float32),
                persistent=False,
            )
        else:
            self.type_loss_weights = None

        cfg.head.in_feat_dim = cfg.backbone.out_feat_dim + cfg.grasp_type_emb.out_feat_dim
        self.output_head = eval(cfg.head.name)(cfg.head)

    def _compute_type_scores(self, global_feature: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Predict type logits and normalized real-type scores.

        Args:
            global_feature: Backbone feature tensor with shape ``(B, C)``.

        Returns:
            Tuple of raw logits and five real-type scores for grasp types
            ``1..5``.
        """
        type_logits = self.type_classifier(global_feature)
        if self.type_objective == "ce":
            type_scores = F.softmax(type_logits, dim=-1)
        else:
            type_scores = torch.sigmoid(type_logits)
        return type_logits, type_scores

    def _compute_ce_loss(self, type_logits: torch.Tensor, data: dict) -> torch.Tensor:
        """Compute record/scene-level 5-way softmax cross-entropy loss.

        Args:
            type_logits: Predicted logits with shape ``(B, 5)`` for real
                grasp types ``1..5``.
            data: Batch dictionary containing ``grasp_type_id`` in ``[1, 5]``.

        Returns:
            Scalar categorical negative-log-likelihood over the 5 real types.
        """
        gt_type = data["grasp_type_id"].long()
        if (gt_type == 0).any():
            raise ValueError("Training data contains grasp_type_id = 0, which should not be used for training")
        gt_index = gt_type - 1
        if ((gt_index < 0) | (gt_index >= REAL_GRASP_TYPE_NUM)).any():
            raise ValueError("CE training requires grasp_type_id in [1, 5]")
        loss = F.cross_entropy(type_logits, gt_index, weight=self.type_loss_weights, reduction="none")
        if self.focal_gamma is not None and float(self.focal_gamma) > 0.0:
            probs = F.softmax(type_logits, dim=-1).gather(1, gt_index.view(-1, 1)).squeeze(1)
            loss = loss * torch.pow(1.0 - probs.clamp_min(1e-8), float(self.focal_gamma))
        return loss.mean()

    def _compute_object_bce_loss(self, type_logits: torch.Tensor, data: dict) -> torch.Tensor:
        """Compute masked BCE loss for object-level type-prior supervision."""
        if "feasible_type_mask" not in data or "tested_type_mask" not in data:
            raise KeyError("object_bce requires feasible_type_mask and tested_type_mask in data")
        feasible_type_mask = data["feasible_type_mask"].to(dtype=type_logits.dtype)
        tested_type_mask = data["tested_type_mask"].to(dtype=type_logits.dtype)
        if feasible_type_mask.shape != type_logits.shape or tested_type_mask.shape != type_logits.shape:
            raise ValueError(
                "feasible/tested masks must match type_logits shape, got "
                f"{tuple(feasible_type_mask.shape)} and {tuple(tested_type_mask.shape)} vs {tuple(type_logits.shape)}"
            )
        loss_per_type = F.binary_cross_entropy_with_logits(type_logits, feasible_type_mask, reduction="none")
        if self.type_loss_weights is not None:
            loss_per_type = loss_per_type * self.type_loss_weights.view(1, -1)
        masked_loss = loss_per_type * tested_type_mask
        normalizer = tested_type_mask.sum().clamp_min(1.0)
        return masked_loss.sum() / normalizer

    def _compute_scene_ranking_loss(self, type_logits: torch.Tensor, data: dict) -> torch.Tensor:
        """Compute sampled-negative pairwise ranking loss."""
        if "negative_type_ids" not in data:
            raise KeyError("scene_ranking requires negative_type_ids in data")
        positive_index = data["grasp_type_id"].long() - 1
        negative_index = data["negative_type_ids"].long() - 1
        if ((positive_index < 0) | (positive_index >= REAL_GRASP_TYPE_NUM)).any():
            raise ValueError("scene_ranking requires positive grasp_type_id in [1, 5]")
        if ((negative_index < 0) | (negative_index >= REAL_GRASP_TYPE_NUM)).any():
            raise ValueError("scene_ranking requires negative_type_ids in [1, 5]")

        positive_logits = type_logits.gather(1, positive_index.view(-1, 1))
        negative_logits = type_logits.gather(1, negative_index)
        logit_diff = positive_logits - negative_logits
        if self.ranking_loss == "margin":
            loss = F.relu(self.ranking_margin - logit_diff)
        elif self.ranking_loss in {"logistic", "bpr"}:
            loss = F.softplus(-logit_diff)
        else:
            raise ValueError(f"Unsupported ranking_loss={self.ranking_loss}")

        if self.type_loss_weights is not None:
            positive_weight = self.type_loss_weights.gather(0, positive_index).view(-1, 1)
            loss = loss * positive_weight
        return loss.mean()

    def _compute_type_loss(self, type_logits: torch.Tensor, data: dict) -> torch.Tensor:
        """Dispatch the configured type objective loss."""
        if self.type_objective == "ce":
            return self._compute_ce_loss(type_logits, data)
        if self.type_objective == "object_bce":
            return self._compute_object_bce_loss(type_logits, data)
        if self.type_objective == "scene_ranking":
            return self._compute_scene_ranking_loss(type_logits, data)
        raise ValueError(f"Unsupported type_objective={self.type_objective}")

    def forward(self, data: dict):
        result_dict = {}
        global_feature, local_feature = self.backbone(data)
        type_logits, type_scores = self._compute_type_scores(global_feature)
        result_dict["loss_type"] = self._compute_type_loss(type_logits, data)

        batch_num, sample_num = data["grasp_type_id"].shape[0], data["right_hand_trans"].shape[1]
        assert sample_num == 1
        grasp_type_id = repeat(data["grasp_type_id"], "b -> (b t)", t=sample_num)
        if ((grasp_type_id <= 0) | (grasp_type_id >= len(GRASP_TYPES))).any():
            raise ValueError("Training data for HierarchicalTypeObjectiveModel must use explicit types in [1, 5]")

        global_feature_expanded = repeat(global_feature, "b c -> (b t) c", t=sample_num)
        cond_feat = torch.cat([global_feature_expanded, self.grasp_type_emb(grasp_type_id)], dim=-1)
        diffusion_dict = self.output_head.forward(data, cond_feat)
        result_dict.update(diffusion_dict)

        with torch.no_grad():
            result_dict["metric_type_score_mean"] = type_scores.mean()
        return result_dict

    def sample(self, data: dict, sample_num: int = 1):
        global_feature, local_feature = self.backbone(data)
        type_logits, type_scores = self._compute_type_scores(global_feature)
        input_type = data["grasp_type_id"]
        if input_type.ndim != 1:
            raise ValueError(f"Expected grasp_type_id to have shape (B,), got {tuple(input_type.shape)}")

        if torch.any(input_type == 0):
            if not torch.all(input_type == 0):
                raise ValueError("Mixed 0_any and explicit grasp types are not supported in one batch")
            pred_type_id = torch.argmax(type_scores, dim=-1) + 1
            return {
                "pred_grasp_type_prob": type_scores.unsqueeze(1),
                "pred_grasp_type_id": pred_type_id.unsqueeze(1),
            }

        if ((input_type <= 0) | (input_type >= len(GRASP_TYPES))).any():
            raise ValueError("Explicit sampling requires grasp_type_id in [1, 5]")

        batch_size = global_feature.shape[0]
        grasp_type_ids = input_type.unsqueeze(1).expand(-1, sample_num)
        global_feature_expanded = repeat(global_feature, "b c -> (b s) c", s=sample_num)
        grasp_type_ids_flat = rearrange(grasp_type_ids, "b s -> (b s)")
        cond_feat = torch.cat([global_feature_expanded, self.grasp_type_emb(grasp_type_ids_flat)], dim=-1)
        robot_pose, log_prob = self.output_head.sample(cond_feat, grasp_type_ids_flat, 1)

        robot_pose = rearrange(robot_pose, "(b s) t ... -> b (s t) ...", b=batch_size, s=sample_num)
        log_prob = rearrange(log_prob, "(b s) t -> b (s t)", b=batch_size, s=sample_num)
        type_scores = repeat(type_scores, "b c -> b s c", s=sample_num)
        return robot_pose, grasp_type_ids, type_scores, log_prob
