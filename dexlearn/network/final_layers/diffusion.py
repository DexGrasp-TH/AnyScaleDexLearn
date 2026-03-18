import torch
import torch.nn.functional as F
from einops import rearrange, repeat

from .diffusion_util import MLPWrapper, GaussianDiffusion1D, GaussianDiffusion1DMask
from .mlp import BasicMLP
from dexlearn.utils.rot import proper_svd
from pytorch3d import transforms as pttf
from dexlearn.utils.RMS import Normalization
from dexlearn.dataset.grasp_types import GRASP_TYPES

class DiffusionBiRT_v2(torch.nn.Module):
    """Diffusion model for bimanual wrist poses (used in hierarchical model)."""

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        N_out = 12 * 2
        policy_mlp_parameters = dict(
            hidden_layers_dim=[512, 256],
            output_dim=N_out,
            act="mish",
        )
        self.policy = MLPWrapper(channels=N_out, feature_dim=cfg.in_feat_dim, **policy_mlp_parameters)
        self.diffusion = GaussianDiffusion1DMask(self.policy, cfg.diffusion)
        self.RMS = Normalization(N_out)

    def forward(self, data, cond_feat):
        result_dict = {}
        batch_num, sample_num, pose_num, _ = data["right_hand_trans"].shape
        right_hand_trans = rearrange(data["right_hand_trans"], "b t n x -> (b t) n x")
        right_hand_rot = rearrange(data["right_hand_rot"], "b t n x y -> (b t) n x y")
        left_hand_trans = rearrange(data["left_hand_trans"], "b t n x -> (b t) n x")
        left_hand_rot = rearrange(data["left_hand_rot"], "b t n x y -> (b t) n x y")

        grasp_rt = torch.cat([
            rearrange(right_hand_rot[:, -1], "b x y -> b (x y)"),
            right_hand_trans[:, -1],
            rearrange(left_hand_rot[:, -1], "b x y -> b (x y)"),
            left_hand_trans[:, -1],
        ], dim=-1)
        grasp_rt_norm = self.RMS(grasp_rt)

        result_dict["loss_diffusion"] = self.diffusion(grasp_rt_norm, cond_feat)
        return result_dict

    def sample(self, cond_feat, grasp_type, sample_num):
        cond_feat = repeat(cond_feat, "b c -> (b n) c", n=sample_num)

        grasp_rt, log_prob = self.diffusion.sample(cond=cond_feat)
        log_prob = rearrange(log_prob, "(b t) -> b t", t=sample_num)
        grasp_rt = self.RMS.inv(grasp_rt)

        r_rot_raw = rearrange(grasp_rt[..., 0:9], "(b t) (x y) -> b t x y", t=sample_num, x=3, y=3)
        r_trans = rearrange(grasp_rt[..., 9:12], "(b t) c -> b t c", t=sample_num)

        l_rot_raw = rearrange(grasp_rt[..., 12:21], "(b t) (x y) -> b t x y", t=sample_num, x=3, y=3)
        l_trans = rearrange(grasp_rt[..., 21:24], "(b t) c -> b t c", t=sample_num)

        r_rot_matrix = proper_svd(r_rot_raw.reshape(-1, 3, 3)).reshape_as(r_rot_raw)
        r_quat = pttf.matrix_to_quaternion(r_rot_matrix)
        right_pose = torch.cat([r_trans.unsqueeze(-2), r_quat.unsqueeze(-2)], dim=-1)

        l_rot_matrix = proper_svd(l_rot_raw.reshape(-1, 3, 3)).reshape_as(l_rot_raw)
        l_quat = pttf.matrix_to_quaternion(l_rot_matrix)
        left_pose = torch.cat([l_trans.unsqueeze(-2), l_quat.unsqueeze(-2)], dim=-1)

        robot_pose = torch.cat([right_pose, left_pose], dim=-1)
        return robot_pose, log_prob




class DiffusionTypeAndBiRT(torch.nn.Module):
    """Joint diffusion model that generates grasp type + bimanual wrist poses from a
    point cloud feature.  No external grasp-type conditioning is needed at inference.

    The denoised vector has 30 dimensions:
        [0 : 6]  – grasp-type logits  (one-hot × TYPE_SCALE during training)
        [6 :18]  – right wrist: rot(9) + trans(3)
        [18:30]  – left  wrist: rot(9) + trans(3)

    During training the left-hand dims are masked out for right-only grasp types
    (types 0/1/2/3), exactly as in DiffusionBiRT.  At inference the full 30-dim space
    is explored; the type is decoded via argmax and the final mask is applied to clean
    up inactive-hand dims.
    """

    N_TYPE = len(GRASP_TYPES)
    N_POSE = 24   # right(12) + left(12)
    N_OUT  = N_TYPE + N_POSE
    TYPE_SCALE = 3.0  # scale one-hot so magnitude matches ~N(0,1) normalised poses

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg

        policy_mlp_parameters = dict(
            hidden_layers_dim=[512, 256],
            output_dim=self.N_OUT,
            act="mish",
        )
        self.policy = MLPWrapper(
            channels=self.N_OUT,
            feature_dim=cfg.in_feat_dim,
            **policy_mlp_parameters,
        )
        self.diffusion = GaussianDiffusion1DMask(self.policy, cfg.diffusion)
        self.RMS = Normalization(self.N_POSE)

    # ------------------------------------------------------------------
    # helpers
    # ------------------------------------------------------------------

    def _get_joint_mask(self, grasp_type_id, device):
        """Return (B, 30) binary mask.

        Type dims  [0 : 6]  → always 1  (always predict the type).
        Right dims [6 :18]  → always 1  (right hand active in all types except 0_any).
        Left  dims [18:30]  → 1 for types 4,5 (both hands); 0 for types 1,2,3.
        """
        B = grasp_type_id.shape[0]
        mask = torch.ones(B, self.N_OUT, device=device)
        right_only = (grasp_type_id >= 1) & (grasp_type_id <= 3)  # types 1, 2, 3
        mask[right_only, self.N_TYPE + 12:] = 0.0
        return mask

    def _build_joint_target(self, grasp_type_id, right_rot, right_trans, left_rot, left_trans):
        """Compose the normalised 30-dim training target and its mask."""
        # Pose part — flatten rotation matrices, concatenate, normalise
        pose = torch.cat(
            [
                rearrange(right_rot, "b x y -> b (x y)"),
                right_trans,
                rearrange(left_rot,  "b x y -> b (x y)"),
                left_trans,
            ],
            dim=-1,
        )  # (B, 24)
        pose_norm = self.RMS(pose)

        # Type part — scaled one-hot
        type_onehot = F.one_hot(grasp_type_id, num_classes=self.N_TYPE).float()
        type_onehot = type_onehot * self.TYPE_SCALE  # (B, 6)

        joint = torch.cat([type_onehot, pose_norm], dim=-1)  # (B, 30)
        mask  = self._get_joint_mask(grasp_type_id, pose.device)
        joint = joint * mask  # zero out inactive dims before diffusion
        return joint, mask

    # ------------------------------------------------------------------
    # training
    # ------------------------------------------------------------------

    def forward(self, data, global_feature):
        result_dict = {}

        batch_num, sample_num, pose_num, _ = data["right_hand_trans"].shape

        right_trans = rearrange(data["right_hand_trans"], "b t n x -> (b t) n x")
        right_rot   = rearrange(data["right_hand_rot"],   "b t n x y -> (b t) n x y")
        left_trans  = rearrange(data["left_hand_trans"],  "b t n x -> (b t) n x")
        left_rot    = rearrange(data["left_hand_rot"],    "b t n x y -> (b t) n x y")

        global_feature = repeat(global_feature, "b c -> (b t) c", t=sample_num)
        grasp_type_id  = repeat(data["grasp_type_id"], "b -> (b t)", t=sample_num)

        assert (grasp_type_id != 0).all(), "grasp_type_id should not be 0 during training"

        joint, mask = self._build_joint_target(
            grasp_type_id,
            right_rot[:, -1], right_trans[:, -1],
            left_rot[:, -1],  left_trans[:, -1],
        )

        result_dict["loss_diffusion"] = self.diffusion(joint, global_feature, mask)
        return result_dict

    # ------------------------------------------------------------------
    # inference
    # ------------------------------------------------------------------

    def sample(self, global_feature, grasp_type, sample_num):
        """Sample grasp type + bimanual wrist poses jointly.

        Args:
            global_feature: (B, C) point cloud feature.
            sample_num:     number of samples per scene.

        Returns:
            robot_pose:  (B, sample_num, 1, 14)  right(trans3+quat4) + left(trans3+quat4)
            grasp_type:  (B, sample_num)          predicted grasp-type id
            log_prob:    (B, sample_num)
        """
        B = global_feature.shape[0]
        global_feature_rep = repeat(global_feature, "b c -> (b n) c", n=sample_num)

        # Full mask — type unknown so we let all dims run free.
        # The model naturally outputs ~0 for left dims when type is right-only
        # because those dims were always masked (set to 0) during training.
        full_mask = torch.ones(B * sample_num, self.N_OUT, device=global_feature.device)

        joint, log_prob = self.diffusion.sample(cond=global_feature_rep, mask=full_mask)

        # ---- Decode type ------------------------------------------------
        type_logits = joint[:, :self.N_TYPE]          # (B*N, 6)
        pred_grasp_type  = type_logits.argmax(dim=-1)       # (B*N,)  int

        # ---- Apply type-conditioned mask to clean up inactive hand ------
        pose_mask = self._get_joint_mask(pred_grasp_type, joint.device)[:, self.N_TYPE:]  # (B*N, 24)
        pose_norm = joint[:, self.N_TYPE:] * pose_mask
        pose      = self.RMS.inv(pose_norm)            # (B*N, 24)

        # ---- Unpack rotation / translation ------------------------------
        r_rot_raw = rearrange(pose[..., 0:9],  "(b t) (x y) -> b t x y", t=sample_num, x=3, y=3)
        r_trans   = rearrange(pose[..., 9:12], "(b t) c     -> b t c",   t=sample_num)
        l_rot_raw = rearrange(pose[..., 12:21],"(b t) (x y) -> b t x y", t=sample_num, x=3, y=3)
        l_trans   = rearrange(pose[..., 21:24],"(b t) c     -> b t c",   t=sample_num)

        r_mask = rearrange(pose_mask[..., 9:10],  "(b t) c -> b t c", t=sample_num)
        l_mask = rearrange(pose_mask[..., 21:22], "(b t) c -> b t c", t=sample_num)

        log_prob   = rearrange(log_prob,   "(b t) -> b t", t=sample_num)
        pred_grasp_type = rearrange(pred_grasp_type, "(b t) -> b t", t=sample_num)

        def process_pose(rot_raw, trans, hand_mask):
            rot_matrix = proper_svd(rot_raw.reshape(-1, 3, 3)).reshape_as(rot_raw)
            quat = pttf.matrix_to_quaternion(rot_matrix)
            return torch.cat(
                [(trans * hand_mask).unsqueeze(-2), (quat * hand_mask).unsqueeze(-2)],
                dim=-1,
            )  # (B, N, 1, 7)

        right_pose = process_pose(r_rot_raw, r_trans, r_mask)
        left_pose  = process_pose(l_rot_raw, l_trans, l_mask)
        robot_pose = torch.cat([right_pose, left_pose], dim=-1)  # (B, N, 1, 14)

        return robot_pose, pred_grasp_type, log_prob





class DiffusionBiRT(torch.nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        N_out = 12 * 2
        policy_mlp_parameters = dict(
            hidden_layers_dim=[512, 256],
            output_dim=N_out,
            act="mish",
        )
        self.policy = MLPWrapper(channels=N_out, feature_dim=cfg.in_feat_dim, **policy_mlp_parameters)
        self.diffusion = GaussianDiffusion1DMask(self.policy, cfg.diffusion)
        self.rms = True
        if self.rms:
            self.RMS = Normalization(N_out)

        return

    def get_mask_by_type(self, grasp_type, device):
        """
        根据 grasp_type 生成 mask。
        假设 x 的维度前一半是左手，后一半是右手。
        grasp_type: list or tensor of strings/ids
        """

        batch_size = len(grasp_type)
        # 假设 model.channels 是总维度 (L_dim + R_dim)
        half_dim = self.policy.channels // 2
        mask = torch.zeros((batch_size, self.policy.channels), device=device)

        # 1. Create a template for all 3 possible states
        # Shape: (3, Total_Dim)
        templates = torch.zeros((3, self.policy.channels), device=device)
        templates[0, :half_dim] = 1.0  # ID 0: Right (First half)
        templates[1, half_dim:] = 1.0  # ID 1: Left (Second half)
        templates[2, :] = 1.0  # ID 2: Both (All)

        # 2. Use the grasp_type tensor as indices to pick the right rows
        # This is a highly optimized operation in PyTorch
        mask = templates[grasp_type]

        return mask

    def forward(self, data, global_feature):
        result_dict = {}

        # Deal with multiple grasps for one vision input
        batch_num, sample_num, pose_num, _ = data["right_hand_trans"].shape
        right_hand_trans = rearrange(data["right_hand_trans"], "b t n x -> (b t) n x")
        right_hand_rot = rearrange(data["right_hand_rot"], "b t n x y -> (b t) n x y")
        left_hand_trans = rearrange(data["left_hand_trans"], "b t n x -> (b t) n x")
        left_hand_rot = rearrange(data["left_hand_rot"], "b t n x y -> (b t) n x y")

        global_feature = repeat(global_feature, "b c -> (b t) c", t=sample_num)

        # Use Flow to predict grasp rot and trans
        grasp_rt = torch.cat(
            [
                repeat(right_hand_rot[:, -1], "b x y -> b (x y)"),
                right_hand_trans[:, -1],
                repeat(left_hand_rot[:, -1], "b x y -> b (x y)"),
                left_hand_trans[:, -1],
            ],
            dim=-1,
        )
        if self.rms:
            grasp_rt_diff = self.RMS(grasp_rt)
        else:
            grasp_rt_diff = grasp_rt

        # get mask
        grasp_type = data["grasp_type_id"]
        mask = self.get_mask_by_type(grasp_type, device=grasp_rt_diff.device)

        result_dict["loss_diffusion"] = self.diffusion(grasp_rt_diff, global_feature, mask)

        return result_dict

    def sample(self, global_feature, grasp_type, sample_num):
        global_feature = repeat(global_feature, "b c -> (b n) c", n=sample_num)
        # Get mask
        mask = self.get_mask_by_type(grasp_type, device=global_feature.device)
        mask = repeat(mask, "b d -> (b n) d", n=sample_num)
        # Sample
        grasp_rt, log_prob = self.diffusion.sample(cond=global_feature, mask=mask)
        log_prob = rearrange(log_prob, "(b t) -> b t", t=sample_num)
        if self.rms:
            grasp_rt = self.RMS.inv(grasp_rt)

        # Right Hand
        r_rot_raw = rearrange(grasp_rt[..., 0:9], "(b t) (x y) -> b t x y", t=sample_num, x=3, y=3)
        r_trans = rearrange(grasp_rt[..., 9:12], "(b t) c -> b t c", t=sample_num, c=3)
        r_mask = rearrange(mask[..., 9:10], "(b t) c -> b t c", t=sample_num, c=1)
        # Left Hand
        l_rot_raw = rearrange(grasp_rt[..., 12:21], "(b t) (x y) -> b t x y", t=sample_num, x=3, y=3)
        l_trans = rearrange(grasp_rt[..., 21:24], "(b t) c -> b t c", t=sample_num, c=3)
        l_mask = rearrange(mask[..., 21:22], "(b t) c -> b t c", t=sample_num, c=1)  # (B, T, 1)

        # --- Post-processing (SVD & Quaternion) ---
        def process_pose(rot_raw, trans, hand_mask):
            # Proper SVD for rotation ortho-normalization
            # We reshape to (-1, 3, 3) for the SVD utility, then restore shape
            rot_matrix = proper_svd(rot_raw.reshape(-1, 3, 3)).reshape_as(rot_raw)
            # Matrix to Quaternion: (B, T, 3, 3) -> (B, T, 4)
            quat = pttf.matrix_to_quaternion(rot_matrix)

            final_trans = trans * hand_mask
            final_quat = quat * hand_mask
            # Cat Trans and Quat: Result (B, T, 7)
            # Note: Added unsqueeze(-2) if you need the singleton dimension (B, T, 1, 7)
            return torch.cat([final_trans.unsqueeze(-2), final_quat.unsqueeze(-2)], dim=-1)

        right_robot_pose = process_pose(r_rot_raw, r_trans, r_mask)
        left_robot_pose = process_pose(l_rot_raw, l_trans, l_mask)
        robot_pose = torch.cat([right_robot_pose, left_robot_pose], dim=-1)

        return robot_pose, log_prob


class DiffusionRT(torch.nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        N_out = 12
        policy_mlp_parameters = dict(
            hidden_layers_dim=[512, 256],
            output_dim=N_out,
            act="mish",
        )
        self.policy = MLPWrapper(channels=N_out, feature_dim=cfg.in_feat_dim, **policy_mlp_parameters)
        self.diffusion = GaussianDiffusion1D(self.policy, cfg.diffusion)
        self.rms = True
        if self.rms:
            self.RMS = Normalization(N_out)

        return

    def forward(self, data, global_feature):
        result_dict = {}

        # Deal with multiple grasps for one vision input
        batch_num, sample_num, pose_num, _ = data["hand_trans"].shape
        hand_trans = rearrange(data["hand_trans"], "b t n x -> (b t) n x")
        hand_rot = rearrange(data["hand_rot"], "b t n x y -> (b t) n x y")
        global_feature = repeat(global_feature, "b c -> (b t) c", t=sample_num)

        # Use Flow to predict grasp rot and trans
        grasp_rt = torch.cat([repeat(hand_rot[:, -1], "b x y -> b (x y)"), hand_trans[:, -1]], dim=-1)
        if self.rms:
            grasp_rt_diff = self.RMS(grasp_rt)
        else:
            grasp_rt_diff = grasp_rt
        result_dict["loss_diffusion"] = self.diffusion(grasp_rt_diff, global_feature)

        return result_dict

    def sample(self, global_feature, grasp_type, sample_num):
        global_feature = repeat(global_feature, "b c -> (b n) c", n=sample_num)
        grasp_rt, log_prob = self.diffusion.sample(cond=global_feature)
        log_prob = rearrange(log_prob, "(b t) -> b t", t=sample_num)
        if self.rms:
            grasp_rt = self.RMS.inv(grasp_rt)
        grasp_rot = rearrange(
            grasp_rt[..., :9],
            "(b t) (x y) -> b t x y",
            t=sample_num,
            x=3,
            y=3,
        )
        grasp_trans = rearrange(
            grasp_rt[..., 9:],
            "(b t) c -> b t c",
            t=sample_num,
            c=3,
        )

        final_trans = grasp_trans.unsqueeze(-2)
        grasp_rot = grasp_rot.unsqueeze(-3)
        final_quat = pttf.matrix_to_quaternion(proper_svd(grasp_rot.reshape(-1, 3, 3)).reshape_as(grasp_rot))
        robot_pose = torch.cat([final_trans, final_quat], dim=-1)

        return robot_pose, log_prob


class DiffusionRTJ(torch.nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        N_out = (cfg.joint_num + 12) * cfg.traj_length
        policy_mlp_parameters = dict(
            hidden_layers_dim=[512, 256],
            output_dim=N_out,
            act="mish",
        )
        self.policy = MLPWrapper(channels=N_out, feature_dim=cfg.in_feat_dim, **policy_mlp_parameters)
        self.diffusion = GaussianDiffusion1D(self.policy, cfg.diffusion)
        self.rms = True
        if self.rms:
            self.RMS = Normalization(N_out)
        return

    def forward(self, data, global_feature):
        result_dict = {}

        # Deal with multiple grasps for one vision input
        batch_num, sample_num, pose_num, _ = data["hand_trans"].shape
        global_feature = repeat(global_feature, "b c -> (b t) c", t=sample_num)
        gt_rtj = torch.cat(
            [
                rearrange(data["hand_rot"], "b t n x y -> (b t) (n x y)"),
                rearrange(data["hand_trans"], "b t n x -> (b t) (n x)"),
                rearrange(data["hand_joint"], "b t n x -> (b t) (n x)"),
            ],
            dim=-1,
        )
        if self.rms:
            gt_rtj = self.RMS(gt_rtj)
        result_dict["loss_diffusion"] = self.diffusion(gt_rtj, global_feature)

        return result_dict

    def sample(self, global_feature, sample_num):
        global_feature = repeat(global_feature, "b c -> (b n) c", n=sample_num)
        pred_rtj, log_prob = self.diffusion.sample(cond=global_feature)
        log_prob = rearrange(log_prob, "(b t) -> b t", t=sample_num)
        if self.rms:
            pred_rtj = self.RMS.inv(pred_rtj)
        pose_num = self.cfg.traj_length
        hand_rot = rearrange(
            pred_rtj[..., : pose_num * 9],
            "(b t) (n x y) -> b t n x y",
            t=sample_num,
            n=pose_num,
            x=3,
        )
        hand_trans = rearrange(
            pred_rtj[..., pose_num * 9 : pose_num * 12],
            "(b t) (n c) -> b t n c",
            t=sample_num,
            n=pose_num,
        )

        hand_joint = rearrange(
            pred_rtj[..., pose_num * 12 :],
            "(b t) (n c) -> b t n c",
            t=sample_num,
            n=pose_num,
        )

        final_quat = pttf.matrix_to_quaternion(proper_svd(hand_rot.reshape(-1, 3, 3)).reshape_as(hand_rot))
        robot_pose = torch.cat([hand_trans, final_quat, hand_joint], dim=-1)
        return robot_pose, log_prob


class DiffusionRT_MLPRTJ(torch.nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        N_out = 12
        policy_mlp_parameters = dict(
            hidden_layers_dim=[512, 256],
            output_dim=N_out,
            act="mish",
        )
        self.policy = MLPWrapper(channels=N_out, feature_dim=cfg.in_feat_dim, **policy_mlp_parameters)
        self.diffusion = GaussianDiffusion1D(self.policy, cfg.diffusion)
        self.rms = True
        if self.rms:
            self.RMS = Normalization(N_out)
        self.joint_mlp = BasicMLP(
            cfg.in_feat_dim + 12,
            (cfg.joint_num + 12) * cfg.traj_length - 12,
        )
        self.joint_loss = torch.nn.SmoothL1Loss(reduction="none")
        return

    def forward(self, data, global_feature):
        result_dict = {}

        # Deal with multiple grasps for one vision input
        batch_num, sample_num, pose_num, _ = data["hand_trans"].shape
        hand_trans = rearrange(data["hand_trans"], "b t n x -> (b t) n x")
        hand_rot = rearrange(data["hand_rot"], "b t n x y -> (b t) n x y")
        hand_joint = rearrange(data["hand_joint"], "b t n x -> (b t) n x")
        global_feature = repeat(global_feature, "b c -> (b t) c", t=sample_num)

        # Use Flow to predict grasp rot and trans
        grasp_rt = torch.cat([repeat(hand_rot[:, -1], "b x y -> b (x y)"), hand_trans[:, -1]], dim=-1)
        if self.rms:
            grasp_rt_diff = self.RMS(grasp_rt)
        else:
            grasp_rt_diff = grasp_rt
        result_dict["loss_diffusion"] = self.diffusion(grasp_rt_diff, global_feature)

        # use MLP to predict other things
        in_mlp_feat = torch.cat([global_feature, grasp_rt], dim=-1)
        out_info = self.joint_mlp(in_mlp_feat)
        gt_info = torch.cat(
            [
                rearrange(hand_trans[:, :-1], "b n x -> b (n x)"),
                rearrange(hand_rot[:, :-1], "b n x y -> b (n x y)"),
                rearrange(hand_joint, "b n x -> b (n x)"),
            ],
            dim=-1,
        )
        loss_others = self.joint_loss(out_info, gt_info)
        pose_num = self.cfg.traj_length - 1
        result_dict["loss_trans"] = loss_others[:, : pose_num * 3].mean()
        result_dict["loss_rot"] = loss_others[:, pose_num * 3 : pose_num * 12].mean()
        result_dict["loss_joint"] = loss_others[:, pose_num * 12 :].mean()
        with torch.no_grad():
            result_dict["abs_dis_joint"] = (out_info - gt_info)[pose_num * 12 :].abs().mean()

        return result_dict

    def sample(self, global_feature, grasp_type, sample_num):
        global_feature = repeat(global_feature, "b c -> (b n) c", n=sample_num)
        grasp_rt, log_prob = self.diffusion.sample(cond=global_feature)
        log_prob = rearrange(log_prob, "(b t) -> b t", t=sample_num)
        if self.rms:
            grasp_rt = self.RMS.inv(grasp_rt)
        grasp_rot = proper_svd(rearrange(grasp_rt[..., :9], "b (x y) -> b x y", x=3))
        grasp_trans = grasp_rt[..., 9:]
        in_mlp_feat = torch.cat(
            [
                global_feature,
                rearrange(grasp_rot, "b x y -> b (x y)"),
                grasp_trans,
            ],
            dim=-1,
        )
        pred_info = self.joint_mlp(in_mlp_feat)
        pose_num = self.cfg.traj_length - 1
        hand_trans = rearrange(
            pred_info[:, : pose_num * 3],
            "(b t) (n c) -> b t n c",
            t=sample_num,
            n=pose_num,
        )
        hand_rot = rearrange(
            pred_info[:, pose_num * 3 : pose_num * 12],
            "(b t) (n x y) -> b t n x y",
            t=sample_num,
            n=pose_num,
            x=3,
        )

        hand_joint = rearrange(
            pred_info[:, pose_num * 12 :],
            "(b t) (n c) -> b t n c",
            t=sample_num,
            n=pose_num + 1,
        )

        final_trans = torch.cat(
            [hand_trans, rearrange(grasp_trans, "(b n) c -> b n 1 c", n=sample_num)],
            dim=-2,
        )
        final_quat = pttf.matrix_to_quaternion(
            torch.cat(
                [
                    proper_svd(hand_rot.reshape(-1, 3, 3)).reshape_as(hand_rot),
                    rearrange(grasp_rot, "(b n) x y -> b n 1 x y", n=sample_num),
                ],
                dim=-3,
            ),
        )
        robot_pose = torch.cat([final_trans, final_quat, hand_joint], dim=-1)
        return robot_pose, log_prob
