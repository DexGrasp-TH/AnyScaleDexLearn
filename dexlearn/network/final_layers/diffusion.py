import torch
from einops import rearrange, repeat

from .diffusion_util import MLPWrapper, GaussianDiffusion1D, GaussianDiffusion1DMask
from .mlp import BasicMLP
from dexlearn.utils.rot import proper_svd
from pytorch3d import transforms as pttf
from dexlearn.utils.RMS import Normalization


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
