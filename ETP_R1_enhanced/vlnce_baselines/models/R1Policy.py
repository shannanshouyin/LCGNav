from copy import deepcopy
import numpy as np
import time
import torch
import torch.nn as nn
import torch.nn.functional as F

from gym import Space
from habitat import Config
from habitat_baselines.common.baseline_registry import baseline_registry
from habitat_baselines.rl.models.rnn_state_encoder import (
    build_rnn_state_encoder,
)
from habitat_baselines.rl.ppo.policy import Net

from vlnce_baselines.models.etp.ETP_R1_vlnbert_init import get_vlnbert_models
from vlnce_baselines.common.aux_losses import AuxLosses
from vlnce_baselines.models.encoders.instruction_encoder import (
    InstructionEncoder,
)
from vlnce_baselines.models.encoders.resnet_encoders import (
    TorchVisionResNet50,
    VlnResnetDepthEncoder,
    CLIPEncoder,
)
from vlnce_baselines.models.policy import ILPolicy

from vlnce_baselines.waypoint_pred.TRM_net import BinaryDistPredictor_TRM
from vlnce_baselines.waypoint_pred.utils import nms
from vlnce_baselines.models.utils import (
    angle_feature_with_ele, dir_angle_feature_with_ele, angle_feature_torch, length2mask)
import math
from vlnce_baselines.models.pointnet_encoder import PointNetEncoder

@baseline_registry.register_policy
class R1Policy(ILPolicy):
    def __init__(
        self,
        observation_space: Space,
        action_space: Space,
        model_config: Config,
        dropout_rate=0.1,
    ):
        super().__init__(
            ETP(
                observation_space=observation_space,
                model_config=model_config,
                num_actions=action_space.n,
                dropout_rate=dropout_rate
            ),
            action_space.n,
        )

    @classmethod
    def from_config(
        cls, config: Config, observation_space: Space, action_space: Space, dropout_rate=0.1
    ):
        config.defrost()
        config.MODEL.TORCH_GPU_ID = config.TORCH_GPU_ID
        config.freeze()

        return cls(
            observation_space=observation_space,
            action_space=action_space,
            model_config=config.MODEL,
            dropout_rate=dropout_rate
        )

class Critic(nn.Module):
    def __init__(self, drop_ratio):
        super(Critic, self).__init__()
        self.state2value = nn.Sequential(
            nn.Linear(768, 512),
            nn.ReLU(),
            nn.Dropout(drop_ratio),
            nn.Linear(512, 1),
        )

    def forward(self, state):
        return self.state2value(state).squeeze()

class ETP(Net):
    def __init__(
        self, observation_space: Space, model_config: Config, num_actions, dropout_rate
    ):
        super().__init__()

        device = (
            torch.device("cuda", model_config.TORCH_GPU_ID)
            if torch.cuda.is_available()
            else torch.device("cpu")
        )
        self.device = device

        print('\nInitalizing the ETP model ...')
        self.vln_bert = get_vlnbert_models(config=model_config, dropout_rate=dropout_rate)
        # if model_config.task_type == 'r2r':
        #     self.rgb_projection = nn.Linear(2048, 768)
        # elif model_config.task_type == 'rxr':
        #     self.rgb_projection = nn.Linear(2048, 512)
        # self.rgb_projection = nn.Linear(2048, 768) # for vit 768 compability
        # if model_config.task_type == 'r2r':
        #     self.rgb_projection = nn.Linear(512, 768)
        # else:
        #     self.rgb_projection = None
        self.drop_env = nn.Dropout(p=0.4)

        # Optional PointNet encoder for ghost node point cloud features.
        pc_cfg = model_config.POINT_CLOUD if hasattr(model_config, "POINT_CLOUD") else None
        self.use_pointnet = bool(getattr(pc_cfg, "enable_pointnet", False)) if pc_cfg is not None else False
        self.use_pc_fusion = bool(getattr(pc_cfg, "enable_fusion", False)) if pc_cfg is not None else False
        self.use_learnable_fusion_lambda = bool(getattr(pc_cfg, "enable_learnable_fusion_lambda", False)) if pc_cfg is not None else False
        self.pc_accumulate_mode = str(getattr(pc_cfg, "accumulate_mode", "mean")).lower() if pc_cfg is not None else "mean"
        self.pc_fusion_scope = str(getattr(pc_cfg, "fusion_scope", "global")).lower() if pc_cfg is not None else "global"
        if self.pc_fusion_scope not in ("global", "connected"):
            raise ValueError(
                f"Unsupported POINT_CLOUD.fusion_scope={self.pc_fusion_scope!r}. "
                "Expected one of: 'global', 'connected'."
            )
        self.pointnet_encoder = None
        self.pc_feat_proj = None
        self.fusion_lambda_raw = None

        if self.use_pointnet:
            self.pointnet_encoder = PointNetEncoder(
                in_dim=int(pc_cfg.POINTNET.in_dim),
                mlp_channels=list(pc_cfg.POINTNET.mlp_channels),
                global_dim=int(pc_cfg.POINTNET.global_dim),
                dropout=float(pc_cfg.POINTNET.dropout),
                use_bn=bool(pc_cfg.POINTNET.use_bn),
            )
            if self.use_pc_fusion:
                hidden_size = int(self.vln_bert.config.hidden_size)
                self.pc_feat_proj = nn.Linear(int(pc_cfg.POINTNET.global_dim), hidden_size)
                if self.use_learnable_fusion_lambda:
                    lam_init = float(getattr(pc_cfg, "fusion_lambda_init", 0.1))
                    # Constrain lambda to (0,1) via sigmoid(raw); initialize so sigmoid(raw)=lam_init.
                    eps = 1e-6
                    lam_init = float(min(max(lam_init, eps), 1.0 - eps))
                    raw_init = float(np.log(lam_init / (1.0 - lam_init)))
                    self.fusion_lambda_raw = nn.Parameter(
                        torch.tensor(raw_init, dtype=torch.float32)
                    )

        if self.pointnet_encoder is not None:
            self.pointnet_encoder.to(self.device)
        if self.pc_feat_proj is not None:
            self.pc_feat_proj.to(self.device)
        # PyTorch 1.x: Parameter.to(device) may return a plain Tensor; assign via .data to keep nn.Parameter.
        if self.fusion_lambda_raw is not None:
            self.fusion_lambda_raw.data = self.fusion_lambda_raw.data.to(self.device)

        # self.pos_encoder = nn.Sequential(
        #     nn.Linear(6, 768),
        #     nn.LayerNorm(768, eps=1e-12)
        # )
        # self.hist_mlp = nn.Sequential(
        #     nn.Linear(768, 768),
        #     nn.ReLU(),
        #     nn.Linear(768, 768)
        # )

        # Init the depth encoder
        assert model_config.DEPTH_ENCODER.cnn_type in [
            "VlnResnetDepthEncoder"
        ], "DEPTH_ENCODER.cnn_type must be VlnResnetDepthEncoder"
        self.depth_encoder = VlnResnetDepthEncoder(
            observation_space,
            output_size=model_config.DEPTH_ENCODER.output_size,
            checkpoint=model_config.DEPTH_ENCODER.ddppo_checkpoint,
            backbone=model_config.DEPTH_ENCODER.backbone,
            spatial_output=model_config.spatial_output,
        )
        self.space_pool_depth = nn.Sequential(nn.AdaptiveAvgPool2d((1,1)), nn.Flatten(start_dim=2))

        # Init the RGB encoder
        # assert model_config.RGB_ENCODER.cnn_type in [
        #     "TorchVisionResNet152", "TorchVisionResNet50"
        # ], "RGB_ENCODER.cnn_type must be TorchVisionResNet152 or TorchVisionResNet50"
        # if model_config.RGB_ENCODER.cnn_type == "TorchVisionResNet50":
        #     self.rgb_encoder = TorchVisionResNet50(
        #         observation_space,
        #         model_config.RGB_ENCODER.output_size,
        #         device,
        #         spatial_output=model_config.spatial_output,
        #     )
        self.rgb_encoder = CLIPEncoder(self.device)
        self.space_pool_rgb = nn.Sequential(nn.AdaptiveAvgPool2d((1,1)), nn.Flatten(start_dim=2))
    
        self.pano_img_idxes = np.arange(0, 12, dtype=np.int64)        # 逆时针
        pano_angle_rad_c = (1-self.pano_img_idxes/12) * 2 * math.pi   # 对应到逆时针
        self.pano_angle_fts = angle_feature_torch(torch.from_numpy(pano_angle_rad_c))

        # Optional depth truncation in waypoint forward:
        # - off: no truncation
        # - global: truncate all 12-view depth maps before depth encoder
        # - candidate: truncate only selected candidate-view depth maps (3m by default)
        depth_trunc_cfg = (
            model_config.DEPTH_TRUNCATION if hasattr(model_config, "DEPTH_TRUNCATION") else None
        )
        self.depth_truncation_mode = (
            str(getattr(depth_trunc_cfg, "mode", "off")).lower()
            if depth_trunc_cfg is not None
            else "off"
        )
        if self.depth_truncation_mode not in ("off", "global", "candidate"):
            raise ValueError(
                f"Unsupported MODEL.DEPTH_TRUNCATION.mode={self.depth_truncation_mode!r}. "
                "Expected one of: 'off', 'global', 'candidate'."
            )
        self.depth_truncation_max_m = (
            float(getattr(depth_trunc_cfg, "max_depth_m", 3.0))
            if depth_trunc_cfg is not None
            else 3.0
        )
        self.depth_truncation_normalize_depth = (
            bool(getattr(depth_trunc_cfg, "normalize_depth", True))
            if depth_trunc_cfg is not None
            else True
        )
        self.depth_truncation_min_depth_m = (
            float(getattr(depth_trunc_cfg, "min_depth_m", 0.0))
            if depth_trunc_cfg is not None
            else 0.0
        )
        self.depth_truncation_max_sensor_depth_m = (
            float(getattr(depth_trunc_cfg, "max_sensor_depth_m", 10.0))
            if depth_trunc_cfg is not None
            else 10.0
        )
        if self.depth_truncation_normalize_depth:
            span = max(
                self.depth_truncation_max_sensor_depth_m - self.depth_truncation_min_depth_m,
                1e-6,
            )
            thr = (self.depth_truncation_max_m - self.depth_truncation_min_depth_m) / span
            self.depth_truncation_value = float(min(max(thr, 0.0), 1.0))
        else:
            self.depth_truncation_value = float(self.depth_truncation_max_m)

    @property  # trivial argument, just for init with habitat
    def output_size(self):
        return 1

    @property
    def is_blind(self):
        return self.rgb_encoder.is_blind or self.depth_encoder.is_blind

    @property
    def num_recurrent_layers(self):
        return 1

    def _truncate_depth_tensor(self, depth: torch.Tensor) -> torch.Tensor:
        """Clip depth values to configured max depth in current tensor scale."""
        return torch.clamp(depth, max=self.depth_truncation_value)

    def forward(self, mode=None, 
                txt_ids=None, txt_task_encoding=None, txt_masks=None, txt_embeds=None, 
                waypoint_predictor=None, observations=None, in_train=True,
                rgb_fts=None, dep_fts=None, loc_fts=None, 
                nav_types=None, view_lens=None,
                gmap_vp_ids=None, gmap_step_ids=None,
                gmap_img_fts=None, gmap_pos_fts=None, 
                gmap_masks=None, gmap_visited_masks=None, gmap_pair_dists=None, gmap_task_embeddings=None,
                gmap_pc_points=None, gmap_pc_masks=None, gmap_pc_fusion_masks=None):

        if mode == 'language':
            encoded_sentence = self.vln_bert.forward_txt(
                txt_ids, txt_task_encoding, txt_masks,
            )
            return encoded_sentence

        elif mode == 'waypoint':
            # batch_size = observations['instruction'].size(0)
            batch_size = observations['rgb'].shape[0]
            ''' encoding rgb/depth at all directions ----------------------------- '''
            NUM_ANGLES = 120    # 120 angles 3 degrees each
            NUM_IMGS = 12
            NUM_CLASSES = 12    # 12 distances at each sector
            depth_batch = torch.zeros_like(observations['depth']).repeat(NUM_IMGS, 1, 1, 1)
            rgb_batch = torch.zeros_like(observations['rgb']).repeat(NUM_IMGS, 1, 1, 1)

            # reverse the order of input images to clockwise
            a_count = 0
            for i, (k, v) in enumerate(observations.items()):
                if 'depth' in k:  # You might need to double check the keys order
                    for bi in range(v.size(0)):
                        ra_count = (NUM_IMGS - a_count) % NUM_IMGS
                        depth_batch[ra_count + bi*NUM_IMGS] = v[bi]
                        rgb_batch[ra_count + bi*NUM_IMGS] = observations[k.replace('depth','rgb')][bi]
                    a_count += 1

            if self.depth_truncation_mode == "global":
                depth_batch = self._truncate_depth_tensor(depth_batch)

            obs_view12 = {}
            obs_view12['depth'] = depth_batch
            obs_view12['rgb'] = rgb_batch
            depth_embedding = self.depth_encoder(obs_view12)  # torch.Size([bs, 128, 4, 4])
            rgb_embedding = self.rgb_encoder(obs_view12)      # torch.Size([bs, 2048, 7, 7])

            ''' waypoint prediction ----------------------------- '''
            waypoint_heatmap_logits = waypoint_predictor(
                rgb_embedding, depth_embedding)

            # reverse the order of images back to counter-clockwise
            rgb_embed_reshape = rgb_embedding.reshape(
                batch_size, NUM_IMGS, 512, 1, 1)
            depth_embed_reshape = depth_embedding.reshape(
                batch_size, NUM_IMGS, 128, 4, 4)
            rgb_feats = torch.cat((
                rgb_embed_reshape[:,0:1,:], 
                torch.flip(rgb_embed_reshape[:,1:,:], [1]),
            ), dim=1)
            depth_feats = torch.cat((
                depth_embed_reshape[:,0:1,:], 
                torch.flip(depth_embed_reshape[:,1:,:], [1]),
            ), dim=1)
            # way_feats = torch.cat((
            #     way_feats[:,0:1,:], 
            #     torch.flip(way_feats[:,1:,:], [1]),
            # ), dim=1)

            # from heatmap to points
            batch_x_norm = torch.softmax(
                waypoint_heatmap_logits.reshape(
                    batch_size, NUM_ANGLES*NUM_CLASSES,
                ), dim=1
            )
            batch_x_norm = batch_x_norm.reshape(
                batch_size, NUM_ANGLES, NUM_CLASSES,
            )
            batch_x_norm_wrap = torch.cat((
                batch_x_norm[:,-1:,:], 
                batch_x_norm, 
                batch_x_norm[:,:1,:]), 
                dim=1)
            batch_output_map = nms(
                batch_x_norm_wrap.unsqueeze(1), 
                max_predictions=5,
                sigma=(7.0,5.0))

            # predicted waypoints before sampling
            batch_output_map = batch_output_map.squeeze(1)[:,1:-1,:]

            # candidate_lengths = ((batch_output_map!=0).sum(-1).sum(-1) + 1).tolist()
            # if isinstance(candidate_lengths, int):
            #     candidate_lengths = [candidate_lengths]
            # max_candidate = max(candidate_lengths)  # including stop
            # cand_mask = length2mask(candidate_lengths, device=self.device)

            if in_train:
                # Waypoint augmentation
                # parts of heatmap for sampling (fix offset first)
                HEATMAP_OFFSET = 5
                batch_way_heats_regional = torch.cat(
                    (waypoint_heatmap_logits[:,-HEATMAP_OFFSET:,:], 
                    waypoint_heatmap_logits[:,:-HEATMAP_OFFSET,:],
                ), dim=1)
                batch_way_heats_regional = batch_way_heats_regional.reshape(batch_size, 12, 10, 12)
                batch_sample_angle_idxes = []
                batch_sample_distance_idxes = []
                # batch_way_log_prob = []
                for j in range(batch_size):
                    # angle indexes with candidates
                    angle_idxes = batch_output_map[j].nonzero()[:, 0]
                    # clockwise image indexes (same as batch_x_norm)
                    img_idxes = ((angle_idxes.cpu().numpy()+5) // 10)
                    img_idxes[img_idxes==12] = 0
                    # # candidate waypoint states
                    # way_feats_regional = way_feats[j][img_idxes]
                    # heatmap regions for sampling
                    way_heats_regional = batch_way_heats_regional[j][img_idxes].view(img_idxes.size, -1)
                    way_heats_probs = F.softmax(way_heats_regional, 1)
                    probs_c = torch.distributions.Categorical(way_heats_probs)
                    way_heats_act = probs_c.sample().detach()
                    sample_angle_idxes = []
                    sample_distance_idxes = []
                    for k, way_act in enumerate(way_heats_act):
                        if img_idxes[k] != 0:
                            angle_pointer = (img_idxes[k] - 1) * 10 + 5
                        else:
                            angle_pointer = 0
                        sample_angle_idxes.append(way_act//12+angle_pointer)
                        sample_distance_idxes.append(way_act%12)
                    batch_sample_angle_idxes.append(sample_angle_idxes)
                    batch_sample_distance_idxes.append(sample_distance_idxes)
                    # batch_way_log_prob.append(
                    #     probs_c.log_prob(way_heats_act))
            else:
                # batch_way_log_prob = None
                None
            
            rgb_feats = self.space_pool_rgb(rgb_feats)
            depth_feats = self.space_pool_depth(depth_feats)

            # for cand
            cand_rgb = []
            cand_depth = []
            cand_angle_fts = []
            cand_img_idxes = []
            cand_angles = []
            cand_distances = []
            for j in range(batch_size):
                if in_train:
                    angle_idxes = torch.tensor(batch_sample_angle_idxes[j])
                    distance_idxes = torch.tensor(batch_sample_distance_idxes[j])
                else:
                    angle_idxes = batch_output_map[j].nonzero()[:, 0]
                    distance_idxes = batch_output_map[j].nonzero()[:, 1]
                # for angle & distance
                angle_rad_c = angle_idxes.cpu().float()/120*2*math.pi       # 顺时针
                angle_rad_cc = 2*math.pi-angle_idxes.float()/120*2*math.pi  # 逆时针
                cand_angle_fts.append( angle_feature_torch(angle_rad_c) )
                cand_angles.append(angle_rad_cc.tolist())
                cand_distances.append( ((distance_idxes + 1)*0.25).tolist() )
                # for img idxes
                img_idxes = 12 - (angle_idxes.cpu().numpy()+5) // 10        # 逆时针
                img_idxes[img_idxes==12] = 0
                cand_img_idxes.append(img_idxes)
                # for rgb & depth
                cand_rgb.append(rgb_feats[j, img_idxes, ...])
                cand_depth.append(depth_feats[j, img_idxes, ...])

            if self.depth_truncation_mode == "candidate":
                depth_batch_candidate = depth_batch.clone()
                # `cand_img_idxes` are CCW pano indices (0..11) in depth_feats order.
                # Convert to depth_batch/depth_embedding order before truncating selected views.
                for bi in range(batch_size):
                    if bi >= len(cand_img_idxes):
                        continue
                    for ccw_idx in cand_img_idxes[bi]:
                        enc_idx = (NUM_IMGS - int(ccw_idx)) % NUM_IMGS
                        flat_idx = bi * NUM_IMGS + enc_idx
                        depth_batch_candidate[flat_idx] = self._truncate_depth_tensor(
                            depth_batch_candidate[flat_idx]
                        )
                obs_view12_candidate = {
                    'depth': depth_batch_candidate,
                    'rgb': rgb_batch,
                }
                depth_embedding_candidate = self.depth_encoder(obs_view12_candidate)
                depth_embed_reshape_candidate = depth_embedding_candidate.reshape(
                    batch_size, NUM_IMGS, 128, 4, 4
                )
                depth_feats_candidate = torch.cat((
                    depth_embed_reshape_candidate[:,0:1,:],
                    torch.flip(depth_embed_reshape_candidate[:,1:,:], [1]),
                ), dim=1)
                depth_feats_candidate = self.space_pool_depth(depth_feats_candidate)
                cand_depth = []
                for j in range(batch_size):
                    img_idxes = cand_img_idxes[j]
                    cand_depth.append(depth_feats_candidate[j, img_idxes, ...])
                # Keep non-candidate pano views unchanged in candidate mode.
                pano_depth = depth_feats
            else:
                pano_depth = depth_feats
            
            # for pano
            pano_rgb = rgb_feats                            # B x 12 x 2048
            pano_depth = pano_depth                         # B x 12 x 128
            pano_angle_fts = deepcopy(self.pano_angle_fts)  # 12 x 4
            pano_img_idxes = deepcopy(self.pano_img_idxes)  # 12

            # cand_angle_fts 顺时针
            # cand_angles 逆时针
            # cand_img_idxes可能内含重复值，即多个waypoints属于同一个角度区间
            outputs = {
                'cand_rgb': cand_rgb,               # [K x 2048]
                'cand_depth': cand_depth,           # [K x 128]
                'cand_angle_fts': cand_angle_fts,   # [K x 4]
                'cand_img_idxes': cand_img_idxes,   # [K]，表示waypoints的角度栅格，用0～11之间的整数值来表示
                'cand_angles': cand_angles,         # [K]，表示waypoints的角度，用0～2pi之间的弧度值来表示
                'cand_distances': cand_distances,   # [K]

                'pano_rgb': pano_rgb,               # B x 12 x 2048
                'pano_depth': pano_depth,           # B x 12 x 128
                'pano_angle_fts': pano_angle_fts,   # 12 x 4
                'pano_img_idxes': pano_img_idxes,   # 12 
            }
            
            return outputs

        elif mode == 'panorama':
            rgb_fts = self.drop_env(rgb_fts)
            outs = self.vln_bert.forward_panorama(
                rgb_fts, dep_fts, loc_fts, nav_types, view_lens,
            )
            return outs

        elif mode == 'navigation':
            # Ghost point cloud feature extraction + residual fusion (方案A: 在训练前向时计算，保证可训练梯度流入 PointNet).
            if (
                self.use_pointnet
                and self.use_pc_fusion
                and (self.pointnet_encoder is not None)
                and (self.pc_feat_proj is not None)
                and (gmap_pc_points is not None)
                and (gmap_pc_masks is not None)
            ):
                if self.use_learnable_fusion_lambda and (self.fusion_lambda_raw is not None):
                    fusion_lambda = torch.sigmoid(self.fusion_lambda_raw).to(
                        gmap_img_fts.device, dtype=gmap_img_fts.dtype
                    )
                else:
                    fusion_lambda = None

                # Apply configured ghost fusion scope.
                # - global: all valid ghost point-cloud slots can fuse.
                # - connected: only one-hop ghosts connected to current node can fuse.
                effective_pc_masks = gmap_pc_masks
                if self.pc_fusion_scope == "connected":
                    if gmap_pc_fusion_masks is None:
                        raise ValueError(
                            "POINT_CLOUD.fusion_scope='connected' requires gmap_pc_fusion_masks."
                        )
                    if gmap_pc_masks.ndim == 3 and gmap_pc_fusion_masks.ndim == 2:
                        gmap_pc_fusion_masks = gmap_pc_fusion_masks.unsqueeze(-1).expand_as(gmap_pc_masks)
                    elif gmap_pc_masks.ndim == 2 and gmap_pc_fusion_masks.ndim == 3:
                        gmap_pc_fusion_masks = gmap_pc_fusion_masks.any(dim=-1)
                    effective_pc_masks = gmap_pc_masks & gmap_pc_fusion_masks

                # Supported shapes:
                # 1) (B, L, N, 3) with masks (B, L)  [legacy]
                # 2) (B, L, S, N, 3) with masks (B, L, S) [feature accumulation across multiple samples]
                if gmap_pc_points.ndim == 4:
                    # gmap_pc_points: (B, L, N, 3), gmap_pc_masks: (B, L)
                    pc_points = gmap_pc_points[effective_pc_masks]  # (M, N, 3)
                    if pc_points.numel() > 0:
                        pc_points = pc_points.to(gmap_img_fts.device, dtype=gmap_img_fts.dtype)
                        pc_feat = self.pointnet_encoder(pc_points)  # (M, pc_global_dim)
                        pc_feat = self.pc_feat_proj(pc_feat)  # (M, hidden_size)
                        if fusion_lambda is not None:
                            pc_feat = pc_feat * fusion_lambda

                        # Non-in-place add: gmap_img_fts often has requires_grad=False (frozen trunk);
                        # in-place slice assignment breaks autograd to pc_feat / fusion_lambda_raw.
                        delta = torch.zeros_like(gmap_img_fts)
                        # autocast may produce fp16 pc_feat while gmap_img_fts is fp32; match for index_put.
                        delta[effective_pc_masks] = pc_feat.to(dtype=delta.dtype)
                        gmap_img_fts = gmap_img_fts + delta
                else:
                    # gmap_pc_points: (B, L, S, N, 3), gmap_pc_masks: (B, L, S)
                    bsz, glen, ssz, npts, _ = gmap_pc_points.shape
                    assert effective_pc_masks.shape == (bsz, glen, ssz), (
                        f"gmap_pc_masks shape mismatch: expected {(bsz, glen, ssz)}, got {effective_pc_masks.shape}"
                    )

                    pc_nonzero = torch.nonzero(effective_pc_masks, as_tuple=False)  # (M2, 3): [b, t, s]
                    if pc_nonzero.numel() > 0:
                        # Gather points for all valid (b,t,s) entries.
                        # Result: (M2, N, 3)
                        pc_points = gmap_pc_points[effective_pc_masks]
                        pc_points = pc_points.to(gmap_img_fts.device, dtype=gmap_img_fts.dtype)

                        pc_feat = self.pointnet_encoder(pc_points)  # (M2, pc_global_dim)
                        pc_feat = self.pc_feat_proj(pc_feat)  # (M2, hidden_size)
                        if fusion_lambda is not None:
                            pc_feat = pc_feat * fusion_lambda

                        # Reduce per token (b,t) over sample dimension (s).
                        token_linear = pc_nonzero[:, 0] * glen + pc_nonzero[:, 1]  # (M2,)
                        total_tokens = bsz * glen

                        feat_sum = torch.zeros(
                            total_tokens, pc_feat.shape[-1],
                            device=pc_feat.device, dtype=pc_feat.dtype
                        )
                        feat_sum.index_add_(0, token_linear, pc_feat)

                        if self.pc_accumulate_mode == "mean":
                            ones = torch.ones(
                                pc_feat.shape[0], device=pc_feat.device, dtype=pc_feat.dtype
                            )
                            counts = torch.zeros(total_tokens, device=pc_feat.device, dtype=pc_feat.dtype)
                            counts.index_add_(0, token_linear, ones)
                            feat_token = feat_sum / counts.clamp(min=1.0).unsqueeze(1)
                        else:
                            feat_token = feat_sum

                        token_has_any = token_linear.unique()
                        if token_has_any.numel() > 0:
                            feat_token_flat = feat_token  # (B*L, H)
                            gmap_img_fts_flat = gmap_img_fts.view(bsz * glen, -1)
                            add = torch.zeros_like(gmap_img_fts_flat)
                            add[token_has_any] = feat_token_flat[token_has_any].to(
                                dtype=add.dtype
                            )
                            gmap_img_fts_flat = gmap_img_fts_flat + add
                            gmap_img_fts = gmap_img_fts_flat.view(bsz, glen, -1)

            outs = self.vln_bert.forward_navigation(
                txt_embeds, txt_masks, 
                gmap_vp_ids, gmap_step_ids,
                gmap_img_fts, gmap_pos_fts, 
                gmap_masks, gmap_visited_masks, gmap_pair_dists, gmap_task_embeddings
            )
            return outs

# class BertLayerNorm(nn.Module):
#     def __init__(self, hidden_size, eps=1e-12):
#         """Construct a layernorm module in the TF style (epsilon inside the square root).
#         """
#         super(BertLayerNorm, self).__init__()
#         self.weight = nn.Parameter(torch.ones(hidden_size))
#         self.bias = nn.Parameter(torch.zeros(hidden_size))
#         self.variance_epsilon = eps

#     def forward(self, x):
#         u = x.mean(-1, keepdim=True)
#         s = (x - u).pow(2).mean(-1, keepdim=True)
#         x = (x - u) / torch.sqrt(s + self.variance_epsilon)
#         return self.weight * x + self.bias
