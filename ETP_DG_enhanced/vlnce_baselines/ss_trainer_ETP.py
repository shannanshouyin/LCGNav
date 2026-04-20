import gc
import os
import sys
import random
import warnings
from collections import defaultdict
from typing import Dict, List
import jsonlines

import lmdb
import msgpack_numpy
import numpy as np
import math
import time
import torch
import torch.nn.functional as F
from torch.autograd import Variable
from torch.nn.parallel import DistributedDataParallel as DDP

import tqdm
from gym import Space
from habitat import Config, logger
from habitat_baselines.common.baseline_registry import baseline_registry
from habitat_baselines.common.environments import get_env_class
from habitat_baselines.common.obs_transformers import (
    apply_obs_transforms_batch,
    apply_obs_transforms_obs_space,
    get_active_obs_transforms,
)
from habitat_baselines.common.tensorboard_utils import TensorboardWriter
from habitat_baselines.utils.common import batch_obs

from vlnce_baselines.common.aux_losses import AuxLosses
from vlnce_baselines.common.base_il_trainer import BaseVLNCETrainer
from vlnce_baselines.common.env_utils import construct_envs, construct_envs_for_rl, is_slurm_batch_job
from vlnce_baselines.common.utils import extract_instruction_tokens
from vlnce_baselines.models.graph_utils import GraphMap, MAX_DIST
from vlnce_baselines.models.pointcloud_sampling import (
    build_depth_feats_maps,
    sample_pointcloud_from_depth,
)
from vlnce_baselines.utils import reduce_loss

from .utils import get_camera_orientations12
from .utils import (
    length2mask, dir_angle_feature_with_ele,
)
from vlnce_baselines.common.utils import dis_to_con, gather_list_and_concat
from habitat_extensions.measures import NDTW, StepsTaken
from fastdtw import fastdtw

with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=FutureWarning)
    import tensorflow as tf  # noqa: F401

import torch.distributed as distr
import gzip
import json
from copy import deepcopy
from torch.cuda.amp import autocast, GradScaler
from vlnce_baselines.common.ops import pad_tensors_wgrad, gen_seq_masks
from torch.nn.utils.rnn import pad_sequence


@baseline_registry.register_trainer(name="SS-ETP")
class RLTrainer(BaseVLNCETrainer):
    def __init__(self, config=None):
        super().__init__(config)
        self.max_len = int(config.IL.max_traj_len) #  * 0.97 transfered gt path got 0.96 spl
        # Used to accumulate dynamic graph weight statistics (every 200 iterations)
        self.dynamic_graph_weight_history = {}  # {layer_idx: {'w1': [], 'w2': [], 'w3': []}}

    def _save_config_snapshot(self) -> None:
        """Write merged config to CHECKPOINT_FOLDER (rank 0 only), analogous to PointCloud."""
        if self.local_rank >= 1:
            return
        os.makedirs(self.config.CHECKPOINT_FOLDER, exist_ok=True)
        config_path = os.path.join(self.config.CHECKPOINT_FOLDER, "config.yaml")
        with open(config_path, "w", encoding="utf-8") as f:
            f.write(self.config.dump())
        logger.info(f"Configuration snapshot saved to {config_path}")
        print(f"[SS-ETP] Configuration snapshot saved to {config_path}")

    def _log_optimizer_groups(self) -> None:
        """Log each AdamW param group LR and tensor counts (rank 0)."""
        if self.local_rank >= 1 or not hasattr(self, "optimizer") or self.optimizer is None:
            return
        lines = ["---------- Optimizer param groups ----------"]
        for i, g in enumerate(self.optimizer.param_groups):
            n = sum(p.numel() for p in g["params"])
            lines.append(
                f"  [{i}] lr={g['lr']:.8g}  weight_decay={g.get('weight_decay', 0)}  tensors={n}"
            )
        if getattr(self, "scheduler", None) is not None:
            lines.append(
                f"  LambdaLR last_epoch={self.scheduler.last_epoch}  base_lrs={self.scheduler.base_lrs}"
            )
        msg = "\n".join(lines)
        logger.info(msg)
        print(msg)

    def _log_run_configuration_summary(self, phase: str = "train") -> None:
        """Print key IL/MODEL/POINT_CLOUD knobs to terminal and habitat logger (rank 0)."""
        if self.local_rank >= 1:
            return
        cfg = self.config
        il = cfg.IL
        m = cfg.MODEL
        ne = getattr(cfg, "NUM_ENVIRONMENTS", "n/a")
        lines = [
            "=" * 62,
            f" SS-ETP ({phase})  |  trainer={cfg.TRAINER_NAME}  env={cfg.ENV_NAME}",
            "=" * 62,
            f"  policy={m.policy_name}  task_type={m.task_type}  NUM_ENVIRONMENTS={ne}",
            f"  IL: lr={il.lr}  pointnet_lr={getattr(il, 'pointnet_lr', -1)}  iters={il.iters}  log_every={getattr(il, 'log_every', 'n/a')}",
            f"  IL: finetune_stage={getattr(il, 'finetune_stage', 'n/a')}  warmup_iters={getattr(il, 'warmup_iters', 0)}  min_lr_ratio={getattr(il, 'min_lr_ratio', 1.0)}",
            *(
                [
                    "  IL (point cloud SFT): align_train_bert=%s global_encoder=%s sap=%s pointcloud=%s rgb_enc=%s depth_enc=%s"
                    % (
                        getattr(il, "align_train_bert", True),
                        bool(getattr(il, "align_train_global_encoder", True)),
                        getattr(il, "align_train_sap", True),
                        getattr(il, "align_train_pointcloud", True),
                        getattr(il, "align_train_rgb_encoder", False),
                        getattr(il, "align_train_depth_encoder", False),
                    ),
                ]
                if self._use_point_cloud_sft(cfg)
                else []
            ),
            f"  IL: load_from_ckpt={il.load_from_ckpt}  is_requeue={il.is_requeue}  ml_weight={il.ml_weight}",
            f"  IL: sample_ratio={il.sample_ratio}  decay_interval={il.decay_interval}  expert_policy={getattr(il, 'expert_policy', 'n/a')}",
            f"  IL: loc_noise={getattr(il, 'loc_noise', 'n/a')}  ghost_aug={getattr(il, 'ghost_aug', 0)}  waypoint_aug={getattr(il, 'waypoint_aug', False)}",
            f"  MODEL: use_dynamic_graph={getattr(m, 'use_dynamic_graph', False)}  dynamic_graph_lr={getattr(m, 'dynamic_graph_lr', 'n/a')}",
            f"  MODEL: use_node_gating={getattr(m, 'use_node_gating', False)}  node_gating_lr={getattr(m, 'node_gating_lr', 'n/a')}",
            f"  MODEL: merge_ghost={getattr(m, 'merge_ghost', True)}  consume_ghost={getattr(m, 'consume_ghost', True)}",
            f"  MODEL: pretrained_path={getattr(m, 'pretrained_path', '')}",
        ]
        pc = getattr(m, "POINT_CLOUD", None)
        if pc is not None:
            lines.extend(
                [
                    "---------- POINT_CLOUD ----------",
                    f"  enable_depth_to_pointcloud={getattr(pc, 'enable_depth_to_pointcloud', False)}",
                    f"  enable_spatial_crop={getattr(pc, 'enable_spatial_crop', False)}  max_depth_m={getattr(pc, 'max_depth_m', 'n/a')}",
                    f"  num_points={getattr(pc, 'num_points', 'n/a')}  fusion_scope={getattr(pc, 'fusion_scope', 'n/a')}",
                    f"  enable_pointnet={getattr(pc, 'enable_pointnet', False)}  enable_fusion={getattr(pc, 'enable_fusion', False)}",
                    f"  accumulate_mode={getattr(pc, 'accumulate_mode', 'n/a')}  fps_seed={getattr(pc, 'fps_seed', 'n/a')}",
                    f"  enable_learnable_fusion_lambda={getattr(pc, 'enable_learnable_fusion_lambda', False)}  fusion_lambda_init={getattr(pc, 'fusion_lambda_init', 'n/a')}",
                ]
            )
        lines.append("=" * 62)
        msg = "\n".join(lines)
        logger.info(msg)
        print(msg)

    def _log_pointcloud_branch_status(self) -> None:
        """PointNet/fusion module status block (PointCloud-style), rank 0 only."""
        if self.local_rank >= 1:
            return
        net_module = self.policy.net.module if hasattr(self.policy.net, "module") else self.policy.net
        pc_cfg = getattr(self.config.MODEL, "POINT_CLOUD", None)
        if pc_cfg is None or not self._use_point_cloud_sft(self.config):
            logger.info("========== Point cloud branch: disabled (POINT_CLOUD off or pointnet/fusion false) ==========")
            print("========== Point cloud branch: disabled ==========")
            return
        pn = (
            sum(p.numel() for p in net_module.pointnet_encoder.parameters())
            if getattr(net_module, "pointnet_encoder", None)
            else 0
        )
        pj = (
            sum(p.numel() for p in net_module.pc_feat_proj.parameters())
            if getattr(net_module, "pc_feat_proj", None)
            else 0
        )
        lam = getattr(net_module, "fusion_lambda_raw", None)
        lam_s = (
            f"{float(torch.sigmoid(lam.detach()).item()):.6f}"
            if lam is not None
            else "n/a"
        )
        block = [
            "========== Point cloud / PointNet module status ==========",
            (
                f"  enable_depth_to_pointcloud={bool(pc_cfg.enable_depth_to_pointcloud)}  "
                f"enable_spatial_crop={bool(pc_cfg.enable_spatial_crop)}  "
                f"max_depth_m={float(pc_cfg.max_depth_m)}  num_points={int(pc_cfg.num_points)}"
            ),
            (
                f"  enable_pointnet={bool(pc_cfg.enable_pointnet)}  enable_fusion={bool(pc_cfg.enable_fusion)}  "
                f"accumulate_mode={str(pc_cfg.accumulate_mode)}  fusion_scope={str(pc_cfg.fusion_scope)}"
            ),
            (
                f"  pointnet_encoder_built={getattr(net_module, 'pointnet_encoder', None) is not None}  "
                f"pc_feat_proj_built={getattr(net_module, 'pc_feat_proj', None) is not None}  "
                f"fusion_lambda_sigmoid={lam_s}"
            ),
            f"  pointnet_params={pn}  fusion_proj_params={pj}",
            "===========================================================",
        ]
        msg = "\n".join(block)
        logger.info(msg)
        print(msg)

    def _make_dirs(self):
        if self.config.local_rank == 0:
            self._make_ckpt_dir()
            # os.makedirs(self.lmdb_features_dir, exist_ok=True)
            if self.config.EVAL.SAVE_RESULTS:
                self._make_results_dir()

    def _current_fusion_lambda_sigmoid(self):
        """sigmoid(fusion_lambda_raw), or None if not using learnable fusion."""
        net = self.policy.net.module if hasattr(self.policy.net, "module") else self.policy.net
        raw = getattr(net, "fusion_lambda_raw", None)
        if raw is None:
            return None
        return float(torch.sigmoid(raw.detach()).float().item())

    def save_checkpoint(self, iteration: int):
        if self.local_rank < 1:
            lam = self._current_fusion_lambda_sigmoid()
            if lam is not None:
                msg = (
                    f"[SS-ETP] Saving ckpt.iter{iteration}.pth  "
                    f"fusion_lambda_sigmoid={lam:.6f}"
                )
                logger.info(msg)
                print(msg)
            else:
                logger.info(f"[SS-ETP] Saving ckpt.iter{iteration}.pth")
                print(f"[SS-ETP] Saving ckpt.iter{iteration}.pth")
        checkpoint_dict = {
            "state_dict": self.policy.state_dict(),
            "config": self.config,
            "iteration": iteration,
        }
        # Only save optimizer state if optimizer exists (in training mode)
        if hasattr(self, 'optimizer'):
            checkpoint_dict["optim_state"] = self.optimizer.state_dict()
        if getattr(self, "scheduler", None) is not None:
            checkpoint_dict["scheduler_state"] = self.scheduler.state_dict()
        torch.save(
            obj=checkpoint_dict,
            f=os.path.join(self.config.CHECKPOINT_FOLDER, f"ckpt.iter{iteration}.pth"),
        )

    def _record_dynamic_graph_weights(self):
        """
        Record current dynamic graph weight values (for statistics)
        """
        try:
            model = self.policy.net
            if hasattr(model, 'vln_bert') and hasattr(model.vln_bert, 'global_encoder'):
                global_encoder = model.vln_bert.global_encoder
                if hasattr(global_encoder, 'encoder') and hasattr(global_encoder.encoder, 'x_layers'):
                    x_layers = global_encoder.encoder.x_layers
                    
                    for layer_idx, layer in enumerate(x_layers):
                        # Record dynamic graph weights
                        if hasattr(layer, 'use_dynamic_graph') and layer.use_dynamic_graph:
                            if layer_idx not in self.dynamic_graph_weight_history:
                                self.dynamic_graph_weight_history[layer_idx] = {
                                    'w1': [], 'w2': [], 'w3': []
                                }
                            
                            if hasattr(layer, 'w1'):
                                self.dynamic_graph_weight_history[layer_idx]['w1'].append(float(layer.w1.item()))
                            if hasattr(layer, 'w2'):
                                self.dynamic_graph_weight_history[layer_idx]['w2'].append(float(layer.w2.item()))
                            if hasattr(layer, 'w3'):
                                self.dynamic_graph_weight_history[layer_idx]['w3'].append(float(layer.w3.item()))
        except Exception as e:
            # Fail silently, do not affect training
            pass

    def save_dynamic_graph_weights(self, iteration: int):
        """
        Save dynamic graph weight information (w1, w2, w3) to JSON file
        Save every 200 iterations, including:
        1. Weight values for each of the 200 iterations (complete history)
        2. Statistics (max, min, mean, std)
        """
        try:
            # Get model
            model = self.policy.net
            if hasattr(model, 'vln_bert') and hasattr(model.vln_bert, 'global_encoder'):
                global_encoder = model.vln_bert.global_encoder
                if hasattr(global_encoder, 'encoder') and hasattr(global_encoder.encoder, 'x_layers'):
                    x_layers = global_encoder.encoder.x_layers
                    
                    # Collect weight information for all layers
                    weights_data = {
                        'iteration': iteration,
                        'layers': []
                    }
                    
                    for layer_idx, layer in enumerate(x_layers):
                        layer_data = {
                            'layer_index': layer_idx,
                            'weights': {}
                        }
                        
                        # Process dynamic graph weights
                        if hasattr(layer, 'use_dynamic_graph') and layer.use_dynamic_graph:
                            # Current values
                            current_w1 = float(layer.w1.item()) if hasattr(layer, 'w1') else None
                            current_w2 = float(layer.w2.item()) if hasattr(layer, 'w2') else None
                            current_w3 = float(layer.w3.item()) if hasattr(layer, 'w3') else None
                            
                            # Prepare weight data
                            weights_info = {}
                            
                            if layer_idx in self.dynamic_graph_weight_history:
                                hist = self.dynamic_graph_weight_history[layer_idx]
                                
                                for weight_name in ['w1', 'w2', 'w3']:
                                    if hist[weight_name]:
                                        values = hist[weight_name]  # Keep all historical values
                                        current_val = current_w1 if weight_name == 'w1' else (current_w2 if weight_name == 'w2' else current_w3)
                                        
                                        # Calculate statistics
                                        values_array = np.array(values)
                                        weights_info[weight_name] = {
                                            'history': values,  # Save all historical values
                                            'current': current_val,
                                            'max': float(np.max(values_array)),
                                            'min': float(np.min(values_array)),
                                            'mean': float(np.mean(values_array)),
                                            'std': float(np.std(values_array)),
                                            'count': len(values)
                                        }
                                    else:
                                        # If no historical data, only save current value
                                        current_val = current_w1 if weight_name == 'w1' else (current_w2 if weight_name == 'w2' else current_w3)
                                        if current_val is not None:
                                            weights_info[weight_name] = {
                                                'history': [current_val],  # At least save current value
                                                'current': current_val,
                                                'max': current_val,
                                                'min': current_val,
                                                'mean': current_val,
                                                'std': 0.0,
                                                'count': 1
                                            }
                            else:
                                # If no historical data, only save current value
                                for weight_name, current_val in [('w1', current_w1), ('w2', current_w2), ('w3', current_w3)]:
                                    if current_val is not None:
                                        weights_info[weight_name] = {
                                            'history': [current_val],  # At least save current value
                                            'current': current_val,
                                            'max': current_val,
                                            'min': current_val,
                                            'mean': current_val,
                                            'std': 0.0,
                                            'count': 1
                                        }
                            
                            layer_data['weights'] = weights_info
                        
                        # Only add to results if layer has dynamic graph weights
                        if layer_data['weights']:
                            weights_data['layers'].append(layer_data)
                    
                    # Save to JSON file
                    weights_dir = os.path.join(self.config.CHECKPOINT_FOLDER, 'dynamic_graph_weights')
                    os.makedirs(weights_dir, exist_ok=True)
                    weights_file = os.path.join(weights_dir, f'weights_iter{iteration}.json')
                    
                    with open(weights_file, 'w', encoding='utf-8') as f:
                        json.dump(weights_data, f, indent=2, ensure_ascii=False)
                    
                    # Clear history for next 200 iterations
                    self.dynamic_graph_weight_history.clear()
                    
                    # logger.info(f'Saved dynamic graph weights to {weights_file}')
        except Exception as e:
            logger.warning(f'Failed to save dynamic graph weights: {e}')

    def _set_config(self):
        self.split = self.config.TASK_CONFIG.DATASET.SPLIT
        self.config.defrost()
        self.config.TASK_CONFIG.TASK.NDTW.SPLIT = self.split
        self.config.TASK_CONFIG.TASK.SDTW.SPLIT = self.split
        self.config.TASK_CONFIG.ENVIRONMENT.ITERATOR_OPTIONS.MAX_SCENE_REPEAT_STEPS = -1
        self.config.SIMULATOR_GPU_IDS = self.config.SIMULATOR_GPU_IDS[self.config.local_rank]
        self.config.use_pbar = not is_slurm_batch_job()
        ''' if choosing image '''
        resize_config = self.config.RL.POLICY.OBS_TRANSFORMS.RESIZER_PER_SENSOR.SIZES
        crop_config = self.config.RL.POLICY.OBS_TRANSFORMS.CENTER_CROPPER_PER_SENSOR.SENSOR_CROPS
        task_config = self.config.TASK_CONFIG
        camera_orientations = get_camera_orientations12()
        for sensor_type in ["RGB", "DEPTH"]:
            resizer_size = dict(resize_config)[sensor_type.lower()]
            cropper_size = dict(crop_config)[sensor_type.lower()]
            sensor = getattr(task_config.SIMULATOR, f"{sensor_type}_SENSOR")
            for action, orient in camera_orientations.items():
                camera_template = f"{sensor_type}_{action}"
                camera_config = deepcopy(sensor)
                camera_config.ORIENTATION = camera_orientations[action]
                camera_config.UUID = camera_template.lower()
                setattr(task_config.SIMULATOR, camera_template, camera_config)
                task_config.SIMULATOR.AGENT_0.SENSORS.append(camera_template)
                resize_config.append((camera_template.lower(), resizer_size))
                crop_config.append((camera_template.lower(), cropper_size))
        self.config.RL.POLICY.OBS_TRANSFORMS.RESIZER_PER_SENSOR.SIZES = resize_config
        self.config.RL.POLICY.OBS_TRANSFORMS.CENTER_CROPPER_PER_SENSOR.SENSOR_CROPS = crop_config
        self.config.TASK_CONFIG = task_config
        self.config.SENSORS = task_config.SIMULATOR.AGENT_0.SENSORS
        if self.config.VIDEO_OPTION:
            self.config.TASK_CONFIG.TASK.MEASUREMENTS.append("TOP_DOWN_MAP_VLNCE")
            self.config.TASK_CONFIG.TASK.MEASUREMENTS.append("DISTANCE_TO_GOAL")
            self.config.TASK_CONFIG.TASK.MEASUREMENTS.append("SUCCESS")
            self.config.TASK_CONFIG.TASK.MEASUREMENTS.append("SPL")
            os.makedirs(self.config.VIDEO_DIR, exist_ok=True)
            shift = 0.
            orient_dict = {
                'Back': [0, math.pi + shift, 0],            # Back
                'Down': [-math.pi / 2, 0 + shift, 0],       # Down
                'Front':[0, 0 + shift, 0],                  # Front
                'Right':[0, math.pi / 2 + shift, 0],        # Right
                'Left': [0, 3 / 2 * math.pi + shift, 0],    # Left
                'Up':   [math.pi / 2, 0 + shift, 0],        # Up
            }
            sensor_uuids = []
            H = 224
            for sensor_type in ["RGB"]:
                sensor = getattr(self.config.TASK_CONFIG.SIMULATOR, f"{sensor_type}_SENSOR")
                for camera_id, orient in orient_dict.items():
                    camera_template = f"{sensor_type}{camera_id}"
                    camera_config = deepcopy(sensor)
                    camera_config.WIDTH = H
                    camera_config.HEIGHT = H
                    camera_config.ORIENTATION = orient
                    camera_config.UUID = camera_template.lower()
                    camera_config.HFOV = 90
                    sensor_uuids.append(camera_config.UUID)
                    setattr(self.config.TASK_CONFIG.SIMULATOR, camera_template, camera_config)
                    self.config.TASK_CONFIG.SIMULATOR.AGENT_0.SENSORS.append(camera_template)
        self.config.freeze()

        self.world_size = self.config.GPU_NUMBERS
        self.local_rank = self.config.local_rank
        self.batch_size = self.config.IL.batch_size
        torch.cuda.set_device(self.device)
        if self.world_size > 1:
            distr.init_process_group(backend='nccl', init_method='env://')
            self.device = self.config.TORCH_GPU_IDS[self.local_rank]
            self.config.defrost()
            self.config.TORCH_GPU_ID = self.config.TORCH_GPU_IDS[self.local_rank]
            self.config.freeze()
            torch.cuda.set_device(self.device)

    def _init_envs(self):
        # for DDP to load different data
        self.config.defrost()
        self.config.TASK_CONFIG.SEED = self.config.TASK_CONFIG.SEED + self.local_rank
        self.config.freeze()

        self.envs = construct_envs(
            self.config, 
            get_env_class(self.config.ENV_NAME),
            auto_reset_done=False
        )
        env_num = self.envs.num_envs
        dataset_len = sum(self.envs.number_of_episodes)
        logger.info(f'LOCAL RANK: {self.local_rank}, ENV NUM: {env_num}, DATASET LEN: {dataset_len}')
        observation_space = self.envs.observation_spaces[0]
        action_space = self.envs.action_spaces[0]
        self.obs_transforms = get_active_obs_transforms(self.config)
        observation_space = apply_obs_transforms_obs_space(
            observation_space, self.obs_transforms
        )

        return observation_space, action_space

    def _use_point_cloud_sft(self, config: Config) -> bool:
        pc = getattr(config.MODEL, "POINT_CLOUD", None)
        if pc is None:
            return False
        return bool(getattr(pc, "enable_pointnet", False) or getattr(pc, "enable_fusion", False))

    def _apply_point_cloud_finetune_stage(self, config: Config) -> None:
        finetune_stage = str(getattr(config.IL, "finetune_stage", "align")).lower()
        pointnet_name_keywords = ("pointnet_encoder", "pc_feat_proj", "fusion_lambda")
        net = self.policy.net.module if hasattr(self.policy.net, "module") else self.policy.net
        mc = config.MODEL

        if finetune_stage == "warmup":
            for n, p in self.policy.named_parameters():
                p.requires_grad_(any(k in n for k in pointnet_name_keywords))
            if self.local_rank < 1:
                logger.info("SFT finetune_stage=warmup: only PointNet/fusion parameters are trainable.")
                print("[SS-ETP] finetune_stage=warmup: only PointNet/fusion trainable.")
        elif finetune_stage == "align":
            align_train_bert = bool(getattr(config.IL, "align_train_bert", True))
            # ETPNav: global planning = vln_bert.global_encoder; SAP = vln_bert.global_sap_head.
            align_train_global_encoder = bool(
                getattr(config.IL, "align_train_global_encoder", True)
            )
            align_train_sap = bool(getattr(config.IL, "align_train_sap", True))
            align_train_pointcloud = bool(getattr(config.IL, "align_train_pointcloud", True))
            align_train_rgb = bool(getattr(config.IL, "align_train_rgb_encoder", False))
            align_train_depth = bool(getattr(config.IL, "align_train_depth_encoder", False))

            for p in self.policy.parameters():
                p.requires_grad_(False)

            for p in net.rgb_encoder.parameters():
                p.requires_grad_(align_train_rgb)
            for p in net.depth_encoder.parameters():
                p.requires_grad_(align_train_depth)

            if align_train_pointcloud:
                if getattr(net, "pointnet_encoder", None) is not None:
                    for p in net.pointnet_encoder.parameters():
                        p.requires_grad_(True)
                if getattr(net, "pc_feat_proj", None) is not None:
                    for p in net.pc_feat_proj.parameters():
                        p.requires_grad_(True)
                if getattr(net, "fusion_lambda_raw", None) is not None:
                    net.fusion_lambda_raw.requires_grad_(True)

            vb = net.vln_bert
            if align_train_bert:
                for p in vb.embeddings.parameters():
                    p.requires_grad_(True)
                for p in vb.lang_encoder.parameters():
                    p.requires_grad_(True)
                for p in vb.img_embeddings.parameters():
                    p.requires_grad_(True)
                if mc.fix_lang_embedding:
                    for p in vb.embeddings.parameters():
                        p.requires_grad_(False)
                    for p in vb.lang_encoder.parameters():
                        p.requires_grad_(False)
                elif not getattr(vb.lang_encoder, "update_lang_bert", True):
                    for p in vb.lang_encoder.layer.parameters():
                        p.requires_grad_(False)
                if mc.fix_pano_embedding:
                    for p in vb.img_embeddings.parameters():
                        p.requires_grad_(False)

            if align_train_global_encoder and getattr(vb, "global_encoder", None) is not None:
                for p in vb.global_encoder.parameters():
                    p.requires_grad_(True)

            if align_train_sap and getattr(vb, "global_sap_head", None) is not None:
                for p in vb.global_sap_head.parameters():
                    p.requires_grad_(True)

            if self.local_rank < 1:
                logger.info(
                    "SFT finetune_stage=align: bert=%s global_encoder=%s sap=%s pointcloud=%s rgb=%s depth=%s"
                    % (
                        align_train_bert,
                        align_train_global_encoder,
                        align_train_sap,
                        align_train_pointcloud,
                        align_train_rgb,
                        align_train_depth,
                    )
                )
                print(
                    "[SS-ETP] finetune_stage=align: "
                    f"bert={align_train_bert} global_encoder={align_train_global_encoder} sap={align_train_sap} "
                    f"pointcloud={align_train_pointcloud} rgb={align_train_rgb} depth={align_train_depth}"
                )
        else:
            raise ValueError(
                f"Unknown IL.finetune_stage: {finetune_stage!r} (expected 'warmup' or 'align')."
            )

    def _build_optimizer_unified(self, config: Config) -> None:
        use_pc = self._use_point_cloud_sft(config)
        use_dynamic = getattr(config.MODEL, "use_dynamic_graph", False)
        use_gating = getattr(config.MODEL, "use_node_gating", False)
        pointnet_lr = float(getattr(config.IL, "pointnet_lr", -1.0))
        if pointnet_lr <= 0:
            pointnet_lr = float(config.IL.lr)

        pointnet_keys = ("pointnet_encoder", "pc_feat_proj", "fusion_lambda")
        trainable = [(n, p) for n, p in self.policy.named_parameters() if p.requires_grad]

        buckets = {"pointnet": [], "dynamic": [], "gating": [], "base": []}
        for n, p in trainable:
            if use_pc and any(k in n for k in pointnet_keys):
                buckets["pointnet"].append((n, p))
            elif use_dynamic and (
                "w1" in n
                or "w2" in n
                or "w3" in n
                or "semantic_sim_mlp" in n
                or "instruction_rel_mlp" in n
            ):
                buckets["dynamic"].append((n, p))
            elif use_gating and "node_gating_mlp" in n:
                buckets["gating"].append((n, p))
            else:
                buckets["base"].append((n, p))

        no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = []

        def add_bucket(pairs, lr):
            if not pairs:
                return
            decay_p = [p for n, p in pairs if not any(nd in n for nd in no_decay)]
            no_d_p = [p for n, p in pairs if any(nd in n for nd in no_decay)]
            if decay_p:
                optimizer_grouped_parameters.append(
                    {"params": decay_p, "weight_decay": 0.01, "lr": float(lr)}
                )
            if no_d_p:
                optimizer_grouped_parameters.append(
                    {"params": no_d_p, "weight_decay": 0.0, "lr": float(lr)}
                )

        dglr = float(getattr(config.MODEL, "dynamic_graph_lr", config.IL.lr))
        nglr = float(getattr(config.MODEL, "node_gating_lr", config.IL.lr))

        add_bucket(buckets["base"], config.IL.lr)
        add_bucket(buckets["dynamic"], dglr)
        add_bucket(buckets["gating"], nglr)
        add_bucket(buckets["pointnet"], pointnet_lr)

        if not optimizer_grouped_parameters:
            raise ValueError("No trainable parameters for optimizer (check freezing / config).")

        self.optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=float(config.IL.lr))
        if self.local_rank < 1:
            logger.info(
                "Optimizer: base_tensors=%d dynamic=%d gating=%d pointnet=%d pointnet_lr=%.6g"
                % (
                    len(buckets["base"]),
                    len(buckets["dynamic"]),
                    len(buckets["gating"]),
                    len(buckets["pointnet"]),
                    pointnet_lr,
                )
            )

    def _build_lr_scheduler_point_cloud(self, config: Config) -> None:
        num_warmup_steps = int(getattr(config.IL, "warmup_iters", 0))
        if num_warmup_steps <= 0:
            self.scheduler = None
            return
        num_training_steps = int(config.IL.iters)
        min_lr_ratio = float(getattr(config.IL, "min_lr_ratio", 1.0))

        def lr_lambda(current_step: int):
            if current_step < num_warmup_steps:
                return float(current_step) / float(max(1, num_warmup_steps))
            progress = float(current_step - num_warmup_steps) / float(
                max(1, num_training_steps - num_warmup_steps)
            )
            progress = min(1.0, progress)
            cosine_decay = 0.5 * (1.0 + math.cos(math.pi * progress))
            return min_lr_ratio + (1.0 - min_lr_ratio) * cosine_decay

        self.scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda)

    def _initialize_policy(
        self,
        config: Config,
        load_from_ckpt: bool,
        observation_space: Space,
        action_space: Space,
        create_optimizer: bool = True,
    ):
        start_iter = 0
        policy = baseline_registry.get_policy(self.config.MODEL.policy_name)
        self.policy = policy.from_config(
            config=config,
            observation_space=observation_space,
            action_space=action_space,
        )
        ''' initialize the waypoint predictor here '''
        from vlnce_baselines.waypoint_pred.TRM_net import BinaryDistPredictor_TRM
        self.waypoint_predictor = BinaryDistPredictor_TRM(device=self.device)
        cwp_fn = 'data/wp_pred/check_cwp_bestdist_hfov63' if self.config.MODEL.task_type == 'rxr' else 'data/wp_pred/check_cwp_bestdist_hfov90'
        self.waypoint_predictor.load_state_dict(torch.load(cwp_fn, map_location = torch.device('cpu'))['predictor']['state_dict'])
        for param in self.waypoint_predictor.parameters():
            param.requires_grad_(False)

        self.policy.to(self.device)
        self.waypoint_predictor.to(self.device)
        self.num_recurrent_layers = self.policy.net.num_recurrent_layers

        if self.config.GPU_NUMBERS > 1:
            print('Using', self.config.GPU_NUMBERS,'GPU!')
            # find_unused_parameters=False fix ddp bug
            self.policy.net = DDP(self.policy.net.to(self.device), device_ids=[self.device],
                output_device=self.device, find_unused_parameters=False, broadcast_buffers=False)

        self.optimizer = None
        self.scheduler = None

        if create_optimizer and bool(config.IL.is_requeue):
            self._build_optimizer_unified(config)
            self._build_lr_scheduler_point_cloud(config)

        if load_from_ckpt:
            if config.IL.is_requeue:
                import glob
                ckpt_list = list(filter(os.path.isfile, glob.glob(config.CHECKPOINT_FOLDER + "/*")) )
                ckpt_list.sort(key=os.path.getmtime)
                ckpt_path = ckpt_list[-1]
            else:
                ckpt_path = config.IL.ckpt_to_load
            ckpt_dict = self.load_checkpoint(ckpt_path, map_location="cpu")
            start_iter = ckpt_dict["iteration"]

            if 'module' in list(ckpt_dict['state_dict'].keys())[0] and self.config.GPU_NUMBERS == 1:
                self.policy.net = torch.nn.DataParallel(self.policy.net.to(self.device),
                    device_ids=[self.device], output_device=self.device)
                incompatible = self.policy.load_state_dict(ckpt_dict["state_dict"], strict=False)
                self.policy.net = self.policy.net.module
                self.waypoint_predictor = torch.nn.DataParallel(self.waypoint_predictor.to(self.device),
                    device_ids=[self.device], output_device=self.device)
            else:
                incompatible = self.policy.load_state_dict(ckpt_dict["state_dict"], strict=False)
            if self.local_rank < 1 and incompatible is not None:
                miss = getattr(incompatible, "missing_keys", []) or []
                unexp = getattr(incompatible, "unexpected_keys", []) or []
                logger.info(
                    f"Checkpoint strict=False: missing_keys={len(miss)}  unexpected_keys={len(unexp)}"
                )
                if len(miss) > 0:
                    preview = miss[:20]
                    logger.info(f"  missing_keys (first {len(preview)}): {preview}")
                if len(unexp) > 0:
                    preview_u = unexp[:20]
                    logger.info(f"  unexpected_keys (first {len(preview_u)}): {preview_u}")
            if config.IL.is_requeue and hasattr(self, 'optimizer') and self.optimizer is not None:
                self.optimizer.load_state_dict(ckpt_dict["optim_state"])
                if "scheduler_state" in ckpt_dict and getattr(self, "scheduler", None) is not None:
                    self.scheduler.load_state_dict(ckpt_dict["scheduler_state"])
            logger.info(f"Loaded weights from checkpoint: {ckpt_path}, iteration: {start_iter}")

        if create_optimizer and not bool(config.IL.is_requeue):
            if self._use_point_cloud_sft(config):
                self._apply_point_cloud_finetune_stage(config)
            self._build_optimizer_unified(config)
            if self._use_point_cloud_sft(config):
                self._build_lr_scheduler_point_cloud(config)

        params = sum(param.numel() for param in self.policy.parameters())
        params_t = sum(
            p.numel() for p in self.policy.parameters() if p.requires_grad
        )
        logger.info(f"Agent parameters: {params/1e6:.2f} MB. Trainable: {params_t/1e6:.2f} MB.")
        logger.info("Finished setting up policy.")

        phase = "train" if create_optimizer else "eval_or_infer"
        self._log_run_configuration_summary(phase)
        self._log_pointcloud_branch_status()
        self._log_optimizer_groups()

        return start_iter

    def _teacher_action(self, batch_angles, batch_distances, candidate_lengths):
        if self.config.MODEL.task_type == 'r2r':
            cand_dists_to_goal = [[] for _ in range(len(batch_angles))]
            oracle_cand_idx = []
            for j in range(len(batch_angles)):
                for k in range(len(batch_angles[j])):
                    angle_k = batch_angles[j][k]
                    forward_k = batch_distances[j][k]
                    dist_k = self.envs.call_at(j, "cand_dist_to_goal", {"angle": angle_k, "forward": forward_k})
                    cand_dists_to_goal[j].append(dist_k)
                curr_dist_to_goal = self.envs.call_at(j, "current_dist_to_goal")
                # if within target range (which def as 3.0)
                if curr_dist_to_goal < 1.5:
                    oracle_cand_idx.append(candidate_lengths[j] - 1)
                else:
                    oracle_cand_idx.append(np.argmin(cand_dists_to_goal[j]))
            return oracle_cand_idx
        elif self.config.MODEL.task_type == 'rxr':
            kargs = []
            current_episodes = self.envs.current_episodes()
            for i in range(self.envs.num_envs):
                kargs.append({
                    'ref_path':self.gt_data[str(current_episodes[i].episode_id)]['locations'],
                    'angles':batch_angles[i],
                    'distances':batch_distances[i],
                    'candidate_length':candidate_lengths[i]
                })
            oracle_cand_idx = self.envs.call(["get_cand_idx"]*self.envs.num_envs, kargs)
            return oracle_cand_idx

    def _teacher_action_new(self, batch_gmap_vp_ids, batch_no_vp_left):
        teacher_actions = []
        cur_episodes = self.envs.current_episodes()
        for i, (gmap_vp_ids, gmap, no_vp_left) in enumerate(zip(batch_gmap_vp_ids, self.gmaps, batch_no_vp_left)):
            curr_dis_to_goal = self.envs.call_at(i, "current_dist_to_goal")
            if curr_dis_to_goal < 1.5:
                teacher_actions.append(0)
            else:
                if no_vp_left:
                    teacher_actions.append(-100)
                elif self.config.IL.expert_policy == 'spl':
                    ghost_vp_pos = [(vp, random.choice(pos)) for vp, pos in gmap.ghost_real_pos.items()]
                    ghost_dis_to_goal = [
                        self.envs.call_at(i, "point_dist_to_goal", {"pos": p[1]})
                        for p in ghost_vp_pos
                    ]
                    target_ghost_vp = ghost_vp_pos[np.argmin(ghost_dis_to_goal)][0]
                    teacher_actions.append(gmap_vp_ids.index(target_ghost_vp))
                elif self.config.IL.expert_policy == 'ndtw':
                    ghost_vp_pos = [(vp, random.choice(pos)) for vp, pos in gmap.ghost_real_pos.items()]
                    target_ghost_vp = self.envs.call_at(i, "ghost_dist_to_ref", {
                        "ghost_vp_pos": ghost_vp_pos,
                        "ref_path": self.gt_data[str(cur_episodes[i].episode_id)]['locations'],
                    })
                    teacher_actions.append(gmap_vp_ids.index(target_ghost_vp))
                else:
                    raise NotImplementedError
       
        return torch.tensor(teacher_actions).cuda()

    def _vp_feature_variable(self, obs):
        batch_rgb_fts, batch_dep_fts, batch_loc_fts = [], [], []
        batch_nav_types, batch_view_lens = [], []

        for i in range(self.envs.num_envs):
            rgb_fts, dep_fts, loc_fts , nav_types = [], [], [], []
            cand_idxes = np.zeros(12, dtype=np.bool)
            cand_idxes[obs['cand_img_idxes'][i]] = True
            # cand
            rgb_fts.append(obs['cand_rgb'][i])
            dep_fts.append(obs['cand_depth'][i])
            loc_fts.append(obs['cand_angle_fts'][i])
            nav_types += [1] * len(obs['cand_angles'][i])
            # non-cand
            rgb_fts.append(obs['pano_rgb'][i][~cand_idxes])
            dep_fts.append(obs['pano_depth'][i][~cand_idxes])
            loc_fts.append(obs['pano_angle_fts'][~cand_idxes])
            nav_types += [0] * (12-np.sum(cand_idxes))
            
            batch_rgb_fts.append(torch.cat(rgb_fts, dim=0))
            batch_dep_fts.append(torch.cat(dep_fts, dim=0))
            batch_loc_fts.append(torch.cat(loc_fts, dim=0))
            batch_nav_types.append(torch.LongTensor(nav_types))
            batch_view_lens.append(len(nav_types))
        # collate
        batch_rgb_fts = pad_tensors_wgrad(batch_rgb_fts)
        batch_dep_fts = pad_tensors_wgrad(batch_dep_fts)
        batch_loc_fts = pad_tensors_wgrad(batch_loc_fts).cuda()
        batch_nav_types = pad_sequence(batch_nav_types, batch_first=True).cuda()
        batch_view_lens = torch.LongTensor(batch_view_lens).cuda()

        return {
            'rgb_fts': batch_rgb_fts, 'dep_fts': batch_dep_fts, 'loc_fts': batch_loc_fts,
            'nav_types': batch_nav_types, 'view_lens': batch_view_lens,
        }
        
    def _nav_gmap_variable(self, cur_vp, cur_pos, cur_ori):
        batch_gmap_vp_ids, batch_gmap_step_ids, batch_gmap_lens = [], [], []
        batch_gmap_img_fts, batch_gmap_pos_fts = [], []
        batch_gmap_pair_dists, batch_gmap_visited_masks = [], []
        batch_no_vp_left = []

        pc_cfg = getattr(self.config.MODEL, "POINT_CLOUD", None)
        pack_pc = (
            pc_cfg is not None
            and bool(getattr(pc_cfg, "enable_pointnet", False))
            and bool(getattr(pc_cfg, "enable_fusion", False))
        )
        num_points = int(pc_cfg.num_points) if pack_pc else 0
        fusion_scope = str(getattr(pc_cfg, "fusion_scope", "connected")).lower() if pack_pc else "connected"
        connected_only = fusion_scope == "connected"
        max_pc_samples = 1
        if pack_pc:
            for bi, gmap in enumerate(self.gmaps):
                for vp in gmap.ghost_pos.keys():
                    if connected_only and (cur_vp[bi] not in gmap.ghost_fronts.get(vp, [])):
                        continue
                    pc_list = gmap.ghost_pc_points.get(vp, None)
                    if pc_list is not None:
                        max_pc_samples = max(max_pc_samples, int(len(pc_list)))

        batch_gmap_pc_points = []
        batch_gmap_pc_masks = []
        batch_gmap_pc_fusion_masks = []

        for i, gmap in enumerate(self.gmaps):
            node_vp_ids = list(gmap.node_pos.keys())
            ghost_vp_ids = list(gmap.ghost_pos.keys())
            if len(ghost_vp_ids) == 0:
                batch_no_vp_left.append(True)
            else:
                batch_no_vp_left.append(False)

            gmap_vp_ids = [None] + node_vp_ids + ghost_vp_ids
            gmap_step_ids = [0] + [gmap.node_stepId[vp] for vp in node_vp_ids] + [0]*len(ghost_vp_ids)
            gmap_visited_masks = [0] + [1] * len(node_vp_ids) + [0] * len(ghost_vp_ids)

            gmap_img_fts = [gmap.get_node_embeds(vp) for vp in node_vp_ids] + \
                           [gmap.get_node_embeds(vp) for vp in ghost_vp_ids]
            gmap_img_fts = torch.stack(
                [torch.zeros_like(gmap_img_fts[0])] + gmap_img_fts, dim=0
            )

            if pack_pc:
                zero_pc = torch.zeros(
                    max_pc_samples, num_points, 3,
                    dtype=torch.float32, device=self.device,
                )
                token_pc_points = []
                token_pc_masks = []
                token_pc_fusion_masks = []

                token_pc_points.append(zero_pc)
                token_pc_masks.append(torch.zeros(max_pc_samples, dtype=torch.bool, device=self.device))
                token_pc_fusion_masks.append(torch.zeros(max_pc_samples, dtype=torch.bool, device=self.device))

                for _ in node_vp_ids:
                    token_pc_points.append(zero_pc)
                    token_pc_masks.append(torch.zeros(max_pc_samples, dtype=torch.bool, device=self.device))
                    token_pc_fusion_masks.append(torch.zeros(max_pc_samples, dtype=torch.bool, device=self.device))

                for vp in ghost_vp_ids:
                    is_connected_ghost = bool(cur_vp[i] in gmap.ghost_fronts.get(vp, []))
                    if connected_only and (not is_connected_ghost):
                        token_pc_points.append(zero_pc)
                        token_pc_masks.append(torch.zeros(max_pc_samples, dtype=torch.bool, device=self.device))
                        token_pc_fusion_masks.append(torch.zeros(max_pc_samples, dtype=torch.bool, device=self.device))
                        continue
                    pc_list = gmap.ghost_pc_points.get(vp, None)
                    if pc_list is not None and len(pc_list) > 0:
                        stack = torch.stack(
                            [pc.to(self.device, dtype=torch.float32) for pc in pc_list], dim=0
                        )
                        t = int(stack.shape[0])
                        pc_padded = torch.zeros(
                            max_pc_samples, num_points, 3,
                            dtype=torch.float32, device=self.device,
                        )
                        pc_padded[:t] = stack
                        mask = torch.zeros(max_pc_samples, dtype=torch.bool, device=self.device)
                        mask[:t] = True
                        token_pc_points.append(pc_padded)
                        token_pc_masks.append(mask)
                        token_pc_fusion_masks.append(
                            torch.full(
                                (max_pc_samples,),
                                is_connected_ghost,
                                dtype=torch.bool,
                                device=self.device,
                            )
                        )
                    else:
                        token_pc_points.append(zero_pc)
                        token_pc_masks.append(torch.zeros(max_pc_samples, dtype=torch.bool, device=self.device))
                        token_pc_fusion_masks.append(torch.zeros(max_pc_samples, dtype=torch.bool, device=self.device))

                gmap_pc_points_tokens = torch.stack(token_pc_points, dim=0)
                gmap_pc_masks_tokens = torch.stack(token_pc_masks, dim=0)
                gmap_pc_fusion_masks_tokens = torch.stack(token_pc_fusion_masks, dim=0)
                batch_gmap_pc_points.append(gmap_pc_points_tokens)
                batch_gmap_pc_masks.append(gmap_pc_masks_tokens)
                batch_gmap_pc_fusion_masks.append(gmap_pc_fusion_masks_tokens)

            gmap_pos_fts = gmap.get_pos_fts(
                cur_vp[i], cur_pos[i], cur_ori[i], gmap_vp_ids
            )
            gmap_pair_dists = np.zeros((len(gmap_vp_ids), len(gmap_vp_ids)), dtype=np.float32)
            for j in range(1, len(gmap_vp_ids)):
                for k in range(j+1, len(gmap_vp_ids)):
                    vp1 = gmap_vp_ids[j]
                    vp2 = gmap_vp_ids[k]
                    if not vp1.startswith('g') and not vp2.startswith('g'):
                        dist = gmap.shortest_dist[vp1][vp2]
                    elif not vp1.startswith('g') and vp2.startswith('g'):
                        front_dis2, front_vp2 = gmap.front_to_ghost_dist(vp2)
                        dist = gmap.shortest_dist[vp1][front_vp2] + front_dis2
                    elif vp1.startswith('g') and vp2.startswith('g'):
                        front_dis1, front_vp1 = gmap.front_to_ghost_dist(vp1)
                        front_dis2, front_vp2 = gmap.front_to_ghost_dist(vp2)
                        dist = front_dis1 + gmap.shortest_dist[front_vp1][front_vp2] + front_dis2
                    else:
                        raise NotImplementedError
                    gmap_pair_dists[j, k] = gmap_pair_dists[k, j] = dist / MAX_DIST
            
            batch_gmap_vp_ids.append(gmap_vp_ids)
            batch_gmap_step_ids.append(torch.LongTensor(gmap_step_ids))
            batch_gmap_lens.append(len(gmap_vp_ids))
            batch_gmap_img_fts.append(gmap_img_fts)
            batch_gmap_pos_fts.append(torch.from_numpy(gmap_pos_fts))
            batch_gmap_pair_dists.append(torch.from_numpy(gmap_pair_dists))
            batch_gmap_visited_masks.append(torch.BoolTensor(gmap_visited_masks))
        
        # collate
        batch_gmap_step_ids = pad_sequence(batch_gmap_step_ids, batch_first=True).cuda()
        batch_gmap_img_fts = pad_tensors_wgrad(batch_gmap_img_fts)
        batch_gmap_pos_fts = pad_tensors_wgrad(batch_gmap_pos_fts).cuda()
        batch_gmap_lens = torch.LongTensor(batch_gmap_lens)
        batch_gmap_masks = gen_seq_masks(batch_gmap_lens).cuda()
        batch_gmap_visited_masks = pad_sequence(batch_gmap_visited_masks, batch_first=True).cuda()

        bs = self.envs.num_envs
        max_gmap_len = max(batch_gmap_lens)
        gmap_pair_dists = torch.zeros(bs, max_gmap_len, max_gmap_len).float()
        for i in range(bs):
            gmap_pair_dists[i, :batch_gmap_lens[i], :batch_gmap_lens[i]] = batch_gmap_pair_dists[i]
        gmap_pair_dists = gmap_pair_dists.cuda()

        out = {
            'gmap_vp_ids': batch_gmap_vp_ids, 'gmap_step_ids': batch_gmap_step_ids,
            'gmap_img_fts': batch_gmap_img_fts, 'gmap_pos_fts': batch_gmap_pos_fts, 
            'gmap_masks': batch_gmap_masks, 'gmap_visited_masks': batch_gmap_visited_masks, 'gmap_pair_dists': gmap_pair_dists,
            'no_vp_left': batch_no_vp_left,
        }
        if pack_pc:
            gmap_pc_points = torch.zeros(
                bs, max_gmap_len, max_pc_samples, num_points, 3,
                dtype=torch.float32, device=self.device,
            )
            gmap_pc_masks = torch.zeros(
                bs, max_gmap_len, max_pc_samples, dtype=torch.bool, device=self.device,
            )
            gmap_pc_fusion_masks = torch.zeros(
                bs, max_gmap_len, max_pc_samples, dtype=torch.bool, device=self.device,
            )
            for i in range(bs):
                gmap_pc_points[i, :batch_gmap_lens[i]] = batch_gmap_pc_points[i]
                gmap_pc_masks[i, :batch_gmap_lens[i]] = batch_gmap_pc_masks[i]
                gmap_pc_fusion_masks[i, :batch_gmap_lens[i]] = batch_gmap_pc_fusion_masks[i]
            out['gmap_pc_points'] = gmap_pc_points
            out['gmap_pc_masks'] = gmap_pc_masks
            out['gmap_pc_fusion_masks'] = gmap_pc_fusion_masks
        return out

    def _history_variable(self, obs):
        batch_size = obs['pano_rgb'].shape[0]
        hist_rgb_fts = obs['pano_rgb'][:, 0, ...].cuda()
        hist_pano_rgb_fts = obs['pano_rgb'].cuda()
        hist_pano_ang_fts = obs['pano_angle_fts'].unsqueeze(0).expand(batch_size, -1, -1).cuda()

        return hist_rgb_fts, hist_pano_rgb_fts, hist_pano_ang_fts

    @staticmethod
    def _pause_envs(envs, batch, envs_to_pause):
        if len(envs_to_pause) > 0:
            state_index = list(range(envs.num_envs))
            for idx in reversed(envs_to_pause):
                state_index.pop(idx)
                envs.pause_at(idx)
            
            for k, v in batch.items():
                batch[k] = v[state_index]

        return envs, batch

    def train(self):
        self._set_config()
        if self.config.MODEL.task_type == 'rxr':
            self.gt_data = {}
            for role in self.config.TASK_CONFIG.DATASET.ROLES:
                with gzip.open(
                    self.config.TASK_CONFIG.TASK.NDTW.GT_PATH.format(
                        split=self.split, role=role
                    ), "rt") as f:
                    self.gt_data.update(json.load(f))

        observation_space, action_space = self._init_envs()
        start_iter = self._initialize_policy(
            self.config,
            self.config.IL.load_from_ckpt,
            observation_space=observation_space,
            action_space=action_space,
        )

        total_iter = self.config.IL.iters
        log_every  = self.config.IL.log_every
        writer     = TensorboardWriter(self.config.TENSORBOARD_DIR if self.local_rank < 1 else None)

        self.scaler = GradScaler()
        logger.info('Traning Starts... GOOD LUCK!')
        print("[SS-ETP] Training starts. GOOD LUCK!")
        if self.local_rank < 1:
            self._save_config_snapshot()

        for idx in range(start_iter, total_iter, log_every):
            interval = min(log_every, max(total_iter-idx, 0))
            cur_iter = idx + interval

            sample_ratio = self.config.IL.sample_ratio ** (idx // self.config.IL.decay_interval + 1)
            # sample_ratio = self.config.IL.sample_ratio ** (idx // self.config.IL.decay_interval)
            if self.local_rank < 1:
                logger.info(f"[SS-ETP] interval end={cur_iter}  sample_ratio={sample_ratio:.6g}  log_every={log_every}")
                print(f"[SS-ETP] iter {cur_iter}  sample_ratio={sample_ratio:.6g}")

            logs = self._train_interval(interval, self.config.IL.ml_weight, sample_ratio)

            if self.local_rank < 1:
                loss_str = f'iter {cur_iter}: '
                for k, v in logs.items():
                    if k == "fusion_lambda":
                        if len(v) == 0:
                            continue
                        # Same as PointCloud: last value in interval (λ after last step)
                        logs[k] = float(v[-1])
                    else:
                        logs[k] = np.mean(v)
                    if k == "fusion_lambda":
                        loss_str += f'{k}: {logs[k]:.6f}, '
                    else:
                        loss_str += f'{k}: {logs[k]:.3f}, '
                    if k == "fusion_lambda":
                        writer.add_scalar(f"train/{k}", logs[k], cur_iter)
                    else:
                        writer.add_scalar(f'loss/{k}', logs[k], cur_iter)
                # Learning rates (all param groups; PointCloud logs group 0 only)
                if self.optimizer is not None:
                    for gi, pg in enumerate(self.optimizer.param_groups):
                        writer.add_scalar(f"train/lr_group_{gi}", pg["lr"], cur_iter)
                    mean_lr = float(np.mean([pg["lr"] for pg in self.optimizer.param_groups]))
                    writer.add_scalar("train/lr_mean", mean_lr, cur_iter)
                    lr_parts = [f"g{i}={pg['lr']:.6g}" for i, pg in enumerate(self.optimizer.param_groups)]
                    logger.info(f"[SS-ETP] lr: {'  '.join(lr_parts)}")
                    print(f"[SS-ETP] lr: {'  '.join(lr_parts)}")
                logger.info(loss_str)
                print(loss_str)
                self.save_checkpoint(cur_iter)
                
                # If dynamic graph or node gating is enabled, save weight information every 200 iterations
                if (getattr(self.config.MODEL, 'use_dynamic_graph', False) or 
                    getattr(self.config.MODEL, 'use_node_gating', False)) and cur_iter % 200 == 0:
                    self.save_dynamic_graph_weights(cur_iter)
        
    def _train_interval(self, interval, ml_weight, sample_ratio):
        self.policy.train()
        if self.world_size > 1:
            self.policy.net.module.rgb_encoder.eval()
            self.policy.net.module.depth_encoder.eval()
        else:
            self.policy.net.rgb_encoder.eval()
            self.policy.net.depth_encoder.eval()
        self.waypoint_predictor.eval()

        if self.local_rank < 1:
            pbar = tqdm.trange(interval, leave=False, dynamic_ncols=True)
        else:
            pbar = range(interval)
        self.logs = defaultdict(list)

        for idx in pbar:
            self.optimizer.zero_grad()
            self.loss = 0.

            with autocast():
                self.rollout('train', ml_weight, sample_ratio)
            self.scaler.scale(self.loss).backward() # self.loss.backward()
            self.scaler.step(self.optimizer)        # self.optimizer.step()
            if getattr(self, "scheduler", None) is not None:
                self.scheduler.step()
            self.scaler.update()

            # fusion_lambda (PointCloud-style): after optimizer step, record current sigmoid(raw)
            net_module = self.policy.net.module if hasattr(self.policy.net, "module") else self.policy.net
            raw = getattr(net_module, "fusion_lambda_raw", None)
            if raw is not None:
                lambda_val = torch.sigmoid(raw).detach().float().item()
                self.logs["fusion_lambda"].append(lambda_val)

            # If dynamic graph is enabled, record weight values (for statistics)
            if self.local_rank < 1 and getattr(self.config.MODEL, 'use_dynamic_graph', False):
                self._record_dynamic_graph_weights()

            if self.local_rank < 1:
                pbar.set_postfix({'iter': f'{idx+1}/{interval}'})
            
        return deepcopy(self.logs)

    def _eval_episodes_allowed(self):
        """Match PointCloud: fast_eval subsample, else optional EPISODE_ID subset, else rank-sharded traj."""
        if getattr(self.config.EVAL, "fast_eval", False):
            return self.traj[::5]
        raw = getattr(self.config.EVAL, "EPISODE_ID", None)
        if raw:
            if isinstance(raw, (str, int)):
                return [str(raw)]
            return [str(x) for x in raw]
        return self.traj

    def _eval_num_episodes_to_collect(self) -> int:
        """
        Number of unique episodes we expect in stat_eps (keys = episode_id).

        sum(envs.number_of_episodes) counts each parallel worker's queue and can be
        num_envs * n when the same EPISODES_ALLOWED list is replicated — that breaks
        `while len(stat_eps) < eps_to_eval` because stat_eps has at most one entry per id.
        """
        raw_eid = getattr(self.config.EVAL, "EPISODE_ID", None)
        if raw_eid:
            return 1 if isinstance(raw_eid, (str, int)) else len(list(raw_eid))
        if getattr(self.config.EVAL, "fast_eval", False):
            return len(self.traj[::5])
        return int(sum(self.envs.number_of_episodes))

    @torch.no_grad()
    def _eval_checkpoint(
        self,
        checkpoint_path: str,
        writer: TensorboardWriter,
        checkpoint_index: int = 0,
    ):
        if self.local_rank < 1:
            logger.info(f"checkpoint_path: {checkpoint_path}")
            ev = self.config.EVAL
            spl = self.config.TASK_CONFIG.DATASET.SPLIT
            epn = ev.EPISODE_COUNT
            fev = getattr(ev, "fast_eval", False)
            eid = getattr(ev, "EPISODE_ID", None)
            if eid:
                n_eid = 1 if isinstance(eid, (str, int)) else len(list(eid))
                eid_info = f"  EPISODE_ID_count={n_eid}"
            else:
                eid_info = ""
            banner = (
                f"[SS-ETP EVAL] ckpt={checkpoint_path}  split={spl}  "
                f"EPISODE_COUNT={epn}  fast_eval={fev}  SAVE_RESULTS={ev.SAVE_RESULTS}{eid_info}"
            )
            logger.info(banner)
            print(banner)
        self.config.defrost()
        self.config.TASK_CONFIG.ENVIRONMENT.ITERATOR_OPTIONS.SHUFFLE = False
        self.config.TASK_CONFIG.ENVIRONMENT.ITERATOR_OPTIONS.MAX_SCENE_REPEAT_STEPS = -1
        self.config.IL.ckpt_to_load = checkpoint_path
        if self.config.VIDEO_OPTION:
            self.config.TASK_CONFIG.TASK.MEASUREMENTS.append("TOP_DOWN_MAP_VLNCE")
            self.config.TASK_CONFIG.TASK.MEASUREMENTS.append("DISTANCE_TO_GOAL")
            self.config.TASK_CONFIG.TASK.MEASUREMENTS.append("SUCCESS")
            self.config.TASK_CONFIG.TASK.MEASUREMENTS.append("SPL")
            os.makedirs(self.config.VIDEO_DIR, exist_ok=True)
            shift = 0.
            orient_dict = {
                'Back': [0, math.pi + shift, 0],            # Back
                'Down': [-math.pi / 2, 0 + shift, 0],       # Down
                'Front':[0, 0 + shift, 0],                  # Front
                'Right':[0, math.pi / 2 + shift, 0],        # Right
                'Left': [0, 3 / 2 * math.pi + shift, 0],    # Left
                'Up':   [math.pi / 2, 0 + shift, 0],        # Up
            }
            sensor_uuids = []
            H = 224
            for sensor_type in ["RGB"]:
                sensor = getattr(self.config.TASK_CONFIG.SIMULATOR, f"{sensor_type}_SENSOR")
                for camera_id, orient in orient_dict.items():
                    camera_template = f"{sensor_type}{camera_id}"
                    camera_config = deepcopy(sensor)
                    camera_config.WIDTH = H
                    camera_config.HEIGHT = H
                    camera_config.ORIENTATION = orient
                    camera_config.UUID = camera_template.lower()
                    camera_config.HFOV = 90
                    sensor_uuids.append(camera_config.UUID)
                    setattr(self.config.TASK_CONFIG.SIMULATOR, camera_template, camera_config)
                    self.config.TASK_CONFIG.SIMULATOR.AGENT_0.SENSORS.append(camera_template)
        self.config.freeze()

        if self.config.EVAL.SAVE_RESULTS:
            fname = os.path.join(
                self.config.RESULTS_DIR,
                f"stats_ckpt_{checkpoint_index}_{self.config.TASK_CONFIG.DATASET.SPLIT}.json",
            )
            if os.path.exists(fname) and not os.path.isfile(self.config.EVAL.CKPT_PATH_DIR):
                print("skipping -- evaluation exists.")
                return
        episodes_allowed = self._eval_episodes_allowed()
        if self.local_rank < 1:
            logger.info(
                f"[SS-ETP EVAL] episodes_allowed count={len(episodes_allowed)} "
                f"(fast_eval={getattr(self.config.EVAL, 'fast_eval', False)})"
            )
        self.envs = construct_envs(
            self.config, 
            get_env_class(self.config.ENV_NAME),
            episodes_allowed=episodes_allowed,
            auto_reset_done=False, # unseen: 11006 
        )
        dataset_length = sum(self.envs.number_of_episodes)
        print('local rank:', self.local_rank, '|', 'dataset length:', dataset_length)

        obs_transforms = get_active_obs_transforms(self.config)
        observation_space = apply_obs_transforms_obs_space(
            self.envs.observation_spaces[0], obs_transforms
        )
        self._initialize_policy(
            self.config,
            load_from_ckpt=True,
            observation_space=observation_space,
            action_space=self.envs.action_spaces[0],
            create_optimizer=False,  # No optimizer needed for evaluation
        )
        self.policy.eval()
        self.waypoint_predictor.eval()

        n_available = self._eval_num_episodes_to_collect()
        if self.config.EVAL.EPISODE_COUNT == -1:
            eps_to_eval = n_available
        else:
            eps_to_eval = min(int(self.config.EVAL.EPISODE_COUNT), n_available)
        if self.local_rank < 1:
            logger.info(
                f"[SS-ETP EVAL] eps_to_eval={eps_to_eval} "
                f"(unique episode ids; dataset_length_sum={dataset_length})"
            )
        self.stat_eps = {}
        # Always initialize loc_noise_history to record loc_noise values for each episode
        self.loc_noise_history = defaultdict(list)
        # Record start time of each episode for calculating episode duration
        self.episode_start_times = {}
        self.pbar = tqdm.tqdm(total=eps_to_eval) if self.config.use_pbar else None

        eval_rollout_t0 = time.perf_counter()
        while len(self.stat_eps) < eps_to_eval:
            self.rollout('eval')
        eval_rollout_sec = time.perf_counter() - eval_rollout_t0
        eval_rollout_min = eval_rollout_sec / 60.0
        eval_rollout_hour = eval_rollout_sec / 3600.0
        num_eval_eps = len(self.stat_eps)
        avg_eval_ep_sec = eval_rollout_sec / max(1, num_eval_eps)
        if self.local_rank < 1:
            logger.info(
                f"Eval rollout total duration: {eval_rollout_sec:.2f}s "
                f"({eval_rollout_min:.2f} min, {eval_rollout_hour:.3f} h)"
            )
            logger.info(
                f"Eval average wall time per episode: {avg_eval_ep_sec:.4f}s "
                "(rollout only, parallel envs; not pure GPU kernel time)"
            )
        self.envs.close()

        if self.world_size > 1:
            distr.barrier()
        aggregated_states = {}
        num_episodes = len(self.stat_eps)
        for stat_key in next(iter(self.stat_eps.values())).keys():
            aggregated_states[stat_key] = (
                sum(v[stat_key] for v in self.stat_eps.values()) / num_episodes
            )
        total = torch.tensor(num_episodes).cuda()
        if self.world_size > 1:
            distr.reduce(total,dst=0)
        total = total.item()

        if self.world_size > 1:
            logger.info(f"rank {self.local_rank}'s {num_episodes}-episode results: {aggregated_states}")
            for k,v in aggregated_states.items():
                v = torch.tensor(v*num_episodes).cuda()
                cat_v = gather_list_and_concat(v,self.world_size)
                v = (sum(cat_v)/total).item()
                aggregated_states[k] = v
        
        split = self.config.TASK_CONFIG.DATASET.SPLIT
        
        # Merge loc_noise_history into stat_eps
        for ep_id, metric in self.stat_eps.items():
            if ep_id in self.loc_noise_history:
                metric['loc_noise_history'] = self.loc_noise_history[ep_id]
            else:
                # If no record, set to empty list
                metric['loc_noise_history'] = []
        
        fname = os.path.join(
            self.config.RESULTS_DIR,
            f"stats_ep_ckpt_{checkpoint_index}_{split}_r{self.local_rank}_w{self.world_size}.json",
        )
        with open(fname, "w") as f:
            json.dump(self.stat_eps, f, indent=2)

        if self.local_rank < 1:
            if self.config.EVAL.SAVE_RESULTS:
                fname = os.path.join(
                    self.config.RESULTS_DIR,
                    f"stats_ckpt_{checkpoint_index}_{split}.json",
                )
                with open(fname, "w") as f:
                    json.dump(aggregated_states, f, indent=2)

            # loc_noise_history has been merged into stat_eps, no need to save separately
            logger.info(f"Episodes evaluated: {total}")
            checkpoint_num = checkpoint_index + 1
            for k, v in aggregated_states.items():
                logger.info(f"Average episode {k}: {v:.6f}")
                writer.add_scalar(f"eval_{k}/{split}", v, checkpoint_num)

    @torch.no_grad()
    def inference(self):
        checkpoint_path = self.config.INFERENCE.CKPT_PATH
        logger.info(f"checkpoint_path: {checkpoint_path}")
        if self.local_rank < 1:
            inf = self.config.INFERENCE
            banner = (
                f"[SS-ETP INFERENCE] ckpt={checkpoint_path}  split={inf.SPLIT}  "
                f"EPISODE_COUNT={inf.EPISODE_COUNT}  SAMPLE={inf.SAMPLE}"
            )
            logger.info(banner)
            print(banner)
        self.config.defrost()
        self.config.IL.ckpt_to_load = checkpoint_path
        self.config.TASK_CONFIG.DATASET.SPLIT = self.config.INFERENCE.SPLIT
        self.config.TASK_CONFIG.DATASET.ROLES = ["guide"]
        self.config.TASK_CONFIG.DATASET.LANGUAGES = self.config.INFERENCE.LANGUAGES
        self.config.TASK_CONFIG.ENVIRONMENT.ITERATOR_OPTIONS.SHUFFLE = False
        self.config.TASK_CONFIG.ENVIRONMENT.ITERATOR_OPTIONS.MAX_SCENE_REPEAT_STEPS = -1
        self.config.TASK_CONFIG.TASK.MEASUREMENTS = ['POSITION_INFER']
        self.config.TASK_CONFIG.TASK.SENSORS = [s for s in self.config.TASK_CONFIG.TASK.SENSORS if "INSTRUCTION" in s]
        self.config.SIMULATOR_GPU_IDS = [self.config.SIMULATOR_GPU_IDS[self.config.local_rank]]
        # if choosing image
        resize_config = self.config.RL.POLICY.OBS_TRANSFORMS.RESIZER_PER_SENSOR.SIZES
        crop_config = self.config.RL.POLICY.OBS_TRANSFORMS.CENTER_CROPPER_PER_SENSOR.SENSOR_CROPS
        task_config = self.config.TASK_CONFIG
        camera_orientations = get_camera_orientations12()
        for sensor_type in ["RGB", "DEPTH"]:
            resizer_size = dict(resize_config)[sensor_type.lower()]
            cropper_size = dict(crop_config)[sensor_type.lower()]
            sensor = getattr(task_config.SIMULATOR, f"{sensor_type}_SENSOR")
            for action, orient in camera_orientations.items():
                camera_template = f"{sensor_type}_{action}"
                camera_config = deepcopy(sensor)
                camera_config.ORIENTATION = camera_orientations[action]
                camera_config.UUID = camera_template.lower()
                setattr(task_config.SIMULATOR, camera_template, camera_config)
                task_config.SIMULATOR.AGENT_0.SENSORS.append(camera_template)
                resize_config.append((camera_template.lower(), resizer_size))
                crop_config.append((camera_template.lower(), cropper_size))
        self.config.RL.POLICY.OBS_TRANSFORMS.RESIZER_PER_SENSOR.SIZES = resize_config
        self.config.RL.POLICY.OBS_TRANSFORMS.CENTER_CROPPER_PER_SENSOR.SENSOR_CROPS = crop_config
        self.config.TASK_CONFIG = task_config
        self.config.SENSORS = task_config.SIMULATOR.AGENT_0.SENSORS
        self.config.freeze()

        torch.cuda.set_device(self.device)
        self.world_size = self.config.GPU_NUMBERS
        self.local_rank = self.config.local_rank
        if self.world_size > 1:
            distr.init_process_group(backend='nccl', init_method='env://')
            self.device = self.config.TORCH_GPU_IDS[self.local_rank]
            torch.cuda.set_device(self.device)
            self.config.defrost()
            self.config.TORCH_GPU_ID = self.config.TORCH_GPU_IDS[self.local_rank]
            self.config.freeze()
        self.traj = self.collect_infer_traj()

        self.envs = construct_envs(
            self.config, 
            get_env_class(self.config.ENV_NAME),
            episodes_allowed=self.traj,
            auto_reset_done=False,
        )

        obs_transforms = get_active_obs_transforms(self.config)
        observation_space = apply_obs_transforms_obs_space(
            self.envs.observation_spaces[0], obs_transforms
        )
        self._initialize_policy(
            self.config,
            load_from_ckpt=True,
            observation_space=observation_space,
            action_space=self.envs.action_spaces[0],
            create_optimizer=False,  # No optimizer needed for inference
        )
        self.policy.eval()
        self.waypoint_predictor.eval()

        if self.config.INFERENCE.EPISODE_COUNT == -1:
            eps_to_infer = sum(self.envs.number_of_episodes)
        else:
            eps_to_infer = min(self.config.INFERENCE.EPISODE_COUNT, sum(self.envs.number_of_episodes))
        self.path_eps = defaultdict(list)
        self.inst_ids: Dict[str, int] = {}   # transfer submit format
        self.pbar = tqdm.tqdm(total=eps_to_infer)
        
        # If dynamic or random loc_noise is enabled, initialize recording
        use_dynamic_loc_noise = getattr(self.config.IL, 'use_dynamic_loc_noise', False)
        use_random_loc_noise = getattr(self.config.IL, 'use_random_loc_noise', False)
        # Always initialize loc_noise_history to record loc_noise values for each episode
        self.loc_noise_history = defaultdict(list)

        while len(self.path_eps) < eps_to_infer:
            self.rollout('infer')
        self.envs.close()

        if self.world_size > 1:
            aggregated_path_eps = [None for _ in range(self.world_size)]
            distr.all_gather_object(aggregated_path_eps, self.path_eps)
            tmp_eps_dict = {}
            for x in aggregated_path_eps:
                tmp_eps_dict.update(x)
            self.path_eps = tmp_eps_dict

            aggregated_inst_ids = [None for _ in range(self.world_size)]
            distr.all_gather_object(aggregated_inst_ids, self.inst_ids)
            tmp_inst_dict = {}
            for x in aggregated_inst_ids:
                tmp_inst_dict.update(x)
            self.inst_ids = tmp_inst_dict


        if self.config.MODEL.task_type == "r2r":
            with open(self.config.INFERENCE.PREDICTIONS_FILE, "w") as f:
                json.dump(self.path_eps, f, indent=2)
            logger.info(f"Predictions saved to: {self.config.INFERENCE.PREDICTIONS_FILE}")
        else:  # use 'rxr' format for rxr-habitat leaderboard
            preds = []
            for k,v in self.path_eps.items():
                # save only positions that changed
                path = [v[0]["position"]]
                for p in v[1:]:
                    if p["position"] != path[-1]: path.append(p["position"])
                preds.append({"instruction_id": self.inst_ids[k], "path": path})
            preds.sort(key=lambda x: x["instruction_id"])
            with jsonlines.open(self.config.INFERENCE.PREDICTIONS_FILE, mode="w") as writer:
                writer.write_all(preds)
            logger.info(f"Predictions saved to: {self.config.INFERENCE.PREDICTIONS_FILE}")
        
        # loc_noise_history is not merged into path_eps in inference mode (because path_eps has different structure)
        # If needed, can be saved separately, but according to user requirements, mainly focus on eval mode stats_ep file

    def get_pos_ori(self):
        pos_ori = self.envs.call(['get_pos_ori']*self.envs.num_envs)
        pos = [x[0] for x in pos_ori]
        ori = [x[1] for x in pos_ori]
        return pos, ori

    def rollout(self, mode, ml_weight=None, sample_ratio=None):
        if mode == 'train':
            feedback = 'sample'
        elif mode == 'eval' or mode == 'infer':
            feedback = 'argmax'
        else:
            raise NotImplementedError

        self.envs.resume_all()
        observations = self.envs.reset()
        instr_max_len = self.config.IL.max_text_len # r2r 80, rxr 200
        instr_pad_id = 1 if self.config.MODEL.task_type == 'rxr' else 0
        observations = extract_instruction_tokens(observations, self.config.TASK_CONFIG.TASK.INSTRUCTION_SENSOR_UUID,
                                                  max_length=instr_max_len, pad_id=instr_pad_id)
        batch = batch_obs(observations, self.device)
        batch = apply_obs_transforms_batch(batch, self.obs_transforms)
        
        if mode == 'eval':
            curr_eps = self.envs.current_episodes()
            # Record start time of new episode
            for i, ep in enumerate(curr_eps):
                ep_id = ep.episode_id
                if ep_id not in self.episode_start_times:
                    self.episode_start_times[ep_id] = time.time()
            
            env_to_pause = [i for i, ep in enumerate(curr_eps) 
                            if ep.episode_id in self.stat_eps]    
            self.envs, batch = self._pause_envs(self.envs, batch, env_to_pause)
            if self.envs.num_envs == 0: return
        if mode == 'infer':
            env_to_pause = [i for i, ep in enumerate(self.envs.current_episodes()) 
                            if ep.episode_id in self.path_eps]    
            self.envs, batch = self._pause_envs(self.envs, batch, env_to_pause)
            if self.envs.num_envs == 0: return
            curr_eps = self.envs.current_episodes()
            for i in range(self.envs.num_envs):
                if self.config.MODEL.task_type == 'rxr':
                    ep_id = curr_eps[i].episode_id
                    k = curr_eps[i].instruction.instruction_id
                    self.inst_ids[ep_id] = int(k)

        # encode instructions
        all_txt_ids = batch['instruction']
        all_txt_masks = (all_txt_ids != instr_pad_id)
        all_txt_embeds = self.policy.net(
            mode='language',
            txt_ids=all_txt_ids,
            txt_masks=all_txt_masks,
        )

        loss = 0.
        total_actions = 0.
        not_done_index = list(range(self.envs.num_envs))

        have_real_pos = (mode == 'train' or self.config.VIDEO_OPTION)
        ghost_aug = self.config.IL.ghost_aug if mode == 'train' else 0
        self.gmaps = [GraphMap(have_real_pos, 
                               self.config.IL.loc_noise, 
                               self.config.MODEL.merge_ghost,
                               ghost_aug) for _ in range(self.envs.num_envs)]
        prev_vp = [None] * self.envs.num_envs

        for stepk in range(self.max_len):
            total_actions += self.envs.num_envs
            txt_masks = all_txt_masks[not_done_index]
            txt_embeds = all_txt_embeds[not_done_index]
            
            # cand waypoint prediction
            wp_outputs = self.policy.net(
                mode = "waypoint",
                waypoint_predictor = self.waypoint_predictor,
                observations = batch,
                in_train = (mode == 'train' and self.config.IL.waypoint_aug),
            )

            # pano encoder
            vp_inputs = self._vp_feature_variable(wp_outputs)
            vp_inputs.update({
                'mode': 'panorama',
            })
            pano_embeds, pano_masks = self.policy.net(**vp_inputs)
            avg_pano_embeds = torch.sum(pano_embeds * pano_masks.unsqueeze(2), 1) / \
                              torch.sum(pano_masks, 1, keepdim=True)

            # get vp_id, vp_pos of cur_node and cand_ndoe
            cur_pos, cur_ori = self.get_pos_ori()
            cur_vp, cand_vp, cand_pos = [], [], []
            for i in range(self.envs.num_envs):
                cur_vp_i, cand_vp_i, cand_pos_i = self.gmaps[i].identify_node(
                    cur_pos[i], cur_ori[i], wp_outputs['cand_angles'][i], wp_outputs['cand_distances'][i]
                )
                cur_vp.append(cur_vp_i)
                cand_vp.append(cand_vp_i)
                cand_pos.append(cand_pos_i)
            
            if mode == 'train' or self.config.VIDEO_OPTION:
                cand_real_pos = []
                for i in range(self.envs.num_envs):
                    cand_real_pos_i = [
                        self.envs.call_at(i, "get_cand_real_pos", {"angle": ang, "forward": dis})
                        for ang, dis in zip(wp_outputs['cand_angles'][i], wp_outputs['cand_distances'][i])
                    ]
                    cand_real_pos.append(cand_real_pos_i)
            else:
                cand_real_pos = [None] * self.envs.num_envs

            pc_cfg = getattr(self.config.MODEL, "POINT_CLOUD", None)
            do_pc_sampling = (
                pc_cfg is not None and bool(getattr(pc_cfg, "enable_depth_to_pointcloud", False))
            )
            if do_pc_sampling:
                depth_hfov = float(self.config.TASK_CONFIG.SIMULATOR.DEPTH_SENSOR.HFOV)
                depth_min = float(getattr(self.config.TASK_CONFIG.SIMULATOR.DEPTH_SENSOR, "MIN_DEPTH", 0.0))
                depth_max = float(getattr(self.config.TASK_CONFIG.SIMULATOR.DEPTH_SENSOR, "MAX_DEPTH", 10.0))
                normalize_depth = bool(getattr(self.config.TASK_CONFIG.SIMULATOR.DEPTH_SENSOR, "NORMALIZE_DEPTH", True))
                depth_feats_maps = build_depth_feats_maps(batch, num_imgs=12)
                num_points_pc = int(pc_cfg.num_points)
                max_depth_m = float(pc_cfg.max_depth_m)
                enable_spatial_crop = bool(pc_cfg.enable_spatial_crop)
                fps_seed_base = int(pc_cfg.fps_seed) if int(pc_cfg.fps_seed) >= 0 else int(self.config.TASK_CONFIG.SEED)
            else:
                depth_feats_maps = None
                depth_hfov = 0.0
                depth_min = 0.0
                depth_max = 0.0
                normalize_depth = False
                num_points_pc = 0
                max_depth_m = 0.0
                enable_spatial_crop = False
                fps_seed_base = 0

            # Calculate loc_noise (priority: dynamic > random > fixed)
            use_dynamic_loc_noise = getattr(self.config.IL, 'use_dynamic_loc_noise', False)
            use_random_loc_noise = getattr(self.config.IL, 'use_random_loc_noise', False)
            loc_noise_values = [None] * self.envs.num_envs
            
            if use_dynamic_loc_noise:
                # Dynamic loc_noise: calculated based on candidate waypoint angle divergence
                loc_noise_min = getattr(self.config.IL, 'dynamic_loc_noise_min', 0.40)
                loc_noise_max = getattr(self.config.IL, 'dynamic_loc_noise_max', 0.60)
                loc_noise_base = getattr(self.config.IL, 'loc_noise', 0.5)  # Base value, used when insufficient candidate points
                # Read formula coefficients from config
                alpha = getattr(self.config.IL, 'dynamic_loc_noise_alpha', 0.65)
                beta = getattr(self.config.IL, 'dynamic_loc_noise_beta', 0.25)
                mapping_type = getattr(self.config.IL, 'dynamic_loc_noise_mapping', 'linear')
                sigmoid_k = getattr(self.config.IL, 'dynamic_loc_noise_sigmoid_k', 12.0)
                exponential_k = getattr(self.config.IL, 'dynamic_loc_noise_exponential_k', 4.0)
                
                def compute_loc_noise_from_std(std_val, mapping='linear'):
                    """
                    Calculate loc_noise from std value, supports three mapping methods:
                    - linear: loc_noise = alpha - beta * std
                    - sigmoid: use sigmoid function mapping, refer to linear_compare.py
                    - exponential: use exponential function mapping, refer to linear_compare.py
                    
                    All mappings use alpha and beta parameters to determine reference points:
                    - When std=0, loc_noise should be close to loc_noise_max (or alpha)
                    - When std increases, loc_noise should decrease
                    - Use alpha and beta to determine the reference range of std
                    """
                    # Determine reference range of std: when std=std_ref, linear mapping's loc_noise reaches minimum
                    # i.e.: alpha - beta * std_ref = loc_noise_min
                    # Therefore: std_ref = (alpha - loc_noise_min) / beta
                    std_ref = (alpha - loc_noise_min) / beta if beta > 0 else 1.0
                    
                    if mapping == 'linear':
                        # Linear mapping: loc_noise = alpha - beta * std
                        loc_noise = alpha - beta * std_val
                    elif mapping == 'sigmoid':
                        # Sigmoid mapping: refer to implementation in linear_compare.py
                        # Use std_ref as reference point, similar to s_max in linear_compare.py
                        # When std=0, loc_noise=loc_noise_max (similar to y_start=0.5)
                        # When std=std_ref, loc_noise=loc_noise_min (similar to y_end=0.25)
                        if std_val <= 0:
                            return loc_noise_max
                        if std_val >= std_ref:
                            return loc_noise_min
                        
                        # Normalize std to [0, 1] range
                        x_norm = std_val / std_ref  # 0 -> 1
                        # Map to sigmoid's effective interval
                        x_mapped = sigmoid_k * (x_norm - 0.5)  # -k/2 -> k/2
                        
                        sigmoid_val = 1 / (1 + np.exp(-x_mapped))
                        
                        # Calibrate boundaries (because sigmoid(-k/2) != 0, sigmoid(k/2) != 1)
                        s_min = 1 / (1 + np.exp(-sigmoid_k * (-0.5)))
                        s_max_val = 1 / (1 + np.exp(-sigmoid_k * (0.5)))
                        
                        ratio = (sigmoid_val - s_min) / (s_max_val - s_min) if (s_max_val - s_min) > 0 else 0
                        
                        # Map to [loc_noise_min, loc_noise_max] range
                        total_drop = loc_noise_max - loc_noise_min
                        loc_noise = loc_noise_max - total_drop * ratio
                    elif mapping == 'exponential':
                        # Exponential mapping: refer to implementation in linear_compare.py
                        if std_val <= 0:
                            return loc_noise_max
                        if std_val >= std_ref:
                            return loc_noise_min
                        
                        # Normalize std to [0, 1] range
                        x_norm = std_val / std_ref
                        
                        # (e^kx - 1) / (e^k - 1)
                        exp_ratio = (np.exp(exponential_k * x_norm) - 1) / (np.exp(exponential_k) - 1)
                        
                        # Map to [loc_noise_min, loc_noise_max] range
                        total_drop = loc_noise_max - loc_noise_min
                        loc_noise = loc_noise_max - total_drop * exp_ratio
                    else:
                        # Default to linear mapping
                        loc_noise = alpha - beta * std_val
                    
                    # Clip to [min, max] range (although theoretically already in range, for safety)
                    return np.clip(loc_noise, loc_noise_min, loc_noise_max)
                
                for i in range(self.envs.num_envs):
                    cand_angles_i = wp_outputs['cand_angles'][i]
                    if len(cand_angles_i) > 1:
                        # Calculate angle standard deviation (in radians)
                        std = float(np.std(cand_angles_i))
                        # Calculate loc_noise based on mapping type
                        dynamic_loc_noise = compute_loc_noise_from_std(std, mapping=mapping_type)
                        loc_noise_values[i] = float(dynamic_loc_noise)
                    else:
                        # If only one or no candidate points, use base value
                        loc_noise_values[i] = loc_noise_base
                
                # Record std and loc_noise in eval/infer mode
                if mode in ['eval', 'infer']:
                    curr_eps = self.envs.current_episodes()
                    for i in range(self.envs.num_envs):
                        ep_id = curr_eps[i].episode_id
                        cand_angles_i = wp_outputs['cand_angles'][i]
                        std = float(np.std(cand_angles_i)) if len(cand_angles_i) > 1 else 0.0
                        self.loc_noise_history[ep_id].append({
                            'step': stepk,
                            'std': std,
                            'loc_noise': loc_noise_values[i],
                            'type': 'dynamic',
                            'mapping': mapping_type
                        })
            elif use_random_loc_noise:
                # Random loc_noise: random sampling within specified range
                random_loc_noise_min = getattr(self.config.IL, 'random_loc_noise_min', 0.40)
                random_loc_noise_max = getattr(self.config.IL, 'random_loc_noise_max', 0.60)
                
                for i in range(self.envs.num_envs):
                    # Independent random sampling for each environment
                    random_loc_noise = random.uniform(random_loc_noise_min, random_loc_noise_max)
                    loc_noise_values[i] = float(random_loc_noise)
                
                # Record random loc_noise in eval/infer mode
                if mode in ['eval', 'infer']:
                    curr_eps = self.envs.current_episodes()
                    for i in range(self.envs.num_envs):
                        ep_id = curr_eps[i].episode_id
                        self.loc_noise_history[ep_id].append({
                            'step': stepk,
                            'loc_noise': loc_noise_values[i],
                            'type': 'random'
                        })
            else:
                # If both are disabled, use fixed loc_noise value, also need to record
                fixed_loc_noise = getattr(self.config.IL, 'loc_noise', 0.5)
                if mode in ['eval', 'infer']:
                    curr_eps = self.envs.current_episodes()
                    for i in range(self.envs.num_envs):
                        ep_id = curr_eps[i].episode_id
                        self.loc_noise_history[ep_id].append({
                            'step': stepk,
                            'loc_noise': fixed_loc_noise,
                            'type': 'fixed'
                        })
            # If both are disabled, loc_noise_values remains None, will use fixed loc_noise value in GraphMap

            for i in range(self.envs.num_envs):
                cur_embeds = avg_pano_embeds[i]
                cand_embeds = pano_embeds[i][vp_inputs['nav_types'][i]==1]
                cand_pc_points = None
                if do_pc_sampling:
                    cand_pc_points = []
                    cand_img_idxes_i = wp_outputs["cand_img_idxes"][i]
                    for k, img_idx in enumerate(cand_img_idxes_i):
                        depth_map = depth_feats_maps[i, int(img_idx), :, :, 0]
                        sampled_points = sample_pointcloud_from_depth(
                            depth_map,
                            hfov_deg=depth_hfov,
                            normalize_depth=normalize_depth,
                            min_depth=depth_min,
                            max_depth=depth_max,
                            enable_spatial_crop=enable_spatial_crop,
                            max_depth_m=max_depth_m,
                            num_points=num_points_pc,
                            seed=int(fps_seed_base + stepk * 1000 + i * 10 + k),
                        )
                        cand_pc_points.append(sampled_points)
                # If dynamic or random loc_noise is enabled, pass calculated value; otherwise pass None to use default value
                loc_noise_to_use = loc_noise_values[i] if (use_dynamic_loc_noise or use_random_loc_noise) else None
                self.gmaps[i].update_graph(prev_vp[i], stepk+1,
                                           cur_vp[i], cur_pos[i], cur_embeds,
                                           cand_vp[i], cand_pos[i], cand_embeds,
                                           cand_real_pos[i], cand_pc_points=cand_pc_points,
                                           loc_noise=loc_noise_to_use)

            nav_inputs = self._nav_gmap_variable(cur_vp, cur_pos, cur_ori)
            nav_inputs.update({
                'mode': 'navigation',
                'txt_embeds': txt_embeds,
                'txt_masks': txt_masks,
            })
            no_vp_left = nav_inputs.pop('no_vp_left')
            nav_outs = self.policy.net(**nav_inputs)
            nav_logits = nav_outs['global_logits']
            nav_probs = F.softmax(nav_logits, 1)
            for i, gmap in enumerate(self.gmaps):
                gmap.node_stop_scores[cur_vp[i]] = nav_probs[i, 0].data.item()

            # random sample demo
            # logits = torch.randn(nav_inputs['gmap_masks'].shape).cuda()
            # logits.masked_fill_(~nav_inputs['gmap_masks'], -float('inf'))
            # logits.masked_fill_(nav_inputs['gmap_visited_masks'], -float('inf'))

            if mode == 'train' or self.config.VIDEO_OPTION:
                teacher_actions = self._teacher_action_new(nav_inputs['gmap_vp_ids'], no_vp_left)
            if mode == 'train':
                loss += F.cross_entropy(nav_logits, teacher_actions, reduction='sum', ignore_index=-100)

            # determine action
            if feedback == 'sample':
                c = torch.distributions.Categorical(nav_probs)
                a_t = c.sample().detach()
                a_t = torch.where(torch.rand_like(a_t, dtype=torch.float)<=sample_ratio, teacher_actions, a_t)
            elif feedback == 'argmax':
                a_t = nav_logits.argmax(dim=-1)
            else:
                raise NotImplementedError
            cpu_a_t = a_t.cpu().numpy()

            # make equiv action
            env_actions = []
            use_tryout = (self.config.IL.tryout and not self.config.TASK_CONFIG.SIMULATOR.HABITAT_SIM_V0.ALLOW_SLIDING)
            for i, gmap in enumerate(self.gmaps):
                if cpu_a_t[i] == 0 or stepk == self.max_len - 1 or no_vp_left[i]:
                    # stop at node with max stop_prob
                    vp_stop_scores = [(vp, stop_score) for vp, stop_score in gmap.node_stop_scores.items()]
                    stop_scores = [s[1] for s in vp_stop_scores]
                    stop_vp = vp_stop_scores[np.argmax(stop_scores)][0]
                    stop_pos = gmap.node_pos[stop_vp]
                    if self.config.IL.back_algo == 'control':
                        back_path = [(vp, gmap.node_pos[vp]) for vp in gmap.shortest_path[cur_vp[i]][stop_vp]]
                        back_path = back_path[1:]
                    else:
                        back_path = None
                    vis_info = {
                            'nodes': list(gmap.node_pos.values()),
                            'ghosts': list(gmap.ghost_aug_pos.values()),
                            'predict_ghost': stop_pos,
                    }
                    env_actions.append(
                        {
                            'action': {
                                'act': 0,
                                'cur_vp': cur_vp[i],
                                'stop_vp': stop_vp, 'stop_pos': stop_pos,
                                'back_path': back_path,
                                'tryout': use_tryout,
                            },
                            'vis_info': vis_info,
                        }
                    )
                else:
                    ghost_vp = nav_inputs['gmap_vp_ids'][i][cpu_a_t[i]]
                    ghost_pos = gmap.ghost_aug_pos[ghost_vp]
                    _, front_vp = gmap.front_to_ghost_dist(ghost_vp)
                    front_pos = gmap.node_pos[front_vp]
                    if self.config.VIDEO_OPTION:
                        teacher_action_cpu = teacher_actions[i].cpu().item()
                        if teacher_action_cpu in [0, -100]:
                            teacher_ghost = None
                        else:
                            teacher_ghost = gmap.ghost_aug_pos[nav_inputs['gmap_vp_ids'][i][teacher_action_cpu]]
                        vis_info = {
                            'nodes': list(gmap.node_pos.values()),
                            'ghosts': list(gmap.ghost_aug_pos.values()),
                            'predict_ghost': ghost_pos,
                            'teacher_ghost': teacher_ghost,
                        }
                    else:
                        vis_info = None
                    # teleport to front, then forward to ghost
                    if self.config.IL.back_algo == 'control':
                        back_path = [(vp, gmap.node_pos[vp]) for vp in gmap.shortest_path[cur_vp[i]][front_vp]]
                        back_path = back_path[1:]
                    else:
                        back_path = None
                    env_actions.append(
                        {
                            'action': {
                                'act': 4,
                                'cur_vp': cur_vp[i],
                                'front_vp': front_vp, 'front_pos': front_pos,
                                'ghost_vp': ghost_vp, 'ghost_pos': ghost_pos,
                                'back_path': back_path,
                                'tryout': use_tryout,
                            },
                            'vis_info': vis_info,
                        }
                    )
                    prev_vp[i] = front_vp
                    if self.config.MODEL.consume_ghost:
                        gmap.delete_ghost(ghost_vp)

            outputs = self.envs.step(env_actions)
            observations, _, dones, infos = [list(x) for x in zip(*outputs)]

            # calculate metric
            if mode == 'eval':
                curr_eps = self.envs.current_episodes()
                for i in range(self.envs.num_envs):
                    if not dones[i]:
                        continue
                    info = infos[i]
                    ep_id = curr_eps[i].episode_id
                    gt_path = np.array(self.gt_data[str(ep_id)]['locations']).astype(np.float)
                    pred_path = np.array(info['position']['position'])
                    distances = np.array(info['position']['distance'])
                    metric = {}
                    metric['steps_taken'] = info['steps_taken']
                    metric['distance_to_goal'] = distances[-1]
                    metric['success'] = 1. if distances[-1] <= 3. else 0.
                    metric['oracle_success'] = 1. if (distances <= 3.).any() else 0.
                    metric['path_length'] = float(np.linalg.norm(pred_path[1:] - pred_path[:-1],axis=1).sum())
                    metric['collisions'] = info['collisions']['count'] / len(pred_path)
                    gt_length = distances[0]
                    metric['spl'] = metric['success'] * gt_length / max(gt_length, metric['path_length'])
                    dtw_distance = fastdtw(pred_path, gt_path, dist=NDTW.euclidean_distance)[0]
                    metric['ndtw'] = np.exp(-dtw_distance / (len(gt_path) * 3.))
                    metric['sdtw'] = metric['ndtw'] * metric['success']
                    metric['ghost_cnt'] = self.gmaps[i].ghost_cnt
                    # Calculate episode duration (in seconds)
                    if ep_id in self.episode_start_times:
                        episode_duration = time.time() - self.episode_start_times[ep_id]
                        metric['episode_time'] = episode_duration
                        # Clean up start time record for completed episode
                        del self.episode_start_times[ep_id]
                    else:
                        # If start time is not recorded, set to 0 (should not happen theoretically)
                        metric['episode_time'] = 0.0
                    self.stat_eps[ep_id] = metric
                    self.pbar.update()

            # record path
            if mode == 'infer':
                curr_eps = self.envs.current_episodes()
                for i in range(self.envs.num_envs):
                    if not dones[i]:
                        continue
                    info = infos[i]
                    ep_id = curr_eps[i].episode_id
                    self.path_eps[ep_id] = [
                        {
                            'position': info['position_infer']['position'][0],
                            'heading': info['position_infer']['heading'][0],
                            'stop': False
                        }
                    ]
                    for p, h in zip(info['position_infer']['position'][1:], info['position_infer']['heading'][1:]):
                        if p != self.path_eps[ep_id][-1]['position']:
                            self.path_eps[ep_id].append({
                                'position': p,
                                'heading': h,
                                'stop': False
                            })
                    self.path_eps[ep_id] = self.path_eps[ep_id][:500]
                    self.path_eps[ep_id][-1]['stop'] = True
                    self.pbar.update()

            # pause env
            if sum(dones) > 0:
                for i in reversed(list(range(self.envs.num_envs))):
                    if dones[i]:
                        not_done_index.pop(i)
                        self.envs.pause_at(i)
                        observations.pop(i)
                        # graph stop
                        self.gmaps.pop(i)
                        prev_vp.pop(i)

            if self.envs.num_envs == 0:
                break

            # obs for next step
            observations = extract_instruction_tokens(observations,self.config.TASK_CONFIG.TASK.INSTRUCTION_SENSOR_UUID)
            batch = batch_obs(observations, self.device)
            batch = apply_obs_transforms_batch(batch, self.obs_transforms)

        if mode == 'train':
            loss = ml_weight * loss / total_actions
            self.loss += loss
            self.logs['IL_loss'].append(loss.item())