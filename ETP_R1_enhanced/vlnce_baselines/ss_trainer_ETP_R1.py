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
from vlnce_baselines.utils import reduce_loss

from .utils import get_camera_orientations12
from .utils import (
    length2mask, dir_angle_feature_with_ele,
)
from vlnce_baselines.models.pointcloud_sampling import (
    build_depth_feats_maps,
    sample_pointcloud_from_depth,
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
import cv2
from collections import OrderedDict

@baseline_registry.register_trainer(name="SS-ETP-R1")
class RLTrainer(BaseVLNCETrainer):
    def __init__(self, config=None):
        super().__init__(config)
        self.max_len = int(config.IL.max_traj_len) #  * 0.97 transfered gt path got 0.96 spl
        self.illegal_episodes_count = 0
        self._module_status_logged_once = False

    def _make_dirs(self):
        if self.config.local_rank == 0:
            self._make_ckpt_dir()
            # os.makedirs(self.lmdb_features_dir, exist_ok=True)
            if self.config.EVAL.SAVE_RESULTS:
                self._make_results_dir()

    def save_checkpoint(self, iteration: int):
        if self.config.ONLY_LAST_SAVEALL and (not iteration == self.config.IL.iters):
            torch.save(
                        obj={
                            "state_dict": self.policy.state_dict(),
                            "config": self.config,
                            "iteration": iteration
                        },
                        f=os.path.join(self.config.CHECKPOINT_FOLDER, f"ckpt.iter{iteration}.pth"),
                    )
        else:
            torch.save(
                obj={
                    "state_dict": self.policy.state_dict(),
                    "config": self.config,
                    "optim_state": self.optimizer.state_dict(),
                    "scheduler_state": self.scheduler.state_dict(),
                    "iteration": iteration,
                },
                f=os.path.join(self.config.CHECKPOINT_FOLDER, f"ckpt.iter{iteration}.pth"),
            )

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
        print(f"init camera information: resize_config:{resize_config}, crop_config:{crop_config}, new_camera_heading:{camera_orientations}")
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

    def _initialize_policy(
        self,
        config: Config,
        load_from_ckpt: bool,
        observation_space: Space,
        action_space: Space,
        setup_optimizer: bool = True,
    ):
        start_iter = 0
        policy = baseline_registry.get_policy(self.config.MODEL.policy_name)
        self.policy = policy.from_config(
            config=config,
            observation_space=observation_space,
            action_space=action_space,
        )
        logger.info(f"-------------------Load pretrain weight: {config.MODEL.pretrained_path}-------------------")
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

        # NOTE:
        # - For requeue/resume, we must build optimizer/scheduler BEFORE loading ckpt so we can restore states.
        # - For normal fine-tuning, we build optimizer/scheduler AFTER ckpt loading and stage-based freezing,
        #   so only trainable params are included.
        self.optimizer = None
        self.scheduler = None
        if bool(config.IL.is_requeue) and setup_optimizer:
            param_optimizer = list(self.policy.named_parameters())
            pointnet_lr = float(getattr(self.config.IL, "pointnet_lr", -1.0))
            if pointnet_lr <= 0:
                pointnet_lr = float(self.config.IL.lr)

            pointnet_name_keywords_rq = ("pointnet_encoder", "pc_feat_proj", "fusion_lambda")
            base_param_optimizer = [
                (n, p) for n, p in param_optimizer
                if not any(k in n for k in pointnet_name_keywords_rq)
            ]
            pointnet_param_optimizer = [
                (n, p) for n, p in param_optimizer
                if any(k in n for k in pointnet_name_keywords_rq)
            ]

            no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
            optimizer_grouped_parameters = []

            base_decay_params = [
                p for n, p in base_param_optimizer
                if not any(nd in n for nd in no_decay)
            ]
            base_no_decay_params = [
                p for n, p in base_param_optimizer
                if any(nd in n for nd in no_decay)
            ]
            if len(base_decay_params) > 0:
                optimizer_grouped_parameters.append({
                    'params': base_decay_params,
                    'weight_decay': 0.01,
                    'lr': float(self.config.IL.lr),
                })
            if len(base_no_decay_params) > 0:
                optimizer_grouped_parameters.append({
                    'params': base_no_decay_params,
                    'weight_decay': 0.0,
                    'lr': float(self.config.IL.lr),
                })

            pointnet_decay_params = [
                p for n, p in pointnet_param_optimizer
                if not any(nd in n for nd in no_decay)
            ]
            pointnet_no_decay_params = [
                p for n, p in pointnet_param_optimizer
                if any(nd in n for nd in no_decay)
            ]
            if len(pointnet_decay_params) > 0:
                optimizer_grouped_parameters.append({
                    'params': pointnet_decay_params,
                    'weight_decay': 0.01,
                    'lr': pointnet_lr,
                })
            if len(pointnet_no_decay_params) > 0:
                optimizer_grouped_parameters.append({
                    'params': pointnet_no_decay_params,
                    'weight_decay': 0.0,
                    'lr': pointnet_lr,
                })

            self.optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=self.config.IL.lr)
            num_warmup_steps = self.config.IL.warmup_iters
            num_training_steps = self.config.IL.iters
            min_lr_ratio = self.config.IL.min_lr_ratio

            def lr_lambda(current_step: int):
                if current_step < num_warmup_steps:
                    return float(current_step) / float(max(1, num_warmup_steps))
                progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
                progress = min(1.0, progress)
                cosine_decay = 0.5 * (1.0 + math.cos(math.pi * progress))
                decayed_lr_multiplier = min_lr_ratio + (1.0 - min_lr_ratio) * cosine_decay
                return decayed_lr_multiplier

            self.scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda)
            if self.local_rank < 1:
                logger.info(
                    f"SFT optimizer LR setup (requeue): base_lr={float(self.config.IL.lr):.6g}, "
                    f"pointnet_lr={pointnet_lr:.6g}, "
                    f"pointnet_param_tensors={len(pointnet_param_optimizer)}"
                )

        if load_from_ckpt:
            if config.IL.is_requeue:
                import glob
                search_pattern = os.path.join(config.CHECKPOINT_FOLDER, "*.pth")
                ckpt_list = glob.glob(search_pattern)
                ckpt_list.sort(key=os.path.getmtime)
                ckpt_path = ckpt_list[-1]
            else:
                ckpt_path = config.IL.ckpt_to_load
            ckpt_dict = self.load_checkpoint(ckpt_path, map_location="cpu")
            if config.IL.is_requeue:
                start_iter = ckpt_dict["iteration"]
            else:
                start_iter = 0

            if 'module' in list(ckpt_dict['state_dict'].keys())[0] and self.config.GPU_NUMBERS == 1:
                self.policy.net = torch.nn.DataParallel(self.policy.net.to(self.device),
                    device_ids=[self.device], output_device=self.device)
                incompatible_keys = self.policy.load_state_dict(ckpt_dict["state_dict"], strict=False)
                self.policy.net = self.policy.net.module
                self.waypoint_predictor = torch.nn.DataParallel(self.waypoint_predictor.to(self.device),
                    device_ids=[self.device], output_device=self.device)
            elif 'module' not in list(ckpt_dict['state_dict'].keys())[0] and self.config.GPU_NUMBERS > 1:
                new_state_dict = OrderedDict()
                for k, v in ckpt_dict['state_dict'].items():
                    if k.startswith("net."):
                        name = k.replace("net.", "net.module.", 1)
                        new_state_dict[name] = v
                    else:
                        new_state_dict[k] = v
                incompatible_keys = self.policy.load_state_dict(new_state_dict, strict=False)
            else:
                incompatible_keys = self.policy.load_state_dict(ckpt_dict["state_dict"], strict=False)
            
            if self.local_rank < 1:
                print("\n" + "="*25 + " Weight loading mismatch report " + "="*25)
                if incompatible_keys.missing_keys:
                    print("The following network layers exist in the model but are missing in the weight file (initial values will be used):")
                    for key in sorted(incompatible_keys.missing_keys):
                        print(f"  - {key}")
                else:
                    print("All network layers present in the model were found in the weight file.")
                if incompatible_keys.unexpected_keys:
                    print("\nThe following network layers exist in the weight file but are missing in the model (will be ignored):")
                    for key in sorted(incompatible_keys.unexpected_keys):
                        print(f"  - {key}")
                else:
                    print("\nThere are no extra network layers in the weight file.")
                print("="*75 + "\n")

            if config.IL.is_requeue:
                if self.optimizer is not None:
                    self.optimizer.load_state_dict(ckpt_dict["optim_state"])
                if self.scheduler is not None and "scheduler_state" in ckpt_dict:
                    self.scheduler.load_state_dict(ckpt_dict["scheduler_state"])
            logger.info(f"Loaded weights from checkpoint: {ckpt_path}, iteration: {start_iter}")

        # Non-requeue fine-tuning: stage-based loading + freezing + optimizer creation
        if not bool(config.IL.is_requeue):
            finetune_stage = str(getattr(self.config.IL, "finetune_stage", "align")).lower()
            pointnet_name_keywords = ("pointnet_encoder", "pc_feat_proj", "fusion_lambda")

            # Freeze/unfreeze according to fine-tuning stage.
            if finetune_stage == "warmup":
                for n, p in self.policy.named_parameters():
                    p.requires_grad_(any(k in n for k in pointnet_name_keywords))
                if self.local_rank < 1:
                    print("[SFT] Fine-tune stage: warmup (trainable: pointnet_encoder + pc_feat_proj)")
                    logger.info("SFT finetune_stage=warmup: only PointNet/fusion parameters are trainable.")
            elif finetune_stage == "align":
                net = self.policy.net.module if hasattr(self.policy.net, "module") else self.policy.net
                mc = config.MODEL

                # Align-stage module switches.
                # Defaults: keep RGB/Depth encoders frozen, and enable BERT/DPFT/SAP.
                align_train_bert = bool(getattr(self.config.IL, "align_train_bert", True))
                align_train_dpft = bool(getattr(self.config.IL, "align_train_dpft", True))
                align_train_sap = bool(getattr(self.config.IL, "align_train_sap", True))
                align_train_pointcloud = bool(getattr(self.config.IL, "align_train_pointcloud", True))
                align_train_rgb = bool(getattr(self.config.IL, "align_train_rgb_encoder", False))
                align_train_depth = bool(getattr(self.config.IL, "align_train_depth_encoder", False))

                # Existing ETP-R1 modules.
                dpft_keywords = ("global_encoder", "graph_query_text", "graph_attentioned_txt_embeds_transform")
                sap_keywords = ("global_sap_head",)

                # Start from fully frozen, then selectively unfreeze by module switches.
                for p in self.policy.parameters():
                    p.requires_grad_(False)

                # Visual encoders.
                for p in net.rgb_encoder.parameters():
                    p.requires_grad_(align_train_rgb)
                for p in net.depth_encoder.parameters():
                    p.requires_grad_(align_train_depth)

                # Newly introduced PointCloud/PointNet branch.
                if align_train_pointcloud:
                    if getattr(net, "pointnet_encoder", None) is not None:
                        for p in net.pointnet_encoder.parameters():
                            p.requires_grad_(True)
                    if getattr(net, "pc_feat_proj", None) is not None:
                        for p in net.pc_feat_proj.parameters():
                            p.requires_grad_(True)
                    if getattr(net, "fusion_lambda_raw", None) is not None:
                        net.fusion_lambda_raw.requires_grad_(True)

                # BERT body.
                vb = net.vln_bert
                if align_train_bert:
                    # text + pano embeddings/encoders
                    for p in vb.embeddings.parameters():
                        p.requires_grad_(True)
                    for p in vb.lang_encoder.parameters():
                        p.requires_grad_(True)
                    for p in vb.img_embeddings.parameters():
                        p.requires_grad_(True)

                    # Keep legacy MODEL.* freeze knobs effective.
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

                # DPFT modules in ETP-R1 global branch.
                if align_train_dpft:
                    for n, p in net.named_parameters():
                        if "vln_bert" in n and any(k in n for k in dpft_keywords):
                            p.requires_grad_(True)

                # SAP heads.
                if align_train_sap:
                    for n, p in net.named_parameters():
                        if "vln_bert" in n and any(k in n for k in sap_keywords):
                            p.requires_grad_(True)

                if self.local_rank < 1:
                    print(
                        "[SFT] Fine-tune stage: align (selective module unfreezing with explicit switches)"
                    )
                    logger.info(
                        "SFT finetune_stage=align switches: bert=%s dpft=%s sap=%s pointcloud=%s rgb_encoder=%s depth_encoder=%s; "
                        "fix_lang_embedding=%s fix_pano_embedding=%s."
                        % (
                            align_train_bert,
                            align_train_dpft,
                            align_train_sap,
                            align_train_pointcloud,
                            align_train_rgb,
                            align_train_depth,
                            bool(mc.fix_lang_embedding),
                            bool(mc.fix_pano_embedding),
                        )
                    )
            else:
                raise ValueError(f"Unknown IL.finetune_stage: {finetune_stage!r} (expected 'warmup' or 'align').")

            # Build optimizer AFTER freezing/unfreezing so it only sees trainable params.
            pointnet_lr = float(getattr(self.config.IL, "pointnet_lr", -1.0))
            if pointnet_lr <= 0:
                pointnet_lr = float(self.config.IL.lr)

            trainable_named_params = [(n, p) for n, p in self.policy.named_parameters() if p.requires_grad]
            base_named_params = [
                (n, p) for n, p in trainable_named_params
                if not any(k in n for k in pointnet_name_keywords)
            ]
            pointnet_named_params = [
                (n, p) for n, p in trainable_named_params
                if any(k in n for k in pointnet_name_keywords)
            ]

            no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
            optimizer_grouped_parameters = []

            base_decay_params = [p for n, p in base_named_params if not any(nd in n for nd in no_decay)]
            base_no_decay_params = [p for n, p in base_named_params if any(nd in n for nd in no_decay)]
            if len(base_decay_params) > 0:
                optimizer_grouped_parameters.append({'params': base_decay_params, 'weight_decay': 0.01, 'lr': float(self.config.IL.lr)})
            if len(base_no_decay_params) > 0:
                optimizer_grouped_parameters.append({'params': base_no_decay_params, 'weight_decay': 0.0, 'lr': float(self.config.IL.lr)})

            pointnet_decay_params = [p for n, p in pointnet_named_params if not any(nd in n for nd in no_decay)]
            pointnet_no_decay_params = [p for n, p in pointnet_named_params if any(nd in n for nd in no_decay)]
            if len(pointnet_decay_params) > 0:
                optimizer_grouped_parameters.append({'params': pointnet_decay_params, 'weight_decay': 0.01, 'lr': float(pointnet_lr)})
            if len(pointnet_no_decay_params) > 0:
                optimizer_grouped_parameters.append({'params': pointnet_no_decay_params, 'weight_decay': 0.0, 'lr': float(pointnet_lr)})

            if setup_optimizer:
                if len(optimizer_grouped_parameters) == 0:
                    raise ValueError(
                        "No trainable parameters found for optimizer setup. "
                        "Check IL.finetune_stage and MODEL.POINT_CLOUD enable flags."
                    )
                self.optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=float(self.config.IL.lr))

                num_warmup_steps = self.config.IL.warmup_iters
                num_training_steps = self.config.IL.iters
                min_lr_ratio = self.config.IL.min_lr_ratio

                def lr_lambda(current_step: int):
                    if current_step < num_warmup_steps:
                        return float(current_step) / float(max(1, num_warmup_steps))
                    progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
                    progress = min(1.0, progress)
                    cosine_decay = 0.5 * (1.0 + math.cos(math.pi * progress))
                    decayed_lr_multiplier = min_lr_ratio + (1.0 - min_lr_ratio) * cosine_decay
                    return decayed_lr_multiplier

                self.scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda)

                if self.local_rank < 1:
                    logger.info(
                        f"SFT optimizer LR setup: base_lr={float(self.config.IL.lr):.6g}, "
                        f"pointnet_lr={float(pointnet_lr):.6g}, "
                        f"trainable_tensors={len(trainable_named_params)}, pointnet_trainable_tensors={len(pointnet_named_params)}"
                    )
                    logger.info(
                        f"SFT checkpoint loading config: load_from_ckpt={bool(load_from_ckpt)}, "
                        f"ckpt_to_load={config.IL.ckpt_to_load}, "
                        f"finetune_stage={finetune_stage}"
                    )
        params = sum(param.numel() for param in self.policy.parameters())
        params_t = sum(
            p.numel() for p in self.policy.parameters() if p.requires_grad
        )
        logger.info(f"Agent parameters: {params/1e6:.2f} MB. Trainable: {params_t/1e6:.2f} MB.")
        logger.info("Finished setting up policy.")
        self._log_module_status_once()

        return start_iter

    def _get_policy_net_module(self):
        return self.policy.net.module if hasattr(self.policy.net, "module") else self.policy.net

    def _log_module_status_once(self):
        if self._module_status_logged_once or self.local_rank >= 1:
            return
        net_module = self._get_policy_net_module()
        pc_cfg = self.config.MODEL.POINT_CLOUD

        pointnet_param_count = 0
        fusion_param_count = 0
        if getattr(net_module, "pointnet_encoder", None) is not None:
            pointnet_param_count = sum(p.numel() for p in net_module.pointnet_encoder.parameters())
        if getattr(net_module, "pc_feat_proj", None) is not None:
            fusion_param_count = sum(p.numel() for p in net_module.pc_feat_proj.parameters())

        logger.info("========== PointCloud/PointNet Module Status ==========")
        logger.info(
            "POINT_CLOUD config: "
            f"enable_depth_to_pointcloud={bool(pc_cfg.enable_depth_to_pointcloud)}, "
            f"enable_spatial_crop={bool(pc_cfg.enable_spatial_crop)}, "
            f"max_depth_m={float(pc_cfg.max_depth_m)}, num_points={int(pc_cfg.num_points)}, "
            f"enable_pointnet={bool(pc_cfg.enable_pointnet)}, enable_fusion={bool(pc_cfg.enable_fusion)}, "
            f"accumulate_mode={str(pc_cfg.accumulate_mode)}"
        )
        logger.info(
            "PointNet module build status: "
            f"pointnet_encoder_built={getattr(net_module, 'pointnet_encoder', None) is not None}, "
            f"pc_feat_proj_built={getattr(net_module, 'pc_feat_proj', None) is not None}, "
            f"pointnet_params={pointnet_param_count}, fusion_proj_params={fusion_param_count}"
        )
        logger.info("======================================================")
        self._module_status_logged_once = True

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

    def _teacher_action_new(self, batch_gmap_vp_ids, batch_no_vp_left, is_train):
        teacher_actions = []
        cur_episodes = self.envs.current_episodes()
        for i, (gmap_vp_ids, gmap, no_vp_left) in enumerate(zip(batch_gmap_vp_ids, self.gmaps, batch_no_vp_left)):
            curr_dis_to_goal = self.envs.call_at(i, "current_dist_to_goal", {"is_train": is_train})
            if curr_dis_to_goal < 1.5:
                teacher_actions.append(0)
            else:
                if no_vp_left:
                    teacher_actions.append(-100)
                elif self.config.IL.expert_policy == 'spl':
                    ghost_vp_pos = [(vp, random.choice(pos)) for vp, pos in gmap.ghost_real_pos.items()]
                    ghost_dis_to_goal = [
                        self.envs.call_at(i, "point_dist_to_goal", {"pos": p[1], "is_train": is_train})
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

            rgb_fts.append(obs['cand_rgb'][i])
            dep_fts.append(obs['cand_depth'][i])
            loc_fts.append(obs['cand_angle_fts'][i])
            nav_types += [1] * len(obs['cand_angles'][i])

            rgb_fts.append(obs['pano_rgb'][i][~cand_idxes])
            dep_fts.append(obs['pano_depth'][i][~cand_idxes])
            loc_fts.append(obs['pano_angle_fts'][~cand_idxes])
            nav_types += [0] * (12-np.sum(cand_idxes))
            
            batch_rgb_fts.append(torch.cat(rgb_fts, dim=0))
            batch_dep_fts.append(torch.cat(dep_fts, dim=0))
            batch_loc_fts.append(torch.cat(loc_fts, dim=0))
            batch_nav_types.append(torch.LongTensor(nav_types))
            batch_view_lens.append(len(nav_types))

        batch_rgb_fts = pad_tensors_wgrad(batch_rgb_fts)
        batch_dep_fts = pad_tensors_wgrad(batch_dep_fts)
        batch_loc_fts = pad_tensors_wgrad(batch_loc_fts).cuda()
        batch_nav_types = pad_sequence(batch_nav_types, batch_first=True).cuda()
        batch_view_lens = torch.LongTensor(batch_view_lens).cuda()

        return {
            'rgb_fts': batch_rgb_fts, 'dep_fts': batch_dep_fts, 'loc_fts': batch_loc_fts,
            'nav_types': batch_nav_types, 'view_lens': batch_view_lens,
        }
        
    def _nav_gmap_variable(self, cur_vp, cur_pos, cur_ori, task_type):
        batch_gmap_vp_ids, batch_gmap_step_ids, batch_gmap_lens = [], [], []
        batch_gmap_img_fts, batch_gmap_pos_fts = [], []
        batch_gmap_pair_dists, batch_gmap_visited_masks = [], []
        batch_gmap_pc_points, batch_gmap_pc_masks = [], []
        batch_gmap_pc_fusion_masks = []
        batch_no_vp_left = []
        batch_gmap_task_embeddings = []

        pc_cfg = self.config.MODEL.POINT_CLOUD
        num_points = int(pc_cfg.num_points)
        accumulate_mode = str(getattr(pc_cfg, "accumulate_mode", "mean")).lower()
        fusion_scope = str(getattr(pc_cfg, "fusion_scope", "global")).lower()
        connected_only = fusion_scope == "connected"
        # How many distinct point clouds each ghost token can accumulate within one rollout.
        # (Used only for feature accumulation in PointNet fusion.)
        max_pc_samples = 1
        for bi, gmap in enumerate(self.gmaps):
            for vp in gmap.ghost_pos.keys():
                if connected_only and (cur_vp[bi] not in gmap.ghost_fronts.get(vp, [])):
                    continue
                pc_list = gmap.ghost_pc_points.get(vp, None)
                if pc_list is not None:
                    max_pc_samples = max(max_pc_samples, int(len(pc_list)))

        for i, gmap in enumerate(self.gmaps):
            node_vp_ids = list(gmap.node_pos.keys())
            ghost_vp_ids = list(gmap.ghost_pos.keys())
            if len(ghost_vp_ids) == 0:
                batch_no_vp_left.append(True)
            else:
                batch_no_vp_left.append(False)

            gmap_vp_ids = [None] + node_vp_ids + ghost_vp_ids
            gmap_step_ids = [0] + [gmap.node_stepId[vp] for vp in node_vp_ids] + [0]*len(ghost_vp_ids)
            mask_visited = bool(getattr(getattr(self.config.MODEL, "SAP", object()), "mask_visited", True))
            gmap_visited_masks = [0] + ([1] * len(node_vp_ids) if mask_visited else [0] * len(node_vp_ids)) + [0] * len(ghost_vp_ids)

            gmap_img_fts = [gmap.get_node_embeds(vp) for vp in node_vp_ids] + \
                           [gmap.get_node_embeds(vp) for vp in ghost_vp_ids]
            gmap_img_fts = torch.stack(
                [torch.zeros_like(gmap_img_fts[0])] + gmap_img_fts, dim=0
            )

            # Build per-token point cloud tensors for ghost nodes.
            # gmap_pc_points: (L, S, N, 3), gmap_pc_masks: (L, S)
            zero_pc = torch.zeros(
                max_pc_samples, num_points, 3,
                dtype=torch.float32, device=self.device
            )
            token_pc_points = []
            token_pc_masks = []
            token_pc_fusion_masks = []

            # Stop token (None)
            token_pc_points.append(zero_pc)
            token_pc_masks.append(torch.zeros(max_pc_samples, dtype=torch.bool, device=self.device))
            token_pc_fusion_masks.append(torch.zeros(max_pc_samples, dtype=torch.bool, device=self.device))

            # Real nodes
            for _ in node_vp_ids:
                token_pc_points.append(zero_pc)
                token_pc_masks.append(torch.zeros(max_pc_samples, dtype=torch.bool, device=self.device))
                token_pc_fusion_masks.append(torch.zeros(max_pc_samples, dtype=torch.bool, device=self.device))

            # Ghost nodes
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
                    )  # (T, N, 3)
                    t = int(stack.shape[0])
                    pc_padded = torch.zeros(
                        max_pc_samples, num_points, 3,
                        dtype=torch.float32, device=self.device
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

            gmap_pc_points_tokens = torch.stack(token_pc_points, dim=0)  # (L, S, N, 3)
            gmap_pc_masks_tokens = torch.stack(token_pc_masks, dim=0)  # (L, S)
            gmap_pc_fusion_masks_tokens = torch.stack(token_pc_fusion_masks, dim=0)  # (L, S)
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
            gmap_step_ids_tensor = torch.LongTensor(gmap_step_ids)
            batch_gmap_step_ids.append(gmap_step_ids_tensor)
            batch_gmap_task_embeddings.append(torch.full_like(gmap_step_ids_tensor, task_type))
            batch_gmap_lens.append(len(gmap_vp_ids))
            batch_gmap_img_fts.append(gmap_img_fts)
            batch_gmap_pos_fts.append(torch.from_numpy(gmap_pos_fts))
            batch_gmap_pair_dists.append(torch.from_numpy(gmap_pair_dists))
            batch_gmap_visited_masks.append(torch.BoolTensor(gmap_visited_masks))
        
        batch_gmap_step_ids = pad_sequence(batch_gmap_step_ids, batch_first=True).cuda()
        batch_gmap_task_embeddings = pad_sequence(batch_gmap_task_embeddings, batch_first=True).cuda()
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

        # Pad point cloud tensors to (B, max_L, S, N, 3) and masks to (B, max_L, S).
        gmap_pc_points = torch.zeros(
            bs, max_gmap_len, max_pc_samples, num_points, 3,
            dtype=torch.float32, device=self.device
        )
        gmap_pc_masks = torch.zeros(
            bs, max_gmap_len, max_pc_samples, dtype=torch.bool, device=self.device
        )
        gmap_pc_fusion_masks = torch.zeros(
            bs, max_gmap_len, max_pc_samples, dtype=torch.bool, device=self.device
        )
        for i in range(bs):
            gmap_pc_points[i, :batch_gmap_lens[i]] = batch_gmap_pc_points[i]
            gmap_pc_masks[i, :batch_gmap_lens[i]] = batch_gmap_pc_masks[i]
            gmap_pc_fusion_masks[i, :batch_gmap_lens[i]] = batch_gmap_pc_fusion_masks[i]

        return {
            'gmap_vp_ids': batch_gmap_vp_ids, 'gmap_step_ids': batch_gmap_step_ids,
            'gmap_img_fts': batch_gmap_img_fts, 'gmap_pos_fts': batch_gmap_pos_fts, 
            'gmap_masks': batch_gmap_masks, 'gmap_visited_masks': batch_gmap_visited_masks, 'gmap_pair_dists': gmap_pair_dists,
            'no_vp_left': batch_no_vp_left, 'gmap_task_embeddings': batch_gmap_task_embeddings,
            'gmap_pc_points': gmap_pc_points, 'gmap_pc_masks': gmap_pc_masks,
            'gmap_pc_fusion_masks': gmap_pc_fusion_masks,
        }

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

        if self.config.local_rank < 1:
            config_path = os.path.join(self.config.CHECKPOINT_FOLDER, "config.yaml")
            with open(config_path, "w") as f:
                f.write(self.config.dump())
            logger.info(f"Configuration saved to {config_path}")
        
        for idx in range(start_iter, total_iter, log_every):
            interval = min(log_every, max(total_iter-idx, 0))
            cur_iter = idx + interval

            sample_ratio = self.config.IL.sample_ratio ** ((idx) // self.config.IL.decay_interval + 1)

            if sample_ratio <= 0.15:
                sample_ratio = 0.0
            logger.info(f"sample ratio: {sample_ratio}")
            logs = self._train_interval(interval, self.config.IL.ml_weight, sample_ratio)

            if self.local_rank < 1:
                loss_str = f'iter {cur_iter}: '
                for k, v in logs.items():
                    if k == "fusion_lambda":
                        logs[k] = float(v[-1])
                    else:
                        logs[k] = np.mean(v)
                    if k == "fusion_lambda":
                        loss_str += f'{k}: {logs[k]:.6f}, '
                    else:
                        loss_str += f'{k}: {logs[k]:.3f}, '
                    writer.add_scalar(f'loss/{k}', logs[k], cur_iter)
                current_lr = self.optimizer.param_groups[0]['lr']
                writer.add_scalar('train/lr', current_lr, cur_iter)
                logger.info(loss_str)
                logger.info(f"lr: {current_lr}")
                self.save_checkpoint(cur_iter)
        
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

        self.sap_loss = 0.
        for idx in pbar:
            self.optimizer.zero_grad()
            self.loss = 0.
            
            with autocast():
                self.rollout('train', ml_weight, sample_ratio)
            self.scaler.scale(self.loss).backward()
            self.scaler.unscale_(self.optimizer)

            # Debug metrics for newly added PointNet/fusion modules.
            net_module = self._get_policy_net_module()
            pointnet_grad_norm_sq = 0.0
            fusion_grad_norm_sq = 0.0
            has_pointnet_grad = False
            has_fusion_grad = False
            if getattr(net_module, "pointnet_encoder", None) is not None:
                for p in net_module.pointnet_encoder.parameters():
                    if p.grad is not None:
                        g = p.grad.detach().float()
                        pointnet_grad_norm_sq += float(torch.sum(g * g).item())
                        has_pointnet_grad = True
            if getattr(net_module, "pc_feat_proj", None) is not None:
                for p in net_module.pc_feat_proj.parameters():
                    if p.grad is not None:
                        g = p.grad.detach().float()
                        fusion_grad_norm_sq += float(torch.sum(g * g).item())
                        has_fusion_grad = True
            self.logs['pointnet_grad_norm'].append(float(np.sqrt(pointnet_grad_norm_sq)) if has_pointnet_grad else 0.0)
            self.logs['fusion_proj_grad_norm'].append(float(np.sqrt(fusion_grad_norm_sq)) if has_fusion_grad else 0.0)

            # Record per-group LR to verify differential LR works in training.
            if len(self.optimizer.param_groups) > 0:
                group_lrs = [float(pg.get('lr', 0.0)) for pg in self.optimizer.param_groups]
                self.logs['optim_min_group_lr'].append(float(min(group_lrs)))
                self.logs['optim_max_group_lr'].append(float(max(group_lrs)))
                if len(group_lrs) >= 4:
                    # Group order follows optimizer construction:
                    # base_decay, base_no_decay, pointnet_decay, pointnet_no_decay.
                    self.logs['optim_pointnet_group_lr'].append(float(np.mean(group_lrs[2:4])))

            self.scaler.step(self.optimizer)
            self.scheduler.step()
            self.scaler.update()

            raw = getattr(net_module, "fusion_lambda_raw", None)
            if raw is not None:
                lambda_val = torch.sigmoid(raw).detach().float().item()
                self.logs["fusion_lambda"].append(lambda_val)

            if self.local_rank < 1:
                pbar.set_postfix({'iter': f'{idx+1}/{interval}'})
        return deepcopy(self.logs)

    @torch.no_grad()
    def _eval_checkpoint(
        self,
        checkpoint_path: str,
        writer: TensorboardWriter,
        checkpoint_index: int = 0,
    ):
        if self.local_rank < 1:
            logger.info(f"checkpoint_path: {checkpoint_path}")
        self.config.defrost()
        self.config.TASK_CONFIG.ENVIRONMENT.ITERATOR_OPTIONS.SHUFFLE = False
        self.config.TASK_CONFIG.ENVIRONMENT.ITERATOR_OPTIONS.MAX_SCENE_REPEAT_STEPS = -1
        self.config.IL.ckpt_to_load = checkpoint_path
        # self.config.TASK_CONFIG.TASK.MEASUREMENTS.append('POSITION_INFER')
        if self.config.VIDEO_OPTION:
            self.config.TASK_CONFIG.TASK.MEASUREMENTS.append("TOP_DOWN_MAP_VLNCE")
            self.config.TASK_CONFIG.TASK.MEASUREMENTS.append("DISTANCE_TO_GOAL")
            self.config.TASK_CONFIG.TASK.MEASUREMENTS.append("SUCCESS")
            self.config.TASK_CONFIG.TASK.MEASUREMENTS.append("SPL")
            self.config.VIDEO_DIR = self.config.VIDEO_DIR + "_" + self.config.EVAL.SPLIT
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

        if self.config.EVAL.fast_eval:
            episodes_allowed = self.traj[::5]
        elif self.config.EVAL.EPISODE_ID:
            episodes_allowed = self.config.EVAL.EPISODE_ID
        else:
            episodes_allowed = self.traj
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
            setup_optimizer=False,
        )
        self.policy.eval()
        self.waypoint_predictor.eval()

        if self.config.EVAL.EPISODE_COUNT == -1:
            eps_to_eval = sum(self.envs.number_of_episodes)
        else:
            eps_to_eval = min(self.config.EVAL.EPISODE_COUNT, sum(self.envs.number_of_episodes))
        self.stat_eps = {}
        self.pbar = tqdm.tqdm(total=eps_to_eval) if self.config.use_pbar else None

        eval_rollout_t0 = time.perf_counter()
        while len(self.stat_eps) < eps_to_eval:
            self.rollout('eval')
        eval_rollout_sec = time.perf_counter() - eval_rollout_t0

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

            logger.info(f"Episodes evaluated: {total}")
            # Wall-clock for rollout loop; use global `total` so multi-GPU avg is eval_wall_time / all_episodes.
            avg_sec_per_ep = (
                eval_rollout_sec / float(total) if total > 0 else float("nan")
            )
            logger.info(
                f"Eval rollout total duration: {eval_rollout_sec:.2f}s "
                f"({eval_rollout_sec / 60.0:.2f} min, {eval_rollout_sec / 3600.0:.3f} h)"
            )
            logger.info(
                f"Eval average wall time per episode: {avg_sec_per_ep:.4f}s "
                f"(rollout only, parallel envs; not pure GPU kernel time)"
            )
            print(
                f"Eval rollout total duration: {eval_rollout_sec:.2f}s | "
                f"avg per episode: {avg_sec_per_ep:.4f}s"
            )
            checkpoint_num = checkpoint_index + 1
            writer.add_scalar(f"eval_time/total_rollout_sec/{split}", eval_rollout_sec, checkpoint_num)
            writer.add_scalar(f"eval_time/avg_sec_per_episode/{split}", avg_sec_per_ep, checkpoint_num)
            for k, v in aggregated_states.items():
                logger.info(f"Average episode {k}: {v:.6f}")
                writer.add_scalar(f"eval_{k}/{split}", v, checkpoint_num)
            print(f"Episodes evaluated: {total}")

    @torch.no_grad()
    def inference(self):
        checkpoint_path = self.config.INFERENCE.CKPT_PATH
        logger.info(f"checkpoint_path: {checkpoint_path}")
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
            setup_optimizer=False,
        )
        self.policy.eval()
        self.waypoint_predictor.eval()

        if self.config.INFERENCE.EPISODE_COUNT == -1:
            eps_to_infer = sum(self.envs.number_of_episodes)
        else:
            eps_to_infer = min(self.config.INFERENCE.EPISODE_COUNT, sum(self.envs.number_of_episodes))
        self.path_eps = defaultdict(list)
        self.inst_ids: Dict[str, int] = {}
        self.pbar = tqdm.tqdm(total=eps_to_infer)

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
 
        instr_max_len = self.config.IL.max_text_len
        instr_pad_id = 1
        if self.config.MODEL.task_type == 'r2r':
            task_type = 1
        elif self.config.MODEL.task_type == 'rxr':
            task_type = 2
        else:
            print("self.config.MODEL.task_type Error")
        observations = extract_instruction_tokens(observations, self.config.TASK_CONFIG.TASK.INSTRUCTION_SENSOR_UUID,
                                                  max_length=instr_max_len, pad_id=instr_pad_id, task_type=task_type)
        batch = batch_obs(observations, self.device)
        batch = apply_obs_transforms_batch(batch, self.obs_transforms)

        if mode == 'eval':
            env_to_pause = [i for i, ep in enumerate(self.envs.current_episodes()) 
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
        all_txt_task_encoding = batch['txt_task_encoding']
        all_txt_masks = (all_txt_ids != instr_pad_id)
        all_txt_embeds = self.policy.net(
            mode='language',
            txt_ids=all_txt_ids,
            txt_task_encoding=all_txt_task_encoding,
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
        # Eval-only stats: whether the planned next ghost target is directly connected
        # to current node (i.e., current node is one of that ghost's frontier nodes).
        eval_plan_ghost_direct_cnt = [0] * self.envs.num_envs
        eval_plan_ghost_indirect_cnt = [0] * self.envs.num_envs

        for stepk in range(self.max_len): 
            total_actions += self.envs.num_envs
            txt_masks = all_txt_masks
            txt_embeds = all_txt_embeds
            
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

            pc_cfg = self.config.MODEL.POINT_CLOUD
            do_pc_sampling = bool(pc_cfg.enable_depth_to_pointcloud)
            if do_pc_sampling:
                depth_hfov = float(self.config.TASK_CONFIG.SIMULATOR.DEPTH_SENSOR.HFOV)
                depth_min = float(getattr(self.config.TASK_CONFIG.SIMULATOR.DEPTH_SENSOR, "MIN_DEPTH", 0.0))
                depth_max = float(getattr(self.config.TASK_CONFIG.SIMULATOR.DEPTH_SENSOR, "MAX_DEPTH", 10.0))
                normalize_depth = bool(getattr(self.config.TASK_CONFIG.SIMULATOR.DEPTH_SENSOR, "NORMALIZE_DEPTH", True))
                depth_feats_maps = build_depth_feats_maps(batch, num_imgs=12)  # (B,12,H,W,1)
                num_points = int(pc_cfg.num_points)
                max_depth_m = float(pc_cfg.max_depth_m)
                enable_spatial_crop = bool(pc_cfg.enable_spatial_crop)
                fps_seed_base = int(pc_cfg.fps_seed) if int(pc_cfg.fps_seed) >= 0 else int(self.config.TASK_CONFIG.SEED)
            else:
                depth_feats_maps = None
                depth_hfov = 0.0
                depth_min = 0.0
                depth_max = 0.0
                normalize_depth = False
                num_points = 0
                max_depth_m = 0.0
                enable_spatial_crop = False
                fps_seed_base = 0

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

            for i in range(self.envs.num_envs):
                cur_embeds = avg_pano_embeds[i]
                cand_embeds = pano_embeds[i][vp_inputs['nav_types'][i]==1] 
                cand_pc_points = None
                if do_pc_sampling:
                    cand_pc_points = []
                    cand_img_idxes_i = wp_outputs["cand_img_idxes"][i]
                    # `cand_img_idxes_i` is aligned with `cand_vp` length returned by `identify_node`.
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
                            num_points=num_points,
                            seed=int(fps_seed_base + stepk * 1000 + i * 10 + k),
                        )
                        cand_pc_points.append(sampled_points)
                self.gmaps[i].update_graph(prev_vp[i], stepk+1,
                                        cur_vp[i], cur_pos[i], cur_embeds,
                                        cand_vp[i], cand_pos[i], cand_embeds,
                                        cand_real_pos[i],
                                        cand_pc_points=cand_pc_points)

            nav_inputs = self._nav_gmap_variable(cur_vp, cur_pos, cur_ori, task_type)
            nav_inputs.update({
                'mode': 'navigation',
                'txt_embeds': txt_embeds, 
                'txt_masks': txt_masks, 
            })
            no_vp_left = nav_inputs.pop('no_vp_left') 

            if mode == 'train':
                pc_cfg = self.config.MODEL.POINT_CLOUD
                self.logs['pc_enable_depth_to_pointcloud'].append(float(bool(pc_cfg.enable_depth_to_pointcloud)))
                self.logs['pc_enable_pointnet'].append(float(bool(pc_cfg.enable_pointnet)))
                self.logs['pc_enable_fusion'].append(float(bool(pc_cfg.enable_fusion)))
                if 'gmap_pc_masks' in nav_inputs:
                    valid_pc_samples = int(nav_inputs['gmap_pc_masks'].sum().item())
                    total_pc_slots = int(nav_inputs['gmap_pc_masks'].numel())
                    ratio = (float(valid_pc_samples) / float(total_pc_slots)) if total_pc_slots > 0 else 0.0
                    self.logs['pc_valid_sample_ratio'].append(ratio)
                    self.logs['pc_valid_sample_count'].append(float(valid_pc_samples))

            nav_outs = self.policy.net(**nav_inputs)
            nav_logits = nav_outs['global_logits']
            nav_probs = F.softmax(nav_logits, 1)
            for i, gmap in enumerate(self.gmaps):
                gmap.node_stop_scores[cur_vp[i]] = nav_probs[i, 0].data.item() 

            if mode == 'train' or self.config.VIDEO_OPTION:
                teacher_actions = self._teacher_action_new(nav_inputs['gmap_vp_ids'], no_vp_left, mode == 'train')
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
                                'tryout': use_tryout
                            },
                            'vis_info': vis_info,
                        }
                    )
                    
                else:
                    ghost_vp = nav_inputs['gmap_vp_ids'][i][cpu_a_t[i]]
                    ghost_pos = gmap.ghost_aug_pos[ghost_vp]
                    if mode == 'eval':
                        # "Directly connected" means current node is a frontier node
                        # that has observed/linked to this ghost.
                        ghost_fronts = gmap.ghost_fronts.get(ghost_vp, [])
                        if cur_vp[i] in ghost_fronts:
                            eval_plan_ghost_direct_cnt[i] += 1
                        else:
                            eval_plan_ghost_indirect_cnt[i] += 1
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
                    metric['high_level_step'] = stepk
                    metric['plan_ghost_direct_conn_cnt'] = float(eval_plan_ghost_direct_cnt[i])
                    metric['plan_ghost_indirect_conn_cnt'] = float(eval_plan_ghost_indirect_cnt[i])
                    total_plan_ghost_cnt = eval_plan_ghost_direct_cnt[i] + eval_plan_ghost_indirect_cnt[i]
                    metric['plan_ghost_direct_conn_ratio'] = (
                        float(eval_plan_ghost_direct_cnt[i]) / float(total_plan_ghost_cnt)
                        if total_plan_ghost_cnt > 0 else 0.0
                    )
                    if ep_id in self.stat_eps:
                        print("ERROR!!!!!!!!!! ", ep_id)
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
                        self.gmaps.pop(i)
                        prev_vp.pop(i)
                        eval_plan_ghost_direct_cnt.pop(i)
                        eval_plan_ghost_indirect_cnt.pop(i)
                        all_txt_ids = torch.cat((all_txt_ids[:i], all_txt_ids[i + 1:]), dim=0)
                        all_txt_task_encoding = torch.cat((all_txt_task_encoding[:i], all_txt_task_encoding[i + 1:]), dim=0)
                        all_txt_masks = torch.cat((all_txt_masks[:i], all_txt_masks[i + 1:]), dim=0)
                        all_txt_embeds = torch.cat((all_txt_embeds[:i], all_txt_embeds[i + 1:]), dim=0)

            if self.envs.num_envs == 0:
                break

            # obs for next step
            observations = extract_instruction_tokens(observations, self.config.TASK_CONFIG.TASK.INSTRUCTION_SENSOR_UUID, \
                                                                        max_length=instr_max_len, pad_id=instr_pad_id, task_type=task_type)
            batch = batch_obs(observations, self.device)
            batch = apply_obs_transforms_batch(batch, self.obs_transforms)

        if mode == 'train':
            loss = ml_weight * loss / total_actions 
            self.loss += loss
            self.logs['IL_loss'].append(loss.item())