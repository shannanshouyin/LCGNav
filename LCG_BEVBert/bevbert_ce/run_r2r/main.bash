export GLOG_minloglevel=2
export MAGNUM_LOG=quiet

flag1="--exp_name bevbert
      --run-type train
      --exp-config run_r2r/iter_train.yaml
      SIMULATOR_GPU_IDS [0]
      TORCH_GPU_IDS [0]
      GPU_NUMBERS 1
      NUM_ENVIRONMENTS 8
      IL.waypoint_aug  True
      TASK_CONFIG.SIMULATOR.HABITAT_SIM_V0.ALLOW_SLIDING True
      MODEL.pretrained_path pretrain/ETP/mlm.sap_r2r/ckpts/model_step_82500.pt
      "

flag2=" --exp_name bevbert
      --run-type eval
      --exp-config run_r2r/iter_train.yaml
      SIMULATOR_GPU_IDS [0]
      TORCH_GPU_IDS [0]
      GPU_NUMBERS 1
      NUM_ENVIRONMENTS 4
      TASK_CONFIG.SIMULATOR.HABITAT_SIM_V0.ALLOW_SLIDING True
      EVAL.CKPT_PATH_DIR data/logs/checkpoints/bevbert/
      IL.back_algo control
      "

flag3="--exp_name bevbert
      --run-type inference
      --exp-config run_r2r/iter_train.yaml
      SIMULATOR_GPU_IDS [0]
      TORCH_GPU_IDS [0]
      GPU_NUMBERS 1
      NUM_ENVIRONMENTS 4
      TASK_CONFIG.SIMULATOR.HABITAT_SIM_V0.ALLOW_SLIDING True
      INFERENCE.CKPT_PATH ckpt/ckpt.iter9600.pth
      INFERENCE.PREDICTIONS_FILE preds.json
      IL.back_algo control
      "

mode=$1
case $mode in 
      train)
      echo "###### train mode ######"
      python -m torch.distributed.launch --nproc_per_node=1 --master_port $2 run.py $flag1
      ;;
      eval)
      echo "###### eval mode ######"
      python -m torch.distributed.launch --nproc_per_node=1 --master_port $2 run.py $flag2
      ;;
      infer)
      echo "###### infer mode ######"
      python -m torch.distributed.launch --nproc_per_node=1 --master_port $2 run.py $flag3
      ;;
esac

# CUDA_VISIBLE_DEVICES=0 bash run_r2r/main.bash [train/eval/infer] 2333
