#!/bin/bash

# Check if the correct number of arguments is provided
if [ "$#" -ne 1 ]; then
    echo "Usage: $0 <path to a txt file generated by prepare_video_file_list.py>"
    exit 1
fi

DATA_PATH=$1
MASTER_ADDRESS=10.102.2.210 # ip address can be checked via `hostname -I`
OUTPUT_DIR='checkpoints/3frame_vitb_patch8x8_mr0.90/'
NNODES=1 # num of nodes to train on
NPROC_PER_NODE=8 # num of gpus per node
NODE_RANK=0 # node rank (needed with using multi-node training)

echo "master addr: $MASTER_ADDRESS"
echo "num of nodes: $NNODES"
echo "node rank: $NODE_RANK"
echo "procs per node: $NPROC_PER_NODE"

OMP_NUM_THREADS=1 torchrun \
    --nproc_per_node=$NPROC_PER_NODE --nnodes=$NNODES --node_rank=$NODE_RANK \
    --master_addr=$MASTER_ADDRESS --master_port=19234 \
    cwm/run_pretraining.py \
    --data_path ${DATA_PATH} \
    --model vitb_8x8patch_3frames \
    --mask_type rotated_table \
    --mask_ratio 0.90 \
    --mask_kwargs '{"tube_length": 1}' \
    --context_frames 2 \
    --target_frames 1 \
    --temporal_units 'ms' \
    --sampling_rate 150 \
    --context_target_gap 150 150 \
    --batch_size 32 \
    --accum_iter 1 \
    --opt adamw \
    --opt_betas 0.9 0.95 \
    --warmup_epochs 40 \
    --save_ckpt_freq 10 \
    --epochs 800 \
    --augmentation_type 'multiscale' \
    --augmentation_scales 1.0 0.875 0.75 0.66 \
    --log_dir ${OUTPUT_DIR} \
    --output_dir ${OUTPUT_DIR} \
    --print_freq 1 \
    --num_workers 16
