#!/bin/bash

# # Shell script to run the Python script with arguments
name=baseline_BLD
CUDA_VISIBLE_DEVICES=0 python examples/brushnet/evaluate_brushnet.py \
    --brushnet_ckpt_path /data1/JM/code/BrushNet/pretrain_model/segmentation_mask_brushnet_ckpt \
    --base_model_path /data1/JM/code/BrushNet/pretrain_model/stable-diffusion-v1-5 \
    --image_save_path /data1/JM/code/BrushNet/data/$name \
    --data_root /data1/JM/code/BrushNet/data \
    --data_name $name

name=baseline_brushnet
CUDA_VISIBLE_DEVICES=0 python examples/brushnet/evaluate_brushnet.py \
    --brushnet_ckpt_path /data1/JM/code/BrushNet/pretrain_model/segmentation_mask_brushnet_ckpt \
    --base_model_path /data1/JM/code/BrushNet/pretrain_model/stable-diffusion-v1-5 \
    --image_save_path /data1/JM/code/BrushNet/data/$name \
    --data_root /data1/JM/code/BrushNet/data \
    --data_name $name


name=baseline_controlnet
CUDA_VISIBLE_DEVICES=0 python examples/brushnet/evaluate_brushnet.py \
    --brushnet_ckpt_path /data1/JM/code/BrushNet/pretrain_model/segmentation_mask_brushnet_ckpt \
    --base_model_path /data1/JM/code/BrushNet/pretrain_model/stable-diffusion-v1-5 \
    --image_save_path /data1/JM/code/BrushNet/data/$name \
    --data_root /data1/JM/code/BrushNet/data \
    --data_name $name

name=baseline_ours
CUDA_VISIBLE_DEVICES=1 python examples/brushnet/evaluate_brushnet.py \
    --brushnet_ckpt_path /data1/JM/code/BrushNet/pretrain_model/segmentation_mask_brushnet_ckpt \
    --base_model_path /data1/JM/code/BrushNet/pretrain_model/stable-diffusion-v1-5 \
    --image_save_path /data1/JM/code/BrushNet/data/$name \
    --data_root /data1/JM/code/BrushNet/data \
    --data_name $name

name=baseline_ppt
CUDA_VISIBLE_DEVICES=1 python examples/brushnet/evaluate_brushnet.py \
    --brushnet_ckpt_path /data1/JM/code/BrushNet/pretrain_model/segmentation_mask_brushnet_ckpt \
    --base_model_path /data1/JM/code/BrushNet/pretrain_model/stable-diffusion-v1-5 \
    --image_save_path /data1/JM/code/BrushNet/data/$name \
    --data_root /data1/JM/code/BrushNet/data \
    --data_name $name

name=baseline_SD
CUDA_VISIBLE_DEVICES=1 python examples/brushnet/evaluate_brushnet.py \
    --brushnet_ckpt_path /data1/JM/code/BrushNet/pretrain_model/segmentation_mask_brushnet_ckpt \
    --base_model_path /data1/JM/code/BrushNet/pretrain_model/stable-diffusion-v1-5 \
    --image_save_path /data1/JM/code/BrushNet/data/$name \
    --data_root /data1/JM/code/BrushNet/data \
    --data_name $name

# /data1/JM/code/BrushNet/data/visual_set/baseline_concat/000196.png