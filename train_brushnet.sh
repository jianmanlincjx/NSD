# sd v1.5
accelerate launch --num_processes 2 examples/brushnet/train_brushnet.py \
--pretrained_model_name_or_path /data1/JM/code/BrushNet/pretrain_model/stable-diffusion-v1-5 \
--output_dir runs/logs/brushnet_itself \
--resolution 512 \
--learning_rate 1e-5 \
--train_batch_size 2 \
--tracker_project_name brushnet \
--report_to tensorboard \
--resume_from_checkpoint latest \
--mixed_precision 'fp16' \
--json_file /data1/JM/code/BrushNet/data/train_combined.json \
--brushnet_model_name_or_path /data1/JM/code/BrushNet/pretrain_model/segmentation_mask_brushnet_ckpt \
--validation_image /data1/JM/code/BrushNet/data/data_train_small/image/000075.png \
--validation_mask /data1/JM/code/BrushNet/data/data_train_small/mask/chair/000075.png \
--validation_prompt 'A delicate small sofa in the room. ' \
--validation_steps 1000 \
--checkpointing_steps 10000 

# 24118 23758

# /data1/JM/code/BrushNet/data/visual_set/baseline_concat/003346.png