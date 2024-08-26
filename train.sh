# # sd v1.5 follow IP-adapter
accelerate launch --num_processes 3 examples/brushnet/train_brushnet_me_follow_ipadapter.py \
--pretrained_model_name_or_path /data1/JM/code/BrushNet/pretrain_model/stable-diffusion-v1-5 \
--output_dir runs/logs/brushnet_segmentationmask \
--resolution 512 \
--learning_rate 1e-5 \
--train_batch_size 2 \
--tracker_project_name brushnet \
--report_to tensorboard \
--resume_from_checkpoint latest \
--checkpointing_steps 100000  \
--json_file /data1/JM/code/BrushNet/data/train_small.json \
--brushnet_model_name_or_path /data1/JM/code/BrushNet/pretrain_model/segmentation_mask_brushnet_ckpt \
--mixed_precision 'fp16' \
--validation_image /data1/JM/code/BrushNet/data/data_train_small/image/000075.png \
--validation_mask /data1/JM/code/BrushNet/data/data_train_small/mask/chair/000075.png \
--validation_prompt 'A delicate sofa in the room. ' \
--validation_steps 1000 \
--image_encoder_path /data1/JM/code/BrushNet/pretrain_model/image_encoder


# https://drive.google.com/drive/folders/1vGfeqTh0G1ZsQLJJUr35ilCY6_ee4I_j?usp=sharing