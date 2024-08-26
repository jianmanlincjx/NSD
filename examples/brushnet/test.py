import os

# names = [
#     "baseline_BLD",
#     "baseline_brushnet",
#     "baseline_controlnet",

# ]

# CUDA_VISIBLE_DEVICES = ["0", "0", "0"]

# for i, name in enumerate(names):
#     cuda_device = CUDA_VISIBLE_DEVICES[i]
#     command = f"""
#     CUDA_VISIBLE_DEVICES={cuda_device} python examples/brushnet/evaluate_brushnet.py \\
#         --brushnet_ckpt_path /data1/JM/code/BrushNet/pretrain_model/segmentation_mask_brushnet_ckpt \\
#         --base_model_path /data1/JM/code/BrushNet/pretrain_model/stable-diffusion-v1-5 \\
#         --image_save_path /data1/JM/code/BrushNet/data/Baseon_4K_dataset/{name} \\
#         --data_root /data1/JM/code/BrushNet/data/Baseon_4K_dataset \\
#         --data_name {name}
#     """
#     os.system(command)




import os

names = [
    "baseline_ours",
    "baseline_BLD",
    "baseline_brushnet",
    "baseline_controlnet",
    "baseline_SD",
    "baseline_ppt",

]

CUDA_VISIBLE_DEVICES = ["1", "1", "1", '1', '1', '1']

for i, name in enumerate(names):
    cuda_device = CUDA_VISIBLE_DEVICES[i]
    command = f"""
    CUDA_VISIBLE_DEVICES={cuda_device} python examples/brushnet/evaluate_brushnet.py \\
        --brushnet_ckpt_path /data1/JM/code/BrushNet/pretrain_model/segmentation_mask_brushnet_ckpt \\
        --base_model_path /data1/JM/code/BrushNet/pretrain_model/stable-diffusion-v1-5 \\
        --image_save_path /data1/JM/code/BrushNet/data/Baseon_4K_dataset/{name} \\
        --data_root /data1/JM/code/BrushNet/data/Baseon_4K_dataset \\
        --data_name {name}
    """
    os.system(command)
