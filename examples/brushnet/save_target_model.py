import torch
import os

def extract_and_save_specific_keys(checkpoint_path, keys_to_extract, output_dir):
    # 加载检查点
    checkpoint = torch.load(checkpoint_path)
    
    # 创建输出目录（如果不存在）
    os.makedirs(output_dir, exist_ok=True)
    
    for key in keys_to_extract:
        # 初始化字典来存储特定的键的参数
        extracted_parameters = {}
        
        # 遍历检查点中的所有参数名称
        for param_tensor in checkpoint:
            key_prefix = param_tensor.split('.')[0]
            if key_prefix == key:
                extracted_parameters[param_tensor] = checkpoint[param_tensor]
        
        # 保存提取的参数到文件
        output_path = os.path.join(output_dir, f"{key}_parameters.pth")
        torch.save(extracted_parameters, output_path)
        print(f"Saved parameters for {key} to {output_path}")

# 路径到你的检查点文件
checkpoint_path = '/data1/JM/code/BrushNet/runs/logs/brushnet_segmentationmask/checkpoint-300000/pytorch_model.bin'

# 定义要提取的键
keys_to_extract = {'image_proj_model', 'adapter_modules'}

# 定义输出目录
output_dir = '/data1/JM/code/BrushNet/runs/logs/brushnet_segmentationmask/checkpoint-300000/extracted_parameters'

# 调用函数提取并保存特定的键
extract_and_save_specific_keys(checkpoint_path, keys_to_extract, output_dir)
