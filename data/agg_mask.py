from PIL import Image

def crop_image(image_path, output_path):
    """
    Crops the image from the 3/5 to 4/5 vertical section.

    Parameters:
    image_path (str): Path to the input image.
    output_path (str): Path to save the cropped image.
    """
    image = Image.open(image_path)
    width, height = image.size
    
    # Calculate the cropping box
    top = int(height * 2.5 / 5)
    bottom = int(height * 4.5 / 5)
    left = 0
    right = width
    
    # Crop the image
    cropped_image = image.crop((left, top, right, bottom))
    
    # Save the cropped image
    cropped_image.save(output_path)

# Example usage
input_image_path = "/data1/JM/code/BrushNet/data/visual_set/baseline_SDI/003210.png"
output_cropped_image_path = "./ttt4.png"
crop_image(input_image_path, output_cropped_image_path)


# import os
# from PIL import Image
# import numpy as np

# # 定义目录路径
# combined_mask_dir = '/data1/JM/code/BrushNet/data/data_train_small/mask/combined'
# image_dir = '/data1/JM/code/BrushNet/data/data_train_small/image'
# output_image_dir = '/data1/JM/code/BrushNet/data/data_train_small/masked_image'

# # 创建输出目录如果不存在的话
# os.makedirs(output_image_dir, exist_ok=True)

# # 获取目录中的所有文件名（假设文件名在各个目录中一致）
# filenames = os.listdir(combined_mask_dir)

# # 遍历每个文件，进行黑白颠倒处理并与原始图像点乘
# for filename in filenames:
#     # 打开每个目录中的对应文件
#     combined_mask_path = os.path.join(combined_mask_dir, filename)
#     image_path = os.path.join(image_dir, filename)
    
#     if os.path.exists(combined_mask_path) and os.path.exists(image_path):
#         combined_mask = Image.open(combined_mask_path).convert('L')
#         original_image = Image.open(image_path).convert('RGB')
        
#         # 将图像转换为numpy数组
#         combined_mask_np = np.array(combined_mask)
#         original_image_np = np.array(original_image)
        
#         # 黑白颠倒处理（255减去当前值）
#         inverted_mask_np = 255 - combined_mask_np
        
#         # 将mask应用到原始图像上（点乘操作）
#         masked_image_np = original_image_np * np.expand_dims(inverted_mask_np / 255, axis=2)
        
#         # 转换回PIL图像并保存
#         masked_image = Image.fromarray(np.uint8(masked_image_np))
#         masked_image.save(os.path.join(output_image_dir, filename))

# print("Mask黑白颠倒并应用到原始图像完成")




# import os
# from PIL import Image
# import numpy as np

# # 定义目录路径
# ceilings_dir = '/data1/JM/code/BrushNet/data/data_train_small/mask/ceilings'
# chair_dir = '/data1/JM/code/BrushNet/data/data_train_small/mask/chair'
# sofa_dir = '/data1/JM/code/BrushNet/data/data_train_small/mask/sofa'
# output_dir = '/data1/JM/code/BrushNet/data/data_train_small/mask/combined'

# # 创建输出目录如果不存在的话
# os.makedirs(output_dir, exist_ok=True)

# # 获取目录中的所有文件名（假设文件名在各个目录中一致）
# filenames = os.listdir(ceilings_dir)

# # 遍历每个文件，进行叠加
# for filename in filenames:
#     # 打开每个目录中的对应文件
#     ceiling_path = os.path.join(ceilings_dir, filename)
#     chair_path = os.path.join(chair_dir, filename)
#     sofa_path = os.path.join(sofa_dir, filename)
    
#     if os.path.exists(ceiling_path) and os.path.exists(chair_path) and os.path.exists(sofa_path):
#         ceiling_mask = Image.open(ceiling_path).convert('L')
#         chair_mask = Image.open(chair_path).convert('L')
#         sofa_mask = Image.open(sofa_path).convert('L')
        
#         # 将图像转换为numpy数组
#         ceiling_mask_np = np.array(ceiling_mask)
#         chair_mask_np = np.array(chair_mask)
#         sofa_mask_np = np.array(sofa_mask)
        
#         # 叠加mask（假设叠加方式为按位或运算）
#         combined_mask_np = np.bitwise_or(np.bitwise_or(ceiling_mask_np, chair_mask_np), sofa_mask_np)
        
#         # 转换回PIL图像并保存
#         combined_mask = Image.fromarray(combined_mask_np)
#         combined_mask.save(os.path.join(output_dir, filename))

# print("Mask叠加完成")
