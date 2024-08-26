

import cv2
import os
import numpy as np

def vertically_concatenate_images_with_gap(input_dir, folders, output_dir, gap=20):
    # 确保输出目录存在，如果不存在则创建
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # 遍历每个文件夹内的文件名
    filenames = os.listdir(os.path.join(input_dir, folders[0]))  # 假设所有文件夹内文件名相同
    
    for filename in filenames:
        images = []
        
        for folder in folders:
            image_path = os.path.join(input_dir, folder, filename)
            image = cv2.imread(image_path)
            if image is not None:
                images.append(image)
        
        if len(images) > 0:
            # 找到最大宽度，调整所有图像的宽度一致
            max_width = max(image.shape[1] for image in images)
            padded_images = []

            for image in images:
                height, width, _ = image.shape
                if width < max_width:
                    # 在右侧填充白色以匹配最大宽度
                    padding = np.full((height, max_width - width, 3), 255, dtype=np.uint8)
                    padded_image = np.hstack((image, padding))
                else:
                    padded_image = image
                padded_images.append(padded_image)

            # 在图像之间添加白色间隙
            white_gap = np.full((gap, max_width, 3), 255, dtype=np.uint8)
            concatenated_image = padded_images[0]

            for i in range(1, len(padded_images)):
                concatenated_image = np.vstack((concatenated_image, white_gap, padded_images[i]))

            # 保存拼接后的图像
            output_path = os.path.join(output_dir, filename)
            cv2.imwrite(output_path, concatenated_image)
            print(f"Processed and saved: {output_path}")

if __name__ == "__main__":
    input_dir = '/data1/JM/code/BrushNet/data/BrushDench'  # 输入根目录
    output_dir = '/data1/JM/code/BrushNet/data/BrushDench/concatenated_results'  # 输出目录

    # 要处理的六个文件夹名
    folders = ['mask_fill', 'result_BLD', 'result_SDI', 'result_CNI', 'result_brushnet', 'result_ours', 'result_pp', 'images']

    vertically_concatenate_images_with_gap(input_dir, folders, output_dir)


# import os
# from PIL import Image

# # 定义路径
# base_path = '/data1/JM/code/BrushNet/data/visual_set/'
# directories = [
#     'masked_image_filled',
#     'baseline_BLD',
#     'baseline_SDI',
#     'baseline_CNI',
#     'baseline_brushnet',
#     'baseline_ppt',
#     'baseline_ours',
#     'image'
# ]
# save_path = os.path.join(base_path, 'baseline_concat_user_study')

# # 如果保存目录不存在，则创建
# os.makedirs(save_path, exist_ok=True)

# # 获取基准尺寸
# def load_baseline_size():
#     baseline_dir = os.path.join(base_path, 'baseline_ours')
#     for file_name in sorted(os.listdir(baseline_dir)):
#         if file_name.endswith('.png'):
#             img_path = os.path.join(baseline_dir, file_name)
#             img = Image.open(img_path)
#             return img.size
#     raise ValueError("No images found in baseline_ours directory")

# # 加载基准尺寸
# baseline_size = load_baseline_size()

# # 确定白缝隙的宽度
# white_gap_width = 20  # 设定白色缝隙宽度

# # 获取所有目录中的文件名列表，假设所有目录中的文件数目和名称一致
# file_names = sorted(os.listdir(os.path.join(base_path, 'baseline_ours')))
# # 遍历每个文件，加载、对齐、拼接和保存
# for file_name in file_names:
#     if file_name.endswith('.png'):
#         images = []
#         for directory in directories:
#             img_path = os.path.join(base_path, directory, file_name)
#             img = Image.open(img_path)
#             if img.size != baseline_size:
#                 img = img.resize(baseline_size, Image.ANTIALIAS)
#             images.append(img)

#         # 拼接图像
#         img_width, img_height = baseline_size
#         combined_height = len(directories) * img_height + (len(directories) - 1) * white_gap_width
#         combined_image = Image.new('RGB', (img_width, combined_height), (255, 255, 255))

#         y_offset = 0
#         for img in images:
#             combined_image.paste(img, (0, y_offset))
#             y_offset += img_height + white_gap_width

#         # 保存拼接后的图像
#         save_image_path = os.path.join(save_path, f'{file_name}')
#         combined_image.save(save_image_path)
#         print(f"Saved combined image {save_image_path}")

# import os
# from PIL import Image


# def resize_image_to_target(image, target_size):
#     return image.resize(target_size, Image.ANTIALIAS)

# def process_and_concatenate_images(root_dir, sub_categories):
#     # The strict order for concatenation
#     sub_dirs_in_order = [
#         'masked_image', 
#         'baseline_BLD', 
#         'baseline_SD', 
#         'baseline_controlnet', 
#         'baseline_ppt', 
#         'baseline_brushnet',  
#         'baseline_ours', 
#         'image'
#     ]
    
#     white_gap_width = 20  # Width of the white gap between images

#     for category in sub_categories:
#         print(f"Processing category: {category}")
        
#         # Get baseline_ours image size as a reference
#         baseline_ours_dir = os.path.join(root_dir, 'baseline_ours', category, 'result_new')
#         baseline_ours_files = sorted(os.listdir(baseline_ours_dir))
        
#         for filename in baseline_ours_files:
#             baseline_ours_image_path = os.path.join(baseline_ours_dir, filename)
#             target_image = Image.open(baseline_ours_image_path)
#             target_size = target_image.size
            
#             images_to_concatenate = []
#             all_files_exist = True
            
#             # Process all directories in the specified order
#             for sub_dir in sub_dirs_in_order:
#                 if sub_dir == 'image' or sub_dir == 'masked_image':  # Handle no 'result' sub-directories
#                     image_path = os.path.join(root_dir, sub_dir, category, filename)
#                 else:
#                     image_path = os.path.join(root_dir, sub_dir, category, 'result', filename)

#                 if sub_dir == 'baseline_ours':
#                     image_path = os.path.join(root_dir, sub_dir, category, 'result_new', filename)
                
#                 if os.path.exists(image_path):
#                     image = Image.open(image_path)
#                     resized_image = resize_image_to_target(image, target_size)
#                     images_to_concatenate.append(resized_image)
#                 else:
#                     print(f"Image {filename} not found in {sub_dir}/{category}, skipping concatenation for this file.")
#                     all_files_exist = False
#                     break
            
#             if not all_files_exist:
#                 continue

#             # Concatenate images if all necessary images were found
#             if images_to_concatenate:
#                 total_height = sum(img.height for img in images_to_concatenate) + (len(images_to_concatenate) - 1) * white_gap_width
#                 max_width = max(img.width for img in images_to_concatenate)
#                 concatenated_image = Image.new('RGB', (max_width, total_height), (255, 255, 255))  # Start with a white background
                
#                 y_offset = 0
#                 for img in images_to_concatenate:
#                     concatenated_image.paste(img, (0, y_offset))
#                     y_offset += img.height + white_gap_width  # Add gap between images
                
#                 # Save the concatenated image
#                 output_dir = os.path.join(root_dir, 'concat', category)
#                 os.makedirs(output_dir, exist_ok=True)
#                 output_path = os.path.join(output_dir, filename)
#                 concatenated_image.save(output_path)
#                 print(f"Saved concatenated image: {output_path}")

# if __name__ == "__main__":
#     root_dir = "/data1/JM/code/BrushNet/data/Baseon_4K_dataset"
#     sub_categories = ['studyroom', 'Meeting_room']
    
#     process_and_concatenate_images(root_dir, sub_categories)
