import os
from PIL import Image, ImageOps
import numpy as np
import shutil

def process_images(image_dir, mask_dir, output_dir):
    # 确保输出目录和原始图像复制目录存在
    os.makedirs(output_dir, exist_ok=True)
    # os.makedirs(original_image_copy_dir, exist_ok=True)

    # 获取所有图像文件名（假设图像和mask的文件名对应）
    image_files = os.listdir(image_dir)

    for image_file in image_files:
        # 构建图像和mask的完整路径
        image_path = os.path.join(image_dir, image_file)
        mask_path = os.path.join(mask_dir, image_file)

        # 打开图像和mask
        image = Image.open(image_path).convert("RGB")
        mask = Image.open(mask_path).convert("L")  # 灰度图

        # 黑白置换 (invert)
        inverted_mask = ImageOps.invert(mask)

        # 将mask转换为与图像相同的RGB模式
        inverted_mask_rgb = Image.merge("RGB", [inverted_mask, inverted_mask, inverted_mask])

        # 将image和mask进行点乘
        masked_image_np = np.array(image) * (np.array(inverted_mask_rgb) // 255)

        # 将点乘结果中的黑色部分（RGB值为 [0, 0, 0]）替换为紫色（RGB值为 [128, 0, 128]）
        purple_color = [128, 0, 128]
        mask_black = np.all(masked_image_np == [0, 0, 0], axis=-1)
        masked_image_np[mask_black] = purple_color

        # 将结果转换回PIL图像
        masked_image = Image.fromarray(masked_image_np.astype('uint8'), 'RGB')

        # 保存处理后的图像
        output_image_path = os.path.join(output_dir, image_file)
        masked_image.save(output_image_path)

        # # 复制原始图像到指定目录
        # original_image_copy_path = os.path.join(original_image_copy_dir, image_file)
        # shutil.copy(image_path, original_image_copy_path)

        print(f"Processed and saved: {output_image_path}")
        # print(f"Copied original image to: {original_image_copy_path}")

if __name__ == "__main__":
    # for vid in sorted(os.listdir('/data1/JM/code/BrushNet/data/Baseon_4K_dataset/data_test_anchor')):
    image_dir = f"/data1/JM/code/BrushNet/data/BrushDench/images"
    mask_dir = f"/data1/JM/code/BrushNet/data/BrushDench/mask"
    output_dir = f"/data1/JM/code/BrushNet/data/BrushDench/mask_fill"
    process_images(image_dir, mask_dir, output_dir)

# from PIL import Image, ImageOps

# # 打开图像
# image_path = '/data1/JM/code/BrushNet/data/Baseon_4K_dataset/concat/Cloakroom/000369.png'
# image = Image.open(image_path)

# # 图像的宽度和高度
# width, height = image.size

# # 每个小图像的高度（不包括间隙）
# single_height = (height - 7 * 20) // 8

# # 计算出每个子图的起始位置
# sub_images = []
# for i in range(8):
#     top = i * (single_height + 20)
#     sub_image = image.crop((0, top, width, top + single_height))
#     sub_images.append(sub_image)

# # 交换1号和5号子图（0开始计数）
# sub_images[1], sub_images[5] = sub_images[5], sub_images[1]

# # 创建一个新图像，背景为白色
# new_image = Image.new('RGB', (width, height), (255, 255, 255))

# # 将子图拼接回新图像中
# for i in range(8):
#     top = i * (single_height + 20)
#     new_image.paste(sub_images[i], (0, top))

# # 保存新的图像
# new_image.save('/data1/JM/code/BrushNet/000337_swapped.png')

