import torch
import os
import numpy as np
import torch.nn as nn
from torchvision import transforms
import cv2
from tqdm import tqdm
from PIL import Image

# 导入您的模型定义和数据加载器
from model_vgg_test import SCT

vgg = nn.Sequential(
    nn.Conv2d(3, 3, (1, 1)),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(3, 64, (3, 3)),
    nn.ReLU(),  # relu1-1
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(64, 64, (3, 3)),
    nn.ReLU(),  # relu1-2
    nn.MaxPool2d((2, 2), (2, 2), (0, 0), ceil_mode=True),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(64, 128, (3, 3)),
    nn.ReLU(),  # relu2-1
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(128, 128, (3, 3)),
    nn.ReLU(),  # relu2-2
    nn.MaxPool2d((2, 2), (2, 2), (0, 0), ceil_mode=True),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(128, 256, (3, 3)),
    nn.ReLU(),  # relu3-1
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(256, 256, (3, 3)),
    nn.ReLU(),  # relu3-2
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(256, 256, (3, 3)),
    nn.ReLU(),  # relu3-3
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(256, 256, (3, 3)),
    nn.ReLU(),  # relu3-4
    nn.MaxPool2d((2, 2), (2, 2), (0, 0), ceil_mode=True),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(256, 512, (3, 3)),
    nn.ReLU(),  # relu4-1, this is the last layer used
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(512, 512, (3, 3)),
    nn.ReLU(),  # relu4-2
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(512, 512, (3, 3)),
    nn.ReLU(),  # relu4-3
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(512, 512, (3, 3)),
    nn.ReLU(),  # relu4-4
    nn.MaxPool2d((2, 2), (2, 2), (0, 0), ceil_mode=True),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(512, 512, (3, 3)),
    nn.ReLU(),  # relu5-1
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(512, 512, (3, 3)),
    nn.ReLU(),  # relu5-2
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(512, 512, (3, 3)),
    nn.ReLU(),  # relu5-3
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(512, 512, (3, 3)),
    nn.ReLU()  # relu5-4
)

# 固定随机种子
def fixed_seed():
    seed = 42
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

# 提取图像的风格特征
def extract_style_features(model, images, device):
    preprocess = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])
    
    processed_images = [preprocess(Image.fromarray(cv2.cvtColor(cv2.resize(cv2.imread(img), (512, 512)), cv2.COLOR_BGR2RGB))).unsqueeze(0) for img in images]
    batch = torch.cat(processed_images, dim=0).to(device)

    with torch.no_grad():
        features_mean, features_std = model(batch)

    return features_mean.cpu().numpy(), features_std.cpu().numpy()

if __name__ == "__main__":
    fixed_seed()
    
    # 加载预训练的模型
    model = SCT(vgg).cuda()
    model.load_state_dict(torch.load('/data0/JM/code/scene_style_learning/finetune_vgg/model_ckpt/000020.pth'))
    model.eval()
    
    img_root = '/data0/JM/code/scene_style_learning/data/train'
    img_list = [os.path.join(img_root, img_name) for img_name in sorted(os.listdir(img_root))]
    
    batch_size = 16  # 设定批处理大小

    feature_mean_dir = '/data0/JM/code/scene_style_learning/finetune_vgg/feature_mean'
    feature_std_dir = '/data0/JM/code/scene_style_learning/finetune_vgg/feature_std'
    os.makedirs(feature_mean_dir, exist_ok=True)
    os.makedirs(feature_std_dir, exist_ok=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    for i in tqdm(range(0, len(img_list), batch_size), desc="Extracting features"):
        batch_paths = img_list[i:i + batch_size]
        features_mean, features_std = extract_style_features(model, batch_paths, device)
        
        for j, img_path in enumerate(batch_paths):
            save_name = os.path.basename(img_path).replace('png', 'npy')
            np.save(os.path.join(feature_mean_dir, save_name), features_mean[j])
            np.save(os.path.join(feature_std_dir, save_name), features_std[j])

        # 每处理一批图像，释放显存
        torch.cuda.empty_cache()

    print("特征提取并保存完成。")
