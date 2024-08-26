from diffusers import StableDiffusionBrushNetPipeline, BrushNetModel, UniPCMultistepScheduler
import torch
import cv2
import json
import os
import numpy as np
from PIL import Image
import argparse
import pandas as pd
import torch
from torchvision.transforms import Resize
from torchvision import transforms
import torch.nn.functional as F
import numpy as np
from torchmetrics.multimodal import CLIPScore
from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
from torchmetrics.regression import MeanSquaredError
from urllib.request import urlretrieve 
from PIL import Image
import open_clip
import os
import hpsv2
import ImageReward as RM
import math
import torch.nn as nn
from sklearn.metrics.pairwise import cosine_similarity
os.environ['CUDA_VISIBLE_DEVICES']= '3'


def make_layers(cfg, batch_norm=True):
    layers = []
    in_channels = 3
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)

def rle2mask(mask_rle, shape): # height, width
    starts, lengths = [np.asarray(x, dtype=int) for x in (mask_rle[0:][::2], mask_rle[1:][::2])]
    starts -= 1
    ends = starts + lengths
    binary_mask = np.zeros(shape[0] * shape[1], dtype=np.uint8)
    for lo, hi in zip(starts, ends):
        binary_mask[lo:hi] = 1
    return binary_mask.reshape(shape)

class SCT(nn.Module):
    def __init__(self, encoder):
        super(SCT, self).__init__()
        enc_layers = list(encoder.children())
        self.enc_1 = nn.Sequential(*enc_layers[:6])  # input -> relu1_1
        self.enc_2 = nn.Sequential(*enc_layers[6:13])  # relu1_1 -> relu2_1
        self.enc_3 = nn.Sequential(*enc_layers[13:20])  # relu2_1 -> relu3_1
        self.enc_4 = nn.Sequential(*enc_layers[20:33])  # relu3_1 -> relu4_1
        self.enc_5 = nn.Sequential(*enc_layers[33:46])  # relu4_1 -> relu5_1
        self.enc_6 = nn.Sequential(*enc_layers[46:70])  # relu5_1 -> maxpool
        self.relu = nn.ReLU(True)
        self.end_layer = 6

        self.conv1x1_5 = nn.Conv2d(1024, 512, kernel_size=1, stride=1, bias=True)
        
        self.projector5 = nn.Sequential(
            nn.Linear(512, 1024),
            nn.ReLU(True),
            nn.Linear(1024, 1024),
            nn.ReLU(True),
            nn.Linear(1024, 1024),
        )
        self.relu = nn.ReLU(True)
        self.transforms_img = transforms.Compose([
            transforms.ToPILImage(),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ])

    def encode_with_intermediate(self, input):
        results = [input]
        for i in range(self.end_layer):
            func = getattr(self, 'enc_{:d}'.format(i + 1))
            results.append(func(results[-1]))
        return results[1:]
    
    def forward(self, img, mask_image):
        cropped_image, random_cropped_image = self.process_image(img, mask_image)
        cropped_image = self.transforms_img(cropped_image).unsqueeze(0).cuda()
        random_cropped_image = self.transforms_img(random_cropped_image).unsqueeze(0).cuda()
        x1 = self.get_feature(cropped_image)
        x2 = self.get_feature(random_cropped_image)

        similarity = self.cosine_similarity(x1, x2)
        return similarity

    def get_feature(self, img):
        f_q = self.encode_with_intermediate(img)[-1]
        gap_q = torch.nn.functional.adaptive_avg_pool2d(f_q, (1,1))
        gmp_q = torch.nn.functional.adaptive_max_pool2d(f_q, (1,1))            
        feature = torch.cat([gap_q, gmp_q], 1)
 
        if feature.dtype != self.conv1x1_5.weight.dtype:
            feature = feature.to(self.conv1x1_5.weight.dtype)
        feature = self.relu(self.conv1x1_5(feature))
        feature = feature.view(feature.size(0), -1)
        feature = self.projector5(feature).view(feature.size(0), -1)
        return feature
    
    def cosine_similarity(self, x1, x2):
        return F.cosine_similarity(x1, x2)

    def process_image(self, image_path, mask_path):
        # 加载图像和掩码
        image = Image.open(image_path).convert('RGB')

        
        mask_np = cv2.resize(mask_path, image.size)

        # 将图像转换为numpy数组
        image_np = np.array(image)

        # 与mask直接相乘
        masked_image = image_np * mask_np[:, :, np.newaxis]

        # 与(1-mask)相乘
        inverse_masked_image = image_np * (1 - mask_np)[:, :, np.newaxis]

        # 确保结果在0-255范围内并转换为uint8
        masked_image = np.clip(masked_image, 0, 255).astype(np.uint8)
        inverse_masked_image = np.clip(inverse_masked_image, 0, 255).astype(np.uint8)
        
        # 转换为图像
        # masked_image = Image.fromarray(np.uint8(masked_image))
        # inverse_masked_image = Image.fromarray(np.uint8(inverse_masked_image))

        return masked_image, inverse_masked_image

    # # 裁剪为224
    # def process_image(self, image_path, mask_path):
    #     # 加载图像和掩码
    #     image = Image.open(image_path).convert('RGB')
    #     mask = Image.open(mask_path).convert('L')  # 将掩码转换为灰度图像
    #     mask = mask.resize(image.size, Image.ANTIALIAS)

    #     image_np = np.array(image)
    #     mask_np = np.array(mask)

    #     # 查找掩码的边界框
    #     coords = np.column_stack(np.where(mask_np > 0))
    #     y_min, x_min = coords.min(axis=0)
    #     y_max, x_max = coords.max(axis=0)

    #     # 计算边界框中心
    #     center_y = (y_min + y_max) // 2
    #     center_x = (x_min + x_max) // 2

    #     # 计算新的边界框坐标，确保其大小为 224x224
    #     half_size = 112  # 224 的一半
    #     new_y_min = max(0, center_y - half_size)
    #     new_x_min = max(0, center_x - half_size)
    #     new_y_max = min(image_np.shape[0], center_y + half_size)
    #     new_x_max = min(image_np.shape[1], center_x + half_size)

    #     # 确保边界框的大小为 224x224，如果图像边界限制了边界框大小，则调整
    #     if new_y_max - new_y_min < 224:
    #         if new_y_min == 0:
    #             new_y_max = min(image_np.shape[0], new_y_min + 224)
    #         else:
    #             new_y_min = max(0, new_y_max - 224)

    #     if new_x_max - new_x_min < 224:
    #         if new_x_min == 0:
    #             new_x_max = min(image_np.shape[1], new_x_min + 224)
    #         else:
    #             new_x_min = max(0, new_x_max - 224)

    #     # 裁剪224x224的矩形框
    #     cropped_region = image_np[new_y_min:new_y_max, new_x_min:new_x_max]

    #     # 定义要避开的随机裁剪区域
    #     avoid_region = (new_y_min, new_y_max, new_x_min, new_x_max)

    #     # Get random crop that doesn't overlap with the mask region
    #     random_crop = self.get_random_crop(image_np, cropped_region.shape, avoid_region)

    #     return cropped_region, random_crop

    def get_random_crop(self, image, crop_size, avoid_region):
        y_min_avoid, y_max_avoid, x_min_avoid, x_max_avoid = avoid_region
        crop_height, crop_width, _ = crop_size

        max_y = image.shape[0] - crop_height
        max_x = image.shape[1] - crop_width

        # 定义四个对角区域
        regions = [
            (0, 0, max_y, max_x),  # Top-left region
            (0, x_max_avoid, max_y, image.shape[1] - crop_width),  # Top-right region
            (y_max_avoid, 0, image.shape[0] - crop_height, max_x),  # Bottom-left region
            (y_max_avoid, x_max_avoid, image.shape[0] - crop_height, image.shape[1] - crop_width)  # Bottom-right region
        ]

        valid_regions = []
        for top_min, left_min, top_max, left_max in regions:
            if top_max > top_min and left_max > left_min:
                valid_regions.append((top_min, left_min, top_max, left_max))

        if not valid_regions:
            raise ValueError("No valid regions found to avoid the specified avoid_region.")

        top_min, left_min, top_max, left_max = valid_regions[np.random.randint(len(valid_regions))]

        top = np.random.randint(top_min, top_max + 1)
        left = np.random.randint(left_min, left_max + 1)

        random_crop = image[top:top + crop_height, left:left + crop_width]
        return random_crop



class MetricsCalculator:
    def __init__(self, device,ckpt_path="/data1/JM/code/BrushNet/pretrain_model") -> None:
        self.device=device
        # clip
        self.clip_metric_calculator = CLIPScore(model_name_or_path=f"{ckpt_path}/models--openai--clip-vit-large-patch14").to(device)
        # lpips
        self.lpips_metric_calculator = LearnedPerceptualImagePatchSimilarity(net_type='squeeze').to(device)
        # aesthetic model
        self.aesthetic_model = torch.nn.Linear(768, 1)
        aesthetic_model_ckpt_path=os.path.join(ckpt_path,"sa_0_4_vit_l_14_linear.pth")
        self.aesthetic_model.load_state_dict(torch.load(aesthetic_model_ckpt_path))
        self.aesthetic_model.eval()
        self.clip_model, _, self.clip_preprocess = open_clip.create_model_and_transforms('ViT-L-14', pretrained='openai')
        # # image reward model
        self.imagereward_model = RM.load(name=f"{ckpt_path}/ImageReward.pt", med_config=f'{ckpt_path}/med_config.json')
        vgg = make_layers([3, 64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 
                512, 512, 512, 512, 'M', 512, 512, 'M', 512, 512, 'M'])
        self.SCT = SCT(encoder=vgg).cuda()
        self.SCT.load_state_dict(torch.load(f'/data1/JM/code/BrushNet/scene_style_learning/ckpt/model_epoch_600.pth'))

    def calculate_image_reward(self,image,prompt):
        reward = self.imagereward_model.score(prompt, [image])
        return reward

    def calculate_hpsv21_score(self,image,prompt):
        result = hpsv2.score(image, prompt, hps_version="v2.1")[0]
        return result.item()

    def calculate_aesthetic_score(self,img):
        image = self.clip_preprocess(img).unsqueeze(0)
        with torch.no_grad():
            image_features = self.clip_model.encode_image(image)
            image_features /= image_features.norm(dim=-1, keepdim=True)
            prediction = self.aesthetic_model(image_features)
        return prediction.cpu().item()

    def calculate_clip_similarity(self, img, txt):
        img = np.array(img)
        
        img_tensor=torch.tensor(img).permute(2,0,1).to(self.device)
        
        score = self.clip_metric_calculator(img_tensor, txt)
        score = score.cpu().item()
        
        return score
    
    def calculate_psnr(self, img_pred, img_gt, mask=None):
        img_pred = np.array(img_pred).astype(np.float32)/255.
        img_gt = np.array(img_gt).astype(np.float32)/255.

        assert img_pred.shape == img_gt.shape, "Image shapes should be the same."
        if mask is not None:
            mask = np.array(mask).astype(np.float32)
            img_pred = img_pred * mask
            img_gt = img_gt * mask

        difference = img_pred - img_gt
        difference_square = difference ** 2
        difference_square_sum = difference_square.sum()
        difference_size = mask.sum()

        mse = difference_square_sum/difference_size

        if mse < 1.0e-10:
            return 1000
        PIXEL_MAX = 1
        return 20 * math.log10(PIXEL_MAX / math.sqrt(mse))
    
    def calculate_style_consistent(self, img, img_mask):
        return self.SCT(img, img_mask).item()

    # def calculate_human_perception(self, img_pred, img_gt, mask=None):
    #     img_pred = np.array(img_pred).astype(np.float32)/255.
    #     img_gt = np.array(img_gt).astype(np.float32)/255.

    #     assert img_pred.shape == img_gt.shape, "Image shapes should be the same."
    #     if mask is not None:
    #         mask = np.array(mask).astype(np.float32)
    #         img_pred = img_pred * mask
    #         img_gt = img_gt * mask

    #     difference = img_pred - img_gt
    #     difference_square = difference ** 2
    #     difference_square_sum = difference_square.sum()
    #     difference_size = mask.sum()

    #     mse = difference_square_sum/difference_size

    #     return mse.item()

    def calculate_human_perception(self, img_pred, img_gt, mask=None):
        img_pred = np.array(img_pred).astype(np.float32) / 255.0
        img_gt = np.array(img_gt).astype(np.float32) / 255.0

        assert img_pred.shape == img_gt.shape, "Image shapes should be the same."

        if mask is not None:
            mask = np.array(mask).astype(np.float32)
            img_pred = img_pred * mask
            img_gt = img_gt * mask
        img_pred_flat = img_pred.flatten()
        img_gt_flat = img_gt.flatten()

        similarity = cosine_similarity([img_pred_flat], [img_gt_flat])

        return similarity[0][0]

    def calculate_lpips(self, img_gt, img_pred, mask=None):
        img_pred = np.array(img_pred).astype(np.float32)/255
        img_gt = np.array(img_gt).astype(np.float32)/255
        assert img_pred.shape == img_gt.shape, "Image shapes should be the same."

        if mask is not None:
            mask = np.array(mask).astype(np.float32)
            img_pred = img_pred * mask 
            img_gt = img_gt * mask
            
        img_pred_tensor=torch.tensor(img_pred).permute(2,0,1).unsqueeze(0).to(self.device)
        img_gt_tensor=torch.tensor(img_gt).permute(2,0,1).unsqueeze(0).to(self.device)
            
        score =  self.lpips_metric_calculator(img_pred_tensor*2-1,img_gt_tensor*2-1)
        score = score.cpu().item()
        
        return score
    
    def calculate_mse(self, img_pred, img_gt, mask=None):
        img_pred = np.array(img_pred).astype(np.float32)/255.
        img_gt = np.array(img_gt).astype(np.float32)/255.

        assert img_pred.shape == img_gt.shape, "Image shapes should be the same."
        if mask is not None:
            mask = np.array(mask).astype(np.float32)
            img_pred = img_pred * mask
            img_gt = img_gt * mask
        
        difference = img_pred - img_gt
        difference_square = difference ** 2
        difference_square_sum = difference_square.sum()
        difference_size = mask.sum()

        mse = difference_square_sum / difference_size

        return mse.item()


parser = argparse.ArgumentParser()
parser.add_argument('--brushnet_ckpt_path', 
                    type=str, 
                    default="/data1/JM/code/BrushNet/pretrain_model/segmentation_mask_brushnet_ckpt")
parser.add_argument('--base_model_path', 
                    type=str, 
                    default="/data1/JM/code/BrushNet/pretrain_model/stable-diffusion-v1-5")
parser.add_argument('--image_save_path', 
                    type=str, 
                    default="/data1/JM/code/BrushNet/data/baseline-brushnet")
parser.add_argument('--paintingnet_conditioning_scale', type=float,default=1.0)
parser.add_argument('--data_root', type=str, default='/data1/JM/code/BrushNet/data')
parser.add_argument('--data_name', type=str, default='baseline-brushnet')

args = parser.parse_args()

device = "cuda" if torch.cuda.is_available() else "cpu"

# evaluation
evaluation_df = pd.DataFrame(columns=['Image ID','Image Reward', 'HPS V2.1', 'Aesthetic Score', 'PSNR', 'LPIPS', 'MSE', 'CLIP Similarity', 'SS', 'HP'])

metrics_calculator=MetricsCalculator(device)

with open('/data1/JM/code/BrushNet/data/BrushDench/mapping_file.json',"r") as f:
    mapping_file=json.load(f)

for key, item in mapping_file.items():
    print(f"generating image {key} ...")
    image_path=item["image"]
    mask=item['inpainting_mask']
    prompt=item["caption"]
   
    init_image = cv2.imread(os.path.join('/data1/JM/code/BrushNet/data/BrushDench', image_path))[:,:,::-1]
    result_image = cv2.imread(os.path.join('/data1/JM/code/BrushNet/data/BrushDench', image_path.replace('images', 'result')))[:,:,::-1]
    mask = rle2mask(mask,(512,512))
    mask_ori = mask[:,:,np.newaxis].copy()
    mask = 1 - mask[:,:,np.newaxis]

    src_image = Image.fromarray(init_image).convert("RGB")
    tgt_image = Image.fromarray(result_image).convert("RGB")

    evaluation_result=[key]

    for metric in evaluation_df.columns.values.tolist()[1:]:

        if metric == 'Image Reward':
            metric_result = metrics_calculator.calculate_image_reward(tgt_image,prompt)
            
        if metric == 'HPS V2.1':
            metric_result = metrics_calculator.calculate_hpsv21_score(tgt_image,prompt)
        
        if metric == 'Aesthetic Score':
            metric_result = metrics_calculator.calculate_aesthetic_score(tgt_image)
        
        if metric == 'PSNR':
            metric_result = metrics_calculator.calculate_psnr(src_image, tgt_image, mask)
        
        if metric == 'LPIPS':
            metric_result = metrics_calculator.calculate_lpips(src_image, tgt_image, mask)
        
        if metric == 'MSE':
            metric_result = metrics_calculator.calculate_mse(src_image, tgt_image, mask)
        
        if metric == 'CLIP Similarity':
            metric_result = metrics_calculator.calculate_clip_similarity(tgt_image, prompt)

        if metric == 'SS':
            metric_result = metrics_calculator.calculate_style_consistent(os.path.join('/data1/JM/code/BrushNet/data/BrushDench', image_path.replace('images', 'result')), mask)

        if metric == 'HP':
            metric_result = metrics_calculator.calculate_human_perception(src_image, tgt_image, mask_ori)

        print(f"evluating metric: {metric}: {metric_result}")

        evaluation_result.append(metric_result)
    
    evaluation_df.loc[len(evaluation_df.index)] = evaluation_result

print("The averaged evaluation result:")
averaged_results=evaluation_df.mean(numeric_only=True)
print(averaged_results)

averaged_results.to_csv(os.path.join('/data1/JM/code/BrushNet/data/BrushDench', "evaluation_result_sum_SS.csv"))
evaluation_df.to_csv(os.path.join('/data1/JM/code/BrushNet/data/BrushDench', "evaluation_result_SS.csv"))


# Residential floor plans: Multi-conditional automatic generation using diffusion models