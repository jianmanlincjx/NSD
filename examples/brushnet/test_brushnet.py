from diffusers import StableDiffusionBrushNetPipeline, BrushNetModel, UniPCMultistepScheduler
import torch
import cv2
import numpy as np
from PIL import Image
import os
import json

base_model_path = "/data1/JM/code/BrushNet/pretrain_model/stable-diffusion-v1-5"
brushnet_path = "/data1/JM/code/BrushNet/pretrain_model/segmentation_mask_brushnet_ckpt"


def rle2mask(mask_rle, shape): # height, width
    starts, lengths = [np.asarray(x, dtype=int) for x in (mask_rle[0:][::2], mask_rle[1:][::2])]
    starts -= 1
    ends = starts + lengths
    binary_mask = np.zeros(shape[0] * shape[1], dtype=np.uint8)
    for lo, hi in zip(starts, ends):
        binary_mask[lo:hi] = 1
    return binary_mask.reshape(shape)
# init model
brushnet_conditioning_scale=1.0

brushnet = BrushNetModel.from_pretrained(brushnet_path, torch_dtype=torch.float16)
pipe = StableDiffusionBrushNetPipeline.from_pretrained(
    base_model_path, brushnet=brushnet, torch_dtype=torch.float16, low_cpu_mem_usage=False
)
pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
pipe.enable_model_cpu_offload()

##################################################################################################
##################################################################################################
##################################################################################################
with open('/data1/JM/code/BrushNet/data/BrushDench/mapping_file.json', "r") as f:
    mapping_file=json.load(f)

for key, item in mapping_file.items():
    print(f"generating image {key} ...")
    image_path=item["image"]
    mask=item['inpainting_mask']
    caption=item["caption"]

    init_image = cv2.imread(os.path.join('/data1/JM/code/BrushNet/data/BrushDench', image_path))[:,:,::-1]
    mask_image = rle2mask(mask,(512,512))[:,:,np.newaxis]
    init_image = init_image * (1-mask_image)

    init_image = Image.fromarray(init_image).convert("RGB")
    mask_image = Image.fromarray(mask_image.repeat(3,-1)*255).convert("RGB")

    generator = torch.Generator('cuda').manual_seed(7777)

    save_path= os.path.join('/data1/JM/code/BrushNet/data/BrushDench/result_ours', os.path.basename(image_path)) 

    generator = torch.Generator("cuda").manual_seed(1234)
    image = pipe(
        image=init_image, 
        mask=mask_image, 
        prompt=caption,
        num_inference_steps=50, 
        generator=generator,
        brushnet_conditioning_scale=1.0,
        guidance_scale=7.5
    ).images[0]
    image.save(save_path)