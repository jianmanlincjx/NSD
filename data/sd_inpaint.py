import os
import torch
from PIL import Image
from glob import glob

from diffusers import StableDiffusionInpaintPipeline
from diffusers.utils import load_image
import cv2
import numpy as np
import json

def rle2mask(mask_rle, shape): # height, width
    starts, lengths = [np.asarray(x, dtype=int) for x in (mask_rle[0:][::2], mask_rle[1:][::2])]
    starts -= 1
    ends = starts + lengths
    binary_mask = np.zeros(shape[0] * shape[1], dtype=np.uint8)
    for lo, hi in zip(starts, ends):
        binary_mask[lo:hi] = 1
    return binary_mask.reshape(shape)

if __name__ == "__main__":

    pipe = StableDiffusionInpaintPipeline.from_pretrained(
        "/data1/JM/code/BrushNet/pretrain_model/models--runwayml--stable-diffusion-inpainting",
        revision="fp16",
        torch_dtype=torch.float16, low_cpu_mem_usage=False
    ).to('cuda')

    image_root = '/data1/JM/code/BrushNet/data/visual_set/image'
    save_root = '/data1/JM/code/BrushNet/data/visual_set/baseline_SDI'
    image_list = sorted(os.listdir(image_root))

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
        save_path= os.path.join('/data1/JM/code/BrushNet/data/BrushDench/result_SDI', os.path.basename(image_path)) 

        generator = torch.manual_seed(7777)

        # Get original dimensions
        original_width, original_height = init_image.size
        mask_width, mask_height = mask_image.size

        try:
            image = pipe(prompt=caption, image=init_image, mask_image=mask_image).images[0]
            image.save(save_path)
        except Exception as e:
            pass
  

