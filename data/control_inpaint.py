from diffusers import StableDiffusionControlNetInpaintPipeline, ControlNetModel, UniPCMultistepScheduler
from diffusers.utils import load_image
import cv2
from PIL import Image  
import numpy as np
import torch
import os
import json


def rle2mask(mask_rle, shape): # height, width
    starts, lengths = [np.asarray(x, dtype=int) for x in (mask_rle[0:][::2], mask_rle[1:][::2])]
    starts -= 1
    ends = starts + lengths
    binary_mask = np.zeros(shape[0] * shape[1], dtype=np.uint8)
    for lo, hi in zip(starts, ends):
        binary_mask[lo:hi] = 1
    return binary_mask.reshape(shape)


# load control net and stable diffusion v1-5
controlnet = ControlNetModel.from_pretrained("/data1/JM/code/BrushNet/pretrain_model/models--lllyasviel--sd-controlnet-canny", torch_dtype=torch.float16).to('cuda')
pipe = StableDiffusionControlNetInpaintPipeline.from_pretrained(
     "/data1/JM/code/BrushNet/pretrain_model/models--runwayml--stable-diffusion-inpainting", controlnet=controlnet, torch_dtype=torch.float16, low_cpu_mem_usage=False
 ).to('cuda')

# speed up diffusion process with faster scheduler and memory optimization
pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
# remove following line if xformers is not installed

image_root = '/data1/JM/code/BrushNet/data/visual_set/image'
save_root = '/data1/JM/code/BrushNet/data/visual_set/baseline_CNI_finetune'
image_list = sorted(os.listdir(image_root))


# image_list = [os.path.basename(i) for i in temp_list]
# for img in image_list:
#     os.makedirs(save_root, exist_ok=True)

#     image_path = os.path.join(image_root, img)
#     mask_path = image_path.replace('image', 'mask')
#     save_path = image_path.replace('image', 'baseline_CNI_finetune')
#     caption = f"Delicate ceilings and chair and sofa. "

#     # Open images
#     image = Image.open(image_path)
#     mask_image = Image.open(mask_path)

#     # Get original dimensions
#     original_width, original_height = image.size
#     mask_width, mask_height = mask_image.size

#     # Define new sizes for downsampling
#     new_size = (original_width // 2, original_height // 2)
#     new_mask_size = (mask_width // 2, mask_height // 2)

#     # Downsample images
#     downsampled_image = image.resize(new_size, Image.ANTIALIAS)
#     downsampled_mask_image = mask_image.resize(new_mask_size, Image.ANTIALIAS)

with open('/data1/JM/code/BrushNet/data/BrushDench/mapping_file.json', "r") as f:
    mapping_file=json.load(f)

for key, item in mapping_file.items():
    print(f"generating image {key} ...")
    image_path=item["image"]
    mask=item['inpainting_mask']
    caption=item["caption"]

    init_image = cv2.imread(os.path.join('/data1/JM/code/BrushNet/data/BrushDench', image_path))[:,:,::-1]
    mask_image = rle2mask(mask,(512,512))[:,:,np.newaxis]
    mask_image = Image.fromarray(mask_image.repeat(3,-1)*255).convert("RGB")
    save_path= os.path.join('/data1/JM/code/BrushNet/data/BrushDench/mask', os.path.basename(image_path)) 
    mask_image.save(save_path)

    # init_image = init_image * (1-mask_image)

    # init_image = Image.fromarray(init_image).convert("RGB")
    # mask_image = Image.fromarray(mask_image.repeat(3,-1)*255).convert("RGB")
    # save_path= os.path.join('/data1/JM/code/BrushNet/data/BrushDench/result_CNI', os.path.basename(image_path)) 

    # # Convert to numpy arrays
    # downsampled_image_np = np.array(init_image)
    # downsampled_mask_np = np.array(mask_image)

    # canny_image = cv2.Canny(downsampled_image_np, 100, 200)
    # canny_image = canny_image[:, :, None]
    # canny_image = np.concatenate([canny_image, canny_image, canny_image], axis=2)
    # canny_image = Image.fromarray(canny_image)
    
    # # if os.path.exists(save_path):
    # #     print(f'{save_path} exist continue')
    # #     continue
    # generator = torch.manual_seed(777777)
    # new_image = pipe(
    #     caption,
    #     num_inference_steps=50,
    #     image=init_image,
    #     control_image=canny_image,
    #     mask_image=mask_image,
    #     generator=generator
    # ).images[0]
    # new_image.save(save_path)