#!/usr/bin/env python
# coding=utf-8
import argparse
import cv2
import os
import imgaug.augmenters as iaa
import torch.nn as nn
import numpy as np
import torch
import torch.utils.checkpoint
from transformers import CLIPVisionModelWithProjection, CLIPImageProcessor
from PIL import Image
from diffusers import (
    AutoencoderKL,
    BrushNetModel,
    DDPMScheduler,
    StableDiffusionBrushNetPipeline,
    UNet2DConditionModel,
    UniPCMultistepScheduler,
)
import json

def rle2mask(mask_rle, shape): # height, width
    starts, lengths = [np.asarray(x, dtype=int) for x in (mask_rle[0:][::2], mask_rle[1:][::2])]
    starts -= 1
    ends = starts + lengths
    binary_mask = np.zeros(shape[0] * shape[1], dtype=np.uint8)
    for lo, hi in zip(starts, ends):
        binary_mask[lo:hi] = 1
    return binary_mask.reshape(shape)

class AttnProcessor(nn.Module):
    r"""
    Default processor for performing attention-related computations.
    """

    def __init__(
        self,
        hidden_size=None,
        cross_attention_dim=None,
    ):
        super().__init__()

    def __call__(
        self,
        attn,
        hidden_states,
        encoder_hidden_states=None,
        attention_mask=None,
        temb=None,
        *args,
        **kwargs,
    ):
        residual = hidden_states

        if attn.spatial_norm is not None:
            hidden_states = attn.spatial_norm(hidden_states, temb)

        input_ndim = hidden_states.ndim

        if input_ndim == 4:
            batch_size, channel, height, width = hidden_states.shape
            hidden_states = hidden_states.view(batch_size, channel, height * width).transpose(1, 2)

        batch_size, sequence_length, _ = (
            hidden_states.shape if encoder_hidden_states is None else encoder_hidden_states.shape
        )
        attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length, batch_size)

        if attn.group_norm is not None:
            hidden_states = attn.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)

        query = attn.to_q(hidden_states)

        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states
        elif attn.norm_cross:
            encoder_hidden_states = attn.norm_encoder_hidden_states(encoder_hidden_states)

        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)

        query = attn.head_to_batch_dim(query)
        key = attn.head_to_batch_dim(key)
        value = attn.head_to_batch_dim(value)

        attention_probs = attn.get_attention_scores(query, key, attention_mask)
        hidden_states = torch.bmm(attention_probs, value)
        hidden_states = attn.batch_to_head_dim(hidden_states)

        # linear proj
        hidden_states = attn.to_out[0](hidden_states)
        # dropout
        hidden_states = attn.to_out[1](hidden_states)

        if input_ndim == 4:
            hidden_states = hidden_states.transpose(-1, -2).reshape(batch_size, channel, height, width)

        if attn.residual_connection:
            hidden_states = hidden_states + residual

        hidden_states = hidden_states / attn.rescale_output_factor

        return hidden_states


class IPAttnProcessor(nn.Module):
    r"""
    Attention processor for IP-Adapater.
    Args:
        hidden_size (`int`):
            The hidden size of the attention layer.
        cross_attention_dim (`int`):
            The number of channels in the `encoder_hidden_states`.
        scale (`float`, defaults to 1.0):
            the weight scale of image prompt.
        num_tokens (`int`, defaults to 4 when do ip_adapter_plus it should be 16):
            The context length of the image features.
    """

    def __init__(self, hidden_size, cross_attention_dim=None, scale=0, num_tokens=4):
        super().__init__()

        self.hidden_size = hidden_size
        self.cross_attention_dim = cross_attention_dim
        self.scale = scale
        self.num_tokens = num_tokens

        self.to_k_ip = nn.Linear(cross_attention_dim or hidden_size, hidden_size, bias=False)
        self.to_v_ip = nn.Linear(cross_attention_dim or hidden_size, hidden_size, bias=False)

    def __call__(
        self,
        attn,
        hidden_states,
        encoder_hidden_states=None,
        attention_mask=None,
        temb=None,
        *args,
        **kwargs,
    ):
        residual = hidden_states

        if attn.spatial_norm is not None:
            hidden_states = attn.spatial_norm(hidden_states, temb)

        input_ndim = hidden_states.ndim

        if input_ndim == 4:
            batch_size, channel, height, width = hidden_states.shape
            hidden_states = hidden_states.view(batch_size, channel, height * width).transpose(1, 2)

        batch_size, sequence_length, _ = (
            hidden_states.shape if encoder_hidden_states is None else encoder_hidden_states.shape
        )
        attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length, batch_size)

        if attn.group_norm is not None:
            hidden_states = attn.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)

        query = attn.to_q(hidden_states)

        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states
        else:
            # get encoder_hidden_states, ip_hidden_states
            end_pos = encoder_hidden_states.shape[1] - self.num_tokens
            encoder_hidden_states, ip_hidden_states = (
                encoder_hidden_states[:, :end_pos, :],
                encoder_hidden_states[:, end_pos:, :],
            )
            if attn.norm_cross:
                encoder_hidden_states = attn.norm_encoder_hidden_states(encoder_hidden_states)

        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)

        query = attn.head_to_batch_dim(query)
        key = attn.head_to_batch_dim(key)
        value = attn.head_to_batch_dim(value)

        attention_probs = attn.get_attention_scores(query, key, attention_mask)
        hidden_states = torch.bmm(attention_probs, value)
        hidden_states = attn.batch_to_head_dim(hidden_states)

        # for ip-adapter
        ip_key = self.to_k_ip(ip_hidden_states)
        ip_value = self.to_v_ip(ip_hidden_states)

        ip_key = attn.head_to_batch_dim(ip_key)
        ip_value = attn.head_to_batch_dim(ip_value)

        ip_attention_probs = attn.get_attention_scores(query, ip_key, None)
        self.attn_map = ip_attention_probs
        ip_hidden_states = torch.bmm(ip_attention_probs, ip_value)
        ip_hidden_states = attn.batch_to_head_dim(ip_hidden_states)

        hidden_states = hidden_states + self.scale * ip_hidden_states 

        # linear proj
        hidden_states = attn.to_out[0](hidden_states)
        # dropout
        hidden_states = attn.to_out[1](hidden_states)

        if input_ndim == 4:
            hidden_states = hidden_states.transpose(-1, -2).reshape(batch_size, channel, height, width)

        if attn.residual_connection:
            hidden_states = hidden_states + residual

        hidden_states = hidden_states / attn.rescale_output_factor

        return hidden_states


class ImageProjModel(torch.nn.Module):
    """Projection Model"""

    def __init__(self, cross_attention_dim=768, clip_embeddings_dim=1024, clip_extra_context_tokens=4):
        super().__init__()

        self.generator = None
        self.cross_attention_dim = cross_attention_dim
        self.clip_extra_context_tokens = clip_extra_context_tokens
        self.proj = torch.nn.Linear(clip_embeddings_dim, self.clip_extra_context_tokens * cross_attention_dim)
        self.norm = torch.nn.LayerNorm(cross_attention_dim)

    def forward(self, image_embeds):

        embeds = image_embeds
        if embeds.dtype != self.proj.weight.dtype:
            embeds = embeds.to(self.proj.weight.dtype)
        clip_extra_context_tokens = self.proj(embeds).reshape(
            -1, self.clip_extra_context_tokens, self.cross_attention_dim
        )
        clip_extra_context_tokens = self.norm(clip_extra_context_tokens)
        return clip_extra_context_tokens


def rename_keys_in_dict(param_dict, prefix):
    # 初始化一个新的字典来存储重命名后的参数
    renamed_dict = {}
    
    # 遍历字典中的所有键并重命名
    for param_tensor in param_dict:
        # 移除指定的前缀
        if param_tensor.startswith(prefix + '.'):
            new_key = param_tensor[len(prefix) + 1:]
            renamed_dict[new_key] = param_dict[param_tensor]
        else:
            renamed_dict[param_tensor] = param_dict[param_tensor]
    
    return renamed_dict


if __name__ == "__main__":
    image_encoder_path = '/data1/JM/code/BrushNet/pretrain_model/image_encoder'
    base_model_path = "/data1/JM/code/BrushNet/pretrain_model/stable-diffusion-v1-5"
    brushnet_path = "/data1/JM/code/BrushNet/pretrain_model/segmentation_mask_brushnet_ckpt"
    linpro_path = '/data1/JM/code/BrushNet/runs/logs/NSD/checkpoint-300000/extracted_parameters/image_proj_model_parameters.pth'
    ip_adapter_path = '/data1/JM/code/BrushNet/runs/logs/NSD/checkpoint-300000/extracted_parameters/adapter_modules_parameters.pth'
    
    ##################################################################################################
    ##################################################################################################
    ##################################################################################################
    image_encoder = CLIPVisionModelWithProjection.from_pretrained(image_encoder_path).cuda().to(torch.float16)
    project = ImageProjModel()
    project_params = torch.load(linpro_path)
    project_params = rename_keys_in_dict(project_params, 'image_proj_model')
    project.load_state_dict(project_params, strict=True)
    project = project.cuda().to(torch.float16)

    vae = AutoencoderKL.from_pretrained(base_model_path, subfolder="vae", torch_dtype=torch.float16)
    unet = UNet2DConditionModel.from_pretrained(base_model_path, subfolder="unet", revision=None, variant=None, torch_dtype=torch.float16)
    brushnet = BrushNetModel.from_pretrained(brushnet_path, torch_dtype=torch.float16)

    attn_procs = {}
    unet_sd = unet.state_dict()
    for name in unet.attn_processors.keys():
        cross_attention_dim = None if name.endswith("attn1.processor") else unet.config.cross_attention_dim
        if name.startswith("mid_block"):
            hidden_size = unet.config.block_out_channels[-1]
        elif name.startswith("up_blocks"):
            block_id = int(name[len("up_blocks.")])
            hidden_size = list(reversed(unet.config.block_out_channels))[block_id]
        elif name.startswith("down_blocks"):
            block_id = int(name[len("down_blocks.")])
            hidden_size = unet.config.block_out_channels[block_id]
        if cross_attention_dim is None:
            attn_procs[name] = AttnProcessor().to(torch.float16)
        else:
            layer_name = name.split(".processor")[0]
            weights = {
                "to_k_ip.weight": unet_sd[layer_name + ".to_k.weight"],
                "to_v_ip.weight": unet_sd[layer_name + ".to_v.weight"],
            }
            attn_procs[name] = IPAttnProcessor(hidden_size=hidden_size, cross_attention_dim=cross_attention_dim).to(torch.float16)
            attn_procs[name].load_state_dict(weights)
    unet.set_attn_processor(attn_procs)

    pipeline = StableDiffusionBrushNetPipeline.from_pretrained(
        base_model_path,
        vae=vae,
        unet=unet,
        brushnet=brushnet,
        revision=None,
        variant=None,
        torch_dtype=torch.float16,
        low_cpu_mem_usage=False
    )
    
    ip_layers = torch.nn.ModuleList(pipeline.unet.attn_processors.values())
    ip_adapter_params = torch.load(ip_adapter_path)
    ip_adapter_params = rename_keys_in_dict(ip_adapter_params, 'adapter_modules')
    ip_layers.load_state_dict(ip_adapter_params, strict=True)

    pipeline.scheduler = UniPCMultistepScheduler.from_config(pipeline.scheduler.config)
    pipeline.enable_model_cpu_offload()

    blended = True
    # conditioning scale
    brushnet_conditioning_scale=1.0
    clip_procee = CLIPImageProcessor()



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

        generator = torch.Generator('cuda').manual_seed(1234)

        save_path= os.path.join('/data1/JM/code/BrushNet/data/BrushDench/result', os.path.basename(image_path)) 

        raw_image = Image.open(os.path.join('/data1/JM/code/BrushNet/data/BrushDench', image_path))
        clip_image = clip_procee(images=raw_image, return_tensors="pt").pixel_values.cuda().to(torch.float16)
        temp = image_encoder(clip_image).image_embeds
        style_code = project(temp)

        with torch.inference_mode():
            prompt_embeds_, negative_prompt_embeds_ = pipeline.encode_prompt(
                caption,
                device='cuda',
                num_images_per_prompt=1,
                do_classifier_free_guidance=True,
                negative_prompt='monochrome, lowres, bad anatomy, worst quality, low quality, black sofa',
            )
        prompt_embeds = torch.cat([prompt_embeds_, style_code], dim=1)
        negative_prompt_embeds = torch.cat([negative_prompt_embeds_, torch.zeros_like(style_code)], dim=1)


        generator = torch.Generator("cuda").manual_seed(1234)
        image = pipeline(
            image=init_image, 
            mask=mask_image, 
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            num_inference_steps=50, 
            generator=generator,
            brushnet_conditioning_scale=1.0,
            guidance_scale=7.5
        ).images[0]
        image.save(save_path)


    # root = '/data1/JM/code/BrushNet/data/Baseon_4K_dataset/baseline_ours'
    # vid_list = sorted(os.listdir(root))
    # vid_list = [i for i in vid_list if not i.endswith('.csv')]
    # for vid in vid_list:
    #     vid_path = os.path.join(root, vid, 'image')
    #     image_root = vid_path
    #     save_root = image_root.replace('image', 'result_new')
    #     image_list = sorted(os.listdir(image_root))


    #     for idx, img in enumerate(image_list):
    #         os.makedirs(save_root, exist_ok=True)

    #         image_path = os.path.join(image_root, img)
    #         mask_path = image_path.replace('image', 'mask')
    #         save_path = image_path.replace('image', 'result_new')
    #         # caption = f"Delicate ceilings and sofa. "
    #         # caption = f"Delicate ceilings and sofa. "
    #         caption = 'Exquisite decorations'
    #         raw_image = Image.open(image_path)
    #         clip_image = clip_procee(images=raw_image, return_tensors="pt").pixel_values.cuda().to(torch.float16)
    #         temp = image_encoder(clip_image).image_embeds
    #         style_code = project(temp)

    #         init_image = cv2.imread(image_path)[:,:,::-1]
    #         mask_image = 1.*(cv2.imread(mask_path).sum(-1)>255)[:,:,np.newaxis]
    #         init_image = init_image * (1-mask_image)
    #         init_image = Image.fromarray(init_image.astype(np.uint8)).convert("RGB")
    #         mask_image = Image.fromarray(mask_image.astype(np.uint8).repeat(3,-1)*255).convert("RGB")

    #         new_size = (init_image.size[0] // 2, init_image.size[1] // 2)
    #         init_image = init_image.resize(new_size, Image.ANTIALIAS)
    #         mask_image = mask_image.resize(new_size, Image.ANTIALIAS)


    #         with torch.inference_mode():
    #             prompt_embeds_, negative_prompt_embeds_ = pipeline.encode_prompt(
    #                 caption,
    #                 device='cuda',
    #                 num_images_per_prompt=1,
    #                 do_classifier_free_guidance=True,
    #                 negative_prompt='monochrome, lowres, bad anatomy, worst quality, low quality, black sofa',
    #             )
    #         prompt_embeds = torch.cat([prompt_embeds_, style_code], dim=1)
    #         negative_prompt_embeds = torch.cat([negative_prompt_embeds_, torch.zeros_like(style_code)], dim=1)


    #         generator = torch.Generator("cuda").manual_seed(777777)
    #         image = pipeline(
    #             image=init_image, 
    #             mask=mask_image, 
    #             prompt_embeds=prompt_embeds,
    #             negative_prompt_embeds=negative_prompt_embeds,
    #             num_inference_steps=50, 
    #             generator=generator,
    #             brushnet_conditioning_scale=1.0,
    #             guidance_scale=7.5
    #         ).images[0]
    #         image.save(save_path)

