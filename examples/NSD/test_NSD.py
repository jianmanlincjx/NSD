#!/usr/bin/env python
# coding=utf-8
import itertools
import argparse
import contextlib
import functools
import gc
import logging
import math
import os
import random
import shutil
from pathlib import Path
import json
import cv2
import imgaug.augmenters as iaa
import torch.nn as nn
import accelerate
import numpy as np
import torch
import torch.nn.functional as F
import torch.utils.checkpoint
from torch.utils.data import Dataset
import transformers
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration, set_seed
from datasets import load_dataset
from huggingface_hub import create_repo, upload_folder
from packaging import version
from PIL import Image, ImageDraw
from torchvision import transforms
from tqdm.auto import tqdm
from transformers import AutoTokenizer, PretrainedConfig
import torchvision
from safetensors import safe_open

import diffusers
from diffusers import (
    AutoencoderKL,
    BrushNetModel,
    DDPMScheduler,
    StableDiffusionBrushNetPipeline,
    UNet2DConditionModel,
    UniPCMultistepScheduler,
)
from diffusers.optimization import get_scheduler
from diffusers.utils import check_min_version, is_wandb_available, make_image_grid
from diffusers.utils.hub_utils import load_or_create_model_card, populate_model_card
from diffusers.utils.import_utils import is_xformers_available
from diffusers.utils.torch_utils import is_compiled_module

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

    def __init__(self, hidden_size, cross_attention_dim=None, scale=1.0, num_tokens=4):
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


class LinerProjection(nn.Module):
    def __init__(self, cross_attention_dim=768, clip_extra_context_tokens=4, clip_embeddings_dim=1024):
        super(LinerProjection, self).__init__()
        self.cross_attention_dim = cross_attention_dim
        self.clip_extra_context_tokens = clip_extra_context_tokens
        self.proj = torch.nn.Linear(clip_embeddings_dim, self.clip_extra_context_tokens * cross_attention_dim)
        self.norm = torch.nn.LayerNorm(cross_attention_dim)
    
    def forward(self, feature):
        if feature.dtype != self.proj.weight.dtype:
            feature = feature.to(self.proj.weight.dtype)
        clip_extra_context_tokens = self.proj(feature).reshape(
            -1, self.clip_extra_context_tokens, self.cross_attention_dim
        )
        clip_extra_context_tokens = self.norm(clip_extra_context_tokens)

        return clip_extra_context_tokens


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
    
    def encode_with_intermediate(self, input):
        results = [input]
        for i in range(self.end_layer):
            func = getattr(self, 'enc_{:d}'.format(i + 1))
            results.append(func(results[-1]))
        return results[1:]
    
    def forward(self, img):
        
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
    vgg_path = '/data1/JM/BrushNet/pretrain_model/model_epoch_20.pth'
    base_model_path = "/data1/JM/BrushNet/pretrain_model/stable-diffusion-v1-5"
    brushnet_path = "/data1/JM/BrushNet/pretrain_model/segmentation_mask_brushnet_ckpt"
    linpro_path = '/data1/JM/BrushNet/runs/logs/brushnet_segmentationmask/checkpoint-1000000/extracted_parameters/linproject_parameters.pth'
    ip_adapter_path = '/data1/JM/BrushNet/runs/logs/brushnet_segmentationmask/checkpoint-1000000/extracted_parameters/adapter_modules_parameters.pth'

    ##################################################################################################
    ##################################################################################################
    ##################################################################################################
    vgg = make_layers([3, 64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 
            512, 512, 512, 512, 'M', 512, 512, 'M', 512, 512, 'M'])
    style_extract = SCT(vgg)
    style_extract.load_state_dict(torch.load(vgg_path), strict=True)
    style_extract = style_extract.cuda().to(torch.float16)

    project = LinerProjection()
    project_params = torch.load(linpro_path)
    project_params = rename_keys_in_dict(project_params, 'linproject')
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
    ##################################################################################################
    ##################################################################################################
    ##################################################################################################
    
    data_root = '/data1/JM/BrushNet/data'
    data_name = 'data_test_single_mask'
    vid_list = ['Bathroom', 'Bedroom', 'Cloakroom', 'Dining_room', 'Gym', 'Kitchen', 'Meeting_room', 'livingroom', 'playroom', 'studyroom']
    vid_list = vid_list[6:]
    for vid in vid_list:
        os.makedirs(os.path.join(data_root, data_name, vid, 'result'), exist_ok=True)
        txt_path = os.path.join(data_root, data_name, vid, 'record.txt')
        with open(txt_path, "r") as file:
            lines = file.readlines()
        for line in lines:
            image, mask, label = line.strip().split()
            image_path = os.path.join(data_root, image)
            mask_path = os.path.join(data_root, mask)
            save_path = image_path.replace('image', 'result')
            caption = f"Delicate {label}."

            image = cv2.imread(image_path)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = (image / 127.5) - 1.0
            pixel_values = torch.tensor(image).permute(2,0,1).unsqueeze(0).to(torch.float16).cuda()
            style_code = project(style_extract(pixel_values))

            init_image = cv2.imread(image_path)[:,:,::-1]
            mask_image = 1.*(cv2.imread(mask_path).sum(-1)>255)[:,:,np.newaxis]
            init_image = init_image * (1-mask_image)
            init_image = Image.fromarray(init_image.astype(np.uint8)).convert("RGB")
            mask_image = Image.fromarray(mask_image.astype(np.uint8).repeat(3,-1)*255).convert("RGB")

            new_size = (init_image.size[0] // 2, init_image.size[1] // 2)
            init_image = init_image.resize(new_size, Image.ANTIALIAS)
            mask_image = mask_image.resize(new_size, Image.ANTIALIAS)

            generator = torch.Generator("cuda").manual_seed(7777777)
            image = pipeline(
                caption, 
                init_image, 
                mask_image, 
                num_inference_steps=50, 
                generator=generator,
                brushnet_conditioning_scale=1.0,
                style_code=style_code
            ).images[0]
            image.save(save_path)

