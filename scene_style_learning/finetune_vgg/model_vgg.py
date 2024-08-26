import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
sys.path.append(os.getcwd())
from torchvision.utils import save_image
import numpy as np


class SCT(nn.Module):
    def __init__(self, net):
        super(SCT, self).__init__()
        self.net = net
        enc_layers = list(net.children())
        self.enc_1 = nn.Sequential(*enc_layers[:4])  # input -> relu1_1
        self.enc_2 = nn.Sequential(*enc_layers[4:11])  # relu1_1 -> relu2_1
        self.enc_3 = nn.Sequential(*enc_layers[11:18])  # relu2_1 -> relu3_1
        self.enc_4 = nn.Sequential(*enc_layers[18:31])  # relu3_1 -> relu4_1
        self.end_layer = 4      

    def encode_with_intermediate(self, input):
        results = [input]
        for i in range(self.end_layer):
            func = getattr(self, 'enc_{:d}'.format(i + 1))
            results.append(func(results[-1]))
        return results[1:]
    
    def forward(self, img_one, img_two):
        loss_style_mean = torch.tensor(0).cuda().float()
        loss_style_std = torch.tensor(0).cuda().float()
        img_one_feat = self.encode_with_intermediate(img_one)
        img_two_feat = self.encode_with_intermediate(img_two)

        for i in range(0, self.end_layer):
            f_q = img_one_feat[i]
            f_k = img_two_feat[i]
            f_q_mean, f_q_std = self.calc_mean_std(f_q)
            f_k_mean, f_k_std = self.calc_mean_std(f_k)
            loss_style_mean += F.mse_loss(f_q_mean, f_k_mean)
            loss_style_std += F.mse_loss(f_q_std, f_k_std)

        return {'total_loss': (loss_style_mean + loss_style_std)}


    def calc_mean_std(self, feat, eps=1e-5):
        # eps is a small value added to the variance to avoid divide-by-zero.
        size = feat.size()
        assert (len(size) == 4)
        N, C = size[:2]
        feat_var = feat.view(N, C, -1).var(dim=2) + eps
        feat_std = feat_var.sqrt().view(N, C, 1, 1)
        feat_mean = feat.view(N, C, -1).mean(dim=2).view(N, C, 1, 1)
        return feat_mean.reshape(N, C), feat_std.reshape(N, C)









