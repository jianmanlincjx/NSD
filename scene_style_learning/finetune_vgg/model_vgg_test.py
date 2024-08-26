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
    
    def forward(self, img):
        f_q = self.encode_with_intermediate(img)[-1]
        f_q_mean, f_q_std = self.calc_mean_std(f_q)
        return f_q_mean, f_q_std

    def calc_mean_std(self, feat, eps=1e-5):
        # eps is a small value added to the variance to avoid divide-by-zero.
        size = feat.size()
        assert (len(size) == 4)
        N, C = size[:2]
        feat_var = feat.view(N, C, -1).var(dim=2) + eps
        feat_std = feat_var.sqrt().view(N, C, 1, 1)
        feat_mean = feat.view(N, C, -1).mean(dim=2).view(N, C, 1, 1)
        return feat_mean.reshape(N, C), feat_std.reshape(N, C)









