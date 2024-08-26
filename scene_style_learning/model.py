import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
sys.path.append(os.getcwd())
from torchvision.utils import save_image
import numpy as np

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



class Normalize(nn.Module):

    def __init__(self, power=2):
        super(Normalize, self).__init__()
        self.power = power

    def forward(self, x):
        norm = x.pow(self.power).sum(1, keepdim=True).pow(1. / self.power)
        out = x.div(norm + 1e-7)
        return out

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
        # fix the encoder
        for name in ['enc_1', 'enc_2','enc_3', 'enc_4', 'enc_5', 'enc_6']:
            for param in getattr(self, name).parameters():
                param.requires_grad = False
        
        self.conv1x1_5 = nn.Conv2d(1024, 512, kernel_size=1, stride=1, bias=True)
        
        self.projector5 = nn.Sequential(
            nn.Linear(512, 1024),
            nn.ReLU(True),
            nn.Linear(1024, 1024),
            nn.ReLU(True),
            nn.Linear(1024, 1024),
        )

        self.Normalize = Normalize(2)

    def encode_with_intermediate(self, input):
        results = [input]
        for i in range(self.end_layer):
            func = getattr(self, 'enc_{:d}'.format(i + 1))
            results.append(func(results[-1]))
        return results[1:]
    
    def forward(self, img_one, img_two):
        
        loss_nce = torch.tensor(0).cuda().float()
        pos = 0
        neg = 0
        f_q = self.encode_with_intermediate(img_one)[-1]
        f_k = self.encode_with_intermediate(img_two)[-1]

        ## f_q
        gap_q = torch.nn.functional.adaptive_avg_pool2d(f_q, (1,1))
        gmp_q = torch.nn.functional.adaptive_max_pool2d(f_q, (1,1))            
        code_q = torch.cat([gap_q, gmp_q], 1)
        code_q = self.relu(self.conv1x1_5(code_q))
        code_q = code_q.view(code_q.size(0), -1)
        projection_q = self.projector5(code_q).view(code_q.size(0), -1)
        projection_q_normalize = self.Normalize(projection_q)

        # f_k
        gap_k = torch.nn.functional.adaptive_avg_pool2d(f_k, (1,1))
        gmp_k = torch.nn.functional.adaptive_max_pool2d(f_k, (1,1))            
        code_k = torch.cat([gap_k, gmp_k], 1)
        code_k = self.relu(self.conv1x1_5(code_k))
        code_k = code_k.view(code_k.size(0), -1)
        projection_k = self.projector5(code_k).view(code_k.size(0), -1)
        projection_k_normalize = self.Normalize(projection_k)

        # # inter-scene contrastive style loss
        loss_nce = self.compute_cosine_embedding_loss(projection_q_normalize, projection_k_normalize)
        pos_similarity, neg_similarity = self.compute_similarity(projection_q_normalize, projection_k_normalize)
        pos = pos_similarity
        neg = neg_similarity

        return {'total_loss': loss_nce, 
                'pos': pos , 
                'neg': neg }

    def compute_similarity(self, projection_q_normalize, projection_k_normalize):
        """
        计算正样本和负样本的相似度。

        Args:
        - projection_q_normalize (torch.Tensor): 形状为 (B, C) 的张量，表示查询样本的归一化特征。
        - projection_k_normalize (torch.Tensor): 形状为 (B, C) 的张量，表示键样本的归一化特征。

        Returns:
        - l_pos (torch.Tensor): 形状为 (B, 1) 的张量，表示正样本相似度。
        - l_neg (torch.Tensor): 形状为 (B, B) 的张量，表示负样本相似度。
        """
        # batch size and channel size
        B, C = projection_q_normalize.shape
        projection_k_normalize = projection_k_normalize.detach()

        # 计算正样本相似度: Bx1
        l_pos = (projection_q_normalize * projection_k_normalize).sum(dim=1, keepdim=True).mean()
        
        # 计算负样本相似度: BxB
        l_neg = torch.mm(projection_q_normalize, projection_k_normalize.t())
        
        # 对角线位置不是负样本，将其置为无穷小
        identity_matrix = torch.eye(B, dtype=torch.bool).to(projection_q_normalize.device)
        l_neg.masked_fill_(identity_matrix, -float('inf'))
        
        # 输出非无穷小的均值
        l_neg_no_inf = l_neg.masked_fill(identity_matrix, 0)
        l_neg_mean = l_neg_no_inf[l_neg_no_inf != 0].mean()

        return l_pos, l_neg_mean
    
    def PatchNCELoss(self, f_q, f_k, tau=0.07):
        # batch size and channel size
        B, C = f_q.shape
        f_k = f_k.detach()
        # calculate v * v+: Bx1
        l_pos = (f_k * f_q).sum(dim=1, keepdim=True)
        # calculate v * v-: BxB
        l_neg = torch.mm(f_q, f_k.t())
        # The diagonal entries are not negatives. Remove them.
        identity_matrix = torch.eye(B, dtype=torch.bool).to(f_q.device)
        l_neg.masked_fill_(identity_matrix, -float('inf'))
        # calculate logits: Bx(B+1)
        logits = torch.cat((l_pos, l_neg), dim=1) / tau
        # return PatchNCE loss
        targets = torch.zeros(B, dtype=torch.long).to(f_q.device)

        return self.cross_entropy_loss(logits, targets)
    
    def compute_cosine_embedding_loss(self, projection_q_normalize, projection_k_normalize):
        # batch size and feature dimension
        B, T = projection_q_normalize.shape
        projection_k_normalize = projection_k_normalize.detach()

        # 创建标签
        target_pos = torch.ones(B, dtype=torch.float32).to(projection_q_normalize.device)

        # 计算正样本相似度
        loss_pos = F.cosine_embedding_loss(projection_q_normalize, projection_k_normalize, target_pos, reduction='mean')

        # 计算负样本相似度
        projection_q_expand = projection_q_normalize.unsqueeze(1).expand(B, B, T).reshape(-1, T)
        projection_k_expand = projection_k_normalize.unsqueeze(0).expand(B, B, T).reshape(-1, T)
        identity_matrix = torch.eye(B, dtype=torch.bool).to(projection_q_normalize.device).view(-1)
        mask = ~identity_matrix
        negative_q = projection_q_expand[mask].view(B * (B - 1), T)
        negative_k = projection_k_expand[mask].view(B * (B - 1), T)
        margin = 0.0
        cosine_similarity = F.cosine_similarity(negative_q, negative_k)
        loss_neg = torch.clamp(cosine_similarity + margin, min=0).mean()
        return loss_pos + loss_neg








