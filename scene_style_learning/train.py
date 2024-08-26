import torch
from torch.utils.data import DataLoader
import sys
import os
sys.path.append("/data1/JM/code/BrushNet/scene_style_learning")
from dataloader import SpatialDataloader
from model import SCT
import random
import numpy as np
from torch.utils.tensorboard import SummaryWriter
import torch.nn as nn

def fixed_seed():
    seed = 42
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark =  True

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



if __name__ == "__main__":
    save_interval = 5
    fixed_seed()
    vgg = make_layers([3, 64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 
            512, 512, 512, 512, 'M', 512, 512, 'M', 512, 512, 'M'])

    log_dir = "/data1/JM/code/BrushNet/scene_style_learning/logs_test"
    os.makedirs(log_dir, exist_ok=True)
    
    writer = SummaryWriter(log_dir=log_dir)
    root_dir = '/data1/JM/code/BrushNet/scene_style_learning/data'
    train_data = SpatialDataloader(root_dir, mode='train', crop_height=512, crop_width=512, iter_num=5000)
    test_data = SpatialDataloader(root_dir, mode='test', crop_height=512, crop_width=512, iter_num=None)

    vgg.load_state_dict(torch.load('/data1/JM/code/BrushNet/scene_style_learning/style_vgg.pth'), strict=True)
    model = SCT(vgg).cuda()

    train_dataloader = DataLoader(train_data, batch_size=16, shuffle=True, num_workers=32)
    test_dataloader = DataLoader(test_data, batch_size=16, shuffle=True, num_workers=32)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=5e-6)

    data_len_train = len(train_dataloader)
    data_len_test = len(test_dataloader)

    iter = 0
    for epoch in range(1000):
        print('<'+'='*55+'train'+'='*55+'>')
        model.train()
        iter_epoch = 0
        total_loss = 0.0
        pos_all = 0
        neg_all = 0

        for batch in train_dataloader:
            crops_one = batch['crops_one'].cuda()
            crops_two = batch['crops_two'].cuda()

            optimizer.zero_grad()
            batchword = model(crops_one, crops_two)
            loss = batchword['total_loss']
            loss.backward()
            optimizer.step()
            total_loss += batchword['total_loss'].item()
            pos_all += batchword['pos']
            neg_all += batchword['neg']
            iter += 1
            iter_epoch += 1
            print(f"train ==> epoch: {epoch:2d}   "
                  f"iter: {iter:4d}   "
                  f"total loss: {batchword['total_loss']:.4f}   "
                  f"pos: {batchword['pos']:.4f}   "
                  f"neg: {batchword['neg']:.4f}")
            if iter % 100 == 0:
                writer.add_scalar(f"train_iter/total loss", total_loss/iter_epoch, iter)   
                writer.add_scalar(f"train_iter/pos", pos_all/iter_epoch, iter)   
                writer.add_scalar(f"train_iter/neg", neg_all/iter_epoch, iter)   
        writer.add_scalar(f"train_epoch/total loss", total_loss/data_len_train, epoch)
        writer.add_scalar(f"train_epoch/pos", pos_all/data_len_train, epoch)
        writer.add_scalar(f"train_epoch/neg", neg_all/data_len_train, epoch)

        print('<'+'='*55+'test'+'='*55+'>')
        model.eval()
        total_loss = 0.0
        pos_all = 0
        neg_all = 0
        with torch.no_grad():  
            for batch in test_dataloader:
                crops_one = batch['crops_one'].cuda()
                crops_two = batch['crops_two'].cuda()
                # img = batch['img'].cuda()
                # img_another = batch['img_another'].cuda()
                batchword = model(crops_one, crops_two)
                total_loss += batchword['total_loss'].item()
                pos_all += batchword['pos']
                neg_all += batchword['neg']
                print(f"test ==> epoch: {epoch:2d}   "
                      f"total loss: {batchword['total_loss']:.4f}   "
                      f"pos: {batchword['pos']:.4f}   "
                      f"neg: {batchword['neg']:.4f}")
            writer.add_scalar(f"test/total loss", total_loss/data_len_test, epoch)
            writer.add_scalar(f"test/pos", pos_all/data_len_test, epoch)
            writer.add_scalar(f"test/neg", neg_all/data_len_test, epoch)

        if (epoch + 1) % save_interval == 0:
            torch.save(model.state_dict(), f'/data1/JM/code/BrushNet/scene_style_learning/ckpt/model_epoch_{epoch + 1}.pth')
            print(f"Model saved at epoch {epoch + 1}")
            


