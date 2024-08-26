import torch
from torch.utils.data import DataLoader
import sys
import os
sys.path.append("/data0/JM/code/scene_style_learning/finetune_vgg")
from dataloader_vgg import SpatialDataloader
from model_vgg import SCT
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


if __name__ == "__main__":
    fixed_seed()
    log_dir = "/data0/JM/code/scene_style_learning/finetune_vgg/logs"
    os.makedirs(log_dir, exist_ok=True)
    
    writer = SummaryWriter(log_dir=log_dir)
    root_dir = '/data0/JM/code/scene_style_learning/data_upsample'
    train_data = SpatialDataloader(root_dir, mode='train', crop_height=512, crop_width=512, iter_num=None)
    test_data = SpatialDataloader(root_dir, mode='test', crop_height=512, crop_width=512, iter_num=None)
    model = SCT(vgg).cuda()
    model.net.load_state_dict(torch.load('/data0/JM/code/scene_style_learning/vgg_normalised.pth'), strict=True)
    train_dataloader = DataLoader(train_data, batch_size=8, shuffle=True, num_workers=16)
    test_dataloader = DataLoader(test_data, batch_size=8, shuffle=True, num_workers=16)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    data_len_train = len(train_dataloader)
    data_len_test = len(test_dataloader)

    iter = 0
    for epoch in range(1000):

        print('<'+'='*55+'test'+'='*55+'>')
        model.eval()
        total_loss = 0.0
        with torch.no_grad():  
            for batch in test_dataloader:
                crops_one = batch['crops_one'].cuda()
                crops_two = batch['crops_two'].cuda()

                batchword = model(crops_one, crops_two)
                total_loss += batchword['total_loss'].item()
                print(f"test ==> epoch: {epoch:2d}   "
                      f"total loss: {batchword['total_loss']:.4f}")
            writer.add_scalar(f"test/total loss", total_loss/data_len_test, epoch)


        print('<'+'='*55+'train'+'='*55+'>')
        model.train()
        iter_epoch = 0
        total_loss = 0.0

        for batch in train_dataloader:
            crops_one = batch['crops_one'].cuda()
            crops_two = batch['crops_two'].cuda()

            optimizer.zero_grad()
            batchword = model(crops_one, crops_two)
            loss = batchword['total_loss']
            loss.backward()
            optimizer.step()
            total_loss += batchword['total_loss'].item()
            iter += 1
            iter_epoch += 1
            print(f"train ==> epoch: {epoch:2d}   "
                  f"iter: {iter:4d}   "
                  f"total loss: {batchword['total_loss']:.4f}")
            if iter % 100 == 0:
                writer.add_scalar(f"train_iter/total loss", total_loss/iter_epoch, iter)   
        writer.add_scalar(f"train_epoch/total loss", total_loss/data_len_train, epoch)

        if epoch != 0 and epoch % 1 == 0:
            torch.save(model.state_dict(), f'/data0/JM/code/scene_style_learning/finetune_vgg/model_ckpt/{str(epoch).zfill(6)}.pth')
            


