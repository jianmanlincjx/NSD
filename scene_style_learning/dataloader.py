from torchvision import transforms
import random
import os
import cv2
import torch
from torch.utils.data import Dataset

class ImageProcessor:
    def __init__(self, crop_height, crop_width):
        self.crop_height = crop_height
        self.crop_width = crop_width

    def crop_from_top_left(self, img):
        _, img_height, img_width = img.shape
        crop_height = min(self.crop_height, img_height)
        crop_width = min(self.crop_width, img_width)
        cropped_img = img[:, 0:crop_height, 0:crop_width]
        resized_img = transforms.Resize((self.crop_height, self.crop_width))(cropped_img)
        return resized_img

    def crop_from_top_right(self, img):
        _, img_height, img_width = img.shape
        crop_height = min(self.crop_height, img_height)
        crop_width = min(self.crop_width, img_width)
        cropped_img = img[:, 0:crop_height, img_width-crop_width:img_width]
        resized_img = transforms.Resize((self.crop_height, self.crop_width))(cropped_img)
        return resized_img

    def crop_from_bottom_left(self, img):
        _, img_height, img_width = img.shape
        crop_height = min(self.crop_height, img_height)
        crop_width = min(self.crop_width, img_width)
        cropped_img = img[:, img_height-crop_height:img_height, 0:crop_width]
        resized_img = transforms.Resize((self.crop_height, self.crop_width))(cropped_img)
        return resized_img

    def crop_from_bottom_right(self, img):
        _, img_height, img_width = img.shape
        crop_height = min(self.crop_height, img_height)
        crop_width = min(self.crop_width, img_width)
        cropped_img = img[:, img_height-crop_height:img_height, img_width-crop_width:img_width]
        resized_img = transforms.Resize((self.crop_height, self.crop_width))(cropped_img)
        return resized_img

    def random_crop(self, img):
        # 随机选择两个不同的裁剪方法
        crop_methods = [self.crop_from_top_left, self.crop_from_top_right, self.crop_from_bottom_left, self.crop_from_bottom_right]
        method_one, method_two = random.sample(crop_methods, 2)
        crop_one = method_one(img)
        crop_two = method_two(img)
        return crop_one, crop_two

class SpatialDataloader(Dataset):
    def __init__(self, root, mode="train", crop_height=100, crop_width=100, iter_num=3000):
        super(SpatialDataloader, self).__init__()
        self.root = os.path.join(root, mode)
        self.img_name_list = sorted(os.listdir(self.root))
        self.crop_height = crop_height
        self.crop_width = crop_width
        self.processor = ImageProcessor(crop_height, crop_width)
        self._img_transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ])
        self.iter_num = iter_num if iter_num is not None else len(self.img_name_list)

    def __len__(self):
        return self.iter_num

    def __getitem__(self, index):
        img_name = random.choice(self.img_name_list)
        img_path = os.path.join(self.root, img_name)
        
        try:
            img = cv2.imread(img_path)
            if img is None:
                raise ValueError(f"Image at path {img_path} could not be read.")
                
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            img = self._img_transform(img)
            crops_one, crops_two = self.processor.random_crop(img)
            return {'crops_one': crops_one, 'crops_two': crops_two}
        
        except Exception as e:
            print(f"Error with image at path {img_path}: {e}")
            # Optionally, return some default value or handle the error
            return {'crops_one': None, 'crops_two': None}

# Example usage
if __name__ == "__main__":
    dataset = SpatialDataloader(root='/path/to/data', mode='train', crop_height=100, crop_width=100, iter_num=3000)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=True)
    
    for batch in dataloader:
        crops_one = batch['crops_one']
        crops_two = batch['crops_two']
        # Your training code here

    
    # def random_crop(self, img):
    #     _, img_height, img_width = img.shape
        
    #     # Ensure crop size is not larger than the image size
    #     crop_height = min(self.crop_height, img_height)
    #     crop_width = min(self.crop_width, img_width)

    #     # Randomly select the starting point for the crop
    #     top = random.randint(0, img_height - crop_height)
    #     left = random.randint(0, img_width - crop_width)

    #     # Perform the crop
    #     cropped_img = img[:, top:top + crop_height, left:left + crop_width]

    #     # Resize to the target dimensions
    #     resized_img = transforms.Resize((self.crop_height, self.crop_width))(cropped_img)

    #     return resized_img
