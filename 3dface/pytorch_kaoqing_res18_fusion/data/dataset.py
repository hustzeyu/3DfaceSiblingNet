import os
from PIL import Image
import torch
from torch.utils import data
import numpy as np
from torchvision import transforms as T
import torchvision
import cv2
import sys

class Dataset(data.Dataset):
    def __init__(self, img_list, phase='train', input_shape=(3, 112, 112)):
        self.phase = phase
        self.input_shape = input_shape
        with open(img_list, 'r') as fd:
            imgs = fd.readlines()
        imgs = [img.rstrip("\n") for img in imgs]
        self.imgs = np.random.permutation(imgs)
        normalize = T.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])


        normalize_d = T.Normalize(mean=[0.5,],
                                  std=[0.5,])

        self.transforms_d = T.Compose([
                #T.RandomCrop(self.input_shape[1:]),
                T.RandomHorizontalFlip(),
                T.ToTensor(),
                normalize_d
            ])


        if self.phase == 'train':
            self.transforms = T.Compose([
                #T.RandomCrop(self.input_shape[1:]),
                T.RandomHorizontalFlip(),
                T.ToTensor(),
                normalize
            ])
        else:
            self.transforms = T.Compose([
                #T.CenterCrop(self.input_shape[1:]),
                T.ToTensor(),
                normalize
            ])

    def __getitem__(self, index):
        sample = self.imgs[index]
        splits = sample.split()
        img_path = splits[0]
        rgb_img = Image.open(img_path)
        rgb_img = rgb_img.resize((112,112))
        rgb_img = self.transforms(rgb_img)

        # get depth_path from rgb_path
        depth_path = img_path.replace("rgb", "rst")

        depth_img = Image.open(depth_path)
        depth_img = depth_img.resize((112,112))
        depth_img = self.transforms_d(depth_img)

        label = np.int32(splits[1])
        return rgb_img.float(), depth_img.float(), label

    def __len__(self):
        return len(self.imgs)


if __name__ == '__main__':
    train_data = Dataset("/home/gp/work/project/pytorch_kaoqing_res18_fusion/img_list/train_rgb.list", "train")
    trainloader = data.DataLoader(train_data, batch_size=64, shuffle=True, num_workers=4)
    for i, (rgb_img, depth_img, label) in enumerate(trainloader):
        import pdb;pdb.set_trace()
        print(rgb_img, depth_img, label)
        
