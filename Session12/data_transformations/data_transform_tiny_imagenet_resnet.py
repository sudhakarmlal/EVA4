from PIL import Image
import cv2
import numpy as np
from albumentations import Compose, RandomCrop, Normalize, HorizontalFlip, Resize,GaussNoise
from albumentations.augmentations.transforms import Cutout,ElasticTransform,PadIfNeeded,RandomCrop
from albumentations.pytorch import ToTensor


class album_Compose_train_64:
    def __init__(self):
        self.transform = Compose(
        [
         PadIfNeeded(min_height=80, min_width=80, border_mode=cv2.BORDER_CONSTANT, value=[0.4914*255, 0.4822*255, 0.4465*255], p=1.0),
         RandomCrop(64,64, p=1.0),
         Cutout(num_holes=1, max_h_size=8, max_w_size=8,  fill_value=[0.4914*255, 0.4822*255, 0.4465*255]),
         HorizontalFlip(p=0.2),
         #GaussNoise(p=0.15),
         #ElasticTransform(p=0.15),
        Normalize((0.4914, 0.4822, 0.4465), ((0.2023, 0.1994, 0.2010))),
        ToTensor(),
        ])
    def __call__(self, img):
        img = np.array(img)
        img = self.transform(image=img)['image']
        return img
		
class album_Compose_train_48:
    def __init__(self):
        self.transform = Compose(
        [
         PadIfNeeded(min_height=64, min_width=64, border_mode=cv2.BORDER_CONSTANT, value=[0.4914*255, 0.4822*255, 0.4465*255], p=1.0),
         RandomCrop(48,48, p=1.0),
         Cutout(num_holes=1, max_h_size=8, max_w_size=8,  fill_value=[0.4914*255, 0.4822*255, 0.4465*255]),
         HorizontalFlip(p=0.2),
         #GaussNoise(p=0.15),
         #ElasticTransform(p=0.15),
        Normalize((0.4914, 0.4822, 0.4465), ((0.2023, 0.1994, 0.2010))),
        ToTensor(),
        ])
    def __call__(self, img):
        img = np.array(img)
        img = self.transform(image=img)['image']
        return img
		
class album_Compose_train_32:
    def __init__(self):
        self.transform = Compose(
        [
         PadIfNeeded(min_height=48, min_width=48, border_mode=cv2.BORDER_CONSTANT, value=[0.4914*255, 0.4822*255, 0.4465*255], p=1.0),
         RandomCrop(32,32, p=1.0),
         Cutout(num_holes=1, max_h_size=8, max_w_size=8,  fill_value=[0.4914*255, 0.4822*255, 0.4465*255]),
         HorizontalFlip(p=0.2),
         #GaussNoise(p=0.15),
         #ElasticTransform(p=0.15),
        Normalize((0.4914, 0.4822, 0.4465), ((0.2023, 0.1994, 0.2010))),
        ToTensor(),
        ])
    def __call__(self, img):
        img = np.array(img)
        img = self.transform(image=img)['image']
        return img
		


class album_Compose_test:
    def __init__(self):
        self.transform = Compose(
        [
        Normalize((0.4914, 0.4822, 0.4465), ((0.2023, 0.1994, 0.2010))),
        ToTensor(),
        ])
    def __call__(self, img):
        img = np.array(img)
        img = self.transform(image=img)['image']
        return img
			
def get_train_transform_64():
    transform = album_Compose_train_64()
    return transform

def get_train_transform_48():
    transform = album_Compose_train_48()
    return transform
	
def get_train_transform_32():
    transform = album_Compose_train_32()
    return transform


def get_test_transform():
    transform = album_Compose_test()
    return transform