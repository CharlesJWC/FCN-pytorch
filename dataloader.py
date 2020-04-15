#-*- coding:utf-8 -*-
'''
[AI502] Deep Learning Assignment
"Fully Convolutional Networks" Implementation
20193640 Jungwon Choi
'''
import torch
import torchvision
import torchvision.transforms as transforms
from PIL import Image

import matplotlib.pyplot as plt
import numpy as np
import os

#===============================================================================
''' VOC2012 Dataset initialization '''
class VOC2012_Dataset():
    DEFAULT_DATA_PATH = '../../dataset/VOC2012/'
    DEFAULT_TRAIN_FILE = 'train.txt'
    DEFAULT_VALID_FILE = 'val.txt'
    # Set labels of 21 classes (20 original classes + 1 background)
    classes = ('background',
               'aeroplane', 'bicycle', 'bird','boat', 'bottle',
               'bus', 'car', 'cat', 'chair', 'cow',
               'diningtable', 'dog', 'horse', 'motorbike', 'person',
               'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor')
    # Set colors of each label
    class_colors = ((  0,   0,   0),
        (128,  0,  0), (  0,128,  0), (128,128,  0), (  0,  0,128), (128,  0,128),
        (  0,128,128), (128,128,128), ( 64,  0,  0), (192,  0,  0), ( 64,128,  0),
        (192,128,  0), ( 64,  0,128), (192,  0,128), ( 64,128,128), (192,128,128),
        (  0, 64,  0), (128, 64,  0), (  0,192,  0), (128,192,  0), (  0, 64,128))
    #===========================================================================
    ''' Initialization '''
    def __init__(self, root=None, train=True):
        # Set dataset root path
        if root == None:
            self.root = self.DEFAULT_DATA_PATH
        else:
            self.root = root
        # Set dataset type
        self.train = train
        # Set dataset transform
        self.transform_train = transforms.Compose([
                        # ** Augmentation yielded no noticeable improvement. **
                        # transforms.Resize((32, 32)),
                        # transforms.RandomHorizontalFlip(p=0.5),
                        # transforms.ColorJitter(brightness=0.4, contrast=0.4,
                        #             saturation=0.4, hue=0),
                        transforms.Resize((256, 256)),
                        transforms.ToTensor(),
                        transforms.Normalize((0.485, 0.456, 0.406),
                                                (0.229, 0.224, 0.225))
                        ])
        self.transform_val = transforms.Compose([
                        transforms.Resize((256, 256)),
                        transforms.ToTensor(),
                        transforms.Normalize((0.485, 0.456, 0.406),
                                                (0.229, 0.224, 0.225))
                        ])
        self.transform_mask = transforms.Compose([
                        transforms.Resize((256, 256)),
                        # transforms.ToTensor(),
                        ])

        # Load the list of images
        if self.train == True:
            LIST_PATH = 'ImageSets/Segmentation/'+self.DEFAULT_TRAIN_FILE
            with open(os.path.join(self.root, LIST_PATH), 'r') as f:
                file_name_list = f.readlines()
        else:
            LIST_PATH = 'ImageSets/Segmentation/'+self.DEFAULT_VALID_FILE
            with open(os.path.join(self.root, LIST_PATH), 'r') as f:
                file_name_list = f.readlines()

        # Save the paths of images
        self.jepg_images = list()
        self.segm_images = list()
        for file_name in file_name_list:
            self.jepg_images.append(os.path.join(self.root, 'JPEGImages', file_name[:-1]+'.jpg'))
            self.segm_images.append(os.path.join(self.root, 'SegmentationClass', file_name[:-1]+'.png'))

        # Sort the lists of paths
        self.jepg_images.sort()
        self.segm_images.sort()

    #===========================================================================
    def __getitem__(self, index):
        jpeg_image = Image.open(self.jepg_images[index])
        segm_image = Image.open(self.segm_images[index]).convert('RGB')

        # Transforming the image
        if self.train == True:
            jpeg_image = self.transform_train(jpeg_image)
        else:
            jpeg_image = self.transform_val(jpeg_image)

        # Transforming the segmentation image
        segm_image = self.transform_mask(segm_image)
        # Convert RGB mask to label mask
        label_mask = self.cvtRGBtoLabel(segm_image)
        # Convert numpy image to tensor of torch
        label_mask = torch.from_numpy(label_mask).long()
        # Check the memory size
        # label_mask_elenum = eval('*'.join([str(n) for n in label_mask.size()]))
        # print(label_mask_elenum*label_mask.element_size())
        return jpeg_image, label_mask

    #===========================================================================
    def __len__(self,):
        return len(self.jepg_images)

    #===========================================================================
    ''' Convert RGB mask image to label mask image '''
    def cvtRGBtoLabel(self, rgb_mask):
        rgb_mask = np.asarray(rgb_mask)
        label_mask = np.zeros((rgb_mask.shape[0],rgb_mask.shape[1]))

        # Insert a label into each pixel corresponding to the class color
        for label, class_color in enumerate(self.class_colors):
            label_mask[np.where(np.all(rgb_mask == class_color, axis=2))] = label
            # axis= 0:W, 1:H, 2:C

        # Reduce the memory of the label mask
        label_mask = label_mask.astype(np.int8)
        # Check the memory size
        # print(label_mask.size*label_mask.itemsize)

        return label_mask

#===============================================================================
''' VOC2011 Dataset initialization '''
class VOC2011_Dataset(VOC2012_Dataset):
    DEFAULT_DATA_PATH = '../../dataset/VOC2011/'
    DEFAULT_TRAIN_FILE = 'train.txt'
    DEFAULT_VALID_FILE = 'seg11valid.txt'

#===============================================================================
''' VOC2012 Dataloader maker '''
class VOC2012_Dataloader():
    DEFAULT_DATA_PATH = '../../dataset/VOC2012/'
    #===========================================================================
    ''' Initialization '''
    def __init__(self, root=None):
        # Set dataset root path
        if root == None:
            self.root = self.DEFAULT_DATA_PATH
        else:
            self.root = root

    #===========================================================================
    ''' Get train dataset loader '''
    def get_train_loader(self, batch_size=20, num_workers=2):
        self.dataset = VOC2012_Dataset(root=self.root, train=True)
        train_loader = torch.utils.data.DataLoader(self.dataset,
                                                batch_size=batch_size,
                                                shuffle=True,
                                                num_workers=num_workers,
                                                pin_memory=True)
        return train_loader

    #===========================================================================
    ''' Get validation dataset loader '''
    def get_val_loader(self, batch_size=20, num_workers=2):
        self.dataset = VOC2012_Dataset(root=self.root, train=False)
        val_loader = torch.utils.data.DataLoader(self.dataset,
                                                batch_size=batch_size,
                                                shuffle=False,
                                                num_workers=num_workers,
                                                pin_memory=True)
        return val_loader

#===============================================================================
''' VOC2011 Dataloader maker '''
class VOC2011_Dataloader(VOC2012_Dataloader):
    DEFAULT_DATA_PATH = '../../dataset/VOC2011/'
    #===========================================================================
    ''' Get train dataset loader '''
    def get_train_loader(self, batch_size=20, num_workers=2):
        self.dataset = VOC2011_Dataset(root=self.root, train=True)
        train_loader = torch.utils.data.DataLoader(self.dataset,
                                                batch_size=batch_size,
                                                shuffle=True,
                                                num_workers=num_workers,
                                                pin_memory=True)
        return train_loader

    #===========================================================================
    ''' Get validation dataset loader '''
    def get_val_loader(self, batch_size=20, num_workers=2):
        self.dataset = VOC2011_Dataset(root=self.root, train=False)
        val_loader = torch.utils.data.DataLoader(self.dataset,
                                                batch_size=batch_size,
                                                shuffle=False,
                                                num_workers=num_workers,
                                                pin_memory=True)
        return val_loader

#===============================================================================
''' Convert label mask image to RGB mask image '''
def cvtLabeltoRGB(label_mask):
    label_mask = np.asarray(label_mask)
    rgb_mask = np.zeros((3, label_mask.shape[0], label_mask.shape[1]))

    # Insert the class color into each pixel corresponding to the label
    for label, class_color in enumerate(VOC2012_Dataset.class_colors):
        rgb_mask[0][label_mask == label] = class_color[0]
        rgb_mask[1][label_mask == label] = class_color[1]
        rgb_mask[2][label_mask == label] = class_color[2]

    # Scale RGB value from 0~255 to 0~1
    rgb_mask /= 255.0
    return rgb_mask

#===============================================================================
''' Image unnormalization '''
def unNormalize(image):
    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)
    for c, m, s in zip(image, mean, std):
        c.mul_(s).add_(m)
    return image

#===============================================================================
''' Test dataloader '''
if __name__ == "__main__":
    # Get an image and label from the dataset
    # voc = VOC2012_Dataset()
    # voc = VOC2011_Dataset()
    # image, label = voc[0]


    # Get images and labels from dataloader
    voc = VOC2012_Dataloader()
    # voc = VOC2011_Dataloader()
    dataloader = voc.get_train_loader(batch_size=20, num_workers=6)
    # dataloader = voc.get_val_loader(batch_size=20, num_workers=6)

    images, labels = iter(dataloader).next()

    # Show images
    unorm_images = list()
    for image in images:
        unorm_images.append(unNormalize(image))
    unorm_images = torch.stack(unorm_images, dim=0)
    unorm_images = torchvision.utils.make_grid(unorm_images)
    plt.imshow(np.transpose(unorm_images.numpy(), (1, 2, 0)))
    plt.show()

    # Show labels
    label_rgb = list()
    for label in labels:
        label_rgb.append(torch.from_numpy(cvtLabeltoRGB(label)))
    label_rgb = torch.stack(label_rgb, dim=0)
    label_rgb = torchvision.utils.make_grid(label_rgb)
    plt.imshow(np.transpose(label_rgb.numpy(), (1, 2, 0)))
    plt.show()
