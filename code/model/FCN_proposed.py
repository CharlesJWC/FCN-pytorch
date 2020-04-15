#-*- coding:utf-8 -*-
'''
[AI502] Deep Learning Assignment
"Fully Convolutional Networks" Implementation
20193640 Jungwon Choi
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
import numpy as np

import warnings; warnings.filterwarnings("ignore");

#===============================================================================
class FCN_32s(nn.Module):
    #===========================================================================
    def __init__(self, num_classes = 21):
        super(FCN_32s, self).__init__()
        self.num_classes = num_classes
        pretrained_vgg16 = models.vgg16(pretrained=True)

        # Separate pre-trained VGG16 features layer to each block
        self.conv_block1 = nn.Sequential()
        for idx in range(0,4+1):
            self.conv_block1.add_module(str(idx), pretrained_vgg16.features[idx])
        self.conv_block2 = nn.Sequential()
        for idx in range(5,9+1):
            self.conv_block2.add_module(str(idx), pretrained_vgg16.features[idx])
        self.conv_block3 = nn.Sequential()
        for idx in range(10,16+1):
            self.conv_block3.add_module(str(idx), pretrained_vgg16.features[idx])
        self.conv_block4 = nn.Sequential()
        for idx in range(17,23+1):
            self.conv_block4.add_module(str(idx), pretrained_vgg16.features[idx])
        self.conv_block5 = nn.Sequential()
        for idx in range(24,30+1):
            self.conv_block5.add_module(str(idx), pretrained_vgg16.features[idx])

        # fully convolutional layers
        self.conv_block6 = nn.Sequential(
                nn.Conv2d(in_channels=512, out_channels=4096, kernel_size=1),
                nn.ReLU(inplace=True),
                nn.Dropout2d(p=0.5),
                )
        self.conv_block7 = nn.Sequential(
                nn.Conv2d(4096, 4096, 1),
                nn.ReLU(inplace=True),
                nn.Dropout2d(p=0.5),
                )
        # last score layer
        self.score_conv7 = nn.Conv2d(4096, self.num_classes, kernel_size=1)
        # upsampling layer
        self.upscore_conv7 = nn.ConvTranspose2d(in_channels=self.num_classes,
                                        out_channels=self.num_classes,
                                        kernel_size=64,
                                        stride=32,  # 32x upscale
                                        padding=16,
                                        groups=self.num_classes,
                                        bias=False)

        # Zero-initialize the class scoring layer
        nn.init.zeros_(self.score_conv7.weight)
        nn.init.zeros_(self.score_conv7.bias)

        # Final layer deconvolutional filters are fixed to bilinear interpolation.
        self.upscore_conv7.weight = nn.Parameter(make_bilinear_weights(self.upscore_conv7.kernel_size[0], self.num_classes))
        self.upscore_conv7.weight.requires_grad = False
        # Equivalent to bilinear deconvolution layer (to handle both odd and even size inputs)
        self.upscore_conv7 = nn.Upsample(mode='bilinear')

    #===========================================================================
    def forward(self, x):
        # Set the proper upsample size
        if isinstance(self.upscore_conv7, nn.Upsample):
            self.upscore_conv7.size = (x.size()[2],x.size()[3],)

        # Pre-trained layers
        # print(x.size())
        out1 = self.conv_block1(x)
        # print('conv1',out1.size())
        out2 = self.conv_block2(out1)
        # print('conv2',out2.size())
        out3 = self.conv_block3(out2)
        # print('conv3',out3.size())
        out4 = self.conv_block4(out3)
        # print('conv4',out4.size())
        out5 = self.conv_block5(out4)
        # print('conv5',out5.size())

        # Expanded fc layers
        out6 = self.conv_block6(out5)
        # print('conv6',out6.size())
        out7 = self.conv_block7(out6)
        # print('conv7',out7.size())
        out_score7 = self.score_conv7(out7)
        # print('score7',out_score7.size())

        # Upsampling layer x32
        out_upscore7 = self.upscore_conv7(out_score7)
        # print('score7*32',out_upscore7.size())
        return out_upscore7

#===============================================================================
class FCN_32s_fixed(nn.Module):
    #===========================================================================
    def __init__(self, num_classes = 21):
        super(FCN_32s_fixed, self).__init__()
        self.num_classes = num_classes
        pretrained_vgg16 = models.vgg16(pretrained=True)

        # freeze the parameters
        for param in pretrained_vgg16.parameters():
            param.requires_grad = False

        # Separate pre-trained VGG16 features layer to each block
        self.conv_block1 = nn.Sequential()
        for idx in range(0,4+1):
            self.conv_block1.add_module(str(idx), pretrained_vgg16.features[idx])
        self.conv_block2 = nn.Sequential()
        for idx in range(5,9+1):
            self.conv_block2.add_module(str(idx), pretrained_vgg16.features[idx])
        self.conv_block3 = nn.Sequential()
        for idx in range(10,16+1):
            self.conv_block3.add_module(str(idx), pretrained_vgg16.features[idx])
        self.conv_block4 = nn.Sequential()
        for idx in range(17,23+1):
            self.conv_block4.add_module(str(idx), pretrained_vgg16.features[idx])
        self.conv_block5 = nn.Sequential()
        for idx in range(24,30+1):
            self.conv_block5.add_module(str(idx), pretrained_vgg16.features[idx])

        # fully convolutional layers
        self.conv_block6 = nn.Sequential(
                nn.Conv2d(in_channels=512, out_channels=4096, kernel_size=1),
                nn.ReLU(inplace=True),
                nn.Dropout2d(p=0.5),
                )
        self.conv_block7 = nn.Sequential(
                nn.Conv2d(4096, 4096, 1),
                nn.ReLU(inplace=True),
                nn.Dropout2d(p=0.5),
                )
        # last score layer
        self.score_conv7 = nn.Conv2d(4096, self.num_classes, kernel_size=1)
        # upsampling layer
        self.upscore_conv7 = nn.ConvTranspose2d(in_channels=self.num_classes,
                                        out_channels=self.num_classes,
                                        kernel_size=64,
                                        stride=32,  # 32x upscale
                                        padding=16,
                                        groups=self.num_classes,
                                        bias=False)

        # Zero-initialize the class scoring layer
        nn.init.zeros_(self.score_conv7.weight)
        nn.init.zeros_(self.score_conv7.bias)

        # Final layer deconvolutional filters are fixed to bilinear interpolation.
        self.upscore_conv7.weight = nn.Parameter(make_bilinear_weights(self.upscore_conv7.kernel_size[0], self.num_classes))
        self.upscore_conv7.weight.requires_grad = False
        # Equivalent to bilinear deconvolution layer (to handle both odd and even size inputs)
        self.upscore_conv7 = nn.Upsample(mode='bilinear')

    #===========================================================================
    def forward(self, x):
        # Set the proper upsample size
        if isinstance(self.upscore_conv7, nn.Upsample):
            self.upscore_conv7.size = (x.size()[2],x.size()[3],)

        # Pre-trained layers
        # print(x.size())
        out1 = self.conv_block1(x)
        # print('conv1',out1.size())
        out2 = self.conv_block2(out1)
        # print('conv2',out2.size())
        out3 = self.conv_block3(out2)
        # print('conv3',out3.size())
        out4 = self.conv_block4(out3)
        # print('conv4',out4.size())
        out5 = self.conv_block5(out4)
        # print('conv5',out5.size())

        # Expanded fc layers
        out6 = self.conv_block6(out5)
        # print('conv6',out6.size())
        out7 = self.conv_block7(out6)
        # print('conv7',out7.size())
        out_score7 = self.score_conv7(out7)
        # print('score7',out_score7.size())

        # Upsampling layer x32
        out_upscore7 = self.upscore_conv7(out_score7)
        # print('score7*32',out_upscore7.size())
        return out_upscore7

#===============================================================================
class FCN_16s(nn.Module):
    #===========================================================================
    def __init__(self, num_classes = 21):
        super(FCN_16s, self).__init__()
        self.num_classes = num_classes
        pretrained_vgg16 = models.vgg16(pretrained=True)

        # Separate pre-trained VGG16 features layer to each block
        self.conv_block1 = nn.Sequential()
        for idx in range(0,4+1):
            self.conv_block1.add_module(str(idx), pretrained_vgg16.features[idx])
        self.conv_block2 = nn.Sequential()
        for idx in range(5,9+1):
            self.conv_block2.add_module(str(idx), pretrained_vgg16.features[idx])
        self.conv_block3 = nn.Sequential()
        for idx in range(10,16+1):
            self.conv_block3.add_module(str(idx), pretrained_vgg16.features[idx])
        self.conv_block4 = nn.Sequential()
        for idx in range(17,23+1):
            self.conv_block4.add_module(str(idx), pretrained_vgg16.features[idx])
        self.conv_block5 = nn.Sequential()
        for idx in range(24,30+1):
            self.conv_block5.add_module(str(idx), pretrained_vgg16.features[idx])

        # fully convolutional layers
        self.conv_block6 = nn.Sequential(
                nn.Conv2d(in_channels=512, out_channels=4096, kernel_size=1),
                nn.ReLU(inplace=True),
                nn.Dropout2d(p=0.5),
                )
        self.conv_block7 = nn.Sequential(
                nn.Conv2d(4096, 4096, 1),
                nn.ReLU(inplace=True),
                nn.Dropout2d(p=0.5),
                )
        # last score layer
        self.score_conv7 = nn.Conv2d(4096, self.num_classes, kernel_size=1)
        self.score_pool4 = nn.Conv2d(512, self.num_classes, kernel_size=1)

        # upsampling layer
        self.upscore_conv7 = nn.ConvTranspose2d(in_channels=self.num_classes,
                                        out_channels=self.num_classes,
                                        kernel_size=4,
                                        stride=2,   # 2x upscale
                                        padding=1,
                                        # groups=self.num_classes,
                                        bias=False)

        self.upscore_pool4 = nn.ConvTranspose2d(in_channels=self.num_classes,
                                        out_channels=self.num_classes,
                                        kernel_size=32,
                                        stride=16,   # 16x upscale
                                        padding=8,
                                        groups=self.num_classes,
                                        bias=False)
        # Zero-initialize the class scoring layer
        nn.init.zeros_(self.score_conv7.weight)
        nn.init.zeros_(self.score_conv7.bias)
        nn.init.zeros_(self.score_pool4.weight)
        nn.init.zeros_(self.score_pool4.bias)

        # Intermediate upsampling layers are initialized to bilinear upsampling, and then learned.
        # self.upscore_conv7.weight = nn.Parameter(make_bilinear_weights(self.upscore_conv7.kernel_size[0], self.num_classes))
        self.upscore_conv7.weight = nn.Parameter(torch.squeeze(torch.stack([make_bilinear_weights(self.upscore_conv7.kernel_size[0], self.num_classes)]*self.num_classes)))
        # Final layer deconvolutional filters are fixed to bilinear interpolation.
        self.upscore_pool4.weight = nn.Parameter(make_bilinear_weights(self.upscore_pool4.kernel_size[0], self.num_classes))
        self.upscore_pool4.weight.requires_grad = False
        # Equivalent to bilinear deconvolution layer (to handle both odd and even size inputs)
        self.upscore_pool4 = nn.Upsample(mode='bilinear')

    #===========================================================================
    def forward(self, x):
        # Set the proper upsample size
        if isinstance(self.upscore_pool4, nn.Upsample):
            self.upscore_pool4.size = (x.size()[2],x.size()[3],)

        # Pre-trained layers
        # print(x.size())
        out1 = self.conv_block1(x)
        # print('1',out1.size())
        out2 = self.conv_block2(out1)
        # print('2',out2.size())
        out3 = self.conv_block3(out2)
        # print('3',out3.size())
        out4 = self.conv_block4(out3)
        # print('4',out4.size())
        out5 = self.conv_block5(out4)
        # print('5',out5.size())

        # Expanded fc layers
        out6 = self.conv_block6(out5)
        # print('6',out6.size())
        out7 = self.conv_block7(out6)
        # print('7',out7.size())

        out_score7 = self.score_conv7(out7)
        # print('score7',out_score7.size())
        # Upsampling layer x2
        out_upscore7 = self.upscore_conv7(out_score7)
        # print('score7*2',out_upscore7.size())

        # skip architecture from pool4
        out_score4 = self.score_pool4(out4)
        # print('score4,',out_score4.size())
        # fuse score
        fuse_score4 = out_score4 + out_upscore7
        # Upsampling layer x16
        out_upscore4 = self.upscore_pool4(fuse_score4)
        # print('score4*16,',out_upscore4.size())
        return out_upscore4

#===============================================================================
class FCN_8s(nn.Module):
    #===========================================================================
    def __init__(self, num_classes = 21):
        super(FCN_8s, self).__init__()
        self.num_classes = num_classes
        pretrained_vgg16 = models.vgg16(pretrained=True)

        # Separate pre-trained VGG16 features layer to each block
        self.conv_block1 = nn.Sequential()
        for idx in range(0,4+1):
            self.conv_block1.add_module(str(idx), pretrained_vgg16.features[idx])
        self.conv_block2 = nn.Sequential()
        for idx in range(5,9+1):
            self.conv_block2.add_module(str(idx), pretrained_vgg16.features[idx])
        self.conv_block3 = nn.Sequential()
        for idx in range(10,16+1):
            self.conv_block3.add_module(str(idx), pretrained_vgg16.features[idx])
        self.conv_block4 = nn.Sequential()
        for idx in range(17,23+1):
            self.conv_block4.add_module(str(idx), pretrained_vgg16.features[idx])
        self.conv_block5 = nn.Sequential()
        for idx in range(24,30+1):
            self.conv_block5.add_module(str(idx), pretrained_vgg16.features[idx])

        # fully convolutional layers
        self.conv_block6 = nn.Sequential(
                nn.Conv2d(in_channels=512, out_channels=4096, kernel_size=1),
                nn.ReLU(inplace=True),
                nn.Dropout2d(p=0.5),
                )
        self.conv_block7 = nn.Sequential(
                nn.Conv2d(4096, 4096, 1),
                nn.ReLU(inplace=True),
                nn.Dropout2d(p=0.5),
                )
        # last score layer
        self.score_conv7 = nn.Conv2d(4096, self.num_classes, kernel_size=1)
        self.score_pool4 = nn.Conv2d(512, self.num_classes, kernel_size=1)
        self.score_pool3 = nn.Conv2d(256, self.num_classes, kernel_size=1)
        # upsampling layer
        self.upscore_conv7 = nn.ConvTranspose2d(in_channels=self.num_classes,
                                        out_channels=self.num_classes,
                                        kernel_size=4,
                                        stride=2,   # 2x upscale
                                        padding=1,
                                        # groups=self.num_classes,
                                        bias=False)
        self.upscore_pool4 = nn.ConvTranspose2d(in_channels=self.num_classes,
                                        out_channels=self.num_classes,
                                        kernel_size=4,
                                        stride=2,   # 2x upscale
                                        padding=1,
                                        # groups=self.num_classes,
                                        bias=False)
        self.upscore_pool3 = nn.ConvTranspose2d(in_channels=self.num_classes,
                                        out_channels=self.num_classes,
                                        kernel_size=16,
                                        stride=8,   # 8x upscale
                                        padding=4,
                                        groups=self.num_classes,
                                        bias=False)

        # Zero-initialize the class scoring layer
        nn.init.zeros_(self.score_conv7.weight)
        nn.init.zeros_(self.score_conv7.bias)
        nn.init.zeros_(self.score_pool4.weight)
        nn.init.zeros_(self.score_pool4.bias)
        nn.init.zeros_(self.score_pool3.weight)
        nn.init.zeros_(self.score_pool3.bias)

        # Intermediate upsampling layers are initialized to bilinear upsampling, and then learned.
        self.upscore_conv7.weight = nn.Parameter(torch.squeeze(torch.stack([make_bilinear_weights(self.upscore_conv7.kernel_size[0], self.num_classes)]*self.num_classes)))
        self.upscore_pool4.weight = nn.Parameter(torch.squeeze(torch.stack([make_bilinear_weights(self.upscore_pool4.kernel_size[0], self.num_classes)]*self.num_classes)))
        # self.upscore_conv7.weight = nn.Parameter(make_bilinear_weights(self.upscore_conv7.kernel_size[0], self.num_classes))
        # self.upscore_pool4.weight = nn.Parameter(make_bilinear_weights(self.upscore_pool4.kernel_size[0], self.num_classes))
        # Final layer deconvolutional filters are fixed to bilinear interpolation.
        self.upscore_pool3.weight = nn.Parameter(make_bilinear_weights(self.upscore_pool3.kernel_size[0], self.num_classes))
        self.upscore_pool3.weight.requires_grad = False
        # Equivalent to bilinear deconvolution layer (to handle both odd and even size inputs)
        self.upscore_pool3 = nn.Upsample(mode='bilinear')

    #===========================================================================
    def forward(self, x):
        # Set the proper upsample size
        if isinstance(self.upscore_pool3, nn.Upsample):
            self.upscore_pool3.size = (x.size()[2],x.size()[3],)

        # Pre-trained layers
        # print(x.size())
        out1 = self.conv_block1(x)
        # print('1',out1.size())
        out2 = self.conv_block2(out1)
        # print('2',out2.size())
        out3 = self.conv_block3(out2)
        # print('3',out3.size())
        out4 = self.conv_block4(out3)
        # print('4',out4.size())
        out5 = self.conv_block5(out4)
        # print('5',out5.size())

        # Expanded fc layers
        out6 = self.conv_block6(out5)
        # print('6',out6.size())
        out7 = self.conv_block7(out6)
        # print('7',out7.size())

        out_score7 = self.score_conv7(out7)
        # print('score7',out_score7.size())
        # Upsampling layer x2
        out_upscore7 = self.upscore_conv7(out_score7)
        # print('score7*2',out_upscore7.size())

        # skip architecture from pool4
        out_score4 = self.score_pool4(out4)
        # print('score4,',out_score4.size())
        # fuse score
        fuse_score4 = out_score4 + out_upscore7
        # Upsampling layer x2
        out_upscore4 = self.upscore_pool4(fuse_score4)
        # print('score4*2,',out_upscore4.size())

        # skip architecture from pool3
        out_score3 = self.score_pool3(out3)
        # print('score3',out_score3.size())
        # fuse score
        fuse_score3 = out_score3 + out_upscore4
        # Upsampling layer x8
        out_upscore3 = self.upscore_pool3(fuse_score3)
        # print('score3*8,',out_upscore3.size())
        return out_upscore3

#===============================================================================
class FCN_4s(nn.Module):
    #===========================================================================
    def __init__(self, num_classes = 21):
        super(FCN_4s, self).__init__()
        self.num_classes = num_classes
        pretrained_vgg16 = models.vgg16(pretrained=True)

        # Separate pre-trained VGG16 features layer to each block
        self.conv_block1 = nn.Sequential()
        for idx in range(0,4+1):
            self.conv_block1.add_module(str(idx), pretrained_vgg16.features[idx])
        self.conv_block2 = nn.Sequential()
        for idx in range(5,9+1):
            self.conv_block2.add_module(str(idx), pretrained_vgg16.features[idx])
        self.conv_block3 = nn.Sequential()
        for idx in range(10,16+1):
            self.conv_block3.add_module(str(idx), pretrained_vgg16.features[idx])
        self.conv_block4 = nn.Sequential()
        for idx in range(17,23+1):
            self.conv_block4.add_module(str(idx), pretrained_vgg16.features[idx])
        self.conv_block5 = nn.Sequential()
        for idx in range(24,30+1):
            self.conv_block5.add_module(str(idx), pretrained_vgg16.features[idx])

        # fully convolutional layers
        self.conv_block6 = nn.Sequential(
                nn.Conv2d(in_channels=512, out_channels=4096, kernel_size=1),
                nn.ReLU(inplace=True),
                nn.Dropout2d(p=0.5),
                )
        self.conv_block7 = nn.Sequential(
                nn.Conv2d(4096, 4096, 1),
                nn.ReLU(inplace=True),
                nn.Dropout2d(p=0.5),
                )
        # last score layer
        self.score_conv7 = nn.Conv2d(4096, self.num_classes, kernel_size=1)
        self.score_pool4 = nn.Conv2d(512, self.num_classes, kernel_size=1)
        self.score_pool3 = nn.Conv2d(256, self.num_classes, kernel_size=1)
        self.score_pool2 = nn.Conv2d(128, self.num_classes, kernel_size=1)
        # upsampling layer
        self.upscore_conv7 = nn.ConvTranspose2d(in_channels=self.num_classes,
                                        out_channels=self.num_classes,
                                        kernel_size=4,
                                        stride=2,   # 2x upscale
                                        padding=1,
                                        # groups=self.num_classes,
                                        bias=False)
        self.upscore_pool4 = nn.ConvTranspose2d(in_channels=self.num_classes,
                                        out_channels=self.num_classes,
                                        kernel_size=4,
                                        stride=2,   # 2x upscale
                                        padding=1,
                                        # groups=self.num_classes,
                                        bias=False)
        self.upscore_pool3 = nn.ConvTranspose2d(in_channels=self.num_classes,
                                        out_channels=self.num_classes,
                                        kernel_size=4,
                                        stride=2,   # 2x upscale
                                        padding=1,
                                        # groups=self.num_classes,
                                        bias=False)
        self.upscore_pool2 = nn.ConvTranspose2d(in_channels=self.num_classes,
                                        out_channels=self.num_classes,
                                        kernel_size=8,
                                        stride=4,   # 4x upscale
                                        padding=2,
                                        groups=self.num_classes,
                                        bias=False)

        # Zero-initialize the class scoring layer
        nn.init.zeros_(self.score_conv7.weight)
        nn.init.zeros_(self.score_conv7.bias)
        nn.init.zeros_(self.score_pool4.weight)
        nn.init.zeros_(self.score_pool4.bias)
        nn.init.zeros_(self.score_pool3.weight)
        nn.init.zeros_(self.score_pool3.bias)
        nn.init.zeros_(self.score_pool2.weight)
        nn.init.zeros_(self.score_pool2.bias)

        # Intermediate upsampling layers are initialized to bilinear upsampling, and then learned.
        self.upscore_conv7.weight = nn.Parameter(torch.squeeze(torch.stack([make_bilinear_weights(self.upscore_conv7.kernel_size[0], self.num_classes)]*self.num_classes)))
        self.upscore_pool4.weight = nn.Parameter(torch.squeeze(torch.stack([make_bilinear_weights(self.upscore_pool4.kernel_size[0], self.num_classes)]*self.num_classes)))
        self.upscore_pool3.weight = nn.Parameter(torch.squeeze(torch.stack([make_bilinear_weights(self.upscore_pool3.kernel_size[0], self.num_classes)]*self.num_classes)))
        # Final layer deconvolutional filters are fixed to bilinear interpolation.
        self.upscore_pool2.weight = nn.Parameter(make_bilinear_weights(self.upscore_pool2.kernel_size[0], self.num_classes))
        self.upscore_pool2.weight.requires_grad = False
        # Equivalent to bilinear deconvolution layer (to handle both odd and even size inputs)
        self.upscore_pool2 = nn.Upsample(mode='bilinear')

    #===========================================================================
    def forward(self, x):
        # Set the proper upsample size
        if isinstance(self.upscore_pool2, nn.Upsample):
            self.upscore_pool2.size = (x.size()[2],x.size()[3],)

        # Pre-trained layers
        # print(x.size())
        out1 = self.conv_block1(x)
        # print('1',out1.size())
        out2 = self.conv_block2(out1)
        # print('2',out2.size())
        out3 = self.conv_block3(out2)
        # print('3',out3.size())
        out4 = self.conv_block4(out3)
        # print('4',out4.size())
        out5 = self.conv_block5(out4)
        # print('5',out5.size())

        # Expanded fc layers
        out6 = self.conv_block6(out5)
        # print('6',out6.size())
        out7 = self.conv_block7(out6)
        # print('7',out7.size())

        out_score7 = self.score_conv7(out7)
        # print('score7',out_score7.size())
        # Upsampling layer x2
        out_upscore7 = self.upscore_conv7(out_score7)
        # print('score7*2',out_upscore7.size())

        # skip architecture from pool4
        out_score4 = self.score_pool4(out4)
        # print('score4,',out_score4.size())
        # fuse score
        fuse_score4 = out_score4 + out_upscore7
        # Upsampling layer x2
        out_upscore4 = self.upscore_pool4(fuse_score4)
        # print('score4*2,',out_upscore4.size())

        # skip architecture from pool3
        out_score3 = self.score_pool3(out3)
        # print('score3',out_score3.size())
        # fuse score
        fuse_score3 = out_score3 + out_upscore4
        # Upsampling layer x2
        out_upscore3 = self.upscore_pool3(fuse_score3)
        # print('score3*2,',out_upscore3.size())

        # skip architecture from pool2
        out_score2 = self.score_pool2(out2)
        # print('score2',out_score2.size())
        # fuse score
        fuse_score2 = out_score2 + out_upscore3
        # Upsampling layer x4
        out_upscore2 = self.upscore_pool2(fuse_score2)
        # print('score2*4,',out_upscore2.size())
        return out_upscore2

#===============================================================================
class FCN_2s(nn.Module):
    #===========================================================================
    def __init__(self, num_classes = 21):
        super(FCN_2s, self).__init__()
        self.num_classes = num_classes
        pretrained_vgg16 = models.vgg16(pretrained=True)

        # Separate pre-trained VGG16 features layer to each block
        self.conv_block1 = nn.Sequential()
        for idx in range(0,4+1):
            self.conv_block1.add_module(str(idx), pretrained_vgg16.features[idx])
        self.conv_block2 = nn.Sequential()
        for idx in range(5,9+1):
            self.conv_block2.add_module(str(idx), pretrained_vgg16.features[idx])
        self.conv_block3 = nn.Sequential()
        for idx in range(10,16+1):
            self.conv_block3.add_module(str(idx), pretrained_vgg16.features[idx])
        self.conv_block4 = nn.Sequential()
        for idx in range(17,23+1):
            self.conv_block4.add_module(str(idx), pretrained_vgg16.features[idx])
        self.conv_block5 = nn.Sequential()
        for idx in range(24,30+1):
            self.conv_block5.add_module(str(idx), pretrained_vgg16.features[idx])

        # fully convolutional layers
        self.conv_block6 = nn.Sequential(
                nn.Conv2d(in_channels=512, out_channels=4096, kernel_size=1),
                nn.ReLU(inplace=True),
                nn.Dropout2d(p=0.5),
                )
        self.conv_block7 = nn.Sequential(
                nn.Conv2d(4096, 4096, 1),
                nn.ReLU(inplace=True),
                nn.Dropout2d(p=0.5),
                )
        # last score layer
        self.score_conv7 = nn.Conv2d(4096, self.num_classes, kernel_size=1)
        self.score_pool4 = nn.Conv2d(512, self.num_classes, kernel_size=1)
        self.score_pool3 = nn.Conv2d(256, self.num_classes, kernel_size=1)
        self.score_pool2 = nn.Conv2d(128, self.num_classes, kernel_size=1)
        self.score_pool1 = nn.Conv2d(64, self.num_classes, kernel_size=1)
        # upsampling layer
        self.upscore_conv7 = nn.ConvTranspose2d(in_channels=self.num_classes,
                                        out_channels=self.num_classes,
                                        kernel_size=4,
                                        stride=2,   # 2x upscale
                                        padding=1,
                                        # groups=self.num_classes,
                                        bias=False)
        self.upscore_pool4 = nn.ConvTranspose2d(in_channels=self.num_classes,
                                        out_channels=self.num_classes,
                                        kernel_size=4,
                                        stride=2,   # 2x upscale
                                        padding=1,
                                        # groups=self.num_classes,
                                        bias=False)
        self.upscore_pool3 = nn.ConvTranspose2d(in_channels=self.num_classes,
                                        out_channels=self.num_classes,
                                        kernel_size=4,
                                        stride=2,   # 2x upscale
                                        padding=1,
                                        # groups=self.num_classes,
                                        bias=False)
        self.upscore_pool2 = nn.ConvTranspose2d(in_channels=self.num_classes,
                                        out_channels=self.num_classes,
                                        kernel_size=4,
                                        stride=2,   # 2x upscale
                                        padding=1,
                                        # groups=self.num_classes,
                                        bias=False)
        self.upscore_pool1 = nn.ConvTranspose2d(in_channels=self.num_classes,
                                        out_channels=self.num_classes,
                                        kernel_size=4,
                                        stride=2,   # 2x upscale
                                        padding=1,
                                        groups=self.num_classes,
                                        bias=False)

        # Zero-initialize the class scoring layer
        nn.init.zeros_(self.score_conv7.weight)
        nn.init.zeros_(self.score_conv7.bias)
        nn.init.zeros_(self.score_pool4.weight)
        nn.init.zeros_(self.score_pool4.bias)
        nn.init.zeros_(self.score_pool3.weight)
        nn.init.zeros_(self.score_pool3.bias)
        nn.init.zeros_(self.score_pool2.weight)
        nn.init.zeros_(self.score_pool2.bias)
        nn.init.zeros_(self.score_pool1.weight)
        nn.init.zeros_(self.score_pool1.bias)

        # Intermediate upsampling layers are initialized to bilinear upsampling, and then learned.
        self.upscore_conv7.weight = nn.Parameter(torch.squeeze(torch.stack([make_bilinear_weights(self.upscore_conv7.kernel_size[0], self.num_classes)]*self.num_classes)))
        self.upscore_pool4.weight = nn.Parameter(torch.squeeze(torch.stack([make_bilinear_weights(self.upscore_pool4.kernel_size[0], self.num_classes)]*self.num_classes)))
        self.upscore_pool3.weight = nn.Parameter(torch.squeeze(torch.stack([make_bilinear_weights(self.upscore_pool3.kernel_size[0], self.num_classes)]*self.num_classes)))
        self.upscore_pool2.weight = nn.Parameter(torch.squeeze(torch.stack([make_bilinear_weights(self.upscore_pool2.kernel_size[0], self.num_classes)]*self.num_classes)))
        # Final layer deconvolutional filters are fixed to bilinear interpolation.
        self.upscore_pool1.weight = nn.Parameter(make_bilinear_weights(self.upscore_pool2.kernel_size[0], self.num_classes))
        self.upscore_pool1.weight.requires_grad = False
        # Equivalent to bilinear deconvolution layer (to handle both odd and even size inputs)
        self.upscore_pool1 = nn.Upsample(mode='bilinear')

    #===========================================================================
    def forward(self, x):
        # Set the proper upsample size
        if isinstance(self.upscore_pool1, nn.Upsample):
            self.upscore_pool1.size = (x.size()[2],x.size()[3],)

        # Pre-trained layers
        # print(x.size())
        out1 = self.conv_block1(x)
        # print('1',out1.size())
        out2 = self.conv_block2(out1)
        # print('2',out2.size())
        out3 = self.conv_block3(out2)
        # print('3',out3.size())
        out4 = self.conv_block4(out3)
        # print('4',out4.size())
        out5 = self.conv_block5(out4)
        # print('5',out5.size())

        # Expanded fc layers
        out6 = self.conv_block6(out5)
        # print('6',out6.size())
        out7 = self.conv_block7(out6)
        # print('7',out7.size())

        out_score7 = self.score_conv7(out7)
        # print('score7',out_score7.size())
        # Upsampling layer x2
        out_upscore7 = self.upscore_conv7(out_score7)
        # print('score7*2',out_upscore16.size())

        # skip architecture from pool4
        out_score4 = self.score_pool4(out4)
        # print('score4,',out_score4.size())
        # fuse score
        fuse_score4 = out_score4 + out_upscore7
        # Upsampling layer x2
        out_upscore4 = self.upscore_pool4(fuse_score4)
        # print('score4*2,',out_upscore4.size())

        # skip architecture from pool3
        out_score3 = self.score_pool3(out3)
        # print('score3',out_score3.size())
        # fuse score
        fuse_score3 = out_score3 + out_upscore4
        # Upsampling layer x2
        out_upscore3 = self.upscore_pool3(fuse_score3)
        # print('score3*2,',out_upscore3.size())

        # skip architecture from pool2
        out_score2 = self.score_pool2(out2)
        # print('score2',out_score2.size())
        # fuse score
        fuse_score2 = out_score2 + out_upscore3
        # Upsampling layer x2
        out_upscore2 = self.upscore_pool2(fuse_score2)
        # print('score2*2,',out_upscore2.size())

        # skip architecture from pool1
        out_score1 = self.score_pool1(out1)
        # print('score1',out_score1.size())
        # fuse score
        fuse_score1 = out_score1 + out_upscore2
        # Upsampling layer x2
        out_upscore1 = self.upscore_pool1(fuse_score1)
        # print('score1*2,',out_upscore1.size())
        return out_upscore1

#===============================================================================
class FCN_1s(nn.Module):
    #===========================================================================
    def __init__(self, num_classes = 21):
        super(FCN_1s, self).__init__()
        self.num_classes = num_classes
        pretrained_vgg16 = models.vgg16(pretrained=True)

        # Separate pre-trained VGG16 features layer to each block
        self.conv_block1 = nn.Sequential()
        for idx in range(0,4+1):
            self.conv_block1.add_module(str(idx), pretrained_vgg16.features[idx])
        self.conv_block2 = nn.Sequential()
        for idx in range(5,9+1):
            self.conv_block2.add_module(str(idx), pretrained_vgg16.features[idx])
        self.conv_block3 = nn.Sequential()
        for idx in range(10,16+1):
            self.conv_block3.add_module(str(idx), pretrained_vgg16.features[idx])
        self.conv_block4 = nn.Sequential()
        for idx in range(17,23+1):
            self.conv_block4.add_module(str(idx), pretrained_vgg16.features[idx])
        self.conv_block5 = nn.Sequential()
        for idx in range(24,30+1):
            self.conv_block5.add_module(str(idx), pretrained_vgg16.features[idx])

        # fully convolutional layers
        self.conv_block6 = nn.Sequential(
                nn.Conv2d(in_channels=512, out_channels=4096, kernel_size=1),
                nn.ReLU(inplace=True),
                nn.Dropout2d(p=0.5),
                )
        self.conv_block7 = nn.Sequential(
                nn.Conv2d(4096, 4096, 1),
                nn.ReLU(inplace=True),
                nn.Dropout2d(p=0.5),
                )
        # last score layer
        self.score_conv7 = nn.Conv2d(4096, self.num_classes, kernel_size=1)
        self.score_pool4 = nn.Conv2d(512, self.num_classes, kernel_size=1)
        self.score_pool3 = nn.Conv2d(256, self.num_classes, kernel_size=1)
        self.score_pool2 = nn.Conv2d(128, self.num_classes, kernel_size=1)
        self.score_pool1 = nn.Conv2d(64, self.num_classes, kernel_size=1)
        self.score_poolx = nn.Conv2d(3, self.num_classes, kernel_size=1)
        # upsampling layer
        self.upscore_conv7 = nn.ConvTranspose2d(in_channels=self.num_classes,
                                        out_channels=self.num_classes,
                                        kernel_size=4,
                                        stride=2,   # 2x upscale
                                        padding=1,
                                        # groups=self.num_classes,
                                        bias=False)
        self.upscore_pool4 = nn.ConvTranspose2d(in_channels=self.num_classes,
                                        out_channels=self.num_classes,
                                        kernel_size=4,
                                        stride=2,   # 2x upscale
                                        padding=1,
                                        # groups=self.num_classes,
                                        bias=False)
        self.upscore_pool3 = nn.ConvTranspose2d(in_channels=self.num_classes,
                                        out_channels=self.num_classes,
                                        kernel_size=4,
                                        stride=2,   # 2x upscale
                                        padding=1,
                                        # groups=self.num_classes,
                                        bias=False)
        self.upscore_pool2 = nn.ConvTranspose2d(in_channels=self.num_classes,
                                        out_channels=self.num_classes,
                                        kernel_size=4,
                                        stride=2,   # 2x upscale
                                        padding=1,
                                        # groups=self.num_classes,
                                        bias=False)
        self.upscore_pool1 = nn.ConvTranspose2d(in_channels=self.num_classes,
                                        out_channels=self.num_classes,
                                        kernel_size=4,
                                        stride=2,   # 2x upscale
                                        padding=1,
                                        # groups=self.num_classes,
                                        bias=False)
        self.upscore_poolx = nn.ConvTranspose2d(in_channels=self.num_classes,
                                        out_channels=self.num_classes,
                                        kernel_size=2,
                                        stride=1,   # 1x upscale
                                        padding=0,
                                        groups=self.num_classes,
                                        bias=False)

        # Zero-initialize the class scoring layer
        nn.init.zeros_(self.score_conv7.weight)
        nn.init.zeros_(self.score_conv7.bias)
        nn.init.zeros_(self.score_pool4.weight)
        nn.init.zeros_(self.score_pool4.bias)
        nn.init.zeros_(self.score_pool3.weight)
        nn.init.zeros_(self.score_pool3.bias)
        nn.init.zeros_(self.score_pool2.weight)
        nn.init.zeros_(self.score_pool2.bias)
        nn.init.zeros_(self.score_pool1.weight)
        nn.init.zeros_(self.score_pool1.bias)
        nn.init.zeros_(self.score_poolx.weight)
        nn.init.zeros_(self.score_poolx.bias)

        # Intermediate upsampling layers are initialized to bilinear upsampling, and then learned.
        self.upscore_conv7.weight = nn.Parameter(torch.squeeze(torch.stack([make_bilinear_weights(self.upscore_conv7.kernel_size[0], self.num_classes)]*self.num_classes)))
        self.upscore_pool4.weight = nn.Parameter(torch.squeeze(torch.stack([make_bilinear_weights(self.upscore_pool4.kernel_size[0], self.num_classes)]*self.num_classes)))
        self.upscore_pool3.weight = nn.Parameter(torch.squeeze(torch.stack([make_bilinear_weights(self.upscore_pool3.kernel_size[0], self.num_classes)]*self.num_classes)))
        self.upscore_pool2.weight = nn.Parameter(torch.squeeze(torch.stack([make_bilinear_weights(self.upscore_pool2.kernel_size[0], self.num_classes)]*self.num_classes)))
        self.upscore_pool1.weight = nn.Parameter(torch.squeeze(torch.stack([make_bilinear_weights(self.upscore_pool1.kernel_size[0], self.num_classes)]*self.num_classes)))
        # Final layer deconvolutional filters are fixed to bilinear interpolation.
        self.upscore_poolx.weight = nn.Parameter(make_bilinear_weights(self.upscore_poolx.kernel_size[0], self.num_classes))
        self.upscore_poolx.weight.requires_grad = False
        # Equivalent to bilinear deconvolution layer (to handle both odd and even size inputs)
        self.upscore_poolx = nn.Upsample(mode='bilinear')

    #===========================================================================
    def forward(self, x):
        # Set the proper upsample size
        if isinstance(self.upscore_poolx, nn.Upsample):
            self.upscore_poolx.size = (x.size()[2],x.size()[3],)

        # Pre-trained layers
        # print(x.size())
        out1 = self.conv_block1(x)
        # print('1',out1.size())
        out2 = self.conv_block2(out1)
        # print('2',out2.size())
        out3 = self.conv_block3(out2)
        # print('3',out3.size())
        out4 = self.conv_block4(out3)
        # print('4',out4.size())
        out5 = self.conv_block5(out4)
        # print('5',out5.size())

        # Expanded fc layers
        out6 = self.conv_block6(out5)
        # print('6',out6.size())
        out7 = self.conv_block7(out6)
        # print('7',out7.size())

        out_score7 = self.score_conv7(out7)
        # print('score7',out_score7.size())
        # Upsampling layer x2
        out_upscore7 = self.upscore_conv7(out_score7)
        # print('score7*2',out_upscore7.size())

        # skip architecture from pool4
        out_score4 = self.score_pool4(out4)
        # print('score4,',out_score4.size())
        # fuse score
        fuse_score4 = out_score4 + out_upscore7
        # Upsampling layer x2
        out_upscore4 = self.upscore_pool4(fuse_score4)
        # print('score4*2,',out_upscore4.size())

        # skip architecture from pool3
        out_score3 = self.score_pool3(out3)
        # print('score3',out_score3.size())
        # fuse score
        fuse_score3 = out_score3 + out_upscore4
        # Upsampling layer x2
        out_upscore3 = self.upscore_pool3(fuse_score3)
        # print('score3*2,',out_upscore3.size())

        # skip architecture from pool2
        out_score2 = self.score_pool2(out2)
        # print('score2',out_score2.size())
        # fuse score
        fuse_score2 = out_score2 + out_upscore3
        # Upsampling layer x2
        out_upscore2 = self.upscore_pool2(fuse_score2)
        # print('score2*2,',out_upscore2.size())

        # skip architecture from pool1
        out_score1 = self.score_pool1(out1)
        # print('score1',out_score1.size())
        # fuse score
        fuse_score1 = out_score1 + out_upscore2
        # Upsampling layer x2
        out_upscore1 = self.upscore_pool1(fuse_score1)
        # print('score1*2,',out_upscore1.size())

        # skip architecture from poolx
        out_scorex = self.score_poolx(x)
        # print('scorex',out_scorex.size())
        # fuse score
        fuse_scorex = out_scorex + out_upscore1
        # Upsampling layer x1
        out_upscorex = self.upscore_poolx(fuse_scorex)
        # print('scorex*1,',out_upscorex.size())
        return out_upscorex

#===============================================================================
''' Make bilinear weight for deconvolution(Transposed Convolution) '''
# Reference: http://warmspringwinds.github.io/tensorflow/tf-slim/2016/11/22/upsampling-and-image-segmentation-with-tensorflow-and-tf-slim/
def make_bilinear_weights(size, num_channels):
    ''' Make a 2D bilinear kernel suitable for upsampling
    Stack the bilinear kernel for application to tensor '''
    factor = (size + 1) // 2
    if size % 2 == 1:
        center = factor - 1
    else:
        center = factor - 0.5
    og = np.ogrid[:size, :size]
    filt = (1 - abs(og[0] - center) / factor) * \
           (1 - abs(og[1] - center) / factor)

    filt = torch.from_numpy(filt)
    w = torch.zeros(num_channels, 1, size, size)
    for i in range(num_channels):
        w[i, 0] = filt
    return w

#===============================================================================
''' Test ResNet '''
if __name__ == '__main__':
    # FCN-32s
    # model = FCN_32s()
    # FCN-32s-fixed
    # model = FCN_32s_fixed()
    # FCN-16s
    # model = FCN_16s()
    # FCN-8s
    model = FCN_8s()
    # FCN-4s
    # model = FCN_4s()
    # FCN-2s
    # model = FCN_2s()
    # FCN-1s
    # model = FCN_1s()

    print(model)
    inputs = torch.randn(20,3,256,256)
    outputs = model(inputs)
    print(outputs.size())
