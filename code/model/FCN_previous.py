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
class FCN_AlexNet(nn.Module):
    #===========================================================================
    def __init__(self, num_classes = 21):
        super(FCN_AlexNet, self).__init__()
        self.num_classes = num_classes
        pretrained_alexnet = models.alexnet(pretrained=True)

        # pre-trained feature layers
        self.features = pretrained_alexnet.features
        # fully convolutional layers
        self.classifier_fc = nn.Sequential(
                nn.Conv2d(in_channels=256, out_channels=4096, kernel_size=1),
                nn.ReLU(inplace=True),
                nn.Dropout2d(p=0.5),
                nn.Conv2d(4096, 4096, 1),
                nn.ReLU(inplace=True),
                nn.Dropout2d(p=0.5)
                )
        # last score layer
        self.score = nn.Conv2d(4096, self.num_classes, kernel_size=1)
        # upsampling layer
        self.upscore = nn.ConvTranspose2d(in_channels=self.num_classes,
                                        out_channels=self.num_classes,
                                        kernel_size=64,
                                        stride=32,
                                        padding=0,  # for odd size feature map
                                        groups=self.num_classes,
                                        bias=False)

        # Zero-initialize the class scoring layer
        nn.init.zeros_(self.score.weight)
        nn.init.zeros_(self.score.bias)

        # Final layer deconvolutional filters are fixed to bilinear interpolation.
        self.upscore.weight = nn.Parameter(make_bilinear_weights(self.upscore.kernel_size[0], self.num_classes))
        self.upscore.weight.requires_grad = False
        # Equivalent to bilinear deconvolution layer (to handle both odd and even size inputs)
        self.upscore = nn.Upsample(mode='bilinear')

    #===========================================================================
    def forward(self, x):
        # Set the proper upsample size
        if isinstance(self.upscore, nn.Upsample):
            self.upscore.size = (x.size()[2],x.size()[3],)

        # Pre-trained layers
        out = self.features(x)
        # print(out.size())
        # Expanded fc layers
        out = self.classifier_fc(out)
        # print(out.size())
        out = self.score(out)
        # print(out.size())

        # Upsampling layer
        upsample = self.upscore(out)
        return upsample

#===============================================================================
class FCN_VGG16(nn.Module):
    #===========================================================================
    def __init__(self, num_classes = 21):
        super(FCN_VGG16, self).__init__()
        self.num_classes = num_classes
        pretrained_vgg16 = models.vgg16(pretrained=True)

        # pre-trained feature layers
        self.features = pretrained_vgg16.features
        # fully convolutional layers
        self.classifier_fc = nn.Sequential(
                nn.Conv2d(in_channels=512, out_channels=4096, kernel_size=1),
                nn.ReLU(inplace=True),
                nn.Dropout2d(p=0.5),
                nn.Conv2d(4096, 4096, 1),
                nn.ReLU(inplace=True),
                nn.Dropout2d(p=0.5)
                )
        # last score layer
        self.score = nn.Conv2d(4096, self.num_classes, kernel_size=1)
        # upsampling layer
        self.upscore = nn.ConvTranspose2d(in_channels=self.num_classes,
                                        out_channels=self.num_classes,
                                        kernel_size=64,
                                        stride=32,
                                        padding=16,
                                        groups=self.num_classes,
                                        bias=False)

        # Zero-initialize the class scoring layer
        nn.init.zeros_(self.score.weight)
        nn.init.zeros_(self.score.bias)

        # Final layer deconvolutional filters are fixed to bilinear interpolation.
        self.upscore.weight = nn.Parameter(make_bilinear_weights(self.upscore.kernel_size[0], self.num_classes))
        self.upscore.weight.requires_grad = False
        # Equivalent to bilinear deconvolution layer (to handle both odd and even size inputs)
        self.upscore = nn.Upsample(mode='bilinear')

    #===========================================================================
    def forward(self, x):
        # Set the proper upsample size
        if isinstance(self.upscore, nn.Upsample):
            self.upscore.size = (x.size()[2],x.size()[3],)

        # Pre-trained layers
        out = self.features(x)

        # Expanded fc layers
        out = self.classifier_fc(out)
        out = self.score(out)

        # Upsampling layer
        upsample = self.upscore(out)
        return upsample

class FCN_GoogLeNet(nn.Module):
    #===========================================================================
    def __init__(self, num_classes = 21):
        super(FCN_GoogLeNet, self).__init__()
        self.num_classes = num_classes
        pretrained_googlenet = models.googlenet(pretrained=True)

        # pre-trained feature layers
        self.entry_block = nn.Sequential()
        self.entry_block.add_module('0', pretrained_googlenet.conv1)
        self.entry_block.add_module('1', pretrained_googlenet.maxpool1)
        self.entry_block.add_module('2', pretrained_googlenet.conv2)
        self.entry_block.add_module('3', pretrained_googlenet.conv3)
        self.entry_block.add_module('4', pretrained_googlenet.maxpool2)

        self.inception3 = nn.Sequential()
        self.inception3.add_module('0', pretrained_googlenet.inception3a)
        self.inception3.add_module('1', pretrained_googlenet.inception3b)
        self.inception3.add_module('2', pretrained_googlenet.maxpool3)

        self.inception4 = nn.Sequential()
        self.inception4.add_module('0', pretrained_googlenet.inception4a)
        self.inception4.add_module('1', pretrained_googlenet.inception4b)
        self.inception4.add_module('2', pretrained_googlenet.inception4c)
        self.inception4.add_module('3', pretrained_googlenet.inception4d)
        self.inception4.add_module('4', pretrained_googlenet.inception4e)
        self.inception4.add_module('5', pretrained_googlenet.maxpool4)

        self.inception5 = nn.Sequential()
        self.inception5.add_module('0', pretrained_googlenet.inception5a)
        self.inception5.add_module('1', pretrained_googlenet.inception5b)

        # fully convolutional layers
        self.dropout = nn.Dropout2d(p=0.5)
        # last score layer
        self.score = nn.Conv2d(1024, self.num_classes, kernel_size=1)
        # upsampling layer
        self.upscore = nn.ConvTranspose2d(in_channels=self.num_classes,
                                        out_channels=self.num_classes,
                                        kernel_size=64,
                                        stride=32,
                                        padding=16,
                                        groups=self.num_classes,
                                        bias=False)

        # Zero-initialize the class scoring layer
        nn.init.zeros_(self.score.weight)
        nn.init.zeros_(self.score.bias)

        # Final layer deconvolutional filters are fixed to bilinear interpolation.
        self.upscore.weight = nn.Parameter(make_bilinear_weights(self.upscore.kernel_size[0], self.num_classes))
        self.upscore.weight.requires_grad = False
        # Equivalent to bilinear deconvolution layer (to handle both odd and even size inputs)
        self.upscore = nn.Upsample(mode='bilinear')

    #===========================================================================
    def forward(self, x):
        # Set the proper upsample size
        if isinstance(self.upscore, nn.Upsample):
            self.upscore.size = (x.size()[2],x.size()[3],)

        # Pre-trained layers
        out = self.entry_block(x)
        out = self.inception3(out)
        out = self.inception4(out)
        out = self.inception5(out)

        # Expanded fc layers
        out = self.dropout(out)
        out = self.score(out)

        # Upsampling layer
        upsample = self.upscore(out)
        return upsample

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
''' Test FCN previous models '''
if __name__ == '__main__':
    # FCN-AlexNet1
    model = FCN_AlexNet()
    # FCN-VGG16
    # model = FCN_VGG16()
    # FCN-GoogLeNet
    # model = FCN_GoogLeNet()

    print(model)
    inputs = torch.randn(20,3,256,256)
    outputs = model(inputs)
    print(outputs.size())
