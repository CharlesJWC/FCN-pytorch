#-*- coding:utf-8 -*-
'''
[AI502] Deep Learning Assignment
"Fully Convolutional Networks" Implementation
20193640 Jungwon Choi
'''
import os
import sys
import time
import numpy as np
import pickle
import matplotlib.pyplot as plt
import matplotlib as mpl
# import seaborn as sns

import PIL
from PIL import Image

import torch
import torch.nn as nn
import torchvision.transforms as transforms

from dataloader import VOC2012_Dataloader, VOC2011_Dataloader, VOC2012_Dataset
from model.FCN_previous import FCN_AlexNet, FCN_VGG16, FCN_GoogLeNet
from model.FCN_proposed import FCN_32s_fixed, FCN_32s, FCN_16s, FCN_8s
from model.FCN_proposed import FCN_4s, FCN_2s, FCN_1s
from model.DeconvNet import DeconvNet

FIGURE_PATH = './figures'
RESULT_PATH = './results'
CHECKPOINT_PATH ='./checkpoints/'

#===============================================================================
# For a single image
TEST_IMAGE_NAME = 'example.jpg'
VGG16_FILE_NAME = 'FCN_VGG16_200_20_SGD_0.00050_0.00010.ckpt'
TEST_IMAGE_PATH = os.path.join(FIGURE_PATH, TEST_IMAGE_NAME)
TARGET_LABLES = [13, 15]
VOC_CLASSES = ('background',                                    # 0
       'aeroplane', 'bicycle', 'bird','boat', 'bottle',         # 1 2 3 4 5
       'bus', 'car', 'cat', 'chair', 'cow',                     # 6 7 8 9 10
       'diningtable', 'dog', 'horse', 'motorbike', 'person',    # 11 12 13 14 15
       'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor')    # 16 17 18 19 20
#===============================================================================
# For the test dataset
TEST_BATCH_SIZE = 20
TEST_RESULT_BATCH = [
    #===========================================================================
    # VOC2011 Adam
    #===========================================================================
    # ('FCN_VGG16', 'VOC2011', 'train', 'FCN_VGG16_200_20_Adam_0.00050_0.00010.ckpt'),
    # ('FCN_VGG16', 'VOC2011', 'val', 'FCN_VGG16_200_20_Adam_0.00050_0.00010.ckpt'),
    # ('FCN_AlexNet', 'VOC2011', 'val', 'FCN_AlexNet_200_20_Adam_0.00050_0.00100.ckpt'),
    # ('FCN_GoogLeNet', 'VOC2011', 'val', 'FCN_GoogLeNet_200_20_Adam_0.00050_0.00005.ckpt'),
    #===========================================================================
    # ('FCN_32s', 'VOC2011', 'val', 'FCN_32s_200_20_Adam_0.00050_0.00010.ckpt'),
    # ('FCN_32s_fixed', 'VOC2011', 'val', 'FCN_32s_fixed_200_20_Adam_0.00050_0.00010.ckpt'),
    # ('FCN_16s', 'VOC2011', 'val', 'FCN_16s_200_20_Adam_0.00050_0.00010.ckpt'),
    # ('FCN_8s', 'VOC2011', 'val', 'FCN_8s_200_20_Adam_0.00050_0.00010.ckpt'),
    # ('FCN_4s', 'VOC2011', 'val', 'FCN_4s_200_20_Adam_0.0005_0.000100.ckpt'),
    # ('FCN_2s', 'VOC2011', 'val', 'FCN_2s_200_20_Adam_0.0005_0.000100.ckpt'),
    # ('FCN_1s', 'VOC2011', 'val', 'FCN_1s_VOC2011_200_20_Adam_0.0005_0.000100.ckpt'),
    #===========================================================================
    # Gourps comparison
    # ('FCN_8s', 'VOC2011', 'val', 'FCN_8s_VOC2011_200_20_Adam_0.0005_0.000100_groups.ckpt'),
    #===========================================================================
    # VOC2011 SGD
    #===========================================================================
    # ('FCN_VGG16', 'VOC2011', 'train', 'FCN_VGG16_200_20_SGD_0.00050_0.00010.ckpt'),
    # ('FCN_VGG16', 'VOC2011', 'val', 'FCN_VGG16_200_20_SGD_0.00050_0.00010.ckpt'),
    # ('FCN_AlexNet', 'VOC2011', 'val', 'FCN_AlexNet_200_20_SGD_0.00050_0.00100.ckpt'),
    # ('FCN_GoogLeNet', 'VOC2011', 'val', 'FCN_GoogLeNet_200_20_SGD_0.0005_0.000050.ckpt'),
    #===========================================================================
    # ('FCN_32s', 'VOC2011', 'val', 'FCN_32s_200_20_SGD_0.0005_0.000100.ckpt'),
    # ('FCN_32s_fixed', 'VOC2011', 'val', 'FCN_32s_fixed_VOC2011_200_20_SGD_0.0005_0.000100.ckpt'),
    # ('FCN_16s', 'VOC2011', 'val', 'FCN_16s_VOC2011_200_20_SGD_0.0005_0.001000.ckpt'),
    # ('FCN_8s', 'VOC2011', 'val', 'FCN_8s_VOC2011_200_20_SGD_0.0005_0.001000.ckpt'),
    # ('FCN_4s', 'VOC2011', 'val', 'FCN_4s_VOC2011_200_20_SGD_0.0005_0.000010.ckpt'),
    # ('FCN_2s', 'VOC2011', 'val', 'FCN_2s_VOC2011_200_20_SGD_0.0005_0.000001.ckpt'),
    # ('FCN_1s', 'VOC2011', 'val', 'FCN_1s_VOC2011_200_20_SGD_0.0005_0.000001.ckpt'),
    #===========================================================================
    # VOC2012 SGD
    #===========================================================================
    # ('FCN_32s', 'VOC2012', 'val', 'FCN_32s_VOC2012_200_20_SGD_0.0005_0.000100.ckpt'),
    # ('FCN_16s', 'VOC2012', 'val', 'FCN_16s_VOC2012_200_20_SGD_0.0005_0.001000.ckpt'),
    # ('FCN_8s', 'VOC2012', 'val', 'FCN_8s_VOC2012_200_20_SGD_0.0005_0.000010.ckpt'),
    ('DeconvNet', 'VOC2012', 'val', 'DeconvNet_VOC2012_500_20_SGD_0.0005_0.010000.ckpt'),
]
#===============================================================================
# For loss graph visualization
RESULTS_FILE_NAME = 'DeconvNet_VOC2012_500_20_SGD_0.0005_0.010000_results.pkl'
#===============================================================================
# For re-sized dateset image
DATASET_BATCH = [
    ('VOC2011', 'train'),
    ('VOC2011', 'val'),
    ('VOC2012', 'train'),
    ('VOC2012', 'val'),
]
#===============================================================================
# For foward time check
TEST_MODEL_BATCH = [
    ('FCN_VGG16', 'FCN_VGG16_200_20_SGD_0.00050_0.00010.ckpt'),
    ('FCN_AlexNet', 'FCN_AlexNet_200_20_SGD_0.00050_0.00100.ckpt'),
    ('FCN_GoogLeNet', 'FCN_GoogLeNet_200_20_SGD_0.0005_0.000050.ckpt'),
]

#===============================================================================
''' Create segmentation label colorbar '''
def makeColorBar():
    color_list = []
    for label in range(20,-1,-1):
        color_list.append(np.full((10,10), label))
    color_label = np.vstack(color_list)
    color_bar = cvtLabeltoRGB(color_label)

    fig = plt.figure()
    plt.imshow(np.transpose(color_bar, (1, 2, 0)))
    plt.xticks([]), plt.yticks(np.arange(205, 0, -10), VOC2012_Dataset.classes)
    plt.ylim(0,210), plt.tight_layout()
    plt.subplots_adjust(left=0, bottom=0, right=1, top=1, hspace=0, wspace=0)
    fig.savefig(os.path.join(FIGURE_PATH, 'color_bar.jpg'),
            bbox_inces='tight', pad_inches=0, dpi=150)
    plt.close(fig)
    print('==> Make colorbar done.')

#===============================================================================
''' Create resized dataset image '''
def make_resized_dataset(dataset_batches):
    for dataset_batch in dataset_batches:
        dataset, data_type = dataset_batch
        #=======================================================================
        # Select the dataset
        if dataset == 'VOC2011':
            voc = VOC2011_Dataloader()
        elif dataset == 'VOC2012':
            voc = VOC2012_Dataloader()

        if data_type == 'train':
            dataloader = voc.get_train_loader(batch_size=1, num_workers=8)
        elif data_type == 'val':
            dataloader = voc.get_val_loader(batch_size=1, num_workers=8)

        #=======================================================================
        # Set the save path
        SAVE_IMG_PATH = os.path.join(FIGURE_PATH, dataset,data_type)
        # Check the directory of the file path
        if not os.path.exists(SAVE_IMG_PATH):
            os.makedirs(SAVE_IMG_PATH)
        print('\nRe-sized image save path:', SAVE_IMG_PATH)
        #=======================================================================
        # Save the each resized image
        total_iter = len(dataloader)
        for ii, (resized_image, _) in enumerate(dataloader):
            IMAGE_NAME = os.path.basename(voc.dataset.segm_images[ii])

            resized_image = unNormalize(torch.squeeze(resized_image))

            fig = plt.figure()
            plt.imshow(np.transpose(resized_image.numpy(), (1, 2, 0)))
            plt.axis('off'), plt.xticks([]), plt.yticks([]), plt.tight_layout()
            plt.subplots_adjust(left=0, bottom=0, right=1, top=1, hspace=0, wspace=0)
            fig.savefig(os.path.join(SAVE_IMG_PATH, 'resized_'+IMAGE_NAME),
                    bbox_inces='tight', pad_inches=0, dpi=150)
            plt.close(fig)

            sys.stdout.write("[{:5d}/{:5d}]\r".format(ii+1, total_iter))
    print('\n==> Save re-sized images done.')

#===============================================================================
''' Image unnormalization '''
def unNormalize(image):
    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)
    for c, m, s in zip(image, mean, std):
        c.mul_(s).add_(m)
    return image

#===============================================================================
''' Re-define the network for watching score map '''
class FCN_model(nn.Module):
    #===========================================================================
    def __init__(self, model=None, ckpt_path=None):
        super(FCN_model, self).__init__()
        assert model != None, 'Please input the model!'
        assert os.path.exists(ckpt_path), 'There is no such file!'
        # Get pre-trained parameters
        model.load_state_dict(torch.load(ckpt_path)['model_state_dict'])
        # pre-trained layers
        self.features = model.features
        self.classifier_fc = model.classifier_fc
        self.score = model.score
        self.upscore = model.upscore

    #===========================================================================
    def forward(self, x):
        # Set the proper upsample size
        if isinstance(self.upscore, nn.Upsample):
            self.upscore.size = (x.size()[2],x.size()[3],)
        # Feature layers
        out = self.features(x)
        # Expanded fc layers
        out = self.classifier_fc(out)
        score = self.score(out)
        # Upsampling layer
        upsample = self.upscore(score)
        return score, upsample

#===============================================================================
''' Visualize the heatmap image of score map '''
def visualize_heatmap_results(model, image_path, ckpt_name):
    #===========================================================================
    # Load the image
    image_origin = Image.open(image_path).convert('RGB')
    # Resize the image
    image_resized = image_origin.resize((256, 256), PIL.Image.BILINEAR)
    # Image normaliztion
    ii = np.transpose(np.asarray(image_resized), (2, 0, 1))/255.0
    transform = transforms.Compose([ transforms.ToTensor(),
                transforms.Normalize((ii[0].mean(), ii[1].mean(), ii[2].mean()),
                                    (ii[0].var(), ii[1].var(), ii[2].var()))
                ])
    # Transfrom the image
    image = transform(image_resized)
    # Make the image as a batch
    image = torch.unsqueeze(image, 0)
    print('==> Image ready.')
    #===========================================================================
    # Load pre-trained model
    ckpt_path = os.path.join(CHECKPOINT_PATH, ckpt_name)
    model = FCN_model(model, ckpt_path)
    print('==> Model ready.')

    #===========================================================================
    # Forward the results
    score, mask_label = model(image)

    # Make outputs as probability distribution
    score = nn.Softmax(dim=1)(score)
    mask_label = nn.Softmax(dim=1)(mask_label)

    # Get the label mask and segmented image.
    mask_label = torch.squeeze(torch.argmax(mask_label, dim=1))
    segmented_image = cvtLabeltoRGB(mask_label)

    # Get the score map.
    score = nn.Upsample((32,32), mode='bilinear')(score)
    print('==> Results ready.')

    #===========================================================================
    # Save the results image.
    IMAGE_NAME = os.path.splitext(TEST_IMAGE_NAME)[0]

    # Re-sized image
    fig = plt.figure()
    plt.imshow(np.asarray(image_resized))
    plt.axis('off'), plt.xticks([]), plt.yticks([]), plt.tight_layout()
    plt.subplots_adjust(left=0, bottom=0, right=1, top=1, hspace=0, wspace=0)
    fig.savefig(os.path.join(FIGURE_PATH, IMAGE_NAME+'_resized.jpg'),
            bbox_inces='tight', pad_inches=0, dpi=150)
    # plt.show()
    plt.close(fig)

    # Heatmap image
    for label in TARGET_LABLES:
        fig = plt.figure()
        plt.pcolor(torch.squeeze(score)[label].detach().numpy())
        # sns.heatmap(cat_scoremap.detach().numpy())
        plt.axis('off'), plt.xticks([]), plt.yticks([]), plt.tight_layout()
        plt.colorbar()
        fig.savefig(os.path.join(FIGURE_PATH, IMAGE_NAME+'_heatmap_'+VOC_CLASSES[label]+'.png'),
                bbox_inces='tight', pad_inches=0, dpi=150, format='png')
        # plt.show()
        plt.close(fig)

    # Segmented image
    fig = plt.figure()
    plt.imshow(np.transpose(segmented_image, (1, 2, 0)))
    plt.axis('off'), plt.xticks([]), plt.yticks([]), plt.tight_layout()
    plt.subplots_adjust(left=0, bottom=0, right=1, top=1, hspace=0, wspace=0)
    fig.savefig(os.path.join(FIGURE_PATH, IMAGE_NAME+'_segmented.png'),
            bbox_inces='tight', pad_inches=0, dpi=150, format='png')
    # plt.show()
    plt.close(fig)
    print('==> Save results done.')

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
''' Visualize the segemetation image by the model '''
def visualize_segmentation_results(result_batches):
    for result_batch in result_batches:
        model_name, dataset, data_type, ckpt_name = result_batch
        #=======================================================================
        # Set the save file path
        MODEL_CONFIG_NAME = os.path.splitext(ckpt_name)[0]
        SAVE_IMG_PATH = os.path.join(FIGURE_PATH, MODEL_CONFIG_NAME)
        if data_type == 'train':
            SAVE_IMG_PATH = os.path.join(SAVE_IMG_PATH, 'train')

        # Check the directory of the file path
        if not os.path.exists(SAVE_IMG_PATH):
            os.makedirs(SAVE_IMG_PATH)
        print('\nSegmented image save path:', SAVE_IMG_PATH)

        #=======================================================================
        # Select the dataset
        if dataset == 'VOC2011':
            voc = VOC2011_Dataloader()
        elif dataset == 'VOC2012':
            voc = VOC2012_Dataloader()

        if data_type == 'train':
            dataloader = voc.get_train_loader(batch_size=TEST_BATCH_SIZE,
                                                    num_workers=8)
        elif data_type == 'val':
            dataloader = voc.get_val_loader(batch_size=TEST_BATCH_SIZE,
                                                    num_workers=8)

        #=======================================================================
        # Load the model
        if model_name == 'FCN_AlexNet': model = FCN_AlexNet()
        elif model_name == 'FCN_VGG16': model = FCN_VGG16()
        elif model_name == 'FCN_GoogLeNet': model = FCN_GoogLeNet()
        elif model_name == 'FCN_32s': model = FCN_32s()
        elif model_name == 'FCN_32s_fixed': model = FCN_32s_fixed()
        elif model_name == 'FCN_16s': model = FCN_16s()
        elif model_name == 'FCN_8s': model = FCN_8s()
        elif model_name == 'FCN_4s': model = FCN_4s()
        elif model_name == 'FCN_2s': model = FCN_2s()
        elif model_name == 'FCN_1s': model = FCN_1s()
        elif model_name == 'DeconvNet': model = DeconvNet()
        else: assert False, "Please select the FCN model"

        # Check DataParallel available
        if torch.cuda.device_count() > 1:
            model = nn.DataParallel(model)

        # Check CUDA available
        if torch.cuda.is_available():
            model.cuda()

        # Load pre-trained parameters
        ckpt_path = os.path.join(CHECKPOINT_PATH, ckpt_name)
        model.load_state_dict(torch.load(ckpt_path)['model_state_dict'])

        #=======================================================================
        # Model evaluation
        model.eval()
        device = next(model.parameters()).device.index
        total_iter = len(voc.dataset)

        if torch.cuda.device_count() > 1:
            n_cl = model.module.num_classes
        else:
            n_cl = model.num_classes

        # Histogram matrix (Row: Original class, Column: Predicted class)
        hist_mat = np.zeros((n_cl, n_cl)) # N x N
                                          # coordinate meaning (A, B) : n_AB
                                          # origin: class A -> predict: class B

        for i, (images, labels) in enumerate(dataloader):
            images, labels = images.cuda(device), labels.cuda(device)
            # Predict labels (Forward propagation)
            pred_labels = model(images)
            # Make outputs as probability distribution
            pred_labels = nn.Softmax(dim=1)(pred_labels)
            # Choose top-1 label
            pred_masks = torch.argmax(pred_labels, dim=1)
            pred_masks, labels = pred_masks.cpu().numpy(), labels.cpu().numpy()
            # Calculate segemetation result histogram
            hist_mat += np.bincount(n_cl*labels.flatten()+pred_masks.flatten(),
                                        minlength=n_cl**2).reshape(n_cl, n_cl)

            # Save the each result image in batch
            for j, pred_mask in enumerate(pred_masks):
                # segmented_image = cvtLabeltoRGB(pred_mask)
                # IMAGE_NAME = os.path.basename(voc.dataset.segm_images[i*TEST_BATCH_SIZE+j])
                #
                # fig = plt.figure()
                # plt.imshow(np.transpose(segmented_image, (1, 2, 0)))
                # plt.axis('off'), plt.xticks([]), plt.yticks([]), plt.tight_layout()
                # plt.subplots_adjust(left=0, bottom=0, right=1, top=1, hspace=0, wspace=0)
                # fig.savefig(os.path.join(SAVE_IMG_PATH, 'segmented_'+IMAGE_NAME),
                #         bbox_inces='tight', pad_inches=0, dpi=150)
                # plt.close(fig)
                sys.stdout.write("[{:5d}/{:5d}]\r".format(i*TEST_BATCH_SIZE+j+1, total_iter))
        #=======================================================================
        # Pixel accuracy
        pixel_acc = np.diag(hist_mat).sum() / hist_mat.sum()
        # Accuracy per class
        acc_pcl = np.diag(hist_mat) / hist_mat.sum(axis=1)
        # Mean accuracy
        mean_acc = (acc_pcl).mean()
        # IoU per-class
        IoU_pcl = np.diag(hist_mat) / (hist_mat.sum(axis=1)+hist_mat.sum(axis=0)
                                                    -np.diag(hist_mat))
        # Mean IoU
        mean_IoU = (IoU_pcl).mean()
        # frequency weight per-class
        fw_pcl = hist_mat.sum(axis=1) / hist_mat.sum()
        # frequency weighted IoU
        fw_IoU = (fw_pcl*IoU_pcl).sum()

        # Save the metric results
        result_file_name = MODEL_CONFIG_NAME+'_metric_results.txt'
        result_file = open(os.path.join(SAVE_IMG_PATH, result_file_name),'w')
        result_file.write("Pixel Acc: {:.1f}\t".format(pixel_acc*100))
        result_file.write("Mean  Acc: {:.1f}\t".format(mean_acc*100))
        result_file.write("Mean  IoU: {:.1f}\t".format(mean_IoU*100))
        result_file.write("frew  IoU: {:.1f}\n".format(fw_IoU)*100)
        result_file.close()

        print("Pixel Acc: {:.1f}\tMean  Acc: {:.1f}\t".format(pixel_acc*100,
                                                              mean_acc*100)
              +"Mean  IoU: {:.1f}\tfrew  IoU: {:.1f}".format(mean_IoU*100,
                                                               fw_IoU*100))
    print('\n==> Segmented image visualization done.')

def visualize_result_graph(plk_file_name):
    #===========================================================================
    # Load results data
    plk_file_name = os.path.join(RESULT_PATH, plk_file_name)
    with open(plk_file_name, 'rb') as pkl_file:
        result_dict = pickle.load(pkl_file)

    train_loss = result_dict['train_loss']
    val_loss = result_dict['val_loss']

    #===========================================================================
    # Save figure
    RESULT_NAME = os.path.splitext(plk_file_name)[0]

    num_epoch = len(train_loss)
    epochs = np.arange(1, num_epoch+1)
    fig = plt.figure(dpi=150)
    plt.title('Train error'), plt.xlabel('Epochs'), plt.ylabel('Loss')
    plt.xlim([0, num_epoch])#, plt.ylim([0, 60])
    plt.plot(epochs, train_loss,'--', markersize=1, alpha=0.8, label='train')
    plt.plot(epochs, val_loss,'-', markersize=1, alpha=0.8, label='val')
    plt.legend()
    file_name = "Loss_graph_{}.png".format(RESULT_NAME)
    fig.savefig(os.path.join(FIGURE_PATH, file_name),format='png')
    print('==> Loss graph visualization done.')

def foward_time_check(model_batches):
    for model_batch in model_batches:
        model_name, ckpt_name = model_batch
        #=======================================================================
        # Load the model
        if model_name == 'FCN_AlexNet': model = FCN_AlexNet()
        elif model_name == 'FCN_VGG16': model = FCN_VGG16()
        elif model_name == 'FCN_GoogLeNet': model = FCN_GoogLeNet()
        elif model_name == 'FCN_32s': model = FCN_32s()
        elif model_name == 'FCN_32s_fixed': model = FCN_32s_fixed()
        elif model_name == 'FCN_16s': model = FCN_16s()
        elif model_name == 'FCN_8s': model = FCN_8s()
        elif model_name == 'FCN_4s': model = FCN_4s()
        elif model_name == 'FCN_2s': model = FCN_2s()
        elif model_name == 'FCN_1s': model = FCN_1s()
        elif model_name == 'DeconvNet': model = DeconvNet()
        else: assert False, "Please select the FCN model"

        # Check DataParallel available
        if torch.cuda.device_count() > 1:
            model = nn.DataParallel(model)

        # Check CUDA available
        if torch.cuda.is_available():
            model.cuda()

        # Load pre-trained parameters
        ckpt_path = os.path.join(CHECKPOINT_PATH, ckpt_name)
        model.load_state_dict(torch.load(ckpt_path)['model_state_dict'])
        #=======================================================================
        # Test the model
        model.eval()
        device = next(model.parameters()).device.index
        input_data = torch.randn(1,3,512,512).cuda(device)

        start_time = time.time()
        #=======================================================================
        for ii in range(20):
            model(input_data)
        #=======================================================================
        end_time = time.time()

        print("Model: {}\t\tFoward time: {:.1f} ms".format(model_name,
                                        (end_time-start_time)*1000/20))
    print('==> Forward time check done.')

#===============================================================================
if __name__ == '__main__':
    # makeColorBar()
    # make_resized_dataset(DATASET_BATCH)
    # visualize_heatmap_results(FCN_VGG16(), TEST_IMAGE_PATH, VGG16_FILE_NAME)
    visualize_segmentation_results(TEST_RESULT_BATCH)
    # visualize_result_graph(RESULTS_FILE_NAME)
    # foward_time_check(TEST_MODEL_BATCH)
    pass
