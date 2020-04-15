#-*- coding:utf-8 -*-
'''
[AI502] Deep Learning Assignment
"Fully Convolutional Networks" Implementation
20193640 Jungwon Choi
'''
import torch
import torch.nn as nn
import numpy as np
import sys

#===============================================================================
''' Train sequence '''
def train(model, train_loader, criterion, optimizer):
    model.train()
    device = next(model.parameters()).device.index
    losses = []
    total_iter = len(train_loader)

    for i, (images, labels) in enumerate(train_loader):
        images, labels = images.cuda(device), labels.cuda(device)

        # Predict labels (Forward propagation)
        pred_labels = model(images)

        # Calculate loss
        loss = criterion(pred_labels, labels)
        losses.append(loss.item())

        # Empty gradients
        optimizer.zero_grad()

        # Calculate gradients (Backpropagation)
        loss.backward()

        # Update parameters
        optimizer.step()

        sys.stdout.write("[{:5d}/{:5d}]\r".format(i+1, total_iter))

    # Calculate average loss
    avg_loss = sum(losses)/len(losses)

    #===========================================================================
    # Check train acc
    model.eval()

    if torch.cuda.device_count() > 1:
        n_cl = model.module.num_classes
    else:
        n_cl = model.num_classes

    # Histogram matrix (Row: Original class, Column: Predicted class)
    hist_mat = np.zeros((n_cl, n_cl)) # N x N
                                      # coordinate meaning (A, B) : n_AB
                                      # origin: class A -> predict: class B

    for i, (images, labels) in enumerate(train_loader):
        images, labels = images.cuda(device), labels.cuda(device)
        # Predict labels (Forward propagation)
        pred_labels = model(images)
        #=======================================================================
        # Make outputs as probability distribution
        pred_labels = nn.Softmax(dim=1)(pred_labels)
        # Choose top-1 label
        pred_masks = torch.argmax(pred_labels, dim=1)
        pred_masks, labels = pred_masks.cpu().numpy(), labels.cpu().numpy()
        # Calculate segemetation result histogram
        hist_mat += np.bincount(n_cl*labels.flatten()+pred_masks.flatten(),
                                    minlength=n_cl**2).reshape(n_cl, n_cl)

        sys.stdout.write("[{:5d}/{:5d}]\r".format(i+1, total_iter))

    # Calculate accuracy
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

    return avg_loss, (pixel_acc, mean_acc, mean_IoU, fw_IoU)
