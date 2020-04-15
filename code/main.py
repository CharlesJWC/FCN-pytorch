#-*- coding:utf-8 -*-
'''
[AI502] Deep Learning Assignment
"Fully Convolutional Networks" Implementation
20193640 Jungwon Choi
'''
import torch
import torch.nn as nn
import torch.optim as optim

import argparse
import random
import pickle
import time
import os

# Implementation files
from dataloader import VOC2012_Dataloader, VOC2011_Dataloader
from model.FCN_previous import FCN_AlexNet, FCN_VGG16, FCN_GoogLeNet
from model.FCN_proposed import FCN_32s_fixed, FCN_32s, FCN_16s, FCN_8s
from model.FCN_proposed import FCN_4s, FCN_2s, FCN_1s
# from model.DeconvNet import DeconvNet
from train import train
from val import val

# from torch.optim.lr_scheduler import ReduceLROnPlateau

# os.environ["CUDA_VISIBLE_DEVICES"]="0,1,2,3"

VERSION_CHECK_MESSAGE = 'NOW 19-10-21 16:00'

# Set the directory paths
RESULTS_PATH = './results/'
CHECKPOINT_PATH ='./checkpoints/'

#===============================================================================
''' Experiment1 : From classifier to dense FCN
    (FCN-AlexNet, FCN-VGG16, FCN-GoogLeNet) '''
''' Experiment2 : Combining what and where
    (FCN-32s-fixed, FCN-32s, FCN-16s, FCN-8s) '''
def main(args):
    #===========================================================================
    # Set the file name format
    FILE_NAME_FORMAT = "{0}_{1}_{2:d}_{3:d}_{4}_{5:.4f}_{6:.6f}{7}".format(
                                    args.model, args.dataset, args.epochs,
                                    args.batch_size, args.optimizer,
                                    args.weight_decay, args.lr, args.flag)
    # Set the results file path
    RESULT_FILE_NAME = FILE_NAME_FORMAT+'_results.pkl'
    RESULT_FILE_PATH = os.path.join(RESULTS_PATH, RESULT_FILE_NAME)
    # Set the checkpoint file path
    CHECKPOINT_FILE_NAME = FILE_NAME_FORMAT+'.ckpt'
    CHECKPOINT_FILE_PATH = os.path.join(CHECKPOINT_PATH, CHECKPOINT_FILE_NAME)
    BEST_CHECKPOINT_FILE_NAME = FILE_NAME_FORMAT+'_best.ckpt'
    BEST_CHECKPOINT_FILE_PATH = os.path.join(CHECKPOINT_PATH,
                                                BEST_CHECKPOINT_FILE_NAME)

    # Set the random seed same
    torch.manual_seed(190811)
    torch.cuda.manual_seed(190811)
    torch.cuda.manual_seed_all(190811)
    random.seed(190811)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # Step1 ====================================================================
    # Load dataset
    if args.dataset == 'VOC2011':
        voc = VOC2011_Dataloader()
    elif args.dataset == 'VOC2012':
        voc = VOC2012_Dataloader()
    else:
        assert False, "Please select the proper dataset!"
    train_loader = voc.get_train_loader(batch_size=args.batch_size,
                                            num_workers=args.num_workers)
    val_loader = voc.get_val_loader(batch_size=args.batch_size,
                                            num_workers=args.num_workers)
    print('==> DataLoader ready.')

    # Step2 ====================================================================
    # Make FCN model
    if args.model == 'FCN_AlexNet':
        model = FCN_AlexNet()
    elif args.model == 'FCN_VGG16':
        model = FCN_VGG16()
    elif args.model == 'FCN_GoogLeNet':
        model = FCN_GoogLeNet()
    elif args.model == 'FCN_32s':
        model = FCN_32s()
    elif args.model == 'FCN_32s_fixed':
        model = FCN_32s_fixed()
    elif args.model == 'FCN_16s':
        model = FCN_16s()
        model.load_state_dict(torch.load('./model/pretrained/FCN32s_'
                                    +args.dataset+'_'+args.optimizer
                                    )['model_state_dict'], strict=False)
    elif args.model == 'FCN_8s':
        model = FCN_8s()
        model.load_state_dict(torch.load('./model/pretrained/FCN16s_'
                                    +args.dataset+'_'+args.optimizer
                                    )['model_state_dict'], strict=False)
    elif args.model == 'FCN_4s':
        model = FCN_4s()
        model.load_state_dict(torch.load('./model/pretrained/FCN8s_'
                                    +args.dataset+'_'+args.optimizer
                                    )['model_state_dict'], strict=False)
    elif args.model == 'FCN_2s':
        model = FCN_2s()
        model.load_state_dict(torch.load('./model/pretrained/FCN4s_'
                                    +args.dataset+'_'+args.optimizer
                                    )['model_state_dict'], strict=False)
    elif args.model == 'FCN_1s':
        model = FCN_1s()
        model.load_state_dict(torch.load('./model/pretrained/FCN2s_'
                                    +args.dataset+'_'+args.optimizer
                                    )['model_state_dict'], strict=False)
    elif args.model == 'DeconvNet':
        model = DeconvNet()
    else:
        assert False, "Please select the FCN model"

    # Check DataParallel available
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)

    # Check CUDA available
    if torch.cuda.is_available():
        model.cuda()
    print('==> Model ready.')

    # Step3 ====================================================================
    # Set loss function and optimizer
    criterion = nn.CrossEntropyLoss()

    # Separate parameters for bias double learning rate
    normal_parameters = []
    double_parameters = []
    for name, parameter in model.named_parameters():
        if 'bias' in name:
            double_parameters.append(parameter)
        else:
            normal_parameters.append(parameter)

    # Select the optimizer
    if args.optimizer == 'SGD':
        optimizer = optim.SGD([
            {'params': normal_parameters},
            {'params': double_parameters, 'lr':args.lr*2,
                                          'weight_decay': args.weight_decay*0},
            ], lr=args.lr, momentum=0.9, weight_decay=args.weight_decay)
    elif args.optimizer == 'Adam':
        optimizer = optim.Adam([
            {'params': normal_parameters},
            {'params': double_parameters, 'lr':args.lr*2,
                                          'weight_decay': args.weight_decay*0},
            ], lr=args.lr, betas=(0.9, 0.999), weight_decay=args.weight_decay)
    else:
        assert False, "Please select the proper optimizer."

    # Set the learning rate scheduler
    # scheduler = ReduceLROnPlateau(optimizer, factor=0.1, patience=10, verbose=True, threshold=1e-4)
    print('==> Criterion and optimizer ready.')

    # Step4 ====================================================================
    # Train and validate the model
    start_epoch = 0
    best_val_mean_IoU = 0

    if args.resume:
        assert os.path.exists(CHECKPOINT_FILE_PATH), 'No checkpoint file!'
        checkpoint = torch.load(CHECKPOINT_FILE_PATH)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch']

    # Save the training information
    result_data = {}
    result_data['model']            = args.model
    result_data['dataset']          = args.dataset
    result_data['target epoch']     = args.epochs
    result_data['batch_size']       = args.batch_size
    result_data['optimizer']        = args.optimizer
    result_data['weight_decay']     = args.weight_decay
    result_data['lr']               = args.lr

    # Initialize the result lists
    train_loss = []
    train_pixel_acc = []
    train_mean_acc = []
    train_mean_IoU = []
    train_frew_IoU = []
    val_loss = []
    val_pixel_acc = []
    val_mean_acc = []
    val_mean_IoU = []
    val_frew_IoU = []

    # Check the directory of the file path
    if not os.path.exists(os.path.dirname(RESULT_FILE_PATH)):
        os.makedirs(os.path.dirname(RESULT_FILE_PATH))
    if not os.path.exists(os.path.dirname(CHECKPOINT_FILE_PATH)):
        os.makedirs(os.path.dirname(CHECKPOINT_FILE_PATH))
    print('==> Train ready.')

    for epoch in range(args.epochs):
        # strat after the checkpoint epoch
        if epoch < start_epoch:
            continue
        print("\n[Epoch: {:3d}/{:3d}]".format(epoch+1, args.epochs))
        epoch_time = time.time()
        #=======================================================================
        # train the model
        tloss, tmetric = train(model, train_loader, criterion, optimizer)
        train_loss.append(tloss)
        train_pixel_acc.append(tmetric[0])
        train_mean_acc.append(tmetric[1])
        train_mean_IoU.append(tmetric[2])
        train_frew_IoU.append(tmetric[3])

        # validate the model
        vloss, vmetric = val(model, val_loader, criterion)
        val_loss.append(vloss)
        val_pixel_acc.append(vmetric[0])
        val_mean_acc.append(vmetric[1])
        val_mean_IoU.append(vmetric[2])
        val_frew_IoU.append(vmetric[3])

        # update learning rate
        # scheduler.step(vloss)
        #=======================================================================
        current = time.time()

        # Save the current result
        result_data['current epoch']    = epoch
        result_data['train_loss']       = train_loss
        result_data['train_pixel_acc']  = train_pixel_acc
        result_data['train_mean_acc']   = train_mean_acc
        result_data['train_mean_IoU']   = train_mean_IoU
        result_data['train_frew_IoU']   = train_frew_IoU
        result_data['val_loss']         = val_loss
        result_data['val_pixel_acc']    = val_pixel_acc
        result_data['val_mean_acc']     = val_mean_acc
        result_data['val_mean_IoU']     = val_mean_IoU
        result_data['val_frew_IoU']     = val_frew_IoU

        # Save result_data as pkl file
        with open(RESULT_FILE_PATH, 'wb') as pkl_file:
            pickle.dump(result_data, pkl_file, protocol=pickle.HIGHEST_PROTOCOL)



        # Save the best checkpoint
        if vmetric[2] > best_val_mean_IoU:
            best_val_mean_IoU = vmetric[2]
            torch.save({
                'epoch': epoch+1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_val_mean_IoU': best_val_mean_IoU,
                }, BEST_CHECKPOINT_FILE_PATH)

        # Save the current checkpoint
        torch.save({
            'epoch': epoch+1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'val_mean_IoU': vmetric[2]
            }, CHECKPOINT_FILE_PATH)

        # Print the information on the console
        print("model                : {}".format(args.model))
        print("dataset              : {}".format(args.dataset))
        print("batch_size           : {}".format(args.batch_size))
        print("optimizer            : {}".format(args.optimizer))
        print("learning rate        : {:f}".format(optimizer.param_groups[0]['lr']))
        print("weight decay         : {:f}".format(optimizer.param_groups[0]['weight_decay']))
        print("train/val loss       : {:f}/{:f}".format(tloss,vloss))
        print("train/val pixel acc  : {:f}/{:f}".format(tmetric[0],vmetric[0]))
        print("train/val mean acc   : {:f}/{:f}".format(tmetric[1],vmetric[1]))
        print("train/val mean IoU   : {:f}/{:f}".format(tmetric[2],vmetric[2]))
        print("train/val frew IoU   : {:f}/{:f}".format(tmetric[3],vmetric[3]))
        print("epoch time     : {0:.3f} sec".format(current - epoch_time))
        print("Current elapsed time: {0:.3f} sec".format(current - start))
    print('==> Train done.')

    print(' '.join(['Results have been saved at', RESULT_FILE_PATH]))
    print(' '.join(['Checkpoints have been saved at', CHECKPOINT_FILE_PATH]))

#===============================================================================
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='FCN Implementation using PASCAL VOC')
    parser.add_argument('--model', default=None, type=str,
                        help='FCN_AlexNet, FCN_GoogLeNet, FCN_VGG16, ' +
                             'FCN_32s_fixed, FCN_32s, FCN_16s, FCN_8s, etc.')
    parser.add_argument('--dataset', default=None, type=str, help='VOC2011, VOC2012')
    parser.add_argument('--epochs', default=200, type=int)
    parser.add_argument('--batch_size', default=20, type=int)
    parser.add_argument('--optimizer', default='Adam', type=str, help='SGD, Adam')
    parser.add_argument('--weight_decay', default=0.0005, type=float)
    parser.add_argument('--lr', default=0.0001, type=float)
    parser.add_argument('--num_workers', default=0, type=int)
    parser.add_argument('--flag', default='', type=str)
    parser.add_argument('--resume', '-r', action='store_true',
                        help='resume from checkpoint')
    args = parser.parse_args()

    # Code version check message
    print(VERSION_CHECK_MESSAGE)

    start = time.time()
    #===========================================================================
    main(args)
    #===========================================================================
    end = time.time()
    print("Total elapsed time: {0:.3f} sec\n".format(end - start))
    print("[Finih time]",time.strftime('%c', time.localtime(time.time())))
