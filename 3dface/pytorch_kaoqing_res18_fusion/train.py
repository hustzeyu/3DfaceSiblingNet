from __future__ import print_function
import os
from data.dataset import Dataset
import torch
from torch.utils import data
import torch.nn.functional as F
from models import resnet, metrics, focal_loss
from models import googlenet
import torchvision
import torch
import numpy as np
import random
import time
from config.config import Config
from torch.nn import DataParallel
from torch.optim.lr_scheduler import StepLR
from tqdm import tqdm
from tensorboard import TensorBoard

def save_model(model, save_path, name, iter_cnt):
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    save_name = os.path.join(save_path, name + '_' + str(iter_cnt) + '.pth')
    torch.save(model.state_dict(), save_name)
    return save_name


if __name__ == '__main__':
    opt = Config()
    device = torch.device("cuda")

    ############ gen trainloader and valloader ##################
    train_dataset = Dataset(opt.train_list, phase='train', input_shape=opt.input_shape)
    val_dataset = Dataset(opt.val_list, phase='val', input_shape=opt.input_shape)
    trainloader = data.DataLoader(train_dataset,
                                  batch_size=opt.train_batch_size,
                                  shuffle=True,
                                  num_workers=opt.num_workers)
    valloader = data.DataLoader(val_dataset,
                                  batch_size=opt.test_batch_size,
                                  shuffle=True,
                                  num_workers=opt.num_workers)
    print('{} train iters per epoch:'.format(len(trainloader)))

        
    ############ choose loss and backbone ######################
    if opt.loss == 'focal_loss':
        criterion = focal_loss.FocalLoss(gamma=2)
    else:
        criterion = torch.nn.CrossEntropyLoss()

    if opt.backbone == 'resnet18':
        model = resnet.resnet_face18(use_se=opt.use_se)
    elif opt.backbone == 'resnet50':
        model = resnet.resnet50()
    elif opt.backbone == 'resnet101':
        model = resnet.resnet101()
    elif opt.backbone == 'resnet152':
        model = resnet.resnet152()
    elif opt.backbone == 'googlenet':
        model = googlenet.GoogLeNet() 

    if opt.metric == 'add_margin':
        metric_fc = metrics.AddMarginProduct(1024, opt.num_classes, s=30, m=0.35)
    elif opt.metric == 'arc_margin':
        metric_fc = metrics.ArcMarginProduct(512, opt.num_classes, s=30, m=0.5, easy_margin=opt.easy_margin)
    elif opt.metric == 'sphere':
        metric_fc = metrics.SphereProduct(512, opt.num_classes, m=4)
    else:
        # metric_fc = nn.Linear(512, opt.num_classes)
        metric_fc = nn.Linear(512, opt.num_classes)

    
    ############ visual_model and model_to_device ##############
    tensor_board = TensorBoard(opt.train_batch_size, 3, 112, 112)
    #tensor_board.visual_model(model)
    model.to(device)
    model = DataParallel(model)
    metric_fc.to(device)
    metric_fc = DataParallel(metric_fc)


    ############ choose optimizer and optimizer ################
    if opt.optimizer == 'sgd':
        optimizer = torch.optim.SGD([{'params': model.parameters()}, {'params': metric_fc.parameters()}],
                                    lr=opt.lr, weight_decay=opt.weight_decay)
    else:
        optimizer = torch.optim.Adam([{'params': model.parameters()}, {'params': metric_fc.parameters()}],
                                     lr=opt.lr, weight_decay=opt.weight_decay)
    scheduler = StepLR(optimizer, step_size=opt.lr_step, gamma=0.1)


    ############ train and test model ##########################
    print ("*"*100)
    print ("Strat training ...")
    for epoch in range(1, opt.max_epoch+1):
        scheduler.step()
        #learn_rate = opt.lr * 0.1**((epoch-1)//opt.lr_step)
        learn_rate = scheduler.get_lr()[0]
        print("learn_rate:%s" % learn_rate)

        model.train()
        batch_idx = 1
        for data in tqdm(trainloader):
            data_input, data_input1, label = data
            data_input = data_input.to(device)
            data_input1 = data_input1.to(device)
            label = label.to(device).long()
            if batch_idx == 1:
                tensor_board.visual_img(data_input, epoch)
            #import pdb;pdb.set_trace()
            feature = model(data_input, data_input1)
            output = metric_fc(feature, label)
            loss = criterion(output, label)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            iters = (epoch - 1) * len(trainloader) + batch_idx

            if iters % opt.print_freq == 0:
                output = output.data.cpu().numpy()
                output = np.argmax(output, axis=1)
                label = label.data.cpu().numpy()
                acc = np.mean((output == label).astype(int))
                time_str = time.asctime(time.localtime(time.time()))
                print('{},  train epoch {},  loss {},  acc {}'.format(time_str, epoch, loss.item()*1.0/opt.train_batch_size, acc))
                tensor_board.visual_loss("train_loss", loss*1.0/opt.train_batch_size, iters)
                tensor_board.visual_acc("train_acc", acc, iters)
            batch_idx += 1

            #if iters % 100 == 0: 
            #    with torch.no_grad():
            #        model.eval()
            #        test_loss_val = 0
            #        correct_val = 0
            #        for data_input_val, data_input1_val, label_val in valloader:
            #            data_input_val, data_input1_val, label_val = data_input_val.to(device), data_input1_val.to(device), label_val.to(device).long()
            #            feature_val = model(data_input_val, data_input1_val)
            #            output_val = metric_fc(feature_val, label_val)
            #            test_loss_val += criterion(output_val, label_val).item()
            #            pred_val = output_val.max(1, keepdim=True)[1]
            #            correct_val += pred_val.eq(label_val.view_as(pred_val)).sum().item()
            #        test_loss_val /= len(valloader.dataset)
            #        acc_val = 1.*correct_val / len(valloader.dataset)
            #        print('{},  val epoch {},  test_loss {},  test_acc {}'.format(time_str, epoch, test_loss_val, acc_val))
            #        tensor_board.visual_loss("test_loss", test_loss_val, iters)
            #        tensor_board.visual_acc("test_acc", acc_val, iters)
   

        if epoch % opt.save_interval == 0:
            print ("*"*80 + "Saving model ...")
            save_model(model, opt.checkpoints_path, opt.backbone, epoch)
