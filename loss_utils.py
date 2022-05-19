import sys
import numpy as np
import torch.nn as nn
import torch

def loss_init(use_weights,loss_type,dataset,num_channels_lab,device):
    if use_weights:
        if dataset == "full":
            if num_channels_lab == 7 :
                class_weights = np.load(r'/home/stefanovicd/DeepSleep/agrovision/class_weights_full_multiclass_with_background.npy')
                # class_weights = [ 0.29468473, 17.64593792, 144.45582386, 61.6571218 , 36.81921342, 49.89357621, 8.40231562]
            elif num_channels_lab ==6:
                class_weights = np.load(r'/home/stefanovicd/DeepSleep/agrovision/class_weights_full_multiclass_without_background.npy')
                # class_weights = [17.64593792, 144.45582386, 61.6571218 , 36.81921342, 49.89357621, 8.40231562]
            elif num_channels_lab == 2:
                class_weights = np.load(r'/home/stefanovicd/DeepSleep/agrovision/class_weights_full_binary_with_background.npy')
            elif num_channels_lab == 1:
                class_weights = np.load(r'/home/stefanovicd/DeepSleep/agrovision/class_weights_full_binary_without_background.npy')
                # class_weights = [2.19672858]
            else:
                print("Error: wrong dataset")
                sys.exit(0)
        elif dataset == "mini":
            if num_channels_lab == 7:
                class_weights = np.load(r'/home/stefanovicd/DeepSleep/agrovision/class_weights_mini_multiclass_with_background.npy')
                # class_weights = [ 0.29468473, 17.64593792, 144.45582386, 61.6571218 , 36.81921342, 49.89357621, 8.40231562]
            elif num_channels_lab ==6:
                class_weights = np.load(r'/home/stefanovicd/DeepSleep/agrovision/class_weights_mini_multiclass_without_background.npy')
                #  class_weights = [17.64593792, 144.45582386, 61.6571218 , 36.81921342, 49.89357621, 8.40231562]
            elif num_channels_lab == 2:
                class_weights = np.load(r'/home/stefanovicd/DeepSleep/agrovision/class_weights_mini_binary_with_background.npy')
            elif num_channels_lab == 1:
                class_weights = np.load(r'/home/stefanovicd/DeepSleep/agrovision/class_weights_mini_binary_with_background.npy')[1:]
            else:
                print("Error: wrong dataset")
                sys.exit(0)
        else:
            print("Error: wrong dataset")
            sys.exit(0)

    if loss_type == "ce":
        if use_weights:
            class_weights = torch.tensor(class_weights, dtype=torch.float,device=device)
            criterion = nn.CrossEntropyLoss(weight=class_weights, reduction="none")
        else:
            criterion = nn.CrossEntropyLoss(reduction="mean")
        return criterion
    elif loss_type == "bce":
        if use_weights:
            class_weights = torch.tensor(class_weights, dtype=torch.float,device=device).reshape(1, num_channels_lab, 1, 1)
            criterion_bce = torch.nn.BCEWithLogitsLoss(pos_weight=class_weights,reduction="none")
        else:
            criterion_bce = torch.nn.BCEWithLogitsLoss()
        return criterion_bce
    elif loss_type == "ce_1":
        if use_weights:
            class_weights = torch.tensor(class_weights, dtype=torch.float,device=device)
            criterion = nn.CrossEntropyLoss(weight=class_weights, reduction="none")
        else:
            criterion = nn.CrossEntropyLoss(reduction="mean")
        return criterion

    


def loss_calc(loss_type ,criterion ,model_output ,target_var ,mask_train = 'None' ,num_channels_lab = 2, use_mask = True): ### num_channels_lab = 2 u slucaju kada imamo 2 klase, bg i fg, Za Saletov slucaj to ce biti 7
                                                                                 ### Kada se koristi bce ili ce kod kog nemamo racunanje verovatnoca argument num_channels_lab nije potrebno    
    if loss_type == "bce":                                                       ### proslediti
        loss = criterion(model_output, target_var)
        if use_mask:
            loss = loss[mask_train.unsqueeze(1)]
            
        loss = loss.mean()
        return loss

    elif loss_type == 'ce':
        loss = criterion(model_output, torch.argmax(target_var, 1))
        if use_mask:
            loss = loss[mask_train.unsqueeze(1)]
        # if use_mask:
        #     loss = torch.multiply(loss, mask_train[:, 0, :, :])
        #     loss = torch.multiply(loss, mask_train[:, 1, :, :])
        loss = loss.mean()
        return loss

    elif loss_type == 'ce_1':
       
        target_var_ce = torch.div(target_var, torch.repeat_interleave(\
            torch.square(torch.sum(target_var, dim=1)).unsqueeze(dim=1), repeats=num_channels_lab, dim=1))
        loss = criterion(model_output, target_var_ce)
        if use_mask:
            loss = torch.multiply(loss, mask_train[:, 0, :, :])
            loss = torch.multiply(loss, mask_train[:, 1, :, :])
        loss = loss.mean()

        return loss