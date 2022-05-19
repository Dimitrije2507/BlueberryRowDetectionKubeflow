from Unet_LtS import UNet3
import torch
import numpy as np
import pandas as pd
import cv2
import os
from metrics_utils import *
from data_utils import *
from loss_utils import *
from tb_utils import *

def set_zero_grad(model):
    for param in model.parameters():
        param.grad = None

def model_init(num_channels,num_channels_lab,img_h,img_w,zscore,net_type,device,server,GPU_list):
    if net_type == "UNet3":
        segmentation_net = UNet3(n_channels=num_channels, n_classes=num_channels_lab, height=img_h, width= img_w, zscore = zscore)
    elif net_type == "UNet":
        segmentation_net = UNet(num_channels, num_channels_lab, img_h, img_w)

    segmentation_net.to(device)
    
    if server:
        segmentation_net = torch.nn.DataParallel(segmentation_net, device_ids=GPU_list)
    
    return segmentation_net

def optimizer_init(segmentation_net,lr,weight_decay,scheduler_lr,lambda_parametri,optimizer_patience):
    if weight_decay != 0:
        optimizer = torch.optim.Adam(params=segmentation_net.parameters(), lr=lr, weight_decay=weight_decay)
    else:
        optimizer = torch.optim.Adam(params=segmentation_net.parameters(), lr=lr)

    if scheduler_lr == 'lambda':
        lmbda = lambda epoch: 0.99
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=[lmbda])
    elif scheduler_lr == 'multiplicative':
        lmbda = lambda epoch: lambda_parametri
        scheduler = torch.optim.lr_scheduler.MultiplicativeLR(optimizer, lr_lambda=lmbda, last_epoch=- 1, verbose=False)
    elif scheduler_lr == 'exponential':
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9)
    elif scheduler_lr == 'reducelr':
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=optimizer_patience)
    elif scheduler_lr == 'cosine':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=lr)
    else:
        scheduler_lr = False

    return optimizer , scheduler

def early_stopping(epoch,val_loss_es,all_validation_losses,es_check,segmentation_net, save_model_path, save_checkpoint_freq,ime_foldera_za_upis,es_min,epoch_model_last_save,es_epoch_count,save_best_model):
    val_loss_es[epoch] = all_validation_losses[epoch]
    if val_loss_es[epoch]< es_min:
        es_min = val_loss_es[epoch]
        es_epoch_count = 0
        save_best_model = True
    elif val_loss_es[epoch] > es_min:
        es_epoch_count += 1
        save_best_model = False
    if es_epoch_count == es_check:
        early_stop = True
    
    
    if save_best_model:
        torch.save(segmentation_net, (save_model_path + 'trained_model_best_epoch' + str(epoch) + ".pt"))
        if os.path.exists(save_model_path + 'trained_model_best_epoch' + str(int(epoch_model_last_save)) + ".pt"):
            os.remove(save_model_path + 'trained_model_best_epoch' + str(int(epoch_model_last_save)) + ".pt")
        epoch_model_last_save = int(epoch / save_checkpoint_freq)
        ispis = ("Model BEST saved at path>> " + save_model_path + 'trained_model_best_epoch' + str(epoch) + ".pt")
        print(ispis)
        upisivanje(ispis, ime_foldera_za_upis)
    ####################################
    ###### Provera ponovljivosti #######
    ####################################
    early_stop = False

    if early_stop:
        torch.save(segmentation_net, (save_model_path + 'trained_model_ES_epoch' + str(epoch) + ".pt"))
        if os.path.exists(save_model_path + 'trained_model_epoch' + str(int(epoch_model_last_save)) + ".pt"):
            os.remove(save_model_path + 'trained_model_epoch' + str(int(epoch_model_last_save)) + ".pt")
        epoch_model_last_save = int(epoch / save_checkpoint_freq)
        ispis = ("Model ES saved at path>> " + save_model_path + 'trained_model_ES_epoch' + str(epoch) + ".pt")
        print(ispis)
        upisivanje(ispis, ime_foldera_za_upis)
        return early_stop

    if (epoch / save_checkpoint_freq).is_integer():
        torch.save(segmentation_net, (save_model_path + 'trained_model_epoch' + str(epoch) + ".pt"))
        if os.path.exists(save_model_path + 'trained_model_epoch' + str(int(epoch_model_last_save)) + ".pt"):
            os.remove(save_model_path + 'trained_model_epoch' + str(int(epoch_model_last_save)) + ".pt")
        epoch_model_last_save = int(epoch / save_checkpoint_freq)
        ispis = ("Model saved at path>> " + save_model_path + 'trained_model_epoch' + str(epoch) + ".pt")
        print(ispis)
        upisivanje(ispis, ime_foldera_za_upis)
        return early_stop



def fully_trained_model_saving(segmentation_net,save_model_path,epoch,ime_foldera_za_upis):
    torch.save(segmentation_net, save_model_path + 'fully_trained_model_epochs_' + str(epoch) + ".pt")
    if os.path.exists(save_model_path + 'trained_model_epoch' + str(int(epoch)) + ".pt"):
        os.remove(save_model_path + 'trained_model_epoch' + str(int(epoch)) + ".pt")
        model_name = save_model_path + 'fully_trained_model_epochs_' + str(epoch) + ".pt"
        ispis = ("Fully Trained Model saved at path>> " + model_name)
        print(ispis)
        upisivanje(ispis, ime_foldera_za_upis)
    elif os.path.exists(save_model_path + 'trained_model_epoch' + str(int(epoch)) + ".pt") == False:
        model_name = save_model_path + 'fully_trained_model_epochs_' + str(epoch) + ".pt"
        ispis = ("Fully Trained Model saved at path>> " + model_name)
        print(ispis)
        upisivanje(ispis, ime_foldera_za_upis)
    else:
        model_name = save_model_path + 'trained_model_epoch' + str(int(epoch)) + ".pt"
        ispis = ("Trained Model saved at path>> " + model_name)
        print(ispis)
        upisivanje(ispis, ime_foldera_za_upis)

def run_testing(segmentation_net, test_loader, ime_foldera_za_upis, device, num_classes, classes_labels, classes_labels2,
                criterion_1,loss_type,tb,zscore):

    ###########################
    ### iscrtavanja legende ###
    ###########################
    
    tb.add_image("Classification legend ",np.moveaxis(cv2.cvtColor(cv2.imread(r"/home/stefanovicd/DeepSleep/agrovision/Legend_Classes.png"),cv2.COLOR_BGR2RGB),2,0),
                            1, dataformats="CHW")
    ### segmentation_net = torch.load(segmentation_net)
    
    segmentation_net.eval()

    tmp = get_args('test')
    globals().update(tmp)
    
    ispis = ("_____________________________________________________________Testing Start ")
    print(ispis)
    upisivanje(ispis, ime_foldera_za_upis)

    index_start = 0
    iou_res = torch.zeros([len(test_loader.dataset.img_names), num_classes * 2])
    iou_res_bg = torch.zeros([len(test_loader.dataset.img_names),2])
    global test_losses
    for input_var, target_var, img_names_test, mask_test in test_loader:
        
        # predikcija
        model_output = segmentation_net(input_var)
        # racunanje loss-a
        mask_test = torch.logical_and(mask_test[:,0,:,:],mask_test[:,1,:,:])
        test_loss = loss_calc(loss_type, criterion_1, model_output,target_var,mask_train=mask_test,use_mask= use_mask)
        # cuvanje loss-a kroz iteracije
        test_losses.append(test_loss.data)
        
        # izvlacenje iou i dice komponenti po batch-u za kasnije racunanje ukupnih metrika
        index_end = index_start + len(img_names_test)
        iou_res[index_start:index_end, :], iou_res_bg[index_start:index_end] = calc_metrics_pix(model_output, target_var,mask_test, num_classes,device,use_mask)    
        index_start += len(img_names_test)

        if binary:
            for target_idx in range(target_var.shape[0]):
                foreground_names.append(img_names_test[target_idx])
                foreground_area.append(target_var[target_idx,0].sum())
        else:
            for target_idx in range(target_var.shape[0]):
                class_area = []
                for target_klasa in range(num_classes):
                    class_area.append(target_var[target_idx, target_klasa].sum())

                for target_klasa in torch.unique(torch.nonzero(torch.tensor(class_area))):
                    # if j == 0:
                    #     background_names.append(img_names_test[i])
                    #     background_area.append(class_area[j])
                    # el
                    if target_klasa == 0:
                        cloud_shadow_names.append(img_names_test[target_idx])
                        cloud_shadow_area.append(class_area[target_klasa])
                    elif target_klasa == 1:
                        double_plant_names.append(img_names_test[target_idx])
                        double_plant_area.append(class_area[target_klasa])
                    elif target_klasa == 2:
                        planter_skip_names.append(img_names_test[target_idx])
                        planter_skip_area.append(class_area[target_klasa])
                    elif target_klasa == 3:
                        standing_water_names.append(img_names_test[target_idx])
                        standing_water_area.append(class_area[target_klasa])
                    elif target_klasa == 4:
                        waterway_names.append(img_names_test[target_idx])
                        waterway_area.append(class_area[target_klasa])
                    elif target_klasa == 5:
                        weed_cluster_names.append(img_names_test[target_idx])
                        weed_cluster_area.append(class_area[target_klasa])
                    else:
                        print("Error: there is no labeled class in images!")
                        print(torch.unique(torch.nonzero(torch.tensor(class_area))))
                        sys.exit(0)
        
    tb.add_figure("Confusion matrix", createConfusionMatrix(test_loader,segmentation_net,classes_labels2),0)

    test_losses = torch.tensor(test_losses,dtype = torch.float32)
    # mean Test loss-ova
    ispis = "Mean Test Loss: " + str(torch.mean(test_losses))
    print(ispis)
    upisivanje(ispis, ime_foldera_za_upis)

    ###################################################################
    # izracunavanje ukupnih metrika nad citavim test setom po klasama #
    ###################################################################
    
    iou_res = torch.tensor(iou_res,dtype = torch.float32,device=device)
    iou_res_bg = torch.tensor(iou_res_bg,dtype = torch.float32,device= device)
    
    IOU = final_metric_calculation(num_channels_lab=num_classes,classes_labels=classes_labels,\
        batch_iou_bg=iou_res_bg,batch_iou= iou_res,train_part='Test',ime_foldera_za_upis=ime_foldera_za_upis)
    FinalTabela = pd.DataFrame()
    FinalTabela['TestSet IoU Metric'] = IOU
    FinalTabela = FinalTabela.set_axis(classes_labels2).T
    FinalTabela.to_csv(os.path.split(ime_foldera_za_upis)[0] +"/" +  "Weighted BCE without background class, Full dataset, BGFG lr 1e-3 50 epochs.csv")
    ###########################################################
    #       Izracunavanje metrike za svaki test uzorak        #
    ###########################################################
    #   Na osnovu dobijen metrika dalje sortiramo uzorke kao  #
    #   priprema za izvlacenje top k i worst k uzoraka        #
    ###########################################################
    
    for im_number in range(len(test_loader.dataset.img_names)):
        iou_tmp = []
        if binary:
            iou_int = iou_res[im_number, 0]
            iou_un = iou_res[im_number, 1]
            # eps = 0.00001
            iou_calc = torch.sum(iou_int) / torch.sum(iou_un)
            iou_tmp.append(iou_calc)            
            iou_per_test_image_fg.append(iou_tmp[0]) 
        else: 
            index_iou = 0
            for klasa in range(num_classes):
                iou_int = iou_res[im_number, index_iou]
                iou_un = iou_res[im_number, index_iou + 1]

                iou_calc = torch.sum(iou_int) / torch.sum(iou_un)

                iou_tmp.append(iou_calc)

                index_iou += 2
            try:
                # if test_loader.dataset.img_names[im_number] in background_names:
                #     iou_per_test_image_bg.append(iou_tmp[0])
                if test_loader.dataset.img_names[im_number] in cloud_shadow_names:
                    iou_per_test_image_cs.append(iou_tmp[0])
                if test_loader.dataset.img_names[im_number] in double_plant_names:
                    iou_per_test_image_dp.append(iou_tmp[1])
                if test_loader.dataset.img_names[im_number] in planter_skip_names:
                    iou_per_test_image_ps.append(iou_tmp[2])
                if test_loader.dataset.img_names[im_number] in standing_water_names:
                    iou_per_test_image_sw.append(iou_tmp[3])
                if test_loader.dataset.img_names[im_number] in waterway_names:
                    iou_per_test_image_ww.append(iou_tmp[4])
                if test_loader.dataset.img_names[im_number] in weed_cluster_names:
                    iou_per_test_image_wc.append(iou_tmp[5])
            except:
                print("Error: Test image is not valid")
                sys.exit(0)

    #################################
    ###     TOP k and WORST k     ###
    #################################
    
    df = pd.DataFrame()
    if num_classes==1:
        df_fg = pd.DataFrame()
        df = [df_fg]
        df_names = [foreground_names]
        df_area = [foreground_area]
        df_iou = [iou_per_test_image_fg]
        for idx, df_iter in enumerate(df):
            df_iter['filenames'] = df_names[idx]; df_iter['broj piksela pozitivne klase'] = torch.tensor(df_area[idx]); df_iter['iou metrika'] = torch.tensor(df_iou[idx]); df_iter['klasa'] = idx

    else:
        df_cs = pd.DataFrame();df_dp = pd.DataFrame();df_ps = pd.DataFrame();df_sw = pd.DataFrame();df_ww = pd.DataFrame();df_wc = pd.DataFrame()
        df = [df_cs,df_dp,df_ps,df_sw,df_ww,df_wc]
        df_names = [cloud_shadow_names,double_plant_names,planter_skip_names,standing_water_names,waterway_names,weed_cluster_names]
        df_area = [cloud_shadow_area,double_plant_area,planter_skip_area,standing_water_area,waterway_area,weed_cluster_area]
        df_iou = [iou_per_test_image_cs,iou_per_test_image_dp,iou_per_test_image_ps,iou_per_test_image_sw,iou_per_test_image_ww,iou_per_test_image_wc]
        for idx, df_iter in enumerate(df):
            df_iter['filenames'] = df_names[idx]; df_iter['broj piksela pozitivne klase'] = torch.tensor(df_area[idx]); df_iter['iou metrika'] = torch.tensor(df_iou[idx]); df_iter['klasa'] = idx
        
    for idx, df_iter in enumerate(df):
        df[idx] = df_iter.sort_values('iou metrika',ascending=False)
                
    tb_top_k_worst_k(df, num_classes, k_index, test_loader, loss_type, zscore, device, segmentation_net, tb, classes_labels,dataset)
    return IOU
    