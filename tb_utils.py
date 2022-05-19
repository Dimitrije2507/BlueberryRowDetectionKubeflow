import tensorboard
import torch
import numpy as np
from data_utils import *
from metrics_utils import *

def tb_num_pix_num_classes(tb,count_train,count_train_tb,count_freq,num_channels_lab,batch_names_train,target_var,classes_labels,loss):
    if (count_train / count_freq).is_integer():
        classes_count = np.zeros(num_channels_lab+1)
        pix_classes_count = np.zeros(num_channels_lab+1)
        for batch in range(len(batch_names_train)):
            if num_channels_lab == 1:
                tresholded = target_var[batch,:,:,:].squeeze().byte()
                classes_count +=1
                pix_classes_count[0] = torch.sum(tresholded)
                pix_classes_count[1] = torch.sum((tresholded==0).byte())
                classes = torch.unique(tresholded)
                tb.add_scalar("Count_Train/background", classes_count[0],
                                count_train_tb)
                tb.add_scalar("Count_Pix_Train/background", pix_classes_count[0],
                                count_train_tb)
                tb.add_scalar("Count_Train/" + str(classes_labels[0]), classes_count[1],
                                count_train_tb)
                tb.add_scalar("Count_Pix_Train/" + str(classes_labels[0]), pix_classes_count[1],
                                count_train_tb)
            else:
                tresholded = torch.argmax(target_var[batch, :, :, :].squeeze(), dim=0)
                tresholded = tresholded.byte()
                classes = torch.unique(tresholded.flatten())
                classes_count[classes] += 1
                for klasa in classes:
                    dummy = torch.zeros([tresholded.shape[0], tresholded.shape[1]])
                    dummy[tresholded == klasa] = 1
                    pix_classes_count[klasa] += torch.sum(dummy)
                    # pix_classes_count[klasa] += np.sum(target_var[batch,klasa, :, :].squeeze().detach().cpu().numpy())

                    tb.add_scalar("Count_Train/" + str(classes_labels[klasa]), classes_count[klasa],
                                    count_train_tb)
                    tb.add_scalar("Count_Pix_Train/" + str(classes_labels[klasa]), pix_classes_count[klasa],
                                    count_train_tb)
        tb.add_scalar("Loss Train Iteration", loss, count_train_tb)

        count_train_tb += 1
    
def tb_write_image(tb, num_classes, epoch, input_var, target_var,mask_var, model_output, index, train_part,
                   tb_img_name,device,use_mask,dataset):
    
    if num_classes >= 1:
    
        tresholded = model_output[index, :, :, :]>0.5
        out = tresholded.byte()
        out = decode_segmap2(out,num_classes,device)
        out = torch.moveaxis(out, 2, 0)

        target = target_var[index, :, :, :]>0.5
        target = target.byte()
        target = decode_segmap2(target,num_classes,device)
        target = torch.moveaxis(target, 2, 0).to(device)

        image = (input_var[index, :, :, :]).reshape(4, 512, 512)
        
        rgb_image = inv_zscore_func(image,dataset)[0:3,:,:]
        nir_image = inv_zscore_func(image,dataset)[3,:,:]
        nir_image= nir_image.repeat(3,1,1)
        tb.add_image("RGB , NIR, Label, Prediction" + tb_img_name + " " + train_part,
                     torch.concat([rgb_image.byte(), torch.ones(size=(3,512,10),device=device,dtype=torch.uint8)*255 , nir_image.byte(), 
                     torch.ones(size=(3,512,10),device=device,dtype=torch.uint8)*255,target.byte(), torch.ones(size=(3,512,10),device=device,dtype=torch.uint8)*255 ,out.byte()], axis=2),
                     epoch, dataformats="CHW")
        
        iou_1 = calc_metrics_tb(model_output[index, :, :, :].unsqueeze(dim=0),
                                          target_var[index, :, :, :].unsqueeze(dim=0),mask_var[index].unsqueeze(dim=0), num_classes,use_mask)
       
        iou_1 = torch.tensor(iou_1,dtype = torch.float32)
        tb.add_scalar("Miou/" + tb_img_name + " " + train_part, iou_1, epoch)


def tb_image_list_plotting(tb,tb_img_list,num_channels_lab,epoch,input_var,target_var, mask_var, model_output,train_part,device,batch_names,use_mask,dataset):
    tb_list = [tb_img for tb_img in tb_img_list if tb_img in batch_names]
    if tb_list:
        for tb_list_index in range(len(tb_list)):
            tb_img_index = batch_names.index(tb_list[tb_list_index])
            tb_write_image(tb, num_channels_lab, epoch, input_var, target_var, mask_var, model_output,
                            tb_img_index, train_part, tb_list[tb_list_index],device,use_mask,dataset)
        
        del tb_list, tb_img_index

def tb_add_epoch_losses(tb,train_losses,validation_losses,epoch):
    tb.add_scalar("Loss/Train", torch.mean(torch.tensor(train_losses,dtype = torch.float32)), epoch)
    tb.add_scalar("Loss/Validation", torch.mean(torch.tensor(validation_losses,dtype = torch.float32)), epoch)

def tb_top_k_worst_k(df, num_classes, k_index, test_loader, loss_type, zscore,device,segmentation_net,tb,classes_labels,dataset):
    for class_iter in range(num_classes):
        df_tmp = df[class_iter]
        # klasa0 = df[df['klasa']==i].reset_index().iloc[:,1:]
        mean_num_pix = df_tmp['broj piksela pozitivne klase'].mean()
        std_num_pix = df_tmp['broj piksela pozitivne klase'].std()
        # print("Donji prag broja piksela za pozitivnu klasu: "+ str(mean_num_pix-std_num_pix))
        # print("Gornji prag broja piksela za pozitivnu klasu: "+ str(mean_num_pix+std_num_pix))
        # df_tmp = df_tmp[(df_tmp["broj piksela pozitivne klase"]>(mean_num_pix-std_num_pix)).values & (df_tmp["broj piksela pozitivne klase"]<(mean_num_pix+std_num_pix)).values]
        df_tmp_top2 = df_tmp.reset_index().iloc[:k_index,1:]
        df_tmp_worst2 = df_tmp.reset_index().iloc[-k_index:,1:]
        
        for k_iter in range(k_index):
            test_img_top_tmp, target_top = load_raw_data(test_loader,df_tmp_top2,k_iter,loss_type)
            test_img_worst_tmp, target_worst = load_raw_data(test_loader,df_tmp_worst2,k_iter,loss_type)

            if zscore:
                test_img_top = zscore_func(test_img_top_tmp,device,dataset)
                test_img_worst = zscore_func(test_img_worst_tmp,device,dataset)
                
            target_top = torch.tensor(target_top)
            target_worst = torch.tensor(target_worst)
            
            if loss_type == 'bce' and num_classes == 1:
                target_top = ((target_top[0]+target_top[1]+target_top[2]+target_top[3]+target_top[4]+target_top[5])>0).float().unsqueeze(0)
                target_worst = ((target_worst[0]+target_worst[1]+target_worst[2]+target_worst[3]+target_worst[4]+target_worst[5])>0).float().unsqueeze(0)
            
            image_top = torch.tensor(test_img_top_tmp[:4],device=device)
            image_worst = torch.tensor(test_img_worst_tmp[:4],device=device)
            
            nir_top = inv_zscore_func(image_top,dataset)[3,:,:]
            rgb_image_top = inv_zscore_func(image_top,dataset)[0:3,:,:]
            nir_top = nir_top.repeat(3,1,1)

            nir_worst = inv_zscore_func(image_worst,dataset)[3,:,:]
            rgb_image_worst = inv_zscore_func(image_worst,dataset)[0:3,:,:]
            nir_worst = nir_worst.repeat(3,1,1)

            model_output_top = segmentation_net(test_img_top.unsqueeze(0))
            out_top = model_output_top[0,:,:,:].squeeze()>0.5
            out_top = out_top.byte().unsqueeze(0)
            out_top = decode_segmap2(out_top, num_classes, device)
            out_top = torch.moveaxis(out_top, 2, 0)
            
            model_output_worst = segmentation_net(test_img_worst.unsqueeze(0))
            out_worst = model_output_worst[0,:,:,:].squeeze()>0.5
            out_worst = out_worst.byte().unsqueeze(0)
            out_worst = decode_segmap2(out_worst, num_classes, device)
            out_worst = torch.moveaxis(out_worst, 2, 0)
            
            target_top = decode_segmap2(target_top,num_classes, device)
            target_top = torch.moveaxis(target_top,2,0)
            
            target_worst = decode_segmap2(target_worst, num_classes, device)
            target_worst = torch.moveaxis(target_worst,2,0)
            

            tb.add_image("Top 2 Test Images/Classwise_"+ classes_labels[class_iter] + "_top_"+str(k_iter+1)+"_"+df_tmp_top2.iloc[k_iter]['filenames']+ " Class area: "+ str(df_tmp_top2.iloc[k_iter]['broj piksela pozitivne klase']) + " IoU metric: " + str(df_tmp_top2.iloc[k_iter]['iou metrika']) ,
                            torch.concat([rgb_image_top.byte(),torch.ones(size=(3,512,10),device=device,dtype=torch.uint8)*255, nir_top.byte() , torch.ones(size=(3,512,10),device=device,dtype=torch.uint8)*255, target_top.byte(), torch.ones(size=(3,512,10),device=device,dtype=torch.uint8)*255, out_top.byte()], axis=2),
                            1, dataformats="CHW")
            tb.add_image("Worst 2 Test Images/Classwise_"+ classes_labels[class_iter] + "_worst_"+str(k_iter+1)+"_"+df_tmp_worst2.iloc[k_iter]['filenames'] + " Class area: "+ str(df_tmp_top2.iloc[k_iter]['broj piksela pozitivne klase']) +" IoU metric: " + str(df_tmp_worst2.iloc[k_iter]['iou metrika']) ,
                            torch.concat([rgb_image_worst.byte(), torch.ones(size=(3,512,10),device=device,dtype=torch.uint8)*255, nir_worst.byte() ,torch.ones(size=(3,512,10),device=device,dtype=torch.uint8)*255, target_worst.byte(),torch.ones(size=(3,512,10),device=device,dtype=torch.uint8)*255, out_worst.byte()], axis=2),
                            1, dataformats="CHW")



def createConfusionMatrix(loader,net,classes_labels):
    y_pred = [] # save predction
    y_true = [] # save ground truth

    for input_var, target_var, img_names_test, z_test in loader:
        for idx in range(target_var.shape[0]):
            target_tmp = target_var[ idx,:, :, :]>0.5
            target_tmp = target_tmp.byte().flatten()
        #   target = target.byte()
            # target_conf = torch.argmax(target_var[idx, :, :, :].squeeze(), dim=0).detach().cpu().numpy().flatten()
            y_true.extend(target_tmp.detach().cpu().numpy())
        # target_conf = np.moveaxis(target_conf, 1, 3)
        # target_conf = np.moveaxis(target_conf, 1, 2)
        model_output = net(input_var)
        for idx in range(model_output.shape[0]):
            output_tmp = model_output[ idx,:, :, :]>0.5
            output_tmp = output_tmp.byte().flatten()
            # pred_conf = torch.argmax(model_output[idx, :, :, :].squeeze(), dim=0).detach().cpu().numpy().flatten()
            y_pred.extend(output_tmp.detach().cpu().numpy())
        # pred_conf = np.moveaxis(pred_conf, 1, 3)
        # pred_conf = np.moveaxis(pred_conf, 1, 2)
        # conf_matrix = confusion_matrix(target_conf, pred_conf)

    # Build confusion matrix
    cf_matrix = confusion_matrix(y_true, y_pred)


    # conf = np.insert(cf_matrix, cf_matrix.shape[0], np.zeros(len(classes_labels)), axis=1)
    # conf = np.insert(conf, cf_matrix.shape[0], np.zeros(len(classes_labels) + 1), axis=0)
    # for i in range(len(classes_labels)):
    #     conf[i,-1] = np.sum(conf[i,:])
    #     conf[-1, i] = np.sum(conf[:, i])

    # classes = classes_labels
    # classes.append("sum")
    df_cm = pd.DataFrame(cf_matrix, index=[i for i in classes_labels],
                         columns=[i for i in classes_labels])
    # Create Heatmap
    plt.figure(figsize=(12, 7))
    return sns.heatmap(df_cm, annot=True,xticklabels=True,yticklabels=True).get_figure()