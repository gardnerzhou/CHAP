import numpy as np
import torch
from medpy import metric
from scipy.ndimage import zoom


def Fmeasure_calu(sMap, gtMap):
    # Function to calculate Fmeasure metrics
    
    gtsize = gtMap.shape
    Label3 = np.zeros(gtsize)
    Label3 = sMap

    NumRec = np.sum(Label3 == 1)  # FP+TP
    NumNoRec = np.sum(Label3 == 0)  # FN+TN
    LabelAnd = np.logical_and(Label3, gtMap)
    NumAnd = np.sum(LabelAnd == 1)  # TP
    num_obj = np.sum(gtMap)  # TP+FN
    num_pred = np.sum(Label3)  # FP+TP

    FN = num_obj - NumAnd
    FP = NumRec - NumAnd
    TN = NumNoRec - FN

    if NumAnd == 0:
        PreFtem = 0
        RecallFtem = 0
        FmeasureF = 0
        Dice = 0
        SpecifTem = 0
        IoU = 0
    else:
        IoU = NumAnd / (FN + NumRec)  # TP/(FN+TP+FP)
        PreFtem = NumAnd / NumRec
        RecallFtem = NumAnd / num_obj
        SpecifTem = TN / (TN + FP)
        Dice = 2 * NumAnd / (num_obj + num_pred)
        FmeasureF = (2.0 * PreFtem * RecallFtem) / (PreFtem + RecallFtem)  # beta = 1.0

    return PreFtem, RecallFtem, SpecifTem, Dice, FmeasureF, IoU


def calculate_metric_percase(pred, gt):
    pred[pred > 0] = 1
    gt[gt > 0] = 1
    if pred.sum() > 0:
        dice = metric.binary.dc(pred, gt)
        hd95 = metric.binary.hd95(pred, gt)
        return dice, hd95
    else:
        return 0, 0


def test_single_volume(image, label, net, classes, patch_size=[256, 256],model_type='unet',device="cuda:0"):
    image, label = image.squeeze(0).cpu().detach().numpy(), label.squeeze(0).cpu().detach().numpy()
    prediction = np.zeros_like(label)
    for ind in range(image.shape[0]):
        slice = image[ind, :, :]
        x, y = slice.shape[0], slice.shape[1]
        slice = zoom(slice, (patch_size[0] / x, patch_size[1] / y), order=0)
        input = torch.from_numpy(slice).unsqueeze(
            0).unsqueeze(0).float().to(device)
        net.eval()
        
        with torch.no_grad():
            if model_type == 'model1':
                output_logits = net(input)[0]
                prob = torch.softmax(output_logits, dim=1)
            elif model_type == 'model2':
                output_logits = net(input)[1]
                prob = torch.softmax(output_logits, dim=1)
            elif model_type == 'logit_ensemble':
                outputs1,outputs2 = net(input)
                output_logits = (outputs1 + outputs2) / 2.0
                prob = torch.softmax(output_logits, dim=1)
            elif model_type == 'prob_ensemble':
                outputs1,outputs2 = net(input)
                prob1 = torch.softmax(outputs1, dim=1)
                prob2 = torch.softmax(outputs2, dim=1)
                prob = (prob1 + prob2) / 2.0
            else:
                outputs = net(input)
                
            # if isinstance(outputs,tuple):
            #     output_logits = outputs[0]
            # else:
            #     output_logits = outputs

            out = torch.argmax(prob, dim=1).squeeze(0)
            out = out.cpu().detach().numpy()
            pred = zoom(out, (x / patch_size[0], y / patch_size[1]), order=0)
            prediction[ind] = pred
    metric_list = []
    for i in range(1, classes):
        metric_list.append(calculate_metric_percase(
            prediction == i, label == i))
    return metric_list


def test_single_volume_ds(image, label, net, classes, patch_size=[256, 256]):
    image, label = image.squeeze(0).cpu().detach(
    ).numpy(), label.squeeze(0).cpu().detach().numpy()
    prediction = np.zeros_like(label)
    for ind in range(image.shape[0]):
        slice = image[ind, :, :]
        x, y = slice.shape[0], slice.shape[1]
        slice = zoom(slice, (patch_size[0] / x, patch_size[1] / y), order=0)
        input = torch.from_numpy(slice).unsqueeze(
            0).unsqueeze(0).float().cuda()
        net.eval()
        with torch.no_grad():
            output_main, _, _, _ = net(input)
            out = torch.argmax(torch.softmax(
                output_main, dim=1), dim=1).squeeze(0)
            out = out.cpu().detach().numpy()
            pred = zoom(out, (x / patch_size[0], y / patch_size[1]), order=0)
            prediction[ind] = pred
    metric_list = []
    for i in range(1, classes):
        metric_list.append(calculate_metric_percase(
            prediction == i, label == i))
    return metric_list


def test_single_adv(image, label, net_g,net_f1, classes, patch_size=[256, 256],device="cuda:0"):
    image, label = image.squeeze(0).cpu().detach().numpy(), label.squeeze(0).cpu().detach().numpy()
    prediction = np.zeros_like(label)
    for ind in range(image.shape[0]):
        slice = image[ind, :, :]
        x, y = slice.shape[0], slice.shape[1]
        slice = zoom(slice, (patch_size[0] / x, patch_size[1] / y), order=0)
        input = torch.from_numpy(slice).unsqueeze(
            0).unsqueeze(0).float().to(device)
        net_g.eval()
        net_f1.eval()

        with torch.no_grad():
            
            outputs = net_g(input)
            if isinstance(outputs,tuple):
                    outputs = outputs[0]
            outputs = net_f1(outputs)
            if isinstance(outputs,tuple):
                    outputs = outputs[0]
            output_logits = outputs

            out = torch.argmax(torch.softmax(output_logits, dim=1), dim=1).squeeze(0)
            out = out.cpu().detach().numpy()
            pred = zoom(out, (x / patch_size[0], y / patch_size[1]), order=0)
            prediction[ind] = pred
    metric_list = []
    for i in range(1, classes):
        metric_list.append(calculate_metric_percase(
            prediction == i, label == i))
    return metric_list


def test_single_volume_poly(image, label, net,device="cuda:0"):
    
    label = label.squeeze(1)

    image, label = image.squeeze(0).cpu().detach().numpy(), label.squeeze(0).cpu().detach().numpy()

    prediction = np.zeros_like(label)
    
    input = torch.from_numpy(image).unsqueeze(
        0).float().to(device)
    net.eval()
    with torch.no_grad():
       
        outputs = net(input)
        if isinstance(outputs,tuple):
            output_logits = outputs[0]
        else:
            output_logits = outputs
        
        out = torch.argmax(torch.softmax(output_logits, dim=1), dim=1).squeeze(0)
        out = out.cpu().detach().numpy()
        
        prediction = out
    
    PreFtem, RecallFtem, SpecifTem, Dice, FmeasureF, IoU = Fmeasure_calu(prediction,label)

    return Dice


def test_single_adv_polyp(image, label, net_g,net_f1, device="cuda:0"):
    label = label.squeeze(1)

    image, label = image.squeeze(0).cpu().detach().numpy(), label.squeeze(0).cpu().detach().numpy()
    prediction = np.zeros_like(label)
    
    input = torch.from_numpy(image).unsqueeze(
        0).float().to(device)
    
    net_g.eval()
    net_f1.eval()

    with torch.no_grad():
            
            outputs = net_g(input)
            outputs = net_f1(outputs)
            output_logits = outputs

            out = torch.argmax(torch.softmax(output_logits, dim=1), dim=1).squeeze(0)
            out = out.cpu().detach().numpy()
            prediction = out

    PreFtem, RecallFtem, SpecifTem, Dice, FmeasureF, IoU = Fmeasure_calu(prediction,label)
    return Dice