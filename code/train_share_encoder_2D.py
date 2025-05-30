import argparse
import logging
import os
import random
import shutil
import sys
import time
from medpy import metric
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tensorboardX import SummaryWriter
from torch.nn import BCEWithLogitsLoss
from torch.nn.modules.loss import CrossEntropyLoss
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.utils import make_grid
from tqdm import tqdm
from networks.net_factory import net_factory
from dataloaders.dataset import *
from utils import losses, metrics, ramps,test_3d_patch,patch,memory
from val_2D import test_single_volume
from utils.launch import init_save_folder
from utils.util import update_values
import yaml

def get_current_consistency_weight(epoch,args):
    # Consistency ramp-up from https://arxiv.org/abs/1610.02242
    return args["consistency"] * ramps.sigmoid_rampup(epoch, args["consistency_rampup"])

def patients_to_slices(dataset, patiens_num):
    ref_dict = None
    if "ACDC" in dataset:
        ref_dict = {"3": 68, "7": 136,
                    "14": 256, "21": 396, "28": 512, "35": 664, "140": 1312}
    elif "Prostate":
        ref_dict = {"2": 27, "4": 53, "8": 120,
                    "12": 179, "16": 256, "21": 312, "42": 623}
    else:
        print("Error")
    return ref_dict[str(patiens_num)]

def update_ema_variables(model, ema_model, alpha, global_step):
    # Use the true average until the exponential average is more correct
    alpha = min(1 - 1 / (global_step + 1), alpha)
    for ema_param, param in zip(ema_model.parameters(), model.parameters()):
        ema_param.data.mul_(alpha).add_(1 - alpha, param.data)

def copy_weights(model_from, model_to):

    model_to.load_state_dict(model_from.state_dict())

def sharpening(P):
    args.temperature = 0.1
    T = 1/args.temperature
    P_sharpen = P ** T / (P ** T + (1-P) ** T)
    return P_sharpen

def kaiming_normal_init_weight(model):
    for m in model.modules():
        if isinstance(m, nn.Conv3d):
            torch.nn.init.kaiming_normal_(m.weight)
        elif isinstance(m, nn.BatchNorm3d):
            m.weight.data.fill_(1)
            m.bias.data.zero_()
    return model


def xavier_normal_init_weight(model):
    for m in model.modules():
        if isinstance(m, nn.Conv3d):
            torch.nn.init.xavier_normal_(m.weight)
        elif isinstance(m, nn.BatchNorm3d):
            m.weight.data.fill_(1)
            m.bias.data.zero_()
    return model

def calculate_metric_percase(pred, gt):
    pred[pred > 0] = 1
    gt[gt > 0] = 1
    dice = metric.binary.dc(pred, gt)

    return dice

def calculate_avg_dice(pred,gt):
    cls1 = calculate_metric_percase(pred==1, gt==1)
    cls2 = calculate_metric_percase(pred==2, gt==2)
    cls3 = calculate_metric_percase(pred==3, gt==3)
    return (cls1+cls2+cls3)/3.0


def acc_conf_analysis(pred,label,labeled_bs,filename):
    
    pred = pred.detach().cpu().numpy()
    label = label.cpu().numpy()

    labeled_prob,labeled_gt = pred[:labeled_bs],label[:labeled_bs]
    unlabeled_prob,unlabeled_gt = pred[labeled_bs:],label[labeled_bs:]

    labeled_conf, labeled_pred = labeled_prob.max(axis=1),labeled_prob.argmax(axis=1)
    unlabeled_conf, unlabeled_pred = unlabeled_prob.max(axis=1),unlabeled_prob.argmax(axis=1)

    # 找到分割错误的地方
    labeled_err_mask = (labeled_pred != labeled_gt)
    unlabeled_err_mask = (unlabeled_pred != unlabeled_gt)

    # dice
    lab_dice = calculate_avg_dice(labeled_pred,labeled_gt)
    unlab_dice = calculate_avg_dice(unlabeled_pred,unlabeled_gt)

    labeled_corr_mask = ~labeled_err_mask
    unlabeled_corr_mask = ~unlabeled_err_mask

    # confidence
    lab_correct_conf = np.sum(labeled_conf * labeled_corr_mask) / (np.sum(labeled_corr_mask)+1e-6)
    unlab_correct_conf = np.sum(unlabeled_conf * unlabeled_corr_mask) / (np.sum(unlabeled_corr_mask)+1e-6)

    lab_err_conf = np.sum(labeled_conf * labeled_err_mask) / (np.sum(labeled_err_mask)+1e-6)
    unlab_err_conf = np.sum(unlabeled_conf * unlabeled_err_mask) / (np.sum(unlabeled_err_mask)+1e-6)

    metrics = {'lab_dice':lab_dice,'lab_corr_conf':lab_correct_conf,'lab_err_conf':lab_err_conf,'unlab_dice':unlab_dice,
               'unlab_corr_conf':unlab_correct_conf,'unlab_err_conf':unlab_err_conf}
    metrics_array = np.array(list(metrics.values()))

    try:
        existing_metrics = np.load(filename)
        updated_metrics = np.vstack([existing_metrics, metrics_array])

    except FileNotFoundError:
        # 如果文件不存在，创建一个新的数组
        updated_metrics = metrics_array.reshape(1, -1)
    
    np.save(filename, updated_metrics)


def train(args, snapshot_path):
    base_lr = args["base_lr"]
    batch_size = args["batch_size"]
    max_iterations = args["max_iterations"]
    num_classes = args["num_classes"]

    device = torch.device("cuda",args["gpu"])

    def create_model(ema=False,model='unet',args=None):
        # Network definition
        model = net_factory(net_type=model, in_chns=1,
                            class_num=num_classes,device=device,args=args)
        if ema:
            for param in model.parameters():
                param.detach_()
        return model
    
    model = create_model(model=args["model"],args=args)
    
    model.train()

    def worker_init_fn(worker_id):
        random.seed(args["seed"] + worker_id)

    db_train = BaseDataSets(base_dir=args["root_path"], split="train", num=None, transform=transforms.Compose
        ([
        RandomGenerator(args["image_size"])
    ]))
    
    db_val = BaseDataSets(base_dir=args["root_path"], split="val")

    total_slices = len(db_train)
    labeled_slice = patients_to_slices(args["root_path"], args["labeled_num"])
    print("Total silices is: {}, labeled slices is: {}".format(
        total_slices, labeled_slice))
    labeled_idxs = list(range(0, labeled_slice))
    unlabeled_idxs = list(range(labeled_slice, total_slices))
    batch_sampler = TwoStreamBatchSampler(
        labeled_idxs, unlabeled_idxs, batch_size, batch_size-args["labeled_bs"])

    trainloader = DataLoader(db_train, batch_sampler=batch_sampler,num_workers=4, pin_memory=True, worker_init_fn=worker_init_fn)

    valloader = DataLoader(db_val, batch_size=1, shuffle=False, num_workers=1)

    optimizer_g = optim.SGD(model.encoder.parameters(), lr=base_lr, momentum=0.9, weight_decay=0.0001)
    optimizer_f = optim.SGD(list(model.decoder1.parameters()) + list(model.decoder2.parameters()), lr=base_lr, momentum=0.9, weight_decay=0.0001)

    if args["extra"]:
        optimizer_f_extra = optim.SGD(list(model.decoder1_extra.parameters()) + list(model.decoder2_extra.parameters()), lr=base_lr, momentum=0.9, weight_decay=0.0001)

    best_performance1 = 0.0
    best_performance2 = 0.0
    best_dice = 0.0
    iter_num = 0
    ratio_global = 0

    ce_loss = CrossEntropyLoss()
    dice_loss = losses.DiceLoss(num_classes)
    worst_case_criterion = losses.WorstCaseEstimationLoss(loss_type=args["worst_losstype"])
    adv_loss = losses.softDiceLoss(num_classes)
    mb = memory.Image_MemoryBank(256,[256,256],args["patch_size"])

    def train_ACAL_one_iter(x,y,weight=1.0,mask=None,loss_type='mse'):

        #logging.info('consistent ratio : %f ' % (weight))

        volume_batch = x
        label_batch = y

        for _ in range(1):

            loss = 0
            if args["extra"]:
                optimizer_f_extra.zero_grad()
            else:
                optimizer_f.zero_grad()

            with torch.no_grad():
                encoder_output = model.forward_encoder(volume_batch)
            
            if args["extra"]:
                outputs1,outputs2 = model.forward_decoders_extra(encoder_output)
            else:
                outputs1,outputs2 = model.forward_decoders(encoder_output)

            outputs_soft1 = torch.softmax(outputs1, dim=1)
            outputs_soft2 = torch.softmax(outputs2, dim=1)

            confidence1,_ = outputs_soft1[args["labeled_bs"]:].detach().max(dim=1)
            confidence2,_ = outputs_soft2[args["labeled_bs"]:].detach().max(dim=1)

            pseudo_outputs1 = torch.argmax(outputs_soft1[args["labeled_bs"]:].detach(), dim=1, keepdim=False)
            pseudo_outputs2 = torch.argmax(outputs_soft2[args["labeled_bs"]:].detach(), dim=1, keepdim=False)

            conflict_mask = (pseudo_outputs1 != pseudo_outputs2)

            #supervised loss
            lab_loss1 = 0.5 * (ce_loss(outputs1[:args["labeled_bs"]], label_batch[:args["labeled_bs"]].long()) + dice_loss(
                outputs_soft1[:args["labeled_bs"]], label_batch[:args["labeled_bs"]].unsqueeze(1)))
            lab_loss2 = 0.5 * (ce_loss(outputs2[:args["labeled_bs"]], label_batch[:args["labeled_bs"]].long()) + dice_loss(
                outputs_soft2[:args["labeled_bs"]], label_batch[:args["labeled_bs"]].unsqueeze(1)))
            
            loss = 0
            if loss_type=='mse':
                if mask is None:
                    dis_loss1 = losses.mse_loss(outputs_soft1[args["labeled_bs"]:], outputs_soft2[args["labeled_bs"]:].detach())
                    dis_loss2 = losses.mse_loss(outputs_soft2[args["labeled_bs"]:], outputs_soft1[args["labeled_bs"]:].detach())
                else:
                    dis_loss1 = losses.mse_loss_noreduction(outputs_soft1[args["labeled_bs"]:], outputs_soft2[args["labeled_bs"]:].detach())
                    dis_loss2 = losses.mse_loss_noreduction(outputs_soft2[args["labeled_bs"]:], outputs_soft1[args["labeled_bs"]:].detach())
                    dis_loss1 = (dis_loss1 * mask.unsqueeze(1)).sum() / mask.sum()
                    dis_loss2 = (dis_loss2 * mask.unsqueeze(1)).sum() / mask.sum()
            elif loss_type == 'softdice':

                dis_loss1 = adv_loss(outputs_soft1[args["labeled_bs"]:], outputs_soft2[args["labeled_bs"]:].detach(),~(mask.bool()))
                dis_loss2 = adv_loss(outputs_soft2[args["labeled_bs"]:], outputs_soft1[args["labeled_bs"]:].detach(),~(mask.bool()))

            dis_loss = (dis_loss1 + dis_loss2)
            loss = ((lab_loss1 + lab_loss2) - dis_loss) * weight
            logging.info('iteration %d : inconsistency loss : %f ' % (iter_num, loss.item()))

            loss.backward()
            if args["extra"]:
                optimizer_f_extra.step()
            else:
                optimizer_f.step()

        for _ in range(1):

            loss = 0
            optimizer_g.zero_grad()

            encoder_output = model.forward_encoder(volume_batch)
            if args["extra"]:
                outputs1,outputs2 = model.forward_decoders_extra(encoder_output)
            else:
                outputs1,outputs2 = model.forward_decoders(encoder_output)

            outputs_soft1 = torch.softmax(outputs1, dim=1)
            outputs_soft2 = torch.softmax(outputs2, dim=1)

            if loss_type=='mse':
                if mask is None:
                    dis_loss1 = losses.mse_loss(outputs_soft1[args["labeled_bs"]:], outputs_soft2[args["labeled_bs"]:].detach())
                    dis_loss2 = losses.mse_loss(outputs_soft2[args["labeled_bs"]:], outputs_soft1[args["labeled_bs"]:].detach())
                else:
                    dis_loss1 = losses.mse_loss_noreduction(outputs_soft1[args["labeled_bs"]:], outputs_soft2[args["labeled_bs"]:].detach())
                    dis_loss2 = losses.mse_loss_noreduction(outputs_soft2[args["labeled_bs"]:], outputs_soft1[args["labeled_bs"]:].detach())
                    dis_loss1 = (dis_loss1 * mask.unsqueeze(1)).sum() / mask.sum()
                    dis_loss2 = (dis_loss2 * mask.unsqueeze(1)).sum() / mask.sum()
            elif loss_type == 'softdice':

                dis_loss1 = adv_loss(outputs_soft1[args["labeled_bs"]:], outputs_soft2[args["labeled_bs"]:].detach(),~(mask.bool()))
                dis_loss2 = adv_loss(outputs_soft2[args["labeled_bs"]:], outputs_soft1[args["labeled_bs"]:].detach(),~(mask.bool()))

            distance_min = (dis_loss1 + dis_loss2)

            loss += distance_min * weight

            loss.backward()
            optimizer_g.step()

    writer = SummaryWriter(snapshot_path + '/log')
    logging.info("{} iterations per epoch".format(len(trainloader)))

    max_epoch = max_iterations // len(trainloader) + 1
    iterator = tqdm(range(max_epoch), ncols=70)
    for epoch_num in iterator:
        for i_batch, sampled_batch in enumerate(trainloader):

            volume_batch, label_batch = sampled_batch['image'], sampled_batch['label']
            volume_batch, label_batch = volume_batch.to(device), label_batch.to(device)
            
            outputs1,outputs2 = model(volume_batch)

            outputs_soft1 = torch.softmax(outputs1, dim=1)
            outputs_soft2 = torch.softmax(outputs2, dim=1)

            confidence1 = torch.max(outputs_soft1,dim=1)[0]
            confidence2 = torch.max(outputs_soft2,dim=1)[0]

            consistency_weight = get_current_consistency_weight(iter_num // 150,args)

            loss1 = 0.5 * (ce_loss(outputs1[:args["labeled_bs"]],
                                   label_batch[:][:args["labeled_bs"]].long()) + dice_loss(
                outputs_soft1[:args["labeled_bs"]], label_batch[:args["labeled_bs"]].unsqueeze(1)))
            loss2 = 0.5 * (ce_loss(outputs2[:args["labeled_bs"]],
                                   label_batch[:][:args["labeled_bs"]].long()) + dice_loss(
                outputs_soft2[:args["labeled_bs"]], label_batch[:args["labeled_bs"]].unsqueeze(1)))
            
            pseudo_outputs1 = torch.argmax(outputs_soft1[args["labeled_bs"]:].detach(), dim=1, keepdim=False)
            pseudo_outputs2 = torch.argmax(outputs_soft2[args["labeled_bs"]:].detach(), dim=1, keepdim=False)

            if args["consistency_type"] == 'ce':
                pseudo_supervision1 = F.cross_entropy(outputs1[args["labeled_bs"]:], pseudo_outputs2.long(),reduction='none')
                pseudo_supervision2 = F.cross_entropy(outputs2[args["labeled_bs"]:], pseudo_outputs1.long(),reduction='none')
            
            if args["consistency_type"] == 'mse':
                pseudo_label1 = sharpening(outputs_soft1[args["labeled_bs"]:])
                pseudo_label2 = sharpening(outputs_soft2[args["labeled_bs"]:])
                pseudo_supervision1 = losses.mse_loss(outputs_soft1[args["labeled_bs"]:], pseudo_label2.detach())
                pseudo_supervision2 = losses.mse_loss(outputs_soft2[args["labeled_bs"]:], pseudo_label1.detach())
            

            knowledge = pseudo_supervision1 + pseudo_supervision2
            mb.add(volume_batch[args["labeled_bs"]:],knowledge,8)
            
            model1_loss = loss1 + consistency_weight * pseudo_supervision1.mean()
            model2_loss = loss2 + consistency_weight * pseudo_supervision2.mean()

            loss = model1_loss + model2_loss

            optimizer_g.zero_grad()
            optimizer_f.zero_grad()
            
            loss.backward()
            optimizer_f.step()
            optimizer_g.step()

            if args["worst"]:
                model.step()
                copy_weights(model.decoder1,model.decoder1_worst)
                copy_weights(model.decoder2,model.decoder2_worst)
            if args["extra"]:
                copy_weights(model.decoder1,model.decoder1_extra)
                copy_weights(model.decoder2,model.decoder2_extra)

            if args["acal"] and iter_num > 10000:
                
                image = mb.get_samples()
                x = image['image'].to(device)
                pixel_mask = image['mask'].to(device)
                new_image  = torch.cat((volume_batch[:args["labeled_bs"]],x))
                train_ACAL_one_iter(new_image, label_batch, mask=pixel_mask, loss_type=args["adv_losstype"])

            iter_num = iter_num + 1

            lr_ = base_lr * (1.0 - iter_num / max_iterations) ** 0.9
            for param_group1 in optimizer_g.param_groups:
                param_group1['lr'] = lr_
            for param_group1 in optimizer_f.param_groups:
                param_group1['lr'] = lr_
            
            writer.add_scalar('lr', lr_, iter_num)

            writer.add_scalar(
                'consistency_weight/consistency_weight', consistency_weight, iter_num)
            writer.add_scalar('loss/model1_loss',
                              model1_loss, iter_num)
            writer.add_scalar('loss/model2_loss',
                              model2_loss, iter_num)
            
            logging.info(
                'iteration %d : model1 loss : %f model2 loss : %f new loss: %f' % (iter_num, model1_loss.item(), model2_loss.item(),worst_case_loss.item()))
            
            if iter_num > 0 and iter_num % 200 == 0:
                model.eval()
                
                metric_list = 0.0
                for i_batch, sampled_batch in enumerate(valloader):
                    metric_i = test_single_volume(
                        sampled_batch["image"], sampled_batch["label"], model, classes=num_classes,model_type='model1',device=device)
                    metric_list += np.array(metric_i)
                metric_list = metric_list / len(db_val)
                for class_i in range(num_classes-1):
                    writer.add_scalar('info/model1_val_{}_dice'.format(class_i+1),
                                      metric_list[class_i, 0], iter_num)
                    writer.add_scalar('info/model1_val_{}_hd95'.format(class_i+1),
                                      metric_list[class_i, 1], iter_num)

                performance1 = np.mean(metric_list, axis=0)[0]

                mean_hd951 = np.mean(metric_list, axis=0)[1]
                writer.add_scalar('info/model1_val_mean_dice', performance1, iter_num)
                writer.add_scalar('info/model1_val_mean_hd95', mean_hd951, iter_num)

                if performance1 > best_performance1:
                    best_performance1 = performance1
                    save_mode_path = os.path.join(snapshot_path,
                                                  'model1_iter_{}_dice_{}.pth'.format(
                                                      iter_num, round(best_performance1, 4)))
                    save_best = os.path.join(snapshot_path,
                                             '{}_best_model1.pth'.format(args["model"]))
                    torch.save(model.state_dict(), save_mode_path)
                    torch.save(model.state_dict(), save_best)

                metric_list = 0.0
                for i_batch, sampled_batch in enumerate(valloader):
                    metric_i = test_single_volume(
                        sampled_batch["image"], sampled_batch["label"], model, classes=num_classes,model_type='model2',device=device)
                    metric_list += np.array(metric_i)
                metric_list = metric_list / len(db_val)
                for class_i in range(num_classes-1):
                    writer.add_scalar('info/model2_val_{}_dice'.format(class_i+1),
                                      metric_list[class_i, 0], iter_num)
                    writer.add_scalar('info/model2_val_{}_hd95'.format(class_i+1),
                                      metric_list[class_i, 1], iter_num)

                performance2 = np.mean(metric_list, axis=0)[0]

                mean_hd952 = np.mean(metric_list, axis=0)[1]
                writer.add_scalar('info/model2_val_mean_dice', performance2, iter_num)
                writer.add_scalar('info/model2_val_mean_hd95', mean_hd952, iter_num)

                if performance2 > best_performance2:
                    best_performance2 = performance2
                    save_mode_path = os.path.join(snapshot_path,
                                                  'model2_iter_{}_dice_{}.pth'.format(
                                                      iter_num, round(best_performance2, 4)))
                    save_best = os.path.join(snapshot_path,
                                             '{}_best_model2.pth'.format(args["model"]))
                    torch.save(model.state_dict(), save_mode_path)
                    torch.save(model.state_dict(), save_best)
                
                logging.info(
                    'iteration %d : model1_mean_dice : %f model1_mean_hd95 : %f' % (iter_num, performance1, mean_hd951))
                
                logging.info(
                    'iteration %d : model2_mean_dice : %f model2_mean_hd95 : %f' % (iter_num, performance2, mean_hd952))

                model.train()

            if iter_num >= max_iterations:
                break
            time1 = time.time()
        if iter_num >= max_iterations:
            iterator.close()
            break
    writer.close()


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument('--cfg', type=str, default='config_2d.yml',help='configuration file')
    
    parser.add_argument('--root_path', type=str,
                        default='/data/zsp/ssl/data/ACDC', help='dataset path')
    parser.add_argument('--exp', type=str,
                        default='danm', help='experiment_name') # 不用加ACDC/
    parser.add_argument('--model', type=str,
                        default='acalnet', help='model_name')
    parser.add_argument('--max_iterations', type=int,
                        default=30000, help='maximum epoch number to train')
    parser.add_argument('--batch_size', type=int, default=24,
                        help='batch_size per gpu')
    parser.add_argument('--deterministic', type=int,  default=1,
                        help='whether use deterministic training')
    parser.add_argument('--base_lr', type=float,  default=0.01,
                        help='segmentation network learning rate')
    parser.add_argument('--image_size', type=list,  default=[256, 256],
                        help='size of network input')
    parser.add_argument('--seed', type=int,  default=1337, help='random seed')
    parser.add_argument('--num_classes', type=int,  default=4,
                        help='output channel of network')
    # label and unlabel
    parser.add_argument('--labeled_bs', type=int, default=12,
                        help='labeled_batch_size per gpu')
    parser.add_argument('--labeled_num', type=int, default=3,
                        help='labeled data')
    
    # costs
    parser.add_argument('--ema_decay', type=float,  default=0.99, help='ema_decay')
    parser.add_argument('--consistency_type', type=str,
                        default="ce", help='consistency_type')
    parser.add_argument('--consistency', type=float,
                        default=0.1, help='consistency')
    parser.add_argument('--consistency_rampup', type=float,
                        default=200.0, help='consistency_rampup')

    parser.add_argument('--gpu', type=int,  default=1, help='GPU to use')

    parser.add_argument("--acal", default=False,action='store_true')

    parser.add_argument("--extra", default=False,action='store_true')
    parser.add_argument("--worst", default=False,action='store_true')
    parser.add_argument('--decoder_type', type=str, default='same', choices=['same','plus','mcnet'])
    parser.add_argument('--worst_losstype', type=str,
                        default="ce", help='loss type of worst loss')
    parser.add_argument('--adv_losstype', type=str,
                        default="mse", help='loss type of adv loss')
    parser.add_argument('--text', type=str, default='null', help='discripition of the experiment')

    parser.add_argument('--patch_size', type=int,  default = 64)

    parser.add_argument('--trade_off_worst', type=float,  default=0.3)

    args = parser.parse_args()
    args = vars(args)

    cfgs_file = args['cfg']
    cfgs_file = os.path.join('../configs',cfgs_file)
    with open(cfgs_file, 'r') as handle:
        options_yaml = yaml.load(handle, Loader=yaml.FullLoader)
    # convert "1e-x" to float
    for each in options_yaml.keys():
        tmp_var = options_yaml[each]
        if type(tmp_var) == str and "1e-" in tmp_var:
            options_yaml[each] = float(tmp_var)
    # update original parameters of argparse
    update_values(options_yaml, args)
    # print confg information
    import pprint

    if not args["deterministic"]:
        cudnn.benchmark = True
        cudnn.deterministic = False
    else:
        cudnn.benchmark = False
        cudnn.deterministic = True

    random.seed(args["seed"])
    np.random.seed(args["seed"])
    torch.manual_seed(args["seed"])
    torch.cuda.manual_seed(args["seed"])

    exp_name = args['exp']
    snapshot_path = '../' + "model/{}/{}_{}_labeled".format('ACDC', exp_name, args['labeled_num'])
    if not os.path.exists(snapshot_path):
        os.makedirs(snapshot_path)
    save_dir = init_save_folder(snapshot_path,args['model'])
    shutil.copy('../code/train_share_encoder_2D.py', save_dir)

    # 写入说明
    if os.path.exists(os.path.join(save_dir,'doc.txt')):
        os.remove(os.path.join(save_dir,'doc.txt'))
    with open(os.path.join(save_dir,'doc.txt'),'w') as f:
       f.write(args['text'])

    logging.basicConfig(filename=save_dir+"/log.txt", level=logging.INFO,
                        format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info("{}".format(pprint.pformat(args)))
    train(args, save_dir)
