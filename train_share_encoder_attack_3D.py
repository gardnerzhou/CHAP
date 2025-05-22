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
from networks.net_factory_3d import net_factory_3d
from dataloaders.dataset import *
from utils import losses,ramps,patch,grad,simi,test_3d_patch
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
   
    dataset_name = args["dataset_name"]

    num_classes = 2

    if dataset_name == "LA":

        patch_size = (112, 112, 80)
        
        max_samples = 80

    elif dataset_name == "Pancreas":

        patch_size = (96, 96, 96)
        
        max_samples = 62

    train_data_path = args["root_path"] + dataset_name

    base_lr = args["base_lr"]
    batch_size = args["batch_size"]
    max_iterations = args["max_iterations"]
    num_classes = args["num_classes"]

    device = torch.device("cuda",args["gpu"])

    def create_model(ema=False,model='unet',args=None):
        # Network definition
        model = net_factory_3d(net_type=model, in_chns=1,
                            class_num=num_classes,device=device,args=args)
        if ema:
            for param in model.parameters():
                param.detach_()
        return model
    
    model = create_model(model=args["model"],args=args)
    #ema_model = create_model(ema=True,model=args["model"],args=args)
    
    model.train()

    def worker_init_fn(worker_id):
        random.seed(args["seed"] + worker_id)

    if dataset_name == "LA":
        
        db_train = LAHeart(base_dir=train_data_path,
                        split='train',
                        transform = transforms.Compose([
                            RandomRotFlip(),
                            RandomCrop(patch_size),
                            ToTensor(),
                            ]))

    elif dataset_name == "Pancreas":
        
        db_train = Pancreas(base_dir=train_data_path,
                    split='train',
                    transform = transforms.Compose([
                        RandomCrop(patch_size),
                        ToTensor(),
                        ]))
    
    
    labelnum = args["labeled_num"]  
    labeled_idxs = list(range(labelnum))
    unlabeled_idxs = list(range(labelnum, max_samples))
    batch_sampler = TwoStreamBatchSampler(labeled_idxs, unlabeled_idxs, args["batch_size"], args["batch_size"]-args["labeled_bs"])

    trainloader = DataLoader(db_train, batch_sampler=batch_sampler,num_workers=4, pin_memory=True, worker_init_fn=worker_init_fn)

    optimizer = optim.SGD(model.parameters(), lr=base_lr, momentum=0.9, weight_decay=0.0001)

    best_performance1 = 0.0
    best_performance2 = 0.0
    best_dice1 = 0.0
    iter_num = 0
    
    ce_loss = CrossEntropyLoss()

    dice_loss = losses.DiceLoss(num_classes)

    gradsim = grad.GradSim(device)
    
    if args["adv_type"] == 'feat':
        adv_loss = losses.VAT2d_feat(xi=1e-1,epi=6.0)
    elif args["adv_type"] == 'image':
        adv_loss = losses.VAT2d(xi=args["noise_mag"],epi=6.0)
        #adv_loss = losses.VAT2d_cutmix(xi=args["noise_mag"],epi=6.0) #
    else:
        raise ValueError('Damn.')
    
    writer = SummaryWriter(snapshot_path + '/log')
    logging.info("{} iterations per epoch".format(len(trainloader)))

    max_epoch = max_iterations // len(trainloader) + 1
    iterator = tqdm(range(max_epoch), ncols=70)
    for epoch_num in iterator:
        for i_batch, sampled_batch in enumerate(trainloader):

            volume_batch, label_batch = sampled_batch['image'], sampled_batch['label']
            volume_batch, label_batch = volume_batch.to(device), label_batch.to(device)
            
            # with torch.no_grad():
            #     ema_model.train()
            #     ema_feats = ema_model.forward_encoder(volume_batch)
            #     ema_output1,ema_output2 = ema_model.forward_decoder(ema_feats)
            #     ema_output1 = torch.split(ema_output1, ema_output1.size(0) * 2 // 3, dim=0)[0]
            #     ema_output2 = torch.split(ema_output2, ema_output2.size(0) * 2 // 3, dim=0)[0]
            #     ema_output_soft1 = torch.softmax(ema_output1, dim=1)
            #     ema_output_soft2 = torch.softmax(ema_output2, dim=1)
            # outputs_soft_ensemble = (ema_output_soft1 + ema_output_soft2) / 2.0
            
            #outputs1,outputs2 = model(volume_batch,args["dropout"],[0,1,2,3,4],gradsim.sim)
            outputs1,outputs2 = model(volume_batch)
            
            outputs1,outputs1_fp = outputs1[:args["batch_size"]], outputs1[args["batch_size"]:]
            outputs2,outputs2_fp = outputs2[:args["batch_size"]], outputs2[args["batch_size"]:]
            
            outputs_soft1 = torch.softmax(outputs1, dim=1)
            outputs_soft2 = torch.softmax(outputs2, dim=1)

            consistency_weight = get_current_consistency_weight(iter_num // 150,args)

            loss1 = 0.5 * (ce_loss(outputs1[:args["labeled_bs"]],
                                   label_batch[:][:args["labeled_bs"]].long()) + dice_loss(
                outputs_soft1[:args["labeled_bs"]], label_batch[:args["labeled_bs"]].unsqueeze(1)))
            loss2 = 0.5 * (ce_loss(outputs2[:args["labeled_bs"]],
                                   label_batch[:][:args["labeled_bs"]].long()) + dice_loss(
                outputs_soft2[:args["labeled_bs"]], label_batch[:args["labeled_bs"]].unsqueeze(1)))
            
            sup_loss = loss1 + loss2

            pseudo_logits1, pseudo_outputs1 = torch.max(outputs_soft1[args["labeled_bs"]:].detach(), dim=1)
            pseudo_logits2, pseudo_outputs2 = torch.max(outputs_soft2[args["labeled_bs"]:].detach(), dim=1)
            
            # pseudo_outputs1_fp = torch.argmax(outputs_soft1_fp.detach(), dim=1, keepdim=False)
            # pseudo_outputs2_fp = torch.argmax(outputs_soft2_fp.detach(), dim=1, keepdim=False)
            
            if args["consistency_type"] == 'ce':

                consistent_mask = (pseudo_outputs1 == pseudo_outputs2)
                confident_mask = (pseudo_logits1.ge(0.9).bool()) & (pseudo_logits2.ge(0.9).bool())
                cc_mask = consistent_mask & confident_mask

                pseudo_supervision1 = F.cross_entropy(outputs1[args["labeled_bs"]:], pseudo_outputs2.long(),reduction='none')
                pseudo_supervision2 = F.cross_entropy(outputs2[args["labeled_bs"]:], pseudo_outputs1.long(),reduction='none')
                
                # pseudo_supervision1 = torch.sum(pseudo_supervision1 * cc_mask) / (torch.sum(cc_mask).item() + 1e-10) 
                # pseudo_supervision2 = torch.sum(pseudo_supervision2 * cc_mask) / (torch.sum(cc_mask).item() + 1e-10) 

                knowledge = pseudo_supervision1 + pseudo_supervision2
                # knowledge1 = F.kl_div(outputs_soft1[args["labeled_bs"]:].detach().log(),outputs_soft2[args["labeled_bs"]:].detach(),reduction='none')
                # knowledge2 = F.kl_div(outputs_soft2[args["labeled_bs"]:].detach().log(),outputs_soft1[args["labeled_bs"]:].detach(),reduction='none')
                # knowledge = 0.5 * (knowledge1.mean(dim=1) + knowledge2.mean(dim=1))

                if args["dropout"] :
                    
                    outputs_soft_ensemble = (outputs_soft1 + outputs_soft2) / 2.0
                    pseudo_outputs_ensemble = torch.argmax(outputs_soft_ensemble[args["labeled_bs"]:].detach(), dim=1, keepdim=False)

                    pseudo_supervision1_fp = F.cross_entropy(outputs1_fp, pseudo_outputs_ensemble.long())
                    pseudo_supervision2_fp = F.cross_entropy(outputs2_fp, pseudo_outputs_ensemble.long())
                    fp_loss = pseudo_supervision1_fp + pseudo_supervision2_fp

                else:
                    fp_loss = torch.tensor(0.0)
                    
                model1_loss = loss1 + consistency_weight * pseudo_supervision1.mean()
                model2_loss = loss2 + consistency_weight * pseudo_supervision2.mean()

            if args["consistency_type"] == 'mse':
                pseudo_label1 = sharpening(outputs_soft1[args["labeled_bs"]:])
                pseudo_label2 = sharpening(outputs_soft2[args["labeled_bs"]:])
                pseudo_supervision1 = losses.mse_loss(outputs_soft1[args["labeled_bs"]:], pseudo_label2.detach())
                pseudo_supervision2 = losses.mse_loss(outputs_soft2[args["labeled_bs"]:], pseudo_label1.detach())

            if args["consistency_type"] == 'dice':
                pseudo_supervision1 = dice_loss(torch.softmax(outputs1[args["labeled_bs"]:],dim=1), pseudo_outputs2.unsqueeze(1).float(),ignore=torch.zeros_like(pseudo_outputs2).float())
                pseudo_supervision2 = dice_loss(torch.softmax(outputs2[args["labeled_bs"]:],dim=1), pseudo_outputs1.unsqueeze(1).float(),ignore=torch.zeros_like(pseudo_outputs1).float())

                model1_loss = loss1 + consistency_weight * pseudo_supervision1
                model2_loss = loss2 + consistency_weight * pseudo_supervision2

            unsup_loss = (pseudo_supervision1.mean()+pseudo_supervision2.mean())
            
            if args["dropout"]:
                gradsim.get_grad(sup_loss,unsup_loss,model,optimizer)

            if args["adv_noise"]:
                 
                # ! image
                #diff_mask = patch.cal_topkmask(16,knowledge,0.3,largest=False)
                #diff_mask = patch.create_maskV1(pseudo_outputs1,pseudo_outputs2,knowledge,scale_factor=4,topk=0.5)
                #diff_mask = cc_mask
                diff_mask = None
                vat_loss = adv_loss(model,volume_batch,outputs_soft1,outputs_soft2,diff_mask,args["adv_losstype"])

            else:
                vat_loss = torch.tensor(0.0)

            loss = model1_loss + model2_loss + consistency_weight * (vat_loss * args["w_adv"] + fp_loss)

            optimizer.zero_grad()
            
            loss.backward()
            optimizer.step()

            iter_num = iter_num + 1

            lr_ = base_lr * (1.0 - iter_num / max_iterations) ** 0.9
            for param_group1 in optimizer.param_groups:
                param_group1['lr'] = lr_
            
            writer.add_scalar('lr', lr_, iter_num)

            writer.add_scalar(
                'consistency_weight/consistency_weight', consistency_weight, iter_num)
            writer.add_scalar('loss/model1_loss',
                              model1_loss, iter_num)
            writer.add_scalar('loss/model2_loss',
                              model2_loss, iter_num)
            writer.add_scalar('loss/vat_loss',
                              vat_loss, iter_num)
            writer.add_scalar('loss/fp_loss',
                              fp_loss, iter_num)
            
            logging.info(
                'iteration %d : model1 loss : %f model2 loss : %f vat loss: %f fp loss: %f' % (iter_num, model1_loss.item(), model2_loss.item(),vat_loss.item(),fp_loss.item()))
            
            if iter_num >= 100 and iter_num % 200 == 0:

                model.eval()
                if dataset_name =="LA":
                    dice_sample = test_3d_patch.var_all_case(model,num_classes=num_classes, device=device,patch_size=patch_size, stride_xy=18, stride_z=4, dataset_name = 'LA')
                elif dataset_name =="Pancreas":
                    dice_sample = test_3d_patch.var_all_case(model,num_classes=num_classes,device=device, patch_size=patch_size, stride_xy=16, stride_z=16, dataset_name = 'Pancreas_CT')
                if dice_sample > best_dice1:
                    best_dice1 = dice_sample
                    save_mode_path = os.path.join(snapshot_path,  'model_iter_{}_dice_{}.pth'.format(iter_num, round(best_dice1,4)))
                    save_best_path = os.path.join(snapshot_path,'{}_best_model.pth'.format(args["model"]))
                    
                    torch.save(model.state_dict(), save_mode_path)
                    torch.save(model.state_dict(), save_best_path)
                    
                    logging.info('iteration %d : dice_score: %f best_dice: %f' % (iter_num, dice_sample, best_dice1))
                    logging.info("save best model to {}".format(save_mode_path))
                
                writer.add_scalar('Var_dice/Dice', dice_sample, iter_num)
                writer.add_scalar('Var_dice/Best_dice', best_dice1, iter_num)
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

    parser.add_argument('--cfg', type=str, default='config_attack_3d.yml',help='configuration file')
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

    parser.add_argument('--gpu', type=int,  default = 1, help='GPU to use')
    parser.add_argument('--w_adv', type=float,  default = 1.0)
    parser.add_argument('--noise_decay', type=float, default = 0.7, help='unimportant regions')
    parser.add_argument('--noise_mag', type=float, default = 0.1, help='magnitude of overall noise')

    parser.add_argument('--decoder_type', type=str, default = 'same', choices=['same','plus','mcnet'])
    parser.add_argument('--adv_losstype', type=str, default = 'softdice', choices=['softdice','kl'])
    parser.add_argument('--adv_type', type=str, default = 'feat', choices=['feat','image'])
    
    parser.add_argument("--adv_noise", default=False,action='store_true')
    parser.add_argument("--dropout", default=False,action='store_true')
    
    parser.add_argument('--text', type=str, default='null', help='discripition of the experiment')


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
    snapshot_path = '../' + "model/{}/{}_{}_labeled".format(args['dataset_name'], exp_name, args['labeled_num'])
    if not os.path.exists(snapshot_path):
        os.makedirs(snapshot_path)
    save_dir = init_save_folder(snapshot_path,args['model'])
    shutil.copy('../code/train_share_encoder_attack_3D.py', save_dir)

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
