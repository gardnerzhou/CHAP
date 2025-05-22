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
from utils import losses,ramps,patch,grad,simi
from val_2D import test_single_volume
from utils.launch import init_save_folder
from utils.util import update_values
import yaml
import wandb
import datetime
from test_2D_func import Inference
from skimage.measure import label

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

def generate_mask(img):
    device = img.device
    batch_size, channel, img_x, img_y = img.shape[0], img.shape[1], img.shape[2], img.shape[3]
    loss_mask = torch.ones(batch_size, img_x, img_y).to(device)
    mask = torch.ones(img_x, img_y).to(device)
    patch_x, patch_y = int(img_x*2/3), int(img_y*2/3)
    w = np.random.randint(0, img_x - patch_x)
    h = np.random.randint(0, img_y - patch_y)
    mask[w:w+patch_x, h:h+patch_y] = 0
    loss_mask[:, w:w+patch_x, h:h+patch_y] = 0
    return mask.long(), loss_mask.long()

def get_ACDC_masks(output, nms=0):
    probs = F.softmax(output, dim=1)
    _, probs = torch.max(probs, dim=1)
    if nms == 1:
        probs = get_ACDC_2DLargestCC(probs)      
    return probs

def get_ACDC_LargestCC(segmentation):
    class_list = []
    for i in range(1, 4):
        temp_prob = segmentation == i * torch.ones_like(segmentation)
        temp_prob = temp_prob.detach().cpu().numpy()
        labels = label(temp_prob)
        # -- with 'try'
        assert(labels.max() != 0)  # assume at least 1 CC
        largestCC = labels == np.argmax(np.bincount(labels.flat)[1:])+1
        class_list.append(largestCC * i)
    acdc_largestCC = class_list[0] + class_list[1] + class_list[2]
    return torch.from_numpy(acdc_largestCC).cuda()

def get_ACDC_2DLargestCC(segmentation):
    device = segmentation.device
    batch_list = []
    N = segmentation.shape[0]
    for i in range(0, N):
        class_list = []
        for c in range(1, 4):
            temp_seg = segmentation[i] #== c *  torch.ones_like(segmentation[i])
            temp_prob = torch.zeros_like(temp_seg)
            temp_prob[temp_seg == c] = 1
            temp_prob = temp_prob.detach().cpu().numpy()
            labels = label(temp_prob)          
            if labels.max() != 0:
                largestCC = labels == np.argmax(np.bincount(labels.flat)[1:])+1
                class_list.append(largestCC * c)
            else:
                class_list.append(temp_prob)
        
        n_batch = class_list[0] + class_list[1] + class_list[2]
        batch_list.append(n_batch)

    return torch.Tensor(batch_list).to(device)

def calculate_avg_dice(pred,gt):
    cls1 = calculate_metric_percase(pred==1, gt==1)
    cls2 = calculate_metric_percase(pred==2, gt==2)
    cls3 = calculate_metric_percase(pred==3, gt==3)
    return (cls1+cls2+cls3) / 3.0

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



dice_loss = losses.DiceLoss_bcp(n_classes=4)
def mix_loss(output, img_l, patch_l, mask, l_weight=1.0, u_weight=0.5, unlab=False):
    CE = nn.CrossEntropyLoss(reduction='none')
    img_l, patch_l = img_l.type(torch.int64), patch_l.type(torch.int64)
    output_soft = F.softmax(output, dim=1)
    image_weight, patch_weight = l_weight, u_weight
    if unlab:
        image_weight, patch_weight = u_weight, l_weight
    patch_mask = 1 - mask
    loss_dice1 = dice_loss(output_soft, img_l.unsqueeze(1), mask.unsqueeze(1)) * image_weight
    loss_dice2 = dice_loss(output_soft, patch_l.unsqueeze(1), patch_mask.unsqueeze(1)) * patch_weight
    loss_ce1 = image_weight * (CE(output, img_l) * mask).sum() / (mask.sum() + 1e-16) 
    loss_ce2 = patch_weight * (CE(output, patch_l) * patch_mask).sum() / (patch_mask.sum() + 1e-16)#loss = loss_ce
    loss_dice = loss_dice1 + loss_dice2
    loss_ce = loss_ce1 + loss_ce2

    loss_image = (loss_dice1 + loss_ce1) / 2.0
    loss_patch = (loss_dice2 + loss_ce2) / 2.0
    
    return loss_image,loss_patch,(loss_dice + loss_ce) / 2.0


def train(args, snapshot_path):

    # test = wandb.init(job_type='test',
    # project='jbhi',
    # name=f"acdc-{datetime.datetime.now()}",
    # notes=args["text"],
    # tags=['vat','cross'],
    # config=args
    # )
    
    # test.config.update({"model_save_folder":snapshot_path})
    
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

    #a = next(model.parameters()).device

    #ema_model = create_model(ema=True,model=args["model"],args=args)
    
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

    optimizer = optim.SGD(model.parameters(), lr=base_lr, momentum=0.9, weight_decay=0.0001)

    best_performance1 = 0.0
    best_performance2 = 0.0
    best_dice = 0.0
    iter_num = 0
    
    ce_loss = CrossEntropyLoss()
    #dice_loss = losses.DiceLoss(num_classes)

    gradsim = grad.GradSim(device,num_classes=num_classes,dir=snapshot_path)

    adv_loss = losses.VAT2d(xi=args["noise_mag"],epi=6.0,num_classes=num_classes)
    
    writer = SummaryWriter(snapshot_path + '/log')
    logging.info("{} iterations per epoch".format(len(trainloader)))
    
    labeled_sub_bs, unlabeled_sub_bs = int(args["labeled_bs"]/2), int((args["batch_size"]-args["labeled_bs"]) / 2)

    sim_score = gradsim.init_simsocre()

    max_epoch = max_iterations // len(trainloader) + 1
    iterator = tqdm(range(max_epoch), ncols=70)
    for epoch_num in iterator:
        for i_batch, sampled_batch in enumerate(trainloader):

            volume_batch, label_batch = sampled_batch['image'], sampled_batch['label']
            volume_batch, label_batch = volume_batch.to(device), label_batch.to(device)

            img_a, img_b = volume_batch[:labeled_sub_bs], volume_batch[labeled_sub_bs:args["labeled_bs"]]
            uimg_a, uimg_b = volume_batch[args["labeled_bs"]:args["labeled_bs"] + unlabeled_sub_bs], volume_batch[args["labeled_bs"] + unlabeled_sub_bs:]
            ulab_a, ulab_b = label_batch[args["labeled_bs"]:args["labeled_bs"] + unlabeled_sub_bs], label_batch[args["labeled_bs"] + unlabeled_sub_bs:]
            lab_a, lab_b = label_batch[:labeled_sub_bs], label_batch[labeled_sub_bs:args["labeled_bs"]]

            uimg_ab = torch.cat((uimg_a,uimg_b))

            with torch.no_grad():
                pre_ab1,pre_ab2 = model(uimg_ab)
                pre_a1,pre_b1 = pre_ab1.chunk(2)
                pre_a2,pre_b2 = pre_ab2.chunk(2)

                outputs_soft1 = F.softmax(pre_ab1, dim=1)
                outputs_soft2 = F.softmax(pre_ab2, dim=1)
                pseudo_outputs1 = torch.argmax(torch.softmax(pre_ab1, dim=1), dim=1)
                pseudo_outputs2 = torch.argmax(torch.softmax(pre_ab2, dim=1), dim=1)
                pseudo_supervision1 = F.cross_entropy(pre_ab1, pseudo_outputs2.long(),reduction='none')
                pseudo_supervision2 = F.cross_entropy(pre_ab2, pseudo_outputs1.long(),reduction='none')
                knowledge = pseudo_supervision1 + pseudo_supervision2

                plab_a1 = get_ACDC_masks(pre_a1.detach(), nms=1)
                plab_b1 = get_ACDC_masks(pre_b1.detach(), nms=1)
                plab_a2 = get_ACDC_masks(pre_a2.detach(), nms=1)
                plab_b2 = get_ACDC_masks(pre_b2.detach(), nms=1)
                img_mask, loss_mask = generate_mask(img_a)
                unl_label = ulab_a * img_mask + lab_a * (1 - img_mask)
                l_label = lab_b * img_mask + ulab_b * (1 - img_mask)

            net_input_unl = uimg_a * img_mask + img_a * (1 - img_mask)
            net_input_l = img_b * img_mask + uimg_b * (1 - img_mask)

            net_input_mix = torch.cat((net_input_l,net_input_unl))
            out_mix1,out_mix2 = model(net_input_mix)

            out_l1,out_unl1 = out_mix1.chunk(2)
            out_l2,out_unl2 = out_mix2.chunk(2)

            # cross supervision
            loss_u_out1,loss_l_in1,mix_loss1 = mix_loss(out_unl1, plab_a2, lab_a, loss_mask, u_weight=0.5, unlab=True)
            loss_u_out2,loss_l_in2,mix_loss2 = mix_loss(out_unl2, plab_a1, lab_a, loss_mask, u_weight=0.5, unlab=True)

            loss_l_out1,loss_u_in1,mix_loss3 = mix_loss(out_l1, lab_b, plab_b2, loss_mask, u_weight=0.5)
            loss_l_out2,loss_u_in2,mix_loss4 = mix_loss(out_l2, lab_b, plab_b1, loss_mask, u_weight=0.5)

            bcp_loss = mix_loss1 + mix_loss2 + mix_loss3 + mix_loss4

            loss_l = loss_l_in1 + loss_l_in2 + loss_l_out1 + loss_l_out2
            loss_u = loss_u_in1 + loss_u_in2 + loss_u_out1 + loss_u_out2
            
            consistency_weight = get_current_consistency_weight(iter_num // 150,args)
            
            # 2) fp
            if args["dropout"]:
                sim_score = gradsim.get_sim()
                outputs1_fp,outputs2_fp = model(uimg_ab,False,args["dropout"],[0,1,2,3,4],sim_score,False)
                pseudo_supervision1_fp = F.cross_entropy(outputs1_fp, pseudo_outputs2.long())
                pseudo_supervision2_fp = F.cross_entropy(outputs2_fp, pseudo_outputs1.long())
                fp_loss = pseudo_supervision1_fp + pseudo_supervision2_fp
                sim_last_iter = gradsim.get_grad_convkernel(loss_l, loss_u, model, optimizer, iter_num)
            else:
                fp_loss = torch.tensor(0.0).to(device)

            if args["adv_noise"]:
                
                diff_mask = patch.create_maskV1(pseudo_outputs1,pseudo_outputs2,knowledge,scale_factor=4,topk=args["topk1"])
                vat_loss = adv_loss(model,volume_batch,outputs_soft1,outputs_soft2,diff_mask,args["adv_losstype"])

            else:
                vat_loss = torch.tensor(0.0)


            loss = bcp_loss + consistency_weight * (fp_loss + vat_loss)


            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            iter_num = iter_num + 1

            lr_ = base_lr * (1.0 - iter_num / max_iterations) ** 0.9
            for param_group1 in optimizer.param_groups:
                param_group1['lr'] = lr_

            writer.add_scalar('lr', lr_, iter_num)

            # writer.add_scalar(
            #     'consistency_weight/consistency_weight', consistency_weight, iter_num)
            # writer.add_scalar('loss/model1_loss',
            #                   model1_loss, iter_num)
            # writer.add_scalar('loss/model2_loss',
            #                   model2_loss, iter_num)
            # writer.add_scalar('loss/vat_loss',
            #                   vat_loss, iter_num)
            # writer.add_scalar('loss/fp_loss',
            #                   fp_loss, iter_num)

            logging.info(
                'iteration %d : l loss : %f unl loss : %f ' % (iter_num, loss_l.item(), loss_u.item()))

            if iter_num > 0 and iter_num % 200 == 0:
                model.eval()

                metric_list = 0.0
                for i_batch, sampled_batch in enumerate(valloader):
                    metric_i = test_single_volume(
                        sampled_batch["image"], sampled_batch["label"], model, classes=num_classes,model_type='logit_ensemble',device=device)
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

                save_mode_path = os.path.join(snapshot_path, 'latest.pth')
                torch.save(model.state_dict(), save_mode_path)

                if performance1 > best_performance1:
                    best_performance1 = performance1
                    save_best = os.path.join(snapshot_path,
                                             '{}_best_model.pth'.format(args["model"]))
                    torch.save(model.state_dict(), save_best)

                    new_row = {
                        'timestamp': time.strftime("%Y-%m-%d %H:%M:%S"),
                        'iteration': iter_num,
                        'val_acc': round(best_performance1, 4),
                    }
                    log_file = os.path.join(snapshot_path, 'val.csv')
                    import pandas as pd
                    pd.DataFrame([new_row]).to_csv(
                        log_file,
                        mode='a',
                        header=False,
                        index=False
                    )

                logging.info(
                    'iteration %d : model1_mean_dice : %f model1_mean_hd95 : %f' % (iter_num, performance1, mean_hd951))



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
    
    parser.add_argument('--root_path', type=str,
                        default='/data/zsp/ssl/data/ACDC', help='dataset path')
    parser.add_argument('--exp', type=str,
                        default='bcp', help='experiment_name') 
    parser.add_argument('--model', type=str,
                        default='dualdecoder', help='model_name')
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
    
    parser.add_argument('--labeled_num', type=int, default=7,
                        help='labeled data')
    
    # costs
    parser.add_argument('--ema_decay', type=float,  default=0.99, help='ema_decay')
    parser.add_argument('--consistency_type', type=str,
                        default="ce", help='consistency_type')
    parser.add_argument('--consistency', type=float,
                        default=1.0, help='consistency')
    parser.add_argument('--consistency_rampup', type=float,
                        default=50.0, help='consistency_rampup')
    
    parser.add_argument('--gpu', type=int,  default = 2, help='GPU to use')

    parser.add_argument('--w_adv', type=float,  default = 1.0)
    parser.add_argument('--w_drop', type=float,  default = 1.0)
    
    parser.add_argument('--noise_mag', type=float, default = 10.0, help='magnitude of overall noise')

    parser.add_argument('--decoder_type', type=str, default = 'mcnet', choices=['same','plus','mcnet'])
    parser.add_argument('--adv_losstype', type=str, default = 'kl', choices=['kl','dice'])
    parser.add_argument("--adv_noise", default = False ,action = 'store_true')
    
    parser.add_argument("--dropout", default = False,action = 'store_true')
    parser.add_argument("--comp_drop", default = False,action = 'store_true')

    parser.add_argument('--text', type=str, default='null', help='discripition of the experiment')

    parser.add_argument('--topk1', type=float, default = 0.1, help='magnitude of overall noise')

    args = parser.parse_args()
    args = vars(args)

    # cfgs_file = args['cfg']
    # cfgs_file = os.path.join('../configs',cfgs_file)
    # with open(cfgs_file, 'r') as handle:
    #     options_yaml = yaml.load(handle, Loader=yaml.FullLoader)
    # # convert "1e-x" to float
    # for each in options_yaml.keys():
    #     tmp_var = options_yaml[each]
    #     if type(tmp_var) == str and "1e-" in tmp_var:
    #         options_yaml[each] = float(tmp_var)
    # # update original parameters of argparse
    # update_values(options_yaml, args)
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
    shutil.copy('../code/train_ours_bcp_2D.py', save_dir)
    
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
    
    # 测试指标
    #test_results = Inference(args,save_dir)

    #wandb.log(test_results)






    
