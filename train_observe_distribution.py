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

    all_pixels = []
    save_path = '/data/zsp/ssl/jbhi2025/acdc.npz'

    max_epoch = max_iterations // len(trainloader) + 1
    iterator = tqdm(range(max_epoch), ncols=70)
    
    for i_batch, sampled_batch in enumerate(trainloader):

        volume_batch = sampled_batch['image']
        img1 = volume_batch[0,:,:,:]
        img2 = volume_batch[1,:,:,:]
        img1 = img1.cpu().numpy().flatten()
        img2 = img2.cpu().numpy().flatten()

        all_pixels.append(img1)
        all_pixels.append(img2)

    all_pixels = np.concatenate(all_pixels)
            
    # 计算统计量
    stats = {
        "data": all_pixels,
        "mean": np.mean(all_pixels),
        "std": np.std(all_pixels),
        "min": np.min(all_pixels),  # 最小值 <button class="citation-flag" data-index="3"><button class="citation-flag" data-index="7">
        "max": np.max(all_pixels)   # 最大值 <button class="citation-flag" data-index="3"><button class="citation-flag" data-index="7">
    }
        
    # 保存为.npz文件（压缩格式）
    np.savez(save_path, **stats)
    print(f"Saved to {save_path}")

  
if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    
    parser.add_argument('--root_path', type=str,
                        default='/data/zsp/ssl/data/ACDC', help='dataset path')
    parser.add_argument('--exp', type=str,
                        default='drop', help='experiment_name') 
    parser.add_argument('--model', type=str,
                        default='dualdecoder', help='model_name')
    parser.add_argument('--max_iterations', type=int,
                        default=30000, help='maximum epoch number to train')
    parser.add_argument('--batch_size', type=int, default=2,
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
    parser.add_argument('--labeled_bs', type=int, default=1,
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
    
    parser.add_argument('--gpu', type=int,  default = 1, help='GPU to use')

    parser.add_argument('--w_adv', type=float,  default = 1.0)
    parser.add_argument('--w_drop', type=float,  default = 1.0)

    parser.add_argument('--noise_mag', type=float, default = 10.0, help='magnitude of overall noise')

    parser.add_argument('--decoder_type', type=str, default = 'mcnet', choices=['same','plus','mcnet'])
    parser.add_argument('--adv_type', type=str, default = 'image', choices=['feat','image'])
    parser.add_argument('--adv_losstype', type=str, default = 'kl', choices=['kl','dice'])
    parser.add_argument("--adv_noise", default = False,action = 'store_true')
    
    parser.add_argument("--dropout", default = False,action = 'store_true')
    parser.add_argument("--comp_drop", default = False,action = 'store_true')

    parser.add_argument('--text', type=str, default='null', help='discripition of the experiment')

    parser.add_argument('--topk1', type=float, default = 0.5, help='magnitude of overall noise')

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
    shutil.copy('../code/train_share_encoder_attack_2D.py', save_dir)
    
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






    
