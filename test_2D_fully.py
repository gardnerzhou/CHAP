import argparse
import os
import shutil
import nibabel as nib
import numpy as np
import SimpleITK as sitk
import torch
from medpy import metric
from scipy.ndimage import zoom
from scipy.ndimage.interpolation import zoom
from tqdm import tqdm
import h5py
import matplotlib.pyplot as plt
from networks.net_factory import net_factory
import wandb

parser = argparse.ArgumentParser()
parser.add_argument('--root_path', type=str,
                    default='/data/zsp/ssl/data/ACDC', help='Name of Experiment')
parser.add_argument('--exp', type=str,
                    default='ACDC/base', help='experiment_name')
parser.add_argument('--runid', type=int,
                    default = 1, help='experiment_name')
parser.add_argument('--model', type=str,
                    default='dualdecoder', help='model_name')
parser.add_argument('--num_classes', type=int,  default=4,
                    help='output channel of network')
parser.add_argument('--labeled_num', type=int, default=7,
                    help='labeled data')
parser.add_argument('--gpu', type=int, default = 1, help='choose gpu')
parser.add_argument('--patchsize', type=int, default=256,help='patch size')
parser.add_argument('--decoder_type', type=str, default='mcnet')

# parser.add_argument('--wandb_id', type=str, default='a')
# parser.add_argument('--wandb_project', type=str, default='jbhi')

def calculate_metric_percase(pred, gt):
    pred[pred > 0] = 1
    gt[gt > 0] = 1
    dice = metric.binary.dc(pred, gt)
    if pred.sum() != 0:
        asd = metric.binary.asd(pred, gt)
        hd95 = metric.binary.hd95(pred, gt)
    else:
        print('bad')
        asd = 0
        hd95 = 0
    # asd = metric.binary.asd(pred, gt)
    # hd95 = metric.binary.hd95(pred, gt)
    jc = metric.binary.jc(pred, gt)
    return dice, hd95, asd, jc


def test_single_volume(case, net, test_save_path, FLAGS,device):
    h5f = h5py.File(FLAGS.root_path + "/data/{}.h5".format(case), 'r')
    image = h5f['image'][:]
    label = h5f['label'][:]
    prediction = np.zeros_like(label)
    patch_size = FLAGS.patchsize
    for ind in range(image.shape[0]):
        slice = image[ind, :, :]
        x, y = slice.shape[0], slice.shape[1]
        slice = zoom(slice, (patch_size / x, patch_size / y), order=0)
        input = torch.from_numpy(slice).unsqueeze(
            0).unsqueeze(0).float().to(device)
        net.eval()
        with torch.no_grad():
            
            out_main = net(input)
            # if isinstance(out_main,tuple):
            #         out_main = out_main[0]
            out_final = (out_main[0] + out_main[1])/2.0
            
            out = torch.argmax(torch.softmax(
                    out_final, dim=1), dim=1).squeeze(0)

            out = out.cpu().detach().numpy()
            pred = zoom(out, (x / patch_size, y / patch_size), order=0)
            prediction[ind] = pred

    first_metric = calculate_metric_percase(prediction == 1, label == 1)
    second_metric = calculate_metric_percase(prediction == 2, label == 2)
    third_metric = calculate_metric_percase(prediction == 3, label == 3)

    # img_itk = sitk.GetImageFromArray(image.astype(np.float32))
    # img_itk.SetSpacing((1, 1, 10))
    # prd_itk = sitk.GetImageFromArray(prediction.astype(np.float32))
    # prd_itk.SetSpacing((1, 1, 10))
    # lab_itk = sitk.GetImageFromArray(label.astype(np.float32))
    # lab_itk.SetSpacing((1, 1, 10))
    # sitk.WriteImage(prd_itk, test_save_path + case + "_pred.nii.gz")
    # sitk.WriteImage(img_itk, test_save_path + case + "_img.nii.gz")
    # sitk.WriteImage(lab_itk, test_save_path + case + "_gt.nii.gz")

    return first_metric, second_metric, third_metric

def Inference(FLAGS):
    with open(FLAGS.root_path + '/test.list', 'r') as f:
        image_list = f.readlines()
    image_list = sorted([item.replace('\n', '').split(".")[0]
                         for item in image_list])
    snapshot_path = "../model/{}_{}_labeled/{}/{}".format(
        FLAGS.exp, FLAGS.labeled_num,FLAGS.model,f"run_{FLAGS.runid}")
    
    test_save_path = "../model/{}_{}_labeled/{}/performances".format(
        FLAGS.exp, FLAGS.labeled_num, FLAGS.model)
    # if os.path.exists(test_save_path):
    #     shutil.rmtree(test_save_path)
    os.makedirs(test_save_path,exist_ok=True)

    net = net_factory(net_type=FLAGS.model, in_chns=1, class_num=FLAGS.num_classes,device=torch.device("cuda",FLAGS.gpu),args=vars(FLAGS))
    
    #a = next(net.parameters()).device

    save_mode_path = os.path.join(snapshot_path, '{}_best_model.pth'.format(FLAGS.model))
    
    net.load_state_dict(torch.load(save_mode_path,map_location=torch.device("cuda",FLAGS.gpu)))
    
    print("init weight from {}".format(save_mode_path))
    net.eval()
    
    first_total = 0.0
    second_total = 0.0
    third_total = 0.0
    for case in tqdm(image_list):
        first_metric, second_metric, third_metric = test_single_volume(
            case, net, test_save_path, FLAGS,device=torch.device("cuda",FLAGS.gpu))
        first_total += np.asarray(first_metric)
        second_total += np.asarray(second_metric)
        third_total += np.asarray(third_metric)
    avg_metric = [first_total / len(image_list), second_total /
                  len(image_list), third_total / len(image_list)]
    return avg_metric,test_save_path

if __name__ == '__main__':
        
    FLAGS = parser.parse_args()

    #run = wandb.init(id=FLAGS.wandb_id, resume="must", project=FLAGS.wandb_project)
    
    metric,test_save_path = Inference(FLAGS)

    average_metric = (metric[0]+metric[1]+metric[2])/3

    print(metric)
    print(average_metric)
    with open(test_save_path+f"/run_{FLAGS.runid}_performance.txt", 'a') as f:
        f.writelines('metric is {} \n'.format(metric))
        f.writelines('average metric is {}\n'.format(average_metric))
    
    #metric_names = ["dice", "hd95", "asd", "jc"]

    #test_results = {name: value for name, value in zip(metric_names, average_metric)}
    
    #wandb.log(test_results)
