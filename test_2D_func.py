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
import torch.nn.functional as F
import csv

parser = argparse.ArgumentParser()
parser.add_argument('--root_path', type=str,
                    default='/data/zsp/ssl/data/ACDC', help='Name of Experiment')
parser.add_argument('--exp', type=str,
                    default='ACDC/vat', help='experiment_name')
parser.add_argument('--runid', type=int,
                    default = 7, help='experiment_name')
parser.add_argument('--model', type=str,
                    default='dualdecoder', help='model_name')
parser.add_argument('--num_classes', type=int,  default=4,
                    help='output channel of network')
parser.add_argument('--labeled_num', type=int, default=7,
                    help='labeled data')
parser.add_argument('--gpu', type=int, default = 0, help='choose gpu')
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


def test_single_volume(case, iter_num, net, test_save_path, FLAGS,device):
    h5f = h5py.File(FLAGS.root_path + "/data/{}.h5".format(case), 'r')
    image = h5f['image'][:]
    label = h5f['label'][:]
    prediction1 = np.zeros_like(label)
    prediction2 = np.zeros_like(label)
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

            out_final = (out_main[0] + out_main[1])/2.0
            
            out1 = torch.argmax(torch.softmax(
                    out_main[0], dim=1), dim=1).squeeze(0)
            out2 = torch.argmax(torch.softmax(
                    out_main[1], dim=1), dim=1).squeeze(0)

            out1 = out1.cpu().detach().numpy()
            out2 = out2.cpu().detach().numpy()

            pred1 = zoom(out1, (x / patch_size, y / patch_size), order=0)
            prediction1[ind] = pred1

            pred2 = zoom(out2, (x / patch_size, y / patch_size), order=0)
            prediction2[ind] = pred2
    
    # first_metric = calculate_metric_percase(prediction == 1, label == 1)
    # second_metric = calculate_metric_percase(prediction == 2, label == 2)
    # third_metric = calculate_metric_percase(prediction == 3, label == 3)

    img_itk = sitk.GetImageFromArray(image.astype(np.float32))
    img_itk.SetSpacing((1, 1, 10))
    prd_itk1 = sitk.GetImageFromArray(prediction1.astype(np.float32))
    prd_itk1.SetSpacing((1, 1, 10))
    prd_itk2 = sitk.GetImageFromArray(prediction2.astype(np.float32))
    prd_itk2.SetSpacing((1, 1, 10))
    lab_itk = sitk.GetImageFromArray(label.astype(np.float32))
    lab_itk.SetSpacing((1, 1, 10))
    folder = test_save_path + '/'+ str(iter_num)
    os.makedirs(folder,exist_ok=True)
    sitk.WriteImage(prd_itk1, folder + '/' + case + "_pred1.nii.gz")
    sitk.WriteImage(prd_itk2,  folder + '/' + case + "_pred2.nii.gz")
    sitk.WriteImage(img_itk,  folder + '/' + case + "_img.nii.gz")
    sitk.WriteImage(lab_itk,  folder + '/' + case + "_gt.nii.gz")

    return 1

def Inference(FLAGS):
    with open(FLAGS.root_path + '/test.list', 'r') as f:
        image_list = f.readlines()
    image_list = sorted([item.replace('\n', '').split(".")[0]
                         for item in image_list])
    snapshot_path = "../model/{}_{}_labeled/{}/{}".format(
        FLAGS.exp, FLAGS.labeled_num,FLAGS.model,f"run_{FLAGS.runid}")
    
    # test_save_path = "../model/{}_{}_labeled/{}/performances".format(
    #     FLAGS.exp, FLAGS.labeled_num, FLAGS.model)
    
    test_save_path = snapshot_path + '/prediction'
    # if os.path.exists(test_save_path):
    #     shutil.rmtree(test_save_path)
    os.makedirs(test_save_path,exist_ok=True)

    net = net_factory(net_type=FLAGS.model, in_chns=1, class_num=FLAGS.num_classes,device=torch.device("cuda",FLAGS.gpu),args=vars(FLAGS))
    
    #a = next(net.parameters()).device
    matched_files = []
    for entry in os.listdir(snapshot_path):
        
        if entry.startswith("model1_iter") and entry.endswith(".pth"):
            matched_files.append(entry)
    
    for model_path in matched_files:
        iter_num = int(model_path.split('_')[2])
        save_mode_path = os.path.join(snapshot_path, model_path)
        
        net.load_state_dict(torch.load(save_mode_path,map_location=torch.device("cuda",FLAGS.gpu)))
        
        net.eval()
        
        first_total = 0.0
        second_total = 0.0
        third_total = 0.0
        dis_ratio_total = 0.0
        for case in tqdm(image_list):
            test_single_volume(
                case, iter_num , net, test_save_path, FLAGS,device = torch.device("cuda",FLAGS.gpu))
            #dis_ratio_total += dis_ratio
            # first_total += np.asarray(first_metric)
            # second_total += np.asarray(second_metric)
            # third_total += np.asarray(third_metric)
        # avg_metric = [first_total / len(image_list), second_total /
        #               len(image_list), third_total / len(image_list)]
        #avg_dis = dis_ratio_total / len(image_list)

        #save_disagreement_ratio(iter_num,avg_dis,csv_path='/data/zsp/ssl/jbhi2025/save/base_kl.csv')
    
    return 


def save_disagreement_ratio(iteration, ratio, csv_path="disagreement.csv"):
    """
    追加保存数据并保证CSV文件始终按迭代次数排序
    - 自动处理文件创建/追加
    - 自动去重相同迭代记录（保留最后一次写入的数据）
    """
    # 读取已有数据（如果文件存在）
    existing_data = {}
    if os.path.exists(csv_path):
        with open(csv_path, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                existing_data[int(row['Iteration'])] = float(row['Disagreement Ratio'])
    
    # 更新/添加新数据
    existing_data[iteration] = ratio
    
    # 转换为排序后的列表
    sorted_iterations = sorted(existing_data.keys())
    sorted_rows = [[iter, existing_data[iter]] for iter in sorted_iterations]
    
    # 写入文件（覆盖模式）
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["Iteration", "Disagreement Ratio"])
        writer.writerows(sorted_rows)
    print(f"数据已排序保存至 {os.path.abspath(csv_path)}")



def compute_kl_divergence(P, Q, epsilon=1e-8):
    """
    计算两个概率图之间的KL散度
    Args:
        P (Tensor): 概率图1，形状[1, C, H, W]
        Q (Tensor): 概率图2，形状[1, C, H, W]
        epsilon (float): 数值稳定系数
    Returns:
        kl_map (Tensor): KL散度图，形状[1, H, W]
    """
    # 确保概率值不小于epsilon，避免log(0)
    P = P.clone().clamp(min=epsilon)
    Q = Q.clone().clamp(min=epsilon)
    
    # 计算KL散度：sum(P * (logP - logQ))，沿通道维度C求和
    kl_map = (P * (torch.log(P) - torch.log(Q))).sum(dim=1)
    return kl_map

def plot_kl_over_img(out_prob1,out_prob2,case,ind,save_path):
    if input.shape[1] == 1:
        original_image = input.repeat(1, 3, 1, 1)  # [1,3,H,W]
    original_np = original_image.squeeze().permute(1,2,0).cpu().numpy()
    original_np = (original_np - original_np.min()) / (original_np.max() - original_np.min())
    
    kl_div = compute_kl_divergence(out_prob1, out_prob2)
    kl_div = (kl_div - kl_div.min()) / (kl_div.max() - kl_div.min())
    kl_normalized = kl_div ** 1.0
    kl_normalized = torch.clamp(kl_normalized, 0, 1)  # 数值保护
    
    kl_np = kl_div.squeeze().cpu().numpy()
    original_size = original_image.shape[-2:]
    kl_resized = F.interpolate(
        kl_normalized.unsqueeze(0), 
        size=original_size, 
        mode='bilinear', 
        align_corners=False
    ).squeeze().cpu().numpy()
    
    cmap = plt.get_cmap('jet')
    heatmap_rgba = cmap(kl_resized)
    
    alpha_scale = 3.0
    threshold= 0.0
    # 创建alpha通道（基于KL值动态调整）
    alpha = np.maximum(kl_resized - threshold, 0)  # 阈值截断
    alpha = alpha / (alpha.max() + 1e-8)           # 重新归一化
    alpha = alpha ** 2 * alpha_scale               # 非线性增强
    alpha = np.clip(alpha, 0, 1)                   # 限制范围
    
    # 分离RGB和alpha通道
    heatmap_rgb = heatmap_rgba[..., :3]
    alpha_mask = np.expand_dims(alpha, axis=-1)
    
    # 智能叠加（仅在显著区域混合）
    overlay = original_np * (1 - alpha_mask) + heatmap_rgb * alpha_mask
    overlay = np.clip(overlay, 0, 1)
    
    # 保存图像
    plt.imsave(save_path+'/kl_divergence_{}_{}.png'.format(case,ind), overlay)

def compute_disagreement(pred1,pred2):

    assert pred1.shape == pred2.shape, "两个张量的形状必须相同"
    
    # 获取预测类别（形状变为[1, H, W]）
    pred1_labels = torch.argmax(pred1, dim=1)
    pred2_labels = torch.argmax(pred2, dim=1)
    
    # 比较不一致的像素（布尔张量）
    disagreement_mask = (pred1_labels != pred2_labels)
    
    # 计算不一致的像素数量
    num_disagree = torch.sum(disagreement_mask).item()
    
    # 总像素数（H * W）
    total_pixels = pred1_labels.shape[1] * pred1_labels.shape[2]
    
    # 计算比例
    disagreement_ratio = num_disagree / total_pixels
    return disagreement_ratio


def plot_multiple_disagreement(csv_paths, labels=None, save_path="/data/zsp/ssl/jbhi2025/save/save.png"):
    """
    同时绘制多个CSV文件的不一致比例曲线
    :param csv_paths: CSV文件路径列表 (至少2个)
    :param labels: 每条曲线的标签列表 (长度需与csv_paths一致)
    :param save_path: 图片保存路径
    """
    # 参数校验
    assert len(csv_paths) >= 2, "至少需要提供两个CSV文件路径"
    if labels:
        assert len(labels) == len(csv_paths), "标签数量必须与文件数量一致"
    else:
        labels = [os.path.basename(p).split('.')[0] for p in csv_paths]

    plt.figure(figsize=(12, 6))
    colors = plt.cm.tab10(np.linspace(0, 1, len(csv_paths)))  # 使用不同颜色
    
    max_iter = 0  # 记录最大迭代次数用于统一x轴
    
    for idx, path in enumerate(csv_paths):
        # 检查文件是否存在
        if not os.path.exists(path):
            raise FileNotFoundError(f"CSV文件 {path} 不存在")
        
        # 读取数据
        iterations, ratios = [], []
        with open(path, 'r') as f:
            reader = csv.reader(f)
            header = next(reader)  # 读取表头
            if header != ["Iteration", "Disagreement Ratio"]:
                print(f"警告：{path} 的表头格式可能不兼容")
            
            for row in reader:
                iterations.append(int(row[0]))
                ratios.append(float(row[1]))
        
        # 更新最大迭代次数
        if len(iterations) > 0:
            max_iter = max(max_iter, max(iterations))
        
        # 绘制曲线
        plt.plot(iterations, ratios, 
                 color=colors[idx], 
                 marker='o' if idx<5 else 's',  # 前5种用圆形标记，其他用方形
                 linestyle='-',
                 linewidth=1.5,
                 markersize=5,
                 label=labels[idx])

    # 图表美化
    plt.title("zz", fontsize=14, pad=20)
    plt.xlabel("aa", fontsize=12)
    plt.ylabel("bb", fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # 统一坐标轴范围
    if max_iter > 0:
        plt.xticks(np.linspace(0, max_iter, num=6, dtype=int))
    plt.ylim(0, 0.02)  # 假设比例范围在0~1之间
    
    # 图例位置优化
    plt.legend(loc='upper right', 
              bbox_to_anchor=(1.15, 1),  # 防止图例遮挡曲线
              frameon=False)
    
    # 保存并显示
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()


if __name__ == '__main__':
        
    FLAGS = parser.parse_args()

    Inference(FLAGS)



