import numpy as np
import matplotlib.pyplot as plt

# root_cps = '/data/zsp/ssl/miccai2025/model/ACDC/cps_unet_unet_7_labeled/unet'
# root_mt = '/data/zsp/ssl/miccai2025/model/ACDC/Mean_Teacher_7_labeled/unet'

# root = root_cps
# # 加载npy文件
# data = np.load(root+'/dice_and_conf.npy')  # 替换为你自己的文件路径

# idx = 1

# lab_metric = data[:, idx]
# unlab_metric = data[:, idx+3]

# # 创建训练进程的x轴（轮次）
# epochs = np.arange(1, len(lab_metric) + 1)

# sampling_interval = 500

# sampled_epochs = epochs[::sampling_interval]
# sampled_lab_metric = lab_metric[::sampling_interval]
# sampled_unlab_metric = unlab_metric[::sampling_interval]

# plt.plot(sampled_epochs, sampled_lab_metric, label='lab', color='b', linestyle='-', marker='o')
# plt.plot(sampled_epochs, sampled_unlab_metric, label='unlab', color='r', linestyle='--', marker='x')

# plt.ylim(0.98, 1.0)

# # 添加图例和标签
# plt.xlabel('Iterations')
# plt.ylabel('Confidence')
# plt.title('Correct Region Confidence Comparison')
# plt.legend()
# plt.grid(True)

# # 保存图片
# plt.savefig(root+'/lab_unlab_comparison.png', dpi=300)  # 保存为PNG格式

# # 显示图形
# plt.show()

txt_file = '/data/zsp/ssl/miccai2025/save/file4.txt'

import numpy as np

def load_epoch_data(file_path, target_epoch):
    """
    从文件中读取指定轮次的 global_channel_sim 数据。
    :param file_path: 保存数据的文件路径。
    :param target_epoch: 需要提取的轮次编号（从 1 开始）。
    :return: 一个 numpy 数组，表示目标轮次的数据；如果目标轮次不存在，返回 None。
    """
    current_epoch = None
    current_data = []

    with open(file_path, 'r') as f:
        for line in f:
            line = line.strip()
            if line.startswith("Epoch"):  # 检测到新的一轮
                if current_epoch == target_epoch:
                    # 如果刚好是目标轮次，返回数据
                    return np.array(current_data)
                # 提取新的 Epoch 编号
                current_epoch = int(line.split()[1].replace(":", ""))
                current_data = []  # 重置数据存储
            elif line:  # 非空行
                # 将当前行解析为浮点数数组
                current_data.append(list(map(float, line.split())))

    # 检查最后一轮是否是目标轮次
    if current_epoch == target_epoch:
        return np.array(current_data)

    return None  


file_path = txt_file
target_epoch = 30000

epoch_data = load_epoch_data(file_path, target_epoch)
avg_data = np.average(epoch_data,axis=0)

import matplotlib.pyplot as plt


# 创建一个柱状图
plt.bar(range(len(avg_data)), avg_data)

# 添加标题和标签
plt.title("channel similarity")
plt.xlabel("channel idx")
plt.ylabel("simi")

# 保存图像到文件
plt.savefig('/data/zsp/ssl/miccai2025/save/bar_chart.png')  # 保存为 PNG 格式的文件

# 显示图形
plt.show()



# if epoch_data is not None:
#     print(f"Epoch {target_epoch} data:")
#     print(epoch_data)
# else:
#     print(f"目标轮次 Epoch {target_epoch} 的数据不存在！")



