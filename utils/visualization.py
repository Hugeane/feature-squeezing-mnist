import os
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


def draw_plot(xs, series_list, label_list, fname):
    fig, ax = plt.subplots()

    for i, series in enumerate(series_list):
        smask = np.isfinite(series)
        ax.plot(xs[smask], series[smask], linestyle='-', marker='o', label=label_list[i])

    legend = ax.legend(loc='best', shadow=True)

    # The frame is matplotlib.patches.Rectangle instance surrounding the legend.
    frame = legend.get_frame()
    frame.set_facecolor('0.90')

    # plt.show()
    plt.savefig(fname)
    plt.close(fig)


# 假设 X_test 和 X_test_adv 是已经生成的 NumPy 数组
def save_images(original_examples, adversarial_examples, target_vectors, output_dir='results/images', prefix='attack'):
    """
    将原始样本和对抗样本保存为图片到指定目录

    :param original_examples: 原始样本的 NumPy 数组，形状为 (batch_size, height, width, channels)
    :param adversarial_examples: 对抗样本的 NumPy 数组，形状为 (batch_size, height, width, channels)
    :param target_vectors: one-hot编码的预测值数组
    :param output_dir: 图片保存的目标路径
    :param prefix: 前缀
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # 定义CIFAR-10名称字典
    name_dict = {
        0: 'airplane',
        1: 'automobile',
        2: 'bird',
        3: 'cat',
        4: 'deer',
        5: 'dog',
        6: 'frog',
        7: 'horse',
        8: 'ship',
        9: 'truck'
    }

    # 还原为类别标签
    target_names = np.argmax(target_vectors, axis=1)

    # 遍历每个样本，保存原始图片和对抗图片 只存10个
    for i in range(10):
        # 获取原始样本和对抗样本
        original_img = original_examples[i]
        adversarial_img = adversarial_examples[i]

        # 将样本转化为 [0, 255] 范围的 uint8 数据类型
        original_img = (original_img * 255).astype(np.uint8)
        adversarial_img = (adversarial_img * 255).astype(np.uint8)

        # 从字典获取名称
        name = name_dict[target_names[i]]

        # 保存原始样本
        original_img_pil = Image.fromarray(original_img)
        original_img_pil.save(os.path.join(output_dir, f'{prefix}_{i}_{name}_origin.png'))

        # 保存对抗样本
        adversarial_img_pil = Image.fromarray(adversarial_img)
        adversarial_img_pil.save(os.path.join(output_dir, f'{prefix}_{i}_{name}_adversarial.png'))

    print(f"成功保存 {original_examples.shape[0]} 个原始样本和对抗样本到 {output_dir}.")
