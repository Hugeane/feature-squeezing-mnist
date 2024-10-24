import os
import matplotlib.pyplot as plt
import numpy as np


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


def save_mnist_examples(original_examples, adversarial_examples, error_index, output_dir='results/images/error'):
    """
    批量保存MNIST原始样本和对抗样本到指定路径
    :param error_index: 需要保存的对象下标
    :param original_examples: 原始样本图像列表 (list of numpy arrays)
    :param adversarial_examples: 对抗样本图像列表 (list of numpy arrays)
    :param output_dir: 保存图像的相对路径
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # 确保两个列表长度相同
    assert len(original_examples) == len(adversarial_examples), "原始样本列表和对抗样本列表长度不一致！"

    err_origin_examples, err_adversarial_examples = original_examples[error_index], adversarial_examples[error_index]

    # 遍历每一个样本并保存
    for i in range(len(err_origin_examples)):
        err_idx = error_index[i]
        original_file_name = os.path.join(output_dir, 'origin_example_%s.png' % err_idx)
        adversarial_file_name = os.path.join(output_dir, 'adversarial_example_%s.png' % err_idx)

        # 保存原始样本
        plt.imsave(original_file_name, original_examples[i].reshape(28, 28), cmap='gray')
        print("Saved original example at:", original_file_name)

        # 保存对抗样本
        plt.imsave(adversarial_file_name, adversarial_examples[i].reshape(28, 28), cmap='gray')
        print("Saved adversarial example at:", adversarial_file_name)
