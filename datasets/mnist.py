import os
import sys

import numpy as np
import urllib3
import tarfile
import tensorflow as tf

from models.carlini_models import carlini_mnist_model
from models.cleverhans_models import cleverhans_mnist_model
from models.pgdtrained_models import pgdtrained_mnist_model

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class MNISTDataset:
    def __init__(self):
        self.dataset_name = "MNIST"
        self.image_size = 28
        self.num_channels = 1
        self.num_classes = 10
        self.mnist_classes = {
            0: '0',
            1: '1',
            2: '2',
            3: '3',
            4: '4',
            5: '5',
            6: '6',
            7: '7',
            8: '8',
            9: '9'
        }

    # load MNIST dataset

    def download_mnist_dataset(self, filepath, url):
        print("Downloading MNIST dataset...")
        http = urllib3.PoolManager()
        try:
            response = http.request('GET', url, preload_content=False)
            with open(filepath, 'wb') as f:
                for chunk in response.stream(1024):
                    f.write(chunk)
            print("Download completed.")
            response.release_conn()
        except Exception as e:
            # 如果下载失败，删除不完整的文件
            if os.path.exists(filepath):
                os.remove(filepath)
            print(f"Error downloading dataset: {e}")
            raise

    def load_mnist_dataset(self, dataset_dir='datasets/mnist'):
        if not os.path.exists(dataset_dir):
            os.makedirs(dataset_dir)

        filepath = os.path.join(dataset_dir, 'mnist.npz')

        # check directory and file
        if not os.path.isfile(filepath) or os.path.getsize(filepath) < 10 * 1024 * 1024:
            url = 'https://s3.amazonaws.com/img-datasets/mnist.npz'
            self.download_mnist_dataset(filepath, url)

        # load dataset
        with np.load(filepath, allow_pickle=True) as f:
            x_train, y_train = f['x_train'], f['y_train']
            x_test, y_test = f['x_test'], f['y_test']

        return (x_train, y_train), (x_test, y_test)

    def get_test_dataset(self):
        (X_train, y_train), (X_test, y_test) = self.load_mnist_dataset()
        X_test = X_test.reshape(X_test.shape[0], self.image_size, self.image_size, self.num_channels)
        X_test = X_test.astype('float32')
        X_test /= 255
        Y_test = tf.keras.utils.to_categorical(y_test, self.num_classes)
        del X_train, y_train
        return X_test, Y_test

    def get_val_dataset(self):
        (X_train, y_train), (X_test, y_test) = self.load_mnist_dataset()
        val_size = 5000
        X_val = X_train[:val_size]
        X_val = X_val.reshape(X_val.shape[0], self.image_size, self.image_size, self.num_channels)
        X_val = X_val.astype('float32') / 255
        y_val = y_train[:val_size]
        Y_val = tf.keras.utils.to_categorical(y_val, self.num_classes)
        del X_train, y_train, X_test, y_test
        return X_val, Y_val

    def load_model_by_name(self, model_name, logits=False, input_range_type=1, pre_filter=lambda x: x):
        """
        :params logits: return logits(input of softmax layer) if True; return softmax output otherwise.
        :params input_range_type: {1: [0,1], 2:[-0.5, 0.5], 3:[-1, 1]...}
        """
        if model_name not in ["cleverhans", 'cleverhans_adv_trained', 'carlini', 'pgdtrained', 'pgdbase']:
            raise NotImplementedError("Undefined model [%s] for MNIST." % model_name)

        download_dir = 'downloads'
        trained_model_dir = 'downloads/trained_models'

        # dir not exist, make dir and download trained models from github, then unzip the url file to directory
        # downloads github url: https://github.com/mzweilin/EvadeML-Zoo/releases/download/v0.1/downloads.tar.gz
        if not os.path.isdir(download_dir):
            os.makedirs(download_dir)
        if not os.listdir(download_dir):
            # 下载和解压模型
            model_archive_url = "https://github.com/mzweilin/EvadeML-Zoo/releases/download/v0.1/downloads.tar.gz"
            model_archive_path = os.path.join(download_dir, 'downloads.tar.gz')
            if not os.path.isfile(model_archive_path):
                print("Downloading trained models...")
                http = urllib3.PoolManager()
                response = http.request('GET', model_archive_url)

                with open(model_archive_path, 'wb') as f:
                    f.write(response.data)
                print("Download completed.")

                # unzip tar.gz file
                with tarfile.open(model_archive_path, 'r:gz') as tar:
                    tar.extractall(path=download_dir)
                print("Extraction completed.")

        model_weights_fpath = "MNIST_%s.keras_weights.h5" % model_name
        model_weights_fpath = os.path.join(trained_model_dir, model_weights_fpath)

        # self.maybe_download_model()
        if model_name in ["cleverhans", 'cleverhans_adv_trained']:
            model = cleverhans_mnist_model(logits=logits, input_range_type=input_range_type, pre_filter=pre_filter)
        elif model_name in ['carlini']:
            model = carlini_mnist_model(logits=logits, input_range_type=input_range_type, pre_filter=pre_filter)
        elif model_name in ['pgdtrained', 'pgdbase']:
            model = pgdtrained_mnist_model(logits=logits, input_range_type=input_range_type, pre_filter=pre_filter)
        print("\n===Defined TensorFlow model graph.")
        model.load_weights(model_weights_fpath)
        print("---Loaded MNIST-%s model.\n" % model_name)
        return model

    def print_predict(self, origin_test_tensor_array, adv_test_tensor_array, standard_result_tensor_array):
        # 假设 Y_test_adv_pred 是对抗样本的预测输出（概率向量），形状为 (batch_size, num_classes)
        # 使用 np.argmax 获取每个样本的预测类别索引
        origin_pred_class_indices = np.argmax(origin_test_tensor_array, axis=1)
        adv_pred_class_indices = np.argmax(adv_test_tensor_array, axis=1)
        standard_class_indices = np.argmax(standard_result_tensor_array, axis=1)

        # 映射类别索引到 分类名称
        origin_pred_class_names = [self.mnist_classes[idx] for idx in origin_pred_class_indices]
        adv_pred_class_names = [self.mnist_classes[idx] for idx in adv_pred_class_indices]
        standard_class_names = [self.mnist_classes[idx] for idx in standard_class_indices]

        error_idx = []

        # 打印每个样本的预测类别
        for i in range(len(origin_test_tensor_array)):
            if standard_class_names[i] != adv_pred_class_names[i]:
                print(f"Sample {i}: Standard class:{standard_class_names[i]}, Origin predicted:{origin_pred_class_names[i]}, adv predicted:{adv_pred_class_names[i]}")
                error_idx.append(i)
        return error_idx


if __name__ == '__main__':
    # from datasets.mnist import *
    dataset = MNISTDataset()
    X_test, Y_test = dataset.get_test_dataset()
    print(X_test.shape)
    print(Y_test.shape)

    model_name = 'cleverhans'
    model = dataset.load_model_by_name(model_name)

    model.compile(loss='categorical_crossentropy', optimizer='sgd', metrics=['acc'])
    _, accuracy = model.evaluate(X_test, Y_test, batch_size=128)
    print("\nTesting accuracy: %.4f" % accuracy)
