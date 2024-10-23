import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.carlini_models import carlini_cifar10_model
from models.cleverhans_models import cleverhans_cifar10_model
from models.densenet_models import densenet_cifar10_model, get_densenet_weights_path

from tensorflow.keras.datasets import cifar10
from tensorflow.python.keras.utils import np_utils
from tensorflow.python.keras import backend as K
from tensorflow.python.keras.datasets.cifar import load_batch
from tensorflow.python.keras.utils.data_utils import get_file
import numpy as np


class CIFAR10Dataset:
    def __init__(self):
        self.dataset_name = "CIFAR-10"
        self.image_size = 32
        self.num_channels = 3
        self.num_classes = 10
        self.cifar10_classes = {
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

    def load_data(self):
        """Loads [CIFAR10 dataset](https://www.cs.toronto.edu/~kriz/cifar.html).

        This is a dataset of 50,000 32x32 color training images and 10,000 test
        images, labeled over 10 categories. See more info at the
        [CIFAR homepage](https://www.cs.toronto.edu/~kriz/cifar.html).

        Returns:
            Tuple of Numpy arrays: `(x_train, y_train), (x_test, y_test)`.

            **x_train, x_test**: uint8 arrays of RGB image data with shape
              `(num_samples, 3, 32, 32)` if `tf.keras.backend.image_data_format()` is
              `'channels_first'`, or `(num_samples, 32, 32, 3)` if the data format
              is `'channels_last'`.

            **y_train, y_test**: uint8 arrays of category labels
              (integers in range 0-9) each with shape (num_samples, 1).
        """
        dirname = 'cifar-10-batches-py'
        origin = 'https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz'
        path = get_file(
            dirname,
            origin=origin,
            untar=True,
            file_hash=
            '6d958be074577803d12ecdefd02955f39262c83c16fe9348329d7fe0b5c001ce')

        num_train_samples = 50000

        x_train = np.empty((num_train_samples, 3, 32, 32), dtype='uint8')
        y_train = np.empty((num_train_samples,), dtype='uint8')

        for i in range(1, 6):
            fpath = os.path.join(path, 'data_batch_' + str(i))
            (x_train[(i - 1) * 10000:i * 10000, :, :, :],
             y_train[(i - 1) * 10000:i * 10000]) = load_batch(fpath)

        fpath = os.path.join(path, 'test_batch')
        x_test, y_test = load_batch(fpath)

        y_train = np.reshape(y_train, (len(y_train), 1))
        y_test = np.reshape(y_test, (len(y_test), 1))

        if K.image_data_format() == 'channels_last':
            x_train = x_train.transpose(0, 2, 3, 1)
            x_test = x_test.transpose(0, 2, 3, 1)

        x_test = x_test.astype(x_train.dtype)
        y_test = y_test.astype(y_train.dtype)

        return (x_train, y_train), (x_test, y_test)

    def get_test_dataset(self):
        (X_train, y_train), (X_test, y_test) = self.load_data()
        X_test = X_test.reshape(X_test.shape[0], self.image_size, self.image_size, self.num_channels)
        X_test = X_test.astype('float32')
        X_test /= 255
        Y_test = np_utils.to_categorical(y_test, self.num_classes)
        del X_train, y_train
        return X_test, Y_test

    def get_val_dataset(self):
        (X_train, y_train), (X_test, y_test) = self.load_data()
        val_size = 5000
        X_val = X_train[:val_size]
        X_val = X_val.reshape(X_val.shape[0], self.image_size, self.image_size, self.num_channels)
        X_val = X_val.astype('float32') / 255
        y_val = y_train[:val_size]
        Y_val = np_utils.to_categorical(y_val, self.num_classes)
        del X_train, y_train, X_test, y_test

        return X_val, Y_val

    def load_model_by_name(self, model_name, logits=False, input_range_type=1, pre_filter=lambda x: x):
        """
        :params logits: return logits(input of softmax layer) if True; return softmax output otherwise.
        :params input_range_type: {1: [0,1], 2:[-0.5, 0.5], 3:[-1, 1]...}
        """
        if model_name not in ["cleverhans", 'cleverhans_adv_trained', 'carlini', 'densenet']:
            raise NotImplementedError("Undefined model [%s] for %s." % (model_name, self.dataset_name))
        self.model_name = model_name

        model_weights_fpath = "%s_%s.keras_weights.h5" % (self.dataset_name, model_name)
        model_weights_fpath = os.path.join('downloads/trained_models', model_weights_fpath)

        if model_name in ["cleverhans", 'cleverhans_adv_trained']:
            model = cleverhans_cifar10_model(logits=logits, input_range_type=input_range_type, pre_filter=pre_filter)
        elif model_name == "carlini":
            model = carlini_cifar10_model(logits=logits, input_range_type=input_range_type, pre_filter=pre_filter)
        elif model_name == "densenet":
            model = densenet_cifar10_model(logits=logits, input_range_type=input_range_type, pre_filter=pre_filter)
            model_weights_fpath = get_densenet_weights_path(self.dataset_name)
        print("\n===Defined TensorFlow model graph.")
        model.load_weights(model_weights_fpath)
        print("---Loaded CIFAR-10-%s model.\n" % model_name)
        return model

    def print_predict(self, origin_test_tensor_array, adv_test_tensor_array, standard_result_tensor_array):
        # 假设 Y_test_adv_pred 是对抗样本的预测输出（概率向量），形状为 (batch_size, num_classes)
        # 使用 np.argmax 获取每个样本的预测类别索引
        origin_pred_class_indices = np.argmax(origin_test_tensor_array, axis=1)
        adv_pred_class_indices = np.argmax(adv_test_tensor_array, axis=1)
        standard_class_indices = np.argmax(standard_result_tensor_array, axis=1)

        # 映射类别索引到 CIFAR-10 分类名称
        origin_pred_class_names = [self.cifar10_classes[idx] for idx in origin_pred_class_indices]
        adv_pred_class_names = [self.cifar10_classes[idx] for idx in adv_pred_class_indices]
        standard_class_names = [self.cifar10_classes[idx] for idx in standard_class_indices]

        # 打印每个样本的预测类别
        for i in range(len(origin_test_tensor_array)):
            print(
                f"Sample {i}: Standard class:{standard_class_names[i]}, Origin predicted class - {origin_pred_class_names[i]}, adv predicted class - {adv_pred_class_names[i]}")


if __name__ == '__main__':
    dataset = CIFAR10Dataset()
    X_test, Y_test = dataset.get_test_dataset()
    print(X_test.shape)
    print(Y_test.shape)
