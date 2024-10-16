from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import numpy as np
import sys
import warnings

from . import utils


def data_mnist(datadir='/tmp/'):
    """
    Load and preprocess MNIST dataset
    :return:
    """

    if 'tensorflow' in sys.modules:

        import utils.tf_mnist_input_data as input_data
        mnist = input_data.read_data_sets(datadir, one_hot=True, reshape=False)
        X_train = np.vstack((mnist.train.images, mnist.validation.images))
        Y_train = np.vstack((mnist.train.labels, mnist.validation.labels))
        X_test = mnist.test.images
        Y_test = mnist.test.labels
    else:
        warnings.warn("cleverhans support for Theano is deprecated and "
                      "will be dropped on 2017-11-08.")
        import tensorflow.keras as keras
        import tensorflow.keras.utils as k_utils
        from tensorflow.keras.datasets import mnist

        # These values are specific to MNIST
        img_rows = 28
        img_cols = 28
        nb_classes = 10

        # the data, shuffled and split between train and test sets
        (X_train, y_train), (X_test, y_test) = mnist.load_data()

        # tf2更新
        # th: Theano 通道在前      th -> channels_first   [样本数量, 通道数, 行, 列]
        # tf: TensorFlow 通道在后  tf -> channels_last    [样本数量, 行, 列, 通道数]
        # if keras.backend.image_dim_ordering() == 'th'
        if keras.backend.image_data_format() == 'channels_first':
            X_train = X_train.reshape(X_train.shape[0], 1, img_rows, img_cols)
            X_test = X_test.reshape(X_test.shape[0], 1, img_rows, img_cols)

        X_train = X_train.astype('float32')
        X_test = X_test.astype('float32')
        X_train /= 255
        X_test /= 255

        # convert class vectors to binary class matrices
        Y_train = k_utils.to_categorical(y_train, nb_classes)
        Y_test = k_utils.to_categorical(y_test, nb_classes)

    print('X_train shape:', X_train.shape)
    print(X_train.shape[0], 'train samples')
    print(X_test.shape[0], 'test samples')

    return X_train, Y_train, X_test, Y_test


def model_mnist(logits=False, input_ph=None, img_rows=28, img_cols=28,
                nb_filters=64, nb_classes=10):
    warnings.warn("`utils_mnist.model_mnist` is deprecated. Switch to"
                  "`utils.cnn_model`. `utils_mnist.model_mnist` will "
                  "be removed after 2017-08-17.")
    return utils.cnn_model(logits=logits, input_ph=input_ph,
                           img_rows=img_rows, img_cols=img_cols,
                           nb_filters=nb_filters, nb_classes=nb_classes)
