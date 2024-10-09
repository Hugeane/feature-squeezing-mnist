from keras.utils import np_utils

import sys, os
import urllib3
import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

class MNISTDataset:
    def __init__(self):
        self.dataset_name = "MNIST"
        self.image_size = 28
        self.num_channels = 1
        self.num_classes = 10

    # load MNIST dataset
    def load_mnist_dataset(self):
        dataset_mnist_dir = 'datasets/mnist'
        if not os.path.exists(dataset_mnist_dir):
            os.makedirs(dataset_mnist_dir)

            # 下载 MNIST 数据集
            url = 'https://s3.amazonaws.com/img-datasets/mnist.npz'
            filepath = os.path.join(dataset_mnist_dir, 'mnist.npz')

            print("Downloading MNIST dataset...")
            http = urllib3.PoolManager()
            response = http.request('GET', url)

            with open(filepath, 'wb') as f:
                f.write(response.data)

            print("Download completed.")

        # load on RAM
        with np.load(os.path.join(dataset_mnist_dir, 'mnist.npz'), allow_pickle=True) as f:
            x_train, y_train = f['x_train'], f['y_train']
            x_test, y_test = f['x_test'], f['y_test']

        return (x_train, y_train), (x_test, y_test)

    def get_test_dataset(self):
        (X_train, y_train), (X_test, y_test) = self.load_mnist_dataset()
        X_test = X_test.reshape(X_test.shape[0], self.image_size, self.image_size, self.num_channels)
        X_test = X_test.astype('float32')
        X_test /= 255
        Y_test = np_utils.to_categorical(y_test, self.num_classes)
        del X_train, y_train
        return X_test, Y_test

    def get_val_dataset(self):
        (X_train, y_train), (X_test, y_test) = self.load_mnist_dataset()
        val_size = 5000
        X_val = X_train[:val_size]
        X_val = X_val.reshape(X_val.shape[0], self.image_size, self.image_size, self.num_channels)
        X_val = X_val.astype('float32') / 255
        y_val = y_train[:val_size]
        Y_val = np_utils.to_categorical(y_val, self.num_classes)
        del X_train, y_train, X_test, y_test
        return X_val, Y_val



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
