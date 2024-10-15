import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten, Lambda
from tensorflow.keras.layers import Conv2D


def conv_2d(filters, kernel_shape, strides, padding, input_shape=None):
    """
    Defines the right convolutional layer according to the
    version of Keras that is installed.
    :param filters: (required integer) the dimensionality of the output
                    space (i.e. the number output of filters in the
                    convolution)
    :param kernel_shape: (required tuple or list of 2 integers) specifies
                         the kernel shape of the convolution
    :param strides: (required tuple or list of 2 integers) specifies
                         the strides of the convolution along the width and
                         height.
    :param padding: (required string) can be either 'valid' (no padding around
                    input or feature map) or 'same' (pad to ensure that the
                    output feature map size is identical to the layer input)
    :param input_shape: (optional) give input shape if this is the first
                        layer of the model
    :return: the Keras layer
    """
    if input_shape is not None:
        return Conv2D(
            filters=filters,
            kernel_size=kernel_shape,
            strides=strides,
            padding=padding,
            input_shape=input_shape,
        )
    else:
        return Conv2D(
            filters=filters, kernel_size=kernel_shape, strides=strides, padding=padding
        )

def cleverhans_mnist_model(logits=False, input_range_type=1, pre_filter=lambda x:x):
    input_shape = (28, 28, 1)
    nb_filters = 64
    nb_classes = 10
    return cleverhans_model(input_shape, nb_filters, nb_classes, logits, input_range_type, pre_filter)


def cleverhans_cifar10_model(logits=False, input_range_type=1, pre_filter=lambda x:x):
    input_shape = (32, 32, 3)
    nb_filters = 64
    nb_classes = 10
    return cleverhans_model(input_shape, nb_filters, nb_classes, logits, input_range_type, pre_filter)


def cleverhans_model(input_shape, nb_filters, nb_classes, logits, input_range_type, pre_filter):
    """
    Defines a CNN model using Keras sequential model
    :params logits: return logits(input of softmax layer) if True; return softmax output otherwise.
    :params input_range_type: the expected input range, {1: [0,1], 2:[-0.5, 0.5], 3:[-1, 1]...}
    :return:
    """
    model = Sequential()

    if input_range_type == 1:
        # The input data range is [0, 1]. 
        # Don't need to do scaling for cleverhans models, as it is the input range by default.
        scaler = lambda x: x
    elif input_range_type == 2:
        # The input data range is [-0.5, 0.5]. Convert to [0,1] by adding 0.5 element-wise.
        scaler = lambda x: x+0.5
    elif input_range_type == 3:
        # The input data range is [-1, 1]. Convert to [0,1] by x/2+0.5.
        scaler = lambda x: x/2+0.5

    layers = [Lambda(scaler, input_shape=input_shape)]
    layers += [Lambda(pre_filter, output_shape=input_shape)]

    layers += [Dropout(0.2),
              conv_2d(nb_filters, (8, 8), (2, 2), "same"),
              Activation('relu'),
              conv_2d((nb_filters * 2), (6, 6), (2, 2), "valid"),
              Activation('relu'),
              conv_2d((nb_filters * 2), (5, 5), (1, 1), "valid"),
              Activation('relu'),
              Dropout(0.5),
              Flatten(),
              Dense(nb_classes)]

    for layer in layers:
        model.add(layer)

    if not logits:
        model.add(Activation('softmax'))

    return model