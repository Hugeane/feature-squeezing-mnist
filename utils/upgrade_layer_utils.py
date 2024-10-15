from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

from tensorflow.python.keras import backend as K


# tensorflow 2.3后移除的两个函数
def convert_kernel(kernel):
    """Converts a Numpy kernel matrix from Theano format to TensorFlow format.

    Also works reciprocally, since the transformation is its own inverse.

    This is used for converting legacy Theano-saved model files.

    Arguments:
        kernel: Numpy array (3D, 4D or 5D).

    Returns:
        The converted kernel.

    Raises:
        ValueError: in case of invalid kernel shape or invalid data_format.
    """
    kernel = np.asarray(kernel)
    if not 3 <= kernel.ndim <= 5:
        raise ValueError('Invalid kernel shape:', kernel.shape)
    slices = [slice(None, None, -1) for _ in range(kernel.ndim)]
    no_flip = (slice(None, None), slice(None, None))
    slices[-2:] = no_flip
    return np.copy(kernel[slices])


def convert_all_kernels_in_model(model):
    """Converts all convolution kernels in a model from Theano to TensorFlow.

    Also works from TensorFlow to Theano.

    Arguments:
        model: target model for the conversion.
    """
    # Note: SeparableConvolution not included
    # since only supported by TF.
    conv_classes = {
        'Conv1D',
        'Conv2D',
        'Conv3D',
        'Conv2DTranspose',
    }
    to_assign = []
    for layer in model.layers:
        if layer.__class__.__name__ in conv_classes:
            original_kernel = K.get_value(layer.kernel)
            converted_kernel = convert_kernel(original_kernel)
            to_assign.append((layer.kernel, converted_kernel))
    K.batch_set_value(to_assign)
