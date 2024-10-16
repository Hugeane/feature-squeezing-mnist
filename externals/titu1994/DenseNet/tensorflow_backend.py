import tensorflow as tf
from tensorflow.keras.backend import image_data_format
from utils.fchollet_tensorflow_backend import _preprocess_conv2d_input
from utils.fchollet_tensorflow_backend import _postprocess_conv2d_output

py_all = all


def depth_to_space(input, scale, data_format=None):
    """ Uses phase shift algorithm to convert channels/depth for spatial resolution """
    if data_format is None:
        data_format = image_data_format()
    data_format = data_format.lower()
    input = _preprocess_conv2d_input(input, data_format)
    out = tf.nn.depth_to_space(input, scale)
    out = _postprocess_conv2d_output(out, data_format)
    return out
