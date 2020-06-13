import functools
import time
from PIL import Image
import numpy as np
import cv2

# import tensorflow_hub as hub
import tensorflow as tf

import IPython.display as display


def tensor_to_image(tensor):
    '''
    Return a tensor as an image
    '''
    tensor = tensor*255
    tensor = np.array(tensor, dtype=np.uint8)
    if np.ndim(tensor) > 3:
        assert tensor.shape[0] == 1
        tensor = tensor[0]
    return PIL.Image.fromarray(tensor)


def load_img(img_byte):
    """
    Loads an image and limit dimension size to 512px
    Scales and convert to new TensorShape
    """
    max_dim = 512

    # Keras preprocessing to convert img bytes to array, then convert to tf.Tensor
    image_array = tf.keras.preprocessing.image.img_to_array(
        img_byte)

    img = tf.image.convert_image_dtype(
        image_array/255, tf.float32)

    # Find scaling factor and cast new shape
    shape = tf.cast(tf.shape(img)[:-1], tf.float32)
    long_dim = max(shape)
    scale = max_dim / long_dim

    # Resize image and cast shape (type: TensorShape)
    new_shape = tf.cast(shape * scale, tf.int32)

    img = tf.image.resize(img, new_shape)
    img = img[tf.newaxis, :]

    return img
