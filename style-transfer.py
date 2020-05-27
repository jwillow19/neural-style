import functools
import time
import PIL.Image
import numpy as np

import tensorflow_hub as hub
import tensorflow as tf

import IPython.display as display

import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.rcParams['figure.figsize'] = (12, 12)
mpl.rcParams['axes.grid'] = False

# %matplotlib inline

# Load the pretrained model
# model = load_vgg_model('pretrained-model/imagenet-vgg-verydeep-19.mat')


def tensor_to_image(tensor):
    tensor = tensor*255
    tensor = np.array(tensor, dtype=np.uint8)
    if np.ndim(tensor) > 3:
        assert tensor.shape[0] == 1
        tensor = tensor[0]
    return PIL.Image.fromarray(tensor)


def load_img(path_to_img):
    """
    Loads an image and limit dimension size to 512px
    """
    max_dim = 512
    img = tf.io.read_file(path_to_img)
    img = tf.image.decode_image(img, channels=3)
    img = tf.image.convert_image_dtype(img, tf.float32)

    shape = tf.cast(tf.shape(img)[:-1], tf.float32)
    long_dim = max(shape)
    scale = max_dim / long_dim

    new_shape = tf.cast(shape * scale, tf.int32)

    img = tf.image.resize(img, new_shape)
    img = img[tf.newaxis, :]

    return img


def imshow(image, title=None):
    """
    function displays an image
    """
    if len(image.shape) > 3:
        image = tf.squeeze(image, axis=0)

    plt.imshow(image)
    plt.show()

    if title:
        plt.title(title)


style_image = load_img('images/claude-monet.jpg')
content_image = load_img('images/louvre.jpg')

plt.plot(1, 2, 1)
imshow(content_image, 'Content Image')

plt.plot(1, 2, 2)
imshow(style_image, 'Style Image')

# Using a preexisting Neural Style module from tensorflow hub to do style transfer
hub_module = hub.load(
    'https://tfhub.dev/google/magenta/arbitrary-image-stylization-v1-256/1')
stylized_image = hub_module(tf.constant(
    content_image), tf.constant(style_image))[0]
tensor_to_image(stylized_image).show()
