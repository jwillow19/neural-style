import functools
import time
from PIL import Image
import numpy as np
import cv2
import streamlit as st
# import tensorflow_hub as hub
import tensorflow as tf

import IPython.display as display


def tensor_to_image(tensor):
    '''
    Return a tensor as an image
    '''
    # Scale pixel values to (0,255) and convert dtype to int
    tensor = tensor*255
    tensor = np.array(tensor, dtype=np.uint8)
    # Prune extra dimension
    if np.ndim(tensor) > 3:
        assert tensor.shape[0] == 1
        tensor = tensor[0]
    return Image.fromarray(tensor)


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


def vgg19():
    base_model = tf.keras.applications.VGG19(
        weights='imagenet',
        include_top=False,
        #         input_shape=(224, 224, 3),
        #         input_tensor=tf.keras.layers.Input(shape=(224, 224, 3)),
    )
    return base_model


def vgg_layers(layer_names):
    '''
    INPUT: a list of layer names
    OUTPUT: a model(functional object: input, output) that returns a list,
            specifically the output for each layer in layer_names
    function returns a vgg19 model (functional object) that return output of layer_name - model requires input,
    '''
    # grabs the pretrained model, remove classification layer, and freeze the weights
    vgg = vgg19()
    vgg.trainable = False

    # store each layer's output of interest to list
    outputs = [vgg.get_layer(name).output for name in layer_names]

    # model = Model(inputs, outputs)
    model = tf.keras.Model([vgg.input], outputs)
    return model


def gram_matrix(input_tensor):
    '''
    INPUT: input_tensor
    OUTPUT: the Gram matrix
    '''
    # computes the element-wise sum of input tensor
    result = tf.linalg.einsum('bijc,bijd->bcd', input_tensor, input_tensor)
    input_shape = tf.shape(input_tensor)
    num_locations = tf.cast(input_shape[1]*input_shape[2], tf.float32)
    return result/(num_locations)


# define a VGG-model class that handles style and content layers
class StyleContentModel(tf.keras.models.Model):
    '''
    A class VGG Model that handles style and content layers outputs
    '''

    def __init__(self, style_layers, content_layers):
        super(StyleContentModel, self).__init__()
        self.vgg = vgg_layers(style_layers + content_layers)
        self.style_layers = style_layers
        self.content_layers = content_layers
        self.num_style_layers = len(style_layers)
        self.vgg.trainable = False

    # call Class Method

    def call(self, inputs):
        '''
        INPUT: image 
        OUTPUT: dictionary of dictionaries with content&style layer:output
        '''
        inputs = inputs*255.0
        preprocessed_input = tf.keras.applications.vgg19.preprocess_input(
            inputs)

        # Get the layer outputs from vgg model
        outputs = self.vgg(preprocessed_input)

        # Split style and content outputs by number of layers for each
        style_outputs, content_outputs = (outputs[:self.num_style_layers],
                                          outputs[self.num_style_layers:])

        # Extra step for style - calculate gram matrix of each style output layer
        style_outputs = [gram_matrix(style_output)
                         for style_output in style_outputs]

        # Store content and style outputs in dictionary
        content_dict = {content_name: value
                        for content_name, value
                        in zip(self.content_layers, content_outputs)}

        style_dict = {style_name: value
                      for style_name, value
                      in zip(self.style_layers, style_outputs)}

        return {'content': content_dict, 'style': style_dict}


def clip_0_1(image):
    '''
    clips image so pixel values remains between (0,1)
    '''
    return tf.clip_by_value(image, clip_value_min=0.0, clip_value_max=1.0)


# Loss function: linear combination of style and content loss
@st.cache
def style_content_loss(outputs, style_target, content_target, style_weight=1e-2, content_weight=1e4, num_style_layers=5, num_content_layers=1):
    '''
    INPUT: output = model(image)
    '''
    style_output = outputs['style']
    content_output = outputs['content']

    # store style mean-square-error across layers then sum values in list to get
    # total style loss
    # tf.add_n() - add all input tensors element-wise
    # tf.reduce_mean() - 1/m * summation(...)
    style_loss = tf.add_n([tf.reduce_mean((style_output[name]-style_target[name])**2)
                           for name in style_output.keys()])
    style_loss *= style_weight / num_style_layers

    content_loss = tf.add_n([tf.reduce_mean((content_output[name]-content_target[name])**2)
                             for name in content_output.keys()])
    content_loss *= content_weight / num_content_layers

    loss = style_loss + content_loss
    return loss


def train_step(image):
    '''
    One training step (one iteration of gradient descend)
    '''
    with tf.GradientTape() as tape:
        outputs = extractor(image)
        loss = style_content_loss(outputs)
        # calculates gradients based on loss?
        grad = tape.gradient(loss, image)
        # apply gradients to optimization algorithm for iamge update
        opt.apply_gradients([(grad, image)])
        image.assign(clip_0_1(image))


@st.cache
def adam_optimizer(lr=1e-2, beta_1=0.99, epsilon=1e-1):
    opt = tf.optimizers.Adam(
        learning_rate=lr, beta_1=beta_1, epsilon=epsilon)
    return opt
