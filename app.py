import streamlit as st
import functools
import time
from PIL import Image
import numpy as np
import cv2

# import tensorflow_hub as hub
import tensorflow as tf

import IPython.display as display

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from utils import *


st.title('Neural TF-Style Transfer')

st.write("This is a small project to showcase my understanding on the topic of Neural Style Transfer (NST). NST builds on the idea that the deep convolution neural network (CNN) learns the 'style' of an image and transfer said style to a content image. Mathematically this is done by minimizing the linear combination of the content and loss functions. In the context of neural style transfer, 'style' is defined as the correlation between activations across channels within the same layer.")

st.write("There are many layers in a deep CNN. To fully capture the 'style' of an image, or the essence of the artist, researchers suggest sampling the outputs of several intermediate network layers. Doing so the style loss function will capture more information in terms of the 'abstractedness' of the style image. And this poses a better minimization problem.")

st.write('Upload a content and style image')


upload_content_file = st.file_uploader(
    "Choose a Content Image", type=["png", "jpg", "jpeg"], key='content_file')

upload_style_file = st.file_uploader(
    "Choose a Style Image", type=["png", "jpg", "jpeg"], key='style_file')

if upload_content_file and upload_style_file is not None:
    content_image_bytes = Image.open(upload_content_file)
    style_image_bytes = Image.open(upload_style_file)

    # Convert to tf.Tensor with new shape
    content_image = load_img(content_image_bytes)
    style_image = load_img(style_image_bytes)

    # st.image(input: img_btyes)
    st.image(
        content_image_bytes,
        caption="Content Image shape: {}".format(content_image.shape),
        use_column_width=True
    )
    st.image(
        style_image_bytes,
        caption="Style Image shape: {}".format(content_image.shape),
        use_column_width=True
    )

    st.write("Much of Neural-Style-Transfer builds on the idea that neural network learns the 'style' of the artist and transfer their style to a content image. And minimize content and style loss to the minimum. In the context of neural style transfer, 'style' is defined as the correlation between activations across channels within the same layer.")

    st.write(
        "'Correlatedness' between channels is done by calculating the Gram matrix")

    content_layers = ['block5_conv2']

    style_layers = ['block1_conv1',
                    'block2_conv1',
                    'block3_conv1',
                    'block4_conv1',
                    'block5_conv1']

    num_content_layers = len(content_layers)
    num_style_layers = len(style_layers)

    # Style-component - Create vgg model from selected layers
    style_representation = vgg_layers(style_layers)
    style_outputs = style_representation(style_image*255)

    # Create model class instance and use call method
    extractor = StyleContentModel(style_layers, content_layers)
    results = extractor(tf.constant(content_image))

    # Set Gram matrices of the StyleImage as target for minimization
    # (so each gradient descend algorithm can learn the "style")
    style_target = extractor(style_image)['style']
    # Set intermediate layer output of a VGG model's with ContentImage input as target
    # (so GD can generate image with abstracted features from ContentImage)
    content_target = extractor(content_image)['content']

    # Generate an image to optimize
    generated_image = tf.Variable(content_image, dtype='float32')

    # Optimizer, style and content weights
    opt = tf.optimizers.Adam(learning_rate=0.02, beta_1=0.99, epsilon=1e-1)
    style_weight = 1e-2
    content_weight = 1e4
    epochs = 1
    steps_per_epoch = 100
    step = 0

    def train_step(image):
        '''
        One training step (one iteration of gradient descend)
        '''
        with tf.GradientTape() as tape:
            outputs = extractor(image)
            loss = style_content_loss(
                outputs, style_target, content_target, style_weight, content_weight)
            # calculates gradients based on loss?
            grad = tape.gradient(loss, image)
            # apply gradients to optimization algorithm for iamge update
            opt.apply_gradients([(grad, image)])
            image.assign(clip_0_1(image))

    for n in range(epochs):
        for m in range(steps_per_epoch):
            step += 1
            train_step(generated_image)
            print(".", end='')
        print("Train step: {}".format(step))

    generated_image = tensor_to_image(generated_image)

    st.image(
        generated_image,
        caption="Generated image",
        use_column_width=True)
