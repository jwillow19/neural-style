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

st.write("This is a small project to showcase my understanding on the topic of Neural Style Transfer (NST). NST builds on the idea that the deep convolution neural network (CNN) learns the 'style' of an image and transfer said style to a content image. Mathematically this is done by minimizing the linear combination of the content and loss functions. Upload a content image that you would like to see transformed.")

upload_content_file = st.file_uploader(
    "Choose a Content Image", type=["png", "jpg", "jpeg"], key='content_file')

if upload_content_file:
    content_image_bytes = Image.open(upload_content_file)
    content_image = load_img(content_image_bytes)
    st.image(
        content_image_bytes,
        caption="Content Image shape: {}".format(content_image.shape),
        use_column_width=True
    )

st.write("In the context of neural style transfer, 'style' is defined as the correlation between activations across channels within the same layer. To fully capture the 'style' of an image, researchers suggest sampling the outputs of several intermediate network layers and calculate the correlation between channels in each layer. This allows the style-loss function to capture more information on the 'abstractedness' of the  image. Which in turn poses a better minimization problem. Upload a content and style image to see for yourself!")

upload_style_file = st.file_uploader(
    "Choose a Style Image", type=["png", "jpg", "jpeg"], key='style_file')

if upload_style_file:
    style_image_bytes = Image.open(upload_style_file)
    # Convert to tf.Tensor with new shape
    style_image = load_img(style_image_bytes)
    # st.image(input: img_btyes)
    st.image(
        style_image_bytes,
        caption="Style Image shape: {}".format(content_image.shape),
        use_column_width=True
    )

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
    st.write('Generating an image to minimize...')
    generated_image = tf.Variable(content_image, dtype='float32')

    st.write("Select hyperparameters on the sidebar and then press 'Finalize' once ready to initiate style transfer")

    add_selectbox_lr = st.sidebar.slider(
        'Adam Optimizer Learning Rate - (Default 0.02)',
        0.01, 0.1, 0.02
    )

    style_weight = st.sidebar.slider(
        'Style layers weight - (Default 0.01)',
        1e-2, 1.0, 0.01
    )

    content_weight = st.sidebar.slider(
        'Content layer weight - (Default 10000)',
        1000, 10000, 10000
    )

    epochs = st.sidebar.radio(
        'Epochs - Number of training passes',
        (10, 20, 30)
    )

    finalize = st.sidebar.button(
        'Finalize')

    if finalize:
        # Optimizer, style and content weights
        # opt = tf.optimizers.Adam(
        #     learning_rate=add_selectbox_lr, beta_1=0.99, epsilon=1e-1)

        opt = adam_optimizer(add_selectbox_lr, beta_1=0.99, epsilon=1e-1)
        # style_weight = 1e-2
        # content_weight = 1e4
        # epochs = 1
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

        st.write(
            "Give it a few seconds, the network is doing the heavylifting and minimizing the loss function...")

        progress_bar = st.progress(0)

        for n in range(epochs):
            for m in range(steps_per_epoch):
                step += 1
                train_step(generated_image)
                print(".", end='')
            print("Train step: {}".format(step))

            progress_bar.progress(n/epochs)

        generated_image = tensor_to_image(generated_image)

        st.success("Style transfer complete!")
        st.balloons()

        st.image(
            generated_image,
            caption="Generated image",
            use_column_width=True)
