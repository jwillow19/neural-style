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
