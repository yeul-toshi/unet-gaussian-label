info = { 'title': 'TensorFlow',
        'requirements': ['numpy>=1.18.2', 'opencv-python>=4.2.0.34', 'tensorflow-cpu>=2.1.0', 'chardet==3.0.4', 'pillow>=7.1.1']}

import numpy as np
import tensorflow as tf
import quantification as qc
import cv2 as cv
from PIL import Image
import math

from tensorflow.keras.models import *
from tensorflow.keras.layers import *
from tensorflow.keras.activations import *
from tensorflow.keras.utils import get_custom_objects

class Mish(Activation):
    def __init__(self, activation, **kwargs):
        super(Mish, self).__init__(activation, **kwargs)
        self.__name__ = 'Mish'

def mish(inputs):
    return inputs * tf.math.tanh(tf.math.softplus(inputs))

get_custom_objects().update({'Mish': Mish(mish)})

def conv_bn(filters, x, act='relu'):
    conv = Conv2D(filters, 3, padding = 'same', kernel_initializer = 'he_normal')(x)
    conv = BatchNormalization()(conv)
    conv = Activation(act)(conv)
    conv = Conv2D(filters, 3, padding = 'same', kernel_initializer = 'he_normal')(conv)
    conv = BatchNormalization()(conv)
    conv = Activation(act)(conv)
    conv = Conv2D(filters, 3, padding = 'same', kernel_initializer = 'he_normal')(conv)
    conv = BatchNormalization()(conv)
    conv = Activation(act)(conv)
    return conv

def unet_bin(input_size = (256, 256, 3)):
    inputs = Input(input_size)
    conv1 = conv_bn(32, inputs, act='Mish')
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
    conv2 = conv_bn(64, pool1, act='Mish')
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
    conv3 = conv_bn(128, pool2, act='Mish')
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)
    
    conv6 = conv_bn(256, pool3, act='Mish')

    up7 = Conv2D(128, 2, activation = 'Mish', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2, 2))(conv6))
    merge7 = concatenate([conv3, up7], axis = 3)
    conv7 = conv_bn(128, merge7, act='Mish')
    up8 = Conv2D(64, 2, activation = 'Mish', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2, 2))(conv7))
    merge8 = concatenate([conv2, up8], axis = 3)
    conv8 = conv_bn(64, merge8, act='Mish')
    up9 = Conv2D(32, 2, activation = 'Mish', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2, 2))(conv8))
    merge9 = concatenate([conv1, up9], axis = 3)
    conv9 = conv_bn(32, merge9, act='Mish')
    conv10 = Conv2D(1, 1, activation = 'sigmoid')(conv9)

    model = Model(inputs = inputs, outputs = conv10)

    return model

loaded = False

def initialize(inp: qc.InitializeInput, out: qc.InitializeOutput):
    out.clusters.add('tissue', '#00FF00')
    out.processing.zoom = 1.0
    out.processing.tile_size = 608
    #out.ui.add_trackbar('value_tbr', 'Detection threshold (0~255):', 0, 255, 128, 10)
    out.ui.add_textbox('value_txb', 'Detection threshold (0~255):', "60")

    out.ui.separated_into_rows = True

    # Check if slide is brightfield
    if inp.slide.info.is_fluorescent:
        raise Exception('This script only works for brightfield slides.')
    
    global loaded, model, model_shape, model_input_size, infer
    if loaded == False:
        # Load tensorflow model
        model = unet_bin((608, 608, 3))
        model.load_weights(inp.environment.app_path + '/scriptquant/unet_bin.h5')
        # Get model's input shape
        model_shape = (1, *model.input_shape[1:])
        model_input_size = model.input_shape[2:0:-1]
        print('model input shape: ', model_input_size)
        loaded = True

def predict_for_block(block: np.ndarray):
    #res = np.asarray([])
    res = block.reshape(model_shape)
    # Execute prediction
    unet_pred = model.predict_on_batch(res)
    # Upscale image
    unet_pred = np.squeeze(unet_pred)
    unet_pred = np.array(unet_pred*255, dtype=np.uint8)

    return unet_pred

def process_tile(inp: qc.ProcessInput, out: qc.ProcessOutput):
    
    # Original source size
    src_size_orig = inp.image.shape[:2]
    # Get inverted size of source image
    src_size = inp.image.shape[:2][::-1]
    # Convert from BGR to RGB
    im = inp.image[:, :, ::-1]/255

    # Get number of block horizontally and vertically
    cols = math.ceil(src_size[0]/model_input_size[0])
    rows = math.ceil(src_size[1]/model_input_size[1])

    # Pad image
    cols_to_add = cols*model_input_size[0] - src_size[0]
    rows_to_add = rows*model_input_size[1] - src_size[1]
    im = np.pad(im, ((0, rows_to_add), (0, cols_to_add), (0, 0)), mode='constant', constant_values=0)

    # Get non-inverted size
    src_size = im.shape[:2]
    
    # The block array will contain the predictions (masks) of the model on the blocks
    block_arr = np.zeros(src_size, dtype=np.float32)

    # Iterate over the image in blocks
    for row in range(rows):
        for col in range(cols):            
            # Get final indices of current block - horizontally and vertically
            block_height = model_input_size[1]
            block_width = model_input_size[0]
            h_end = row*block_height + block_height
            w_end = col*block_width + block_width

            # Extract block of current position from image
            block = im[row*block_height:h_end, col*block_width:w_end].copy()
            # Predict for current block
            res = predict_for_block(block)
            # Copy prediction values into block array
            block_arr[row*block_height : h_end, col*block_width : w_end] = res.copy()[0:block_height, 0:block_width]
           
    # Stretch out the image back in the 0..255 range, convert to uint8, reshape
    res = res.reshape(src_size)

    # Crop prediction mask to original size
    res = res[:src_size_orig[0], :src_size_orig[1]]
    th = int(int(inp.ui_values['value_txb']))
    if th > 255 or th < 0:
        print("Set detection threshold between 0~255.")
        exit
    _, img_th = cv.threshold(res, th, 255, cv.THRESH_BINARY)
    label = cv.connectedComponentsWithStats(img_th)
    n = label[0] - 1
    center_pre = np.delete(label[3], 0, 0)
    
    img_bbox = np.zeros(res.shape[:2], np.uint8)
    for x, y in center_pre:
        img_bbox = cv.rectangle(img_bbox, (int(x)-32, int(y)-32), (int(x)+32, int(y)+32), 255, 3)
    
    out.results.add_polygons_by_mask_image(img_bbox, 0)