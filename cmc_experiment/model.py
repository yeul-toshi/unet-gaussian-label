import numpy as np
import tensorflow as tf
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
    conv1 = conv_bn(64, inputs, act='Mish')
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
    conv2 = conv_bn(128, pool1, act='Mish')
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
    conv3 = conv_bn(256, pool2, act='Mish')
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)
    
    conv6 = conv_bn(512, pool3, act='Mish')

    up7 = Conv2D(256, 2, activation = 'Mish', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2, 2))(conv6))
    merge7 = concatenate([conv3, up7], axis = 3)
    conv7 = conv_bn(256, merge7, act='Mish')
    up8 = Conv2D(128, 2, activation = 'Mish', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2, 2))(conv7))
    merge8 = concatenate([conv2, up8], axis = 3)
    conv8 = conv_bn(128, merge8, act='Mish')
    up9 = Conv2D(64, 2, activation = 'Mish', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2, 2))(conv8))
    merge9 = concatenate([conv1, up9], axis = 3)
    conv9 = conv_bn(64, merge9, act='Mish')
    conv10 = Conv2D(1, 1, activation = 'sigmoid')(conv9)

    model = Model(inputs = inputs, outputs = conv10)

    return model

def unet(input_size = (256, 256, 3)):
    inputs = Input(input_size)
    conv1 = conv_bn(64, inputs, act='Mish')
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
    conv2 = conv_bn(128, pool1, act='Mish')
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
    conv3 = conv_bn(256, pool2, act='Mish')
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)
    
    conv6 = conv_bn(512, pool3, act='Mish')

    up7 = Conv2D(256, 2, activation = 'Mish', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2, 2))(conv6))
    merge7 = concatenate([conv3, up7], axis = 3)
    conv7 = conv_bn(256, merge7, act='Mish')
    up8 = Conv2D(128, 2, activation = 'Mish', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2, 2))(conv7))
    merge8 = concatenate([conv2, up8], axis = 3)
    conv8 = conv_bn(128, merge8, act='Mish')
    up9 = Conv2D(64, 2, activation = 'Mish', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2, 2))(conv8))
    merge9 = concatenate([conv1, up9], axis = 3)
    conv9 = conv_bn(64, merge9, act='Mish')
    conv10 = Conv2D(3, 1, activation = 'softmax')(conv9)

    model = Model(inputs = inputs, outputs = conv10)

    return model