import tensorflow as tf
import numpy as np
import tensorflow.keras.backend as K
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Conv2D, ZeroPadding2D, Add, Input, LeakyReLU, \
                                    UpSampling2D, Concatenate, Lambda
from tensorflow.keras.regularizers import l2
from tensorflow.keras.losses import binary_crossentropy, sparse_categorical_crossentropy
from batch_norm import BatchNormalization

yolo_anchors = np.array([(10, 13), (16, 30), (33, 23), (30, 61), (62, 45),
                         (59, 119), (116, 90), (156, 198), (373, 326)], np.float32) / 416

yolo_anchor_masks = np.array([[6, 7, 8], [3, 4, 5], [0, 1, 2]])

# Same conv layer as Darknet
def Conv(x, filters, size, strides=1, batch_norm=True):
    if strides == 1:
        padding = 'same'
    else:
        x = ZeroPadding2D(((1, 0), (1, 0)))(x)  # top left half-padding
        padding = 'valid'
        
    x = Conv2D(filters=filters, kernel_size=size, strides=strides, padding=padding,
               use_bias=not batch_norm, kernel_regularizer=l2(0.0005))(x)
    if batch_norm:
        x = BatchNormalization()(x)
        x = LeakyReLU(alpha=0.1)(x)
    return x
    

def ConvRes(x, filters):
    prev = x
    x = Conv(x, filters=filters // 2, size=1)
    x = Conv(x, filters=filters, size=3)  
    x = Add()([prev, x])
    
    return x

def ConvResBlock(x, filters, num_blocks):
    x = Conv(x, filters, 3, strides=2)
    
    for i in range(num_blocks):
        x = ConvRes(x, filters)
    
    return x

def MNET(name=None):
    x = inputs = Input([416, 416, 3])
    
    x = Conv(x, 32, 3)
    
    x = ConvResBlock(x, 64, 1)
    x = ConvResBlock(x, 128, 2)
    x = x_1 = ConvResBlock(x, 256, 4)
    x = x_2 = ConvResBlock(x, 512, 4)
    x = ConvResBlock(x, 1024, 2)
    
    return tf.keras.Model(inputs, (x_1, x_2, x), name=name)






def MNETConv(filters, name=None):
    def mnet_conv(x_in):
        if isinstance(x_in, tuple):
            inputs = Input(x_in[0].shape[1:]), Input(x_in[1].shape[1:])
            x, x_skip = inputs

            # concat with skip connection
            x = Conv(x, filters, 1)
            x = UpSampling2D(2)(x)
            x = Concatenate()([x, x_skip])
        else:
            x = inputs = Input(x_in.shape[1:])

#       Same feature detector as YoloV3
        x = Conv(x, filters, 1)
        x = Conv(x, filters * 2, 3)
        x = Conv(x, filters, 1)
        x = Conv(x, filters * 2, 3)
        x = Conv(x, filters, 1)
        return Model(inputs, x, name=name)(x_in)
    return mnet_conv

# Createst the (size/32, size/32, anchors_size, classes_5) encoding (stolen from yolov3)
def MNETOutput(filters, anchors, classes, name=None):
#   Final part the of the yolo feature detector
    def mnet_output(x_in):
        x = inputs = Input(x_in.shape[1:])
        x = Conv(x, filters * 2, 3)
        x = Conv(x, anchors * (classes + 5), 1, batch_norm=False)
        x = Lambda(lambda x: tf.reshape(x, (-1, tf.shape(x)[1], tf.shape(x)[2], anchors, classes + 5)))(x)
        return tf.keras.Model(inputs, x, name=name)(x_in)
    return mnet_output



