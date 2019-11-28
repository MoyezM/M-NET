import tensorflow as tf
import numpy as np
import tensorflow.keras.backend as K
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Conv2D, ZeroPadding2D, Add, Input, LeakyReLU, \
                                    UpSampling2D, Concatenate, Lambda
from tensorflow.keras.regularizers import l2
from tensorflow.keras.losses import binary_crossentropy, sparse_categorical_crossentropy
from batch_norm import BatchNormalization

mnet_anchors = np.array([(10, 13), (16, 30), (33, 23), (30, 61), (62, 45),
                         (59, 119), (116, 90), (156, 198), (373, 326)], np.float32) / 416

mnet_anchor_masks = np.array([[6, 7, 8], [3, 4, 5], [0, 1, 2]])

iou_threshold = 0.5

score_threshold = 0.5

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
def output(filters, anchors, classes, name=None):
#   Final part the of the yolo feature detector
    def mnet_output(x_in):
        x = inputs = Input(x_in.shape[1:])
        x = Conv(x, filters * 2, 3)
        x = Conv(x, anchors * (classes + 5), 1, batch_norm=False)
        x = Lambda(lambda x: tf.reshape(x, (-1, tf.shape(x)[1], tf.shape(x)[2], anchors, classes + 5)))(x)
        return tf.keras.Model(inputs, x, name=name)(x_in)
    return mnet_output

# Idea taken from the YoloV3 paper with sigmoids during BB 
def boxes(pred, anchors, classes):
    grid_size = tf.shape(pred)[1]
    
    box_xy, box_wh, objectness, class_probs = tf.split(pred, (2, 2, 1, classes))
    
    box_xy = tf.sigmoid(box_xy)
    objectness = tf.sigmoid(objectness)
    class_probs = tf.sigmoid(class_probs)
    pred_box = tf.concat((box_xy, box_wh), axis=-1)
    
#   Creates the grid of size (grid_size x grid_size)
    grid = tf.meshgrid(tf.range(grid_size), tf.range(grid_size))
    grid = tf.expand_dims(tf.stack(grid, axis=-1), axis=2)
    
    box_xy = (box_xy + tf.cast(grid, tf.float32)) / tf.cast(grid_sizem, tf.float32)
    box_wh = tf.exp(box_wh) * anchors
    
    box_x1y1 = box_xy - box_wh / 2
    box_x2y2 = box_xy + box_wh / 2
    bbox = tf.concat([box_x1y1, box_x2y2], axis=-1)

    return bbox, objectness, class_probs, pred_box
    
    

def nms(outputs, anchors, masks, classes):
#   Boxes, Conf, Type
    b, c, t = [], [], []
    
    for o in outputs:
        b.append(tf.reshape(o[0], (tf.shape(o[0])[0], -1, tf.shape(o[0])[-1])))
        c.append(tf.reshape(o[1], (tf.shape(o[1])[0], -1, tf.shape(o[1])[-1])))
        t.append(tf.reshape(o[2], (tf.shape(o[2])[0], -1, tf.shape(o[2])[-1])))
        
    bbox = tf.concat(b, axis=1)
    confidence = tf.concat(c, axis=1)
    class_probs = tf.concat(t, axis=1)
    
    scores = confidence * class_probs
    
    boxes, scores, classes, valid_detections = tf.image.combined_non_max_suppression(
        boxes=tf.reshape(bbox, (tf.shape(bbox)[0], -1, 1, 4)),
        scores=tf.reshape(scores, (tf.shape(scores)[0], -1, tf.shape(scores)[-1])),
        max_output_size_per_class=100,
        max_total_size=100,
        iou_threshold=iou_threshold,
        score_threshold=score_threshold
    )
    
    return boxes, scores, classes, valid_detections
    

def MNET_complete(size=None, channels=3, anchors=mnet_anchors, masks=mnet_anchor_masks, classes=80, training=False):
    x = inputs = Input([size, size, channels])
    
    x_1, x_2, x = MNET(name="MNET")(x)
    
    output_0 = output(512, len(masks[0]), classes, name='mnet_output_0')(x)

    x = MNETConv(256, name='mnet_conv_1')((x, x_2))
    output_1 = output(256, len(masks[1]), classes, name='mnet_output_1')(x)

    x = MNETConv(128, name='mnet_conv_2')((x, x_1))
    output_2 = output(128, len(masks[2]), classes, name='mnet_output_2')(x)
    
    if training:
        return Model(inputs, (output_0, output_1, output_2), name='mnet')
    
    boxes_0 = Lambda(lambda x: boxes(x, anchors[masks[0]], classes), name='mnet_boxes_0')(output_0)
    boxes_1 = Lambda(lambda x: boxes(x, anchors[masks[1]], classes), name='mnet_boxes_1')(output_1)
    boxes_2 = Lambda(lambda x: boxes(x, anchors[masks[1]], classes), name='mnet_boxes_2')(output_2)

    outputs = Lambda(lambda x: nms(x, anchors, masks, classes),
                     name='mnet_nms')((boxes_0[:3], boxes_1[:3], boxes_2[:3]))

    return Model(inputs, outputs, name='mnet')

# Completely stolen from Yolo paper and TF2 implementation
def Loss(anchors, classes=80, ignore_thresh=0.5):
    def loss(y_true, y_pred):
        # 1. transform all pred outputs
        # y_pred: (batch_size, grid, grid, anchors, (x, y, w, h, obj, ...cls))
        pred_box, pred_obj, pred_class, pred_xywh = boxes(
            y_pred, anchors, classes)
        pred_xy = pred_xywh[..., 0:2]
        pred_wh = pred_xywh[..., 2:4]

        # 2. transform all true outputs
        # y_true: (batch_size, grid, grid, anchors, (x1, y1, x2, y2, obj, cls))
        true_box, true_obj, true_class_idx = tf.split(y_true, (4, 1, 1), axis=-1)
        true_xy = (true_box[..., 0:2] + true_box[..., 2:4]) / 2
        true_wh = true_box[..., 2:4] - true_box[..., 0:2]

        # give higher weights to small boxes
        box_loss_scale = 2 - true_wh[..., 0] * true_wh[..., 1]

        # 3. inverting the pred box equations
        grid_size = tf.shape(y_true)[1]
        grid = tf.meshgrid(tf.range(grid_size), tf.range(grid_size))
        grid = tf.expand_dims(tf.stack(grid, axis=-1), axis=2)
        true_xy = true_xy * tf.cast(grid_size, tf.float32) - \
            tf.cast(grid, tf.float32)
        true_wh = tf.math.log(true_wh / anchors)
        true_wh = tf.where(tf.math.is_inf(true_wh), tf.zeros_like(true_wh), true_wh)

        # 4. calculate all masks
        obj_mask = tf.squeeze(true_obj, -1)
        # ignore false positive when iou is over threshold
        true_box_flat = tf.boolean_mask(true_box, tf.cast(obj_mask, tf.bool))
        best_iou = tf.reduce_max(broadcast_iou(pred_box, true_box_flat), axis=-1)
        ignore_mask = tf.cast(best_iou < ignore_thresh, tf.float32)

        # 5. calculate all losses
        xy_loss = obj_mask * box_loss_scale * tf.reduce_sum(tf.square(true_xy - pred_xy), axis=-1)
        wh_loss = obj_mask * box_loss_scale * tf.reduce_sum(tf.square(true_wh - pred_wh), axis=-1)
        obj_loss = binary_crossentropy(true_obj, pred_obj)
        obj_loss = obj_mask * obj_loss + (1 - obj_mask) * ignore_mask * obj_loss
        # TODO: use binary_crossentropy instead
        
        class_loss = obj_mask * sparse_categorical_crossentropy(true_class_idx, pred_class)

        # 6. sum over (batch, gridx, gridy, anchors) => (batch, 1)
        xy_loss = tf.reduce_sum(xy_loss, axis=(1, 2, 3))
        wh_loss = tf.reduce_sum(wh_loss, axis=(1, 2, 3))
        obj_loss = tf.reduce_sum(obj_loss, axis=(1, 2, 3))
        class_loss = tf.reduce_sum(class_loss, axis=(1, 2, 3))

        return xy_loss + wh_loss + obj_loss + class_loss
    return loss




