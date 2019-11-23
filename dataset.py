import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

image_feature_description  = {
    'image/width': tf.io.FixedLenFeature([], tf.int64),
    'image/height': tf.io.FixedLenFeature([], tf.int64),
    'image/encoded': tf.io.FixedLenFeature([], tf.string),
    'image/object/bbox/xmin': tf.io.VarLenFeature(tf.float32),
    'image/object/bbox/ymin': tf.io.VarLenFeature(tf.float32),
    'image/object/bbox/xmax': tf.io.VarLenFeature(tf.float32),
    'image/object/bbox/ymax': tf.io.VarLenFeature(tf.float32),
    'image/object/class/label': tf.io.VarLenFeature(tf.int64),
}

def _parse_image_function(example_proto):
    return tf.io.parse_single_example(example_proto, image_feature_description)

def print_img(img):
    imgplot = plt.imshow(img.numpy().astype(dtype='uint8'))

def parse_tfrecord(example_proto):
    x_train = tf.image.decode_jpeg(example_proto['image/encoded'], channels=3)
    x_train = tf.image.resize(x_train, (416, 416))

    labels = tf.cast(tf.sparse.to_dense(example_proto['image/object/class/label']), tf.float32)
    y_train = tf.stack([tf.sparse.to_dense(example_proto['image/object/bbox/xmin']),
                        tf.sparse.to_dense(example_proto['image/object/bbox/ymin']),
                        tf.sparse.to_dense(example_proto['image/object/bbox/xmax']),
                        tf.sparse.to_dense(example_proto['image/object/bbox/ymax']),
                        labels], axis=1)

    paddings = [[0, 100 - tf.shape(y_train)[0]], [0, 0]]
    y_train = tf.pad(y_train, paddings)
    train = {
        "y_train" : y_train,
        "x_train" : x_train
    }

    return train

def createDataset(tfrecord_path):
    raw_dataset = tf.data.TFRecordDataset(tfrecord_path)
    parsed_dataset = raw_dataset.map(_parse_image_function)
    dataset = parsed_dataset.map(parse_tfrecord)

    return dataset

