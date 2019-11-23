from absl import app
from absl import flags

import tensorflow as tf
import matplotlib as plt
import matplotlib.image as mpimg

FLAGS = flags.FLAGS

flags.DEFINE_string('train', 'False', 'Path the the tf record of the train set')


IMAGE_FEATURE_MAP = {
    'image/width': tf.io.FixedLenFeature([], tf.int64),
    'image/height': tf.io.FixedLenFeature([], tf.int64),
    'image/encoded': tf.io.FixedLenFeature([], tf.string),
    'image/object/bbox/xmin': tf.io.VarLenFeature(tf.float32),
    'image/object/bbox/ymin': tf.io.VarLenFeature(tf.float32),
    'image/object/bbox/xmax': tf.io.VarLenFeature(tf.float32),
    'image/object/bbox/ymax': tf.io.VarLenFeature(tf.float32),
    'image/object/class/label': tf.io.VarLenFeature(tf.int64),
}

def main(argv):
    dataset = tf.data.TFRecordDataset(FLAGS.train)
    print(dataset)

    dataset = dataset.map(_parse_function)
    print(dataset)
    
    # dataset = parse_tfrecord(dataset)

    dataset = dataset.map(parse_tfrecord)
    print(dataset)
    
    img = dataset.take(1)

    # img = mpimg.imread('../../doc/_static/stinkbug.png')
    print(img)
    




    print('-------------')


def _parse_function(example_proto):
    return tf.io.parse_single_example(example_proto, IMAGE_FEATURE_MAP)

def parse_tfrecord(x):
    x_train = tf.image.decode_jpeg(x['image/encoded'], channels=3)
    x_train = tf.image.resize(x_train, (416, 416))

    # class_text = tf.sparse.to_dense(
    #     x['image/object/class/text'], default_value='')
    # labels = tf.cast(class_table.lookup(class_text), tf.float32)
    y_train = tf.stack([tf.sparse.to_dense(x['image/object/bbox/xmin']),
                        tf.sparse.to_dense(x['image/object/bbox/ymin']),
                        tf.sparse.to_dense(x['image/object/bbox/xmax']),
                        tf.sparse.to_dense(x['image/object/bbox/ymax'])]
                        , axis=1)

    paddings = [[0, 100 - tf.shape(y_train)[0]], [0, 0]]
    y_train = tf.pad(y_train, paddings)

    return x_train, y_train


if __name__ == '__main__':
  app.run(main)