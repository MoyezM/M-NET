import tensorflow as tf
import tensorflow.keras as K
import numpy as np
import cv2
from tensorflow.keras.callbacks import (
    ReduceLROnPlateau,
    EarlyStopping,
    ModelCheckpoint,
    TensorBoard,
    TerminateOnNaN,

)
import model as mnet
import dataset as dataset
from datetime import datetime


train_path = "/home/moyez/Documents/Code/Python/M-NET/coco_train.record"
train_size = 118287
val_path = "/home/moyez/Documents/Code/Python/M-NET/coco_val.record"
val_size = 5000


def main():
    config = tf.compat.v1.ConfigProto()
    config.gpu_options.allow_growth = True
    session = tf.compat.v1.Session(config=config)

    # K.set_epsilon(1e-4)
    # K.backend.set_floatx('float16')


    model = mnet.MNET_complete(416, training=True)
    anchors = mnet.mnet_anchors
    anchor_masks = mnet.mnet_anchor_masks

    batch_size = 8
    
#     Get the training set
    train_dataset = dataset.load_tfrecord_dataset(train_path)

    # train_dataset = train_dataset.take(200)

    train_dataset = train_dataset.shuffle(buffer_size=1024)
    train_dataset = train_dataset.batch(batch_size)
    
    train_dataset = train_dataset.map(lambda x, y: (
        dataset.transform_images(x, 416),
        dataset.transform_targets(y, anchors, anchor_masks, 80)))
    
    train_dataset = train_dataset.prefetch(
        buffer_size=tf.data.experimental.AUTOTUNE)

    
    val_dataset = dataset.load_tfrecord_dataset(val_path)

    # val_dataset = val_dataset.take(50)

    val_dataset = val_dataset.batch(batch_size)
    val_dataset = val_dataset.map(lambda x, y: (
        dataset.transform_images(x, 416),
        dataset.transform_targets(y, anchors, anchor_masks, 80)))

    optimizer = tf.keras.optimizers.Adam(lr = 1e-3)
    loss = [mnet.Loss(anchors[mask], classes = 80) for mask in anchor_masks]
    mAP = [mnet.map(anchors[mask]) for mask in anchor_masks]
    
    avg_loss = tf.keras.metrics.Mean('loss', dtype=tf.float32)
    avg_val_loss = tf.keras.metrics.Mean('val_loss', dtype=tf.float32)

    eager = False


    logdir = "logs/" + datetime.now().strftime("%Y%m%d-%H%M%S")
    tensorboard_callback = K.callbacks.TensorBoard(log_dir=logdir)


    model.compile(optimizer=optimizer, loss=loss, run_eagerly=(False), metrics=[*mAP])
    callbacks = [
        ReduceLROnPlateau(verbose=1),
        EarlyStopping(patience=2, verbose=1),
        ModelCheckpoint('checkpoints/mnet_train_{epoch}.tf', verbose=1, save_weights_only=True),
        tensorboard_callback]

    # history = model.fit(train_dataset, validation_data=val_dataset, epochs=2, callbacks=callbacks)
    history = model.fit(train_dataset, validation_data=val_dataset, epochs=20, callbacks=callbacks, validation_steps=int(val_size/batch_size))



main()
