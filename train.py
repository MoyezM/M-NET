import tensorflow as tf
import numpy as np
import cv2
from tensorflow.keras.callbacks import (
    ReduceLROnPlateau,
    EarlyStopping,
    ModelCheckpoint,
    TensorBoard
)
import model as mnet
import dataset as dataset

train_path = "D:\\Coco\\coco_train"
val_path = "D:\\Coco\\coco_val"

def main(_argv):
    model = mnet.MNET_complete(416, training=true)
    anchors = mnet.mnet_anchors
    anchor_masks = mnet.mnet_anchor_masks
    
#     Get the training set
    train_dataset = dataset.createDataset(train_path)
    train_dataset = train_dataset.shuffle(buffer_size=1024)
    train_dataset = train_dataset.batch(FLAGS.batch_size)
    
    train_dataset = train_dataset.map(lambda x, y: (
        dataset.transform_images(x, 416),
        dataset.transform_targets(y, anchors, anchor_masks, 80)))
    
    train_dataset = train_dataset.prefetch(
        buffer_size=tf.data.experimental.AUTOTUNE)
    
    
    val_path = dataset.createDataset(val_path)
    
    val_dataset = val_dataset.batch(FLAGS.batch_size)
    val_dataset = val_dataset.map(lambda x, y: (
        dataset.transform_images(x, 416),
        dataset.transform_targets(y, anchors, anchor_masks, 80)))

    optimizer = tf.keras.optimizers.Adam(lr = 1e-3)
    loss = [mnet.Loss(anchors[mask], classes = 80) for mask in anchor_masks]
    
    model.compile(optimizer=optimizer, loss=loss, run_eagerly = False)

        callbacks = [
            ReduceLROnPlateau(verbose=1),
            EarlyStopping(patience=3, verbose=1),
            ModelCheckpoint('checkpoints/mnet_train_{epoch}.tf',
                            verbose=1, save_weights_only=True),
            TensorBoard(log_dir='logs')
        ]

        history = model.fit(train_dataset,
                            epochs=FLAGS.epochs,
                            callbacks=callbacks,
                            validation_data=val_dataset)

if __name__ == '__main__':
    try:
        app.run(main)
    except SystemExit:
        pass


