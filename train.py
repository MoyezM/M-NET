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

train_path = "D:\\Coco\\coco_train.tfrecord"
val_path = "D:\\Coco\\coco_val.tfrecord"

def main():
    model = mnet.MNET_complete(416, training=True)
    anchors = mnet.mnet_anchors
    anchor_masks = mnet.mnet_anchor_masks
    
#     Get the training set
    train_dataset = dataset.createDataset(train_path)
    train_dataset = train_dataset.shuffle(buffer_size=1024)
    train_dataset = train_dataset.batch(8)
    
    train_dataset = train_dataset.map(lambda x, y: (
        dataset.transform_images(x, 416),
        dataset.transform_targets(y, anchors, anchor_masks, 80)))
    
    train_dataset = train_dataset.prefetch(
        buffer_size=tf.data.experimental.AUTOTUNE)
    
    
    val_dataset = dataset.createDataset(val_path)
    
    val_dataset = val_dataset.batch(8)
    val_dataset = val_dataset.map(lambda x, y: (
        dataset.transform_images(x, 416),
        dataset.transform_targets(y, anchors, anchor_masks, 80)))

    optimizer = tf.keras.optimizers.Adam(lr = 1e-3)
    loss = [mnet.Loss(anchors[mask], classes = 80) for mask in anchor_masks]
    
    avg_loss = tf.keras.metrics.Mean('loss', dtype=tf.float32)
    avg_val_loss = tf.keras.metrics.Mean('val_loss', dtype=tf.float32)

    for epoch in range(1, 2 + 1):
        for batch, (images, labels) in enumerate(train_dataset):
            with tf.GradientTape() as tape:
                outputs = model(images, training=True)
                regularization_loss = tf.reduce_sum(model.losses)
                pred_loss = []
                for output, label, loss_fn in zip(outputs, labels, loss):
                    pred_loss.append(loss_fn(label, output))
                total_loss = tf.reduce_sum(pred_loss) + regularization_loss

            grads = tape.gradient(total_loss, model.trainable_variables)
            optimizer.apply_gradients(
                zip(grads, model.trainable_variables))

            logging.info("{}_train_{}, {}, {}".format(
                epoch, batch, total_loss.numpy(),
                list(map(lambda x: np.sum(x.numpy()), pred_loss))))
            avg_loss.update_state(total_loss)

        for batch, (images, labels) in enumerate(val_dataset):
            outputs = model(images)
            regularization_loss = tf.reduce_sum(model.losses)
            pred_loss = []
            for output, label, loss_fn in zip(outputs, labels, loss):
                pred_loss.append(loss_fn(label, output))
            total_loss = tf.reduce_sum(pred_loss) + regularization_loss

            logging.info("{}_val_{}, {}, {}".format(
                epoch, batch, total_loss.numpy(),
                list(map(lambda x: np.sum(x.numpy()), pred_loss))))
            avg_val_loss.update_state(total_loss)

        logging.info("{}, train: {}, val: {}".format(
            epoch,
            avg_loss.result().numpy(),
            avg_val_loss.result().numpy()))

        avg_loss.reset_states()
        avg_val_loss.reset_states()
        model.save_weights(
            'checkpoints/yolov3_train_{}.tf'.format(epoch))




main()



