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

    batch_size = 4
    
#     Get the training set
    train_dataset = dataset.load_tfrecord_dataset(train_path)
    train_dataset = train_dataset.shuffle(buffer_size=1024)
    train_dataset = train_dataset.batch(batch_size)
    
    train_dataset = train_dataset.map(lambda x, y: (
        dataset.transform_images(x, 416),
        dataset.transform_targets(y, anchors, anchor_masks, 80)))
    
    train_dataset = train_dataset.prefetch(
        buffer_size=tf.data.experimental.AUTOTUNE)

    
    val_dataset = dataset.load_tfrecord_dataset(val_path)
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

    if eager:

        for epoch in range(1, 5 + 1):
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

                print("{}_train_{}, {}, {}".format(
                    epoch, batch, total_loss.numpy(),
                    list(map(lambda x: np.sum(x.numpy()), pred_loss))))
                avg_loss.update_state(total_loss)
                print(avg_loss.result().numpy())


            print("{}, train: {}".format(
                epoch,
                avg_loss.result().numpy()))

            avg_loss.reset_states()
            model.save_weights(
                'checkpoints/yolov3_train_{}.tf'.format(epoch))

    else:

        model.compile(optimizer=optimizer, loss=loss, run_eagerly=(False), metrics=[*mAP])
        callbacks = [
            # ReduceLROnPlateau(verbose=1),
            # EarlyStopping(patience=3, verbose=1),
            ModelCheckpoint('checkpoints/yolov3_train_{epoch}.tf', verbose=1, save_weights_only=True),
            TensorBoard(log_dir='logs')]

        # history = model.fit(train_dataset, validation_data=val_dataset, epochs=2, callbacks=callbacks)
        history = model.fit(train_dataset, validation_data=val_dataset, epochs=25, callbacks=callbacks)



main()
