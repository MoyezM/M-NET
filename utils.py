import dataset
import tensorflow as tf
import cv2
import numpy as np
import matplotlib.pyplot as plt

def draw_labels(x, y):
    img = x.numpy()
    boxes, classes = tf.split(y, (4, 1), axis=-1)
    classes = classes[..., 0]
    wh = np.flip(img.shape[0])
    for i in range(len(boxes)):
        temp = boxes[i] * wh
        x1y1 = tuple((np.array(temp[0:2])).astype(np.int32))
        x2y2 = tuple((np.array(temp[2:4])).astype(np.int32))       
        img = cv2.rectangle(img, x1y1, x2y2, (255, 0, 0), 2)
        img = cv2.putText(img, category_map[classes[i].numpy()],
                          x1y1, cv2.FONT_HERSHEY_COMPLEX_SMALL,
                          1, (0, 0, 255), 2)
    return img

# for image_features in dataset.take(2):
# #     print(image_features['y_train'])
#     img = draw_labels(image_features['x_train'], image_features['y_train'])  
# #     print(img)

#     plt.imshow(img.astype(dtype='uint8'))

def draw_outputs(img, outputs, class_names):
    boxes, objectness, classes, nums = outputs
    boxes, objectness, classes, nums = boxes[0], objectness[0], classes[0], nums[0]
    wh = np.flip(img.shape[0:2])
    for i in range(nums):
        x1y1 = tuple((np.array(boxes[i][0:2]) * wh).astype(np.int32))
        x2y2 = tuple((np.array(boxes[i][2:4]) * wh).astype(np.int32))
        img = cv2.rectangle(img, x1y1, x2y2, (255, 0, 0), 2)
        img = cv2.putText(img, '{} {:.4f}'.format(
            class_names[int(classes[i])], objectness[i]),
            x1y1, cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 0, 255), 2)
    return img




