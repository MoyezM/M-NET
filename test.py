import dataset as dataset
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import tensorflow as tf
import numpy as np
import cv2

category_map = {
    -1:'',
    0:'person',
    1:'bicycle',
    2:'car',
    3:'motorcycle',
    4:'airplane',
    5:'bus',
    6:'train',
    7:'truck',
    8:'boat',
    9:'traffic light',
    10:'fire hydrant',
    11:'stop sign',
    12:'parking meter',
    13:'bench',
    14:'bird',
    15:'cat',
    16:'dog',
    17:'horse',
    18:'sheep',
    19:'cow',
    20:'elephant',
    21:'bear',
    22:'zebra',
    23:'giraffe',
    24:'backpack',
    25:'umbrella',
    26:'handbag',
    27:'tie',
    28:'suitcase',
    29:'frisbee',
    30:'skis',
    31:'snowboard',
    32:'sports ball',
    33:'kite',
    34:'baseball bat',
    35:'baseball glove',
    36:'skateboard',
    37:'surfboard',
    38:'tennis racket',
    39:'bottle',
    40:'wine glass',
    41:'cup',
    42:'fork',
    43:'knife',
    44:'spoon',
    45:'bowl',
    46:'banana',
    47:'apple',
    48:'sandwich',
    49:'orange',
    50:'broccoli',
    51:'carrot',
    52:'hot dog',
    53:'pizza',
    54:'donut',
    55:'cake',
    56:'chair',
    57:'couch',
    58:'potted plant',
    59:'bed',
    60:'dining table',
    61:'toilet',
    62:'tv',
    63:'laptop',
    64:'mouse',
    65:'remote',
    66:'keyboard',
    67:'cell phone',
    68:'microwave',
    69:'oven',
    70:'toaster',
    71:'sink',
    72:'refrigerator',
    73:'book',
    74:'clock',
    75:'vase',
    76:'scissors',
    77:'teddy bear',
    78:'hair drier',
    79:'toothbrush',
}

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

train_path = "/home/moyez/Documents/Code/Python/M-NET/coco_train.record"

train_dataset = dataset.load_tfrecord_dataset(train_path)


test = train_dataset.take(4)

for x, y in test:
    img = draw_labels(x, y)
    plt.imshow(img.astype(dtype='uint8'))
