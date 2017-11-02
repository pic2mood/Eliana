
import os
import sys

sys.path.append(os.path.dirname(os.path.realpath(__file__)) + "/../lib/annotator/")


import tensorflow as tf
from annotator import Annotator
from PIL import Image
from image import ElianaImage

CWD_PATH = os.getcwd()


# First test on images
PATH_TO_TEST_IMAGES_DIR = '../Eliana/lib/annotator/object_detection/test_images'
TEST_IMAGE_PATHS = [ os.path.join(PATH_TO_TEST_IMAGES_DIR, 'image{}.jpg'.format(i)) for i in range(1, 3) ]

# Size, in inches, of the output images.
IMAGE_SIZE = (12, 8)


# Path to frozen detection graph. This is the actual model that is used for the object detection.
MODEL_NAME = 'ssd_mobilenet_v1_coco_11_06_2017'
PATH_TO_CKPT = os.path.join(CWD_PATH, 'lib', 'annotator', 'object_detection', MODEL_NAME, 'frozen_inference_graph.pb')

# List of the strings that is used to add correct label for each box.
PATH_TO_LABELS = os.path.join(CWD_PATH, 'lib', 'annotator', 'object_detection', 'data', 'mscoco_label_map.pbtxt')

NUM_CLASSES = 90


detection_graph = tf.Graph()

with detection_graph.as_default():
    od_graph_def = tf.GraphDef()
    with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
        serialized_graph = fid.read()
        od_graph_def.ParseFromString(serialized_graph)
        tf.import_graph_def(od_graph_def, name='')


with detection_graph.as_default():
    with tf.Session(graph=detection_graph) as sess:
        for image_path in TEST_IMAGE_PATHS:
            image = Image.open(image_path)
            image_np = ElianaImage(image).img

            (img_w, img_h) = image.size

            annotator = Annotator(
                img_w, 
                img_h,
                MODEL_NAME,
                PATH_TO_CKPT,
                PATH_TO_LABELS,
                NUM_CLASSES
            )

            annotator.annotate(image_np, sess, detection_graph)



