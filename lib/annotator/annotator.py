"""
.. module:: annotator
    :platform: Linux
    :synopsis: main module for object detection/annotation in images

.. moduleauthor:: Raymel Francisco <franciscoraymel@gmail.com>
"""

import os
import cv2
import time
import argparse
import multiprocessing
import numpy as np
import tensorflow as tf
from matplotlib import pyplot as plt

from object_detection.utils import label_map_util as lbl_util
from object_detection.utils import visualization_utils as vis_util


class Annotator:
    """Class for object detection/annotation in images.
    """

    def __init__(
            self,
            img_w,
            img_h,
            model: str,
            ckpt: str,
            labels: str,
            classes: int):

        """Annotator class constructor.

        Args:
            img_w (int): Width of the image in pixels.
            img_h (int): Height of the image in pixels.
        """

        self.IMAGE_SIZE = (12, 8)

        self.img_w = img_w
        self.img_h = img_h
        self.model = model
        self.ckpt = ckpt
        self.labels = labels
        self.classes = classes

        self.label_map = lbl_util.load_labelmap(labels)

        self.categories = lbl_util.convert_label_map_to_categories(
            self.label_map,
            max_num_classes=classes,
            use_display_name=True
        )

        self.category_index = lbl_util.create_category_index(
            self.categories
        )

    def annotate(self, img: np, session, detection_graph):
        """Method for annotation action.

        Args:
            img (np): image in numpy array representation
            session (Session): tensorflow session
            detection_graph: pass
        """

        dg = detection_graph
        img_expanded: np = np.expand_dims(img, axis=0)
        img_tensor = dg.get_tensor_by_name('image_tensor:0')

        boxes: np = dg.get_tensor_by_name('detection_boxes:0')

        scores: np = dg.get_tensor_by_name('detection_scores:0')
        classes: np = dg.get_tensor_by_name('detection_classes:0')
        detections: np = dg.get_tensor_by_name('num_detections:0')

        (boxes, scores, classes, detections) = self.__annotate(
            session, boxes, scores, classes, detections,
            img_tensor, img_expanded
        )

        count: int = self.__get_object_count(scores)

        #
        #
        for i in range(count):

            (xminn, xmaxx, yminn, ymaxx) = self.__get_box_coords(boxes, i)
            img_crop = self.__crop(img, (xminn, xmaxx, yminn, ymaxx))

            self.show_img(img_crop, session)

    def __annotate(
            self,
            session,
            boxes,
            scores,
            classes,
            detections,
            img_tensor,
            img_expanded):

        """
        Helper method for Annotate.annotate(). Runs a tensorflow session for
        object detection/annotation.

        Args:
            session (Session): tensorflow session.
        """

        (boxes, scores, classes, detections) = session.run(
            [boxes, scores, classes, detections],
            feed_dict={img_tensor: img_expanded}
        )

        return (boxes, scores, classes, detections)

    #
    def __crop(self, img: np, coords: tuple):

        (xminn, xmaxx, yminn, ymaxx) = coords
        img_crop = tf.image.crop_to_bounding_box(
            img, yminn, xminn,
            ymaxx - yminn,
            xmaxx - xminn
        )
        return img_crop

    #
    def show_img(self, img, session):

        img_data = session.run(img)
        plt.figure(figsize=self.IMAGE_SIZE)
        plt.imshow(img_data)
        plt.show()

    def _draw_boxes_and_labels(self, img, boxes, classes, scores):

        vis_util.visualize_boxes_and_labels_on_image_array(
            img,
            np.squeeze(boxes),
            np.squeeze(classes).astype(np.int32),
            np.squeeze(scores),
            category_index,
            use_normalized_coordinates=True,
            line_thickness=8
        )

        return img_np

    def __get_box_coords(self, boxes, i) -> tuple:

        ymin: float = boxes[0, i, 0]
        xmin: float = boxes[0, i, 1]
        ymax: float = boxes[0, i, 2]
        xmax: float = boxes[0, i, 3]

        xminn = int(xmin * self.img_w)
        xmaxx = int(xmax * self.img_w)
        yminn = int(ymin * self.img_h)
        ymaxx = int(ymax * self.img_h)

        return (xminn, xmaxx, yminn, ymaxx)

    def __get_object_count(self, scores: np) -> int:

        final_scores: np = np.squeeze(scores)
        count: int = 0

        for i in range(100):
            if scores is None or final_scores[i] > 0.5:
                count += 1

        return count
