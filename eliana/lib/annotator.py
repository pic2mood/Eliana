"""
.. module:: annotator
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

from object_detection.utils import label_map_util as lbl
from object_detection.utils import visualization_utils as vis

from eliana.lib.eliana_image import ElianaImage


class Annotator:
    """.. class:: Annotator

    Class for object detection/annotation in images.
    """

    def __init__(
            self,
            img,
            model: str,
            ckpt: str,
            labels: str,
            classes: int,
            session,
            detection_graph):

        """Annotator class constructor.

        Args:
            img (ElianaImage): Input image.
            session (Session): tensorflow session
            detection_graph: pass
        """

        self.img = img

        self.model = model
        self.ckpt = ckpt
        self.labels = labels
        self.classes = classes

        self.session = session
        self.detection_graph = detection_graph

        self.label_map = lbl.load_labelmap(labels)

        self.categories = lbl.convert_label_map_to_categories(
            self.label_map,
            max_num_classes=classes,
            use_display_name=True
        )

        self.category_index = lbl.create_category_index(
            self.categories
        )

    def annotate(self):
        """annotate()

        Method for annotation action.
        """

        def __annotate_init_params():

            dg = self.detection_graph
            img_expanded: np = np.expand_dims(self.img.as_numpy, axis=0)
            img_tensor = dg.get_tensor_by_name('image_tensor:0')

            boxes: np = dg.get_tensor_by_name('detection_boxes:0')

            scores: np = dg.get_tensor_by_name('detection_scores:0')
            classes: np = dg.get_tensor_by_name('detection_classes:0')
            detections: np = dg.get_tensor_by_name('num_detections:0')

            return (boxes, scores, classes, detections,
                    img_expanded, img_tensor)

        def __annotate(
            boxes,
            scores,
            classes,
            detections,
            img_tensor,
            img_expanded
        ):

            """
            Helper method for Annotate.annotate().
            Runs a tensorflow session for
            object detection/annotation.

            Args:
                session (Session): tensorflow session.
            """

            (boxes, scores, classes, detections) = self.session.run(
                [boxes, scores, classes, detections],
                feed_dict={img_tensor: img_expanded}
            )

            return (boxes, scores, classes, detections)

        #
        #
        (boxes, scores, classes, detections,
            img_expanded, img_tensor) = __annotate_init_params()

        (boxes, scores, classes, detections) = __annotate(
            boxes,
            scores,
            classes,
            detections,
            img_tensor,
            img_expanded
        )

        boxed_img = self.__draw_boxes_and_labels(boxes, classes, scores)
        boxed_img.show(use='plt')
        # print(boxed_img)

        cropped_images = self.__crop_batch(boxes, scores)
        for img in cropped_images:
            img.show(use='pil')

    #
    #
    def __crop_batch(self, boxes, scores):

        def __get_box_coords(boxes, i) -> tuple:

            ymin: float = boxes[0, i, 0]
            xmin: float = boxes[0, i, 1]
            ymax: float = boxes[0, i, 2]
            xmax: float = boxes[0, i, 3]

            xminn = int(xmin * self.img.width)
            xmaxx = int(xmax * self.img.width)
            yminn = int(ymin * self.img.height)
            ymaxx = int(ymax * self.img.height)

            return (xminn, xmaxx, yminn, ymaxx)

        def __get_object_count(scores: np) -> int:

            final_scores: np = np.squeeze(scores)
            count: int = 0

            for i in range(100):
                if scores is None or final_scores[i] > 0.5:
                    count += 1

            return count

        #
        #
        cropped_images = []
        count: int = __get_object_count(scores)

        for i in range(count):

            (xminn, xmaxx, yminn, ymaxx) = __get_box_coords(boxes, i)
            img_crop = self.__crop_single((xminn, xmaxx, yminn, ymaxx))

            cropped_images.append(img_crop)

        return cropped_images

    #
    #
    def __crop_single(self, coords: tuple):
        """
        Helper method for Annotate.annotate(). Crops an image object segment.

        Args:
            coords (tuple): anchor points for cropping
        """

        def ___crop_using_pil(coords: tuple):

            (xminn, xmaxx, yminn, ymaxx) = coords
            img_crop = self.img.as_pil.crop(
                (xminn, yminn, xmaxx, ymaxx)
            )
            return ElianaImage(pil=img_crop)

        def ___crop_using_tf(coords: tuple):

            (xminn, xmaxx, yminn, ymaxx) = coords
            img_crop = tf.image.crop_to_bounding_box(
                self.img.as_numpy, yminn, xminn,
                ymaxx - yminn,
                xmaxx - xminn
            )
            return img_crop

        #
        img_crop = ___crop_using_pil(coords)
        return img_crop

    def __draw_boxes_and_labels(self, boxes, classes, scores):

        boxed_img = self.img.as_numpy
        vis.visualize_boxes_and_labels_on_image_array(
            boxed_img,
            np.squeeze(boxes),
            np.squeeze(classes).astype(np.int32),
            np.squeeze(scores),
            self.category_index,
            use_normalized_coordinates=True,
            line_thickness=8
        )

        return ElianaImage(np=boxed_img)
