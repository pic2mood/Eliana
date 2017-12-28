"""
.. module:: annotator
    :synopsis: main module for object detection/annotation in images

.. moduleauthor:: Raymel Francisco <franciscoraymel@gmail.com>
"""

import os
#import cv2
import time
import argparse
import multiprocessing
import numpy as np
import tensorflow as tf
from matplotlib import pyplot as plt

from eliana.api.object_detection.utils import label_map_util as lbl
from eliana.api.object_detection.utils import visualization_utils as vis


class Annotator:
    """.. class:: Annotator

    Class for object detection/annotation in images.
    """

    def __init__(
            self,
            model: str,
            ckpt: str,
            labels: str,
            classes: int):

        """Annotator class constructor.

        Args:
            img (ElianaImage): Input image.
            session (Session): tensorflow session
            detection_graph: pass
        """

        self.ckpt = ckpt
        self.detection_graph = tf.Graph()

        with self.detection_graph.as_default():

            graph_def = tf.GraphDef()
            fid = tf.gfile.GFile(self.ckpt, 'rb')
            graph_serialized = fid.read()
            graph_def.ParseFromString(graph_serialized)
            tf.import_graph_def(graph_def, name='')

            self.session = tf.Session(graph=self.detection_graph)

        # self.img = img

        self.model = model
        self.labels = labels
        self.classes = classes

        # self.session = session
        # self.detection_graph = detection_graph

        self.label_map = lbl.load_labelmap(labels)

        self.categories = lbl.convert_label_map_to_categories(
            self.label_map,
            max_num_classes=classes,
            use_display_name=True
        )

        self.category_index = lbl.create_category_index(
            self.categories
        )

    # def __reshape_np_image(self, img: ElianaImage):
    #     return np.array(img.as_pil).reshape(
    #         (img.height, img.width, 3)
    #     ).astype(
    #         np.uint8
    #     )

    def __reshape_np_image(self, img):
        return img.reshape(
            (img.shape[0], img.shape[1], -1)
        ).astype(
            np.uint8
        )

    def annotate(self, img):
        """annotate()

        Method for annotation action.
        """

        self.img = img
        self.cropped_images = []

        def __annotate_init_params():

            dg = self.detection_graph
            # img_expanded: np = np.expand_dims(
            #     self.img.as_numpy,
            #     axis=0
            # )
            img_expanded: np = np.expand_dims(
                self.__reshape_np_image(self.img),
                axis=0
            )

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

        # for b, c, s in boxes, classes, scores:
        #     pass

        # print(boxes)
        # print(classes)
        # print(scores)

        # print(np.shape(boxes))
        # print(np.shape(classes))
        # print(np.shape(scores))
        # # annotated_array = np.dstack((boxes, classes, scores))
        # # print(annotated_array)

        # scores_temp = np.zeros([1, 100, 4])
        # boxes_temp = np.zeros([1, 100])
        # classes_temp = np.zeros([1, 100])

        # i = 0
        # for a in scores[0]:
        #     if a > 0.5:
        #         # scores = np.delete(scores[0], i)
        #         # boxes = np.delete(boxes, i)
        #         # classes = np.delete(classes, i)
        #         np.put(scores_temp, i, a)
        #         np.put(boxes_temp, i, boxes[0][i])
        #         np.put(classes_temp, i, classes[0][i])
        #         i += 1

        # print(boxes_temp)
        # print(classes_temp)
        # print(scores_temp)

        # print(np.shape(boxes_temp))
        # print(np.shape(classes_temp))
        # print(np.shape(scores_temp))

        boxed_img = self.__draw_boxes_and_labels(boxes, classes, scores)
        # boxed_img.show(use='plt')
        # print(boxed_img)

        self.cropped_images = self.__crop_batch(boxes, scores)
        # for img in self.cropped_images:
        #     img.show(use='pil')

        # print(classes)
        # print(scores)
        # print(detections)

        # print(np.squeeze(boxes))
        # print(np.squeeze(classes).astype(np.int32))
        # print(np.squeeze(scores))
        # print(self.category_index)

        # print()

        scores = np.squeeze(scores)
        classes = np.squeeze(classes)

        # print(scores)

        result = []

        for i in range(scores.size):
            if scores[i] > 0.5:
                # print(
                #     scores[i],
                #     self.category_index[classes[i]]
                # )
                category = self.category_index[classes[i]]
                result.append(
                    (scores[i], self.cropped_images[i], category['id'], category['name'])
                )

        return result

        # input("Press...")

    #
    #
    def __crop_batch(self, boxes, scores):

        def __get_box_coords(boxes, i) -> tuple:

            ymin: float = boxes[0, i, 0]
            xmin: float = boxes[0, i, 1]
            ymax: float = boxes[0, i, 2]
            xmax: float = boxes[0, i, 3]

            xminn = int(xmin * self.img.shape[1])
            xmaxx = int(xmax * self.img.shape[1])
            yminn = int(ymin * self.img.shape[0])
            ymaxx = int(ymax * self.img.shape[0])

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
            # img_crop = self.img.as_pil.crop(
            #     (xminn, yminn, xmaxx, ymaxx)
            # )
            img_crop = self.img[yminn:ymaxx, xminn:xmaxx]
            #return ElianaImage(pil=img_crop)
            return img_crop

        def ___crop_using_tf(coords: tuple):

            (xminn, xmaxx, yminn, ymaxx) = coords
            img_crop = tf.image.crop_to_bounding_box(
                self.__reshape_np_image(self.img), yminn, xminn,
                # self.img.as_numpy, yminn, xminn,
                ymaxx - yminn,
                xmaxx - xminn
            )
            return img_crop

        #
        img_crop = ___crop_using_pil(coords)
        return img_crop

    def __draw_boxes_and_labels(self, boxes, classes, scores):

        # boxed_img = self.img.as_numpy
        boxed_img = self.__reshape_np_image(self.img)

        vis.visualize_boxes_and_labels_on_image_array(
            boxed_img,
            np.squeeze(boxes),
            np.squeeze(classes).astype(np.int32),
            np.squeeze(scores),
            self.category_index,
            use_normalized_coordinates=True,
            line_thickness=8,
            min_score_thresh=0.5
        )

        return boxed_img
