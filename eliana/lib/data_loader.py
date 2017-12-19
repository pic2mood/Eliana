"""
.. module:: data loader
    :synopsis: loads necessary data

.. moduleauthor:: Raymel Francisco <franciscoraymel@gmail.com>
.. created:: Dec 12, 2017
"""

from eliana.lib.image_batch_loader import ImageBatchLoader

import os
import tensorflow as tf


class DataLoader:

    def __init__(self):
        self.training_images = []

        self.dir_working = os.getcwd()

        self.__dir_env_modules = os.path.join(
            self.dir_working,
            'env',
            'eliana',
            'lib',
            'python3.6',
            'site-packages'
        )
        self.__dir_local_modules = os.path.join(
            self.dir_working,
            'lib'
        )
        self.training_data = os.path.join(
            self.dir_working,
            'training'
        )
        self.__dir_images = os.path.join(
            self.training_data,
            'data',
            'test_images'
        )

    def training(self):
        return os.path.join(
            self.__dir_images,
            'manual_annotator.txt'
        )

    def images(self):
        return ImageBatchLoader(
            self.__dir_images, limit=None
        ).images

    def annotator(self):

        model = 'ssd_mobilenet_v1_coco_11_06_2017'
        graph = 'frozen_inference_graph.pb'
        file_ckpt = os.path.join(
            self.training_data,
            'models',
            model,
            graph
        )
        label = 'mscoco_label_map.pbtxt'
        file_label = os.path.join(
            self.training_data,
            'data',
            label
        )
        num_classes = 90
        detection_graph = tf.Graph()

        with detection_graph.as_default():

            graph_def = tf.GraphDef()
            fid = tf.gfile.GFile(file_ckpt, 'rb')
            graph_serialized = fid.read()
            graph_def.ParseFromString(graph_serialized)
            tf.import_graph_def(graph_def, name='')

            sess = tf.Session(graph=detection_graph)

        return (
            model,
            file_ckpt,
            file_label,
            num_classes,
            sess,
            detection_graph
        )
