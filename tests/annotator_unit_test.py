"""
.. test:: annotator_unit_test.py
    :platform: Linux
    :synopsis: unit test for annotator module

.. created:: Nov 2, 2017
.. author:: Raymel Francisco <franciscoraymel@gmail.com>
"""
import os
from tests.eliana_test import ElianaUnitTest

import tensorflow as tf
from eliana.lib.annotator import Annotator
from eliana.lib.eliana_image import ElianaImage

import traceback


class AnnotatorUnitTest(ElianaUnitTest):

    def __init__(self):
        ElianaUnitTest.__init__(self)

    def __init_test_images(self):

        self.__dir_test_images = os.path.join(
            self.training_data,
            'data',
            'test_images'
        )
        self.__test_images = [
            os.path.join(
                self.__dir_test_images, 'img{}.jpg'.format(i)
            )
            for i in range(1, 3)
        ]
        self.__image_size = (12, 8)

    def __init_model(self):

        self.__model = 'ssd_mobilenet_v1_coco_11_06_2017'
        self.__graph = 'frozen_inference_graph.pb'
        self.__file_ckpt = os.path.join(
            self.training_data,
            'models',
            self.__model,
            self.__graph
        )

    def __init_label_file(self):

        self.__label = 'mscoco_label_map.pbtxt'
        self.__file_label = os.path.join(
            self.training_data,
            'data',
            self.__label
        )

    def run(self):

        self.eliana_log.steps = 6

        self.eliana_log.log('Starting unit test for Annotator module')
        print(self.eliana_log.log_ok)

        self.eliana_log.log('Loading test images')
        self.test(self.__init_test_images)

        #
        #
        self.eliana_log.log('Loading model')
        self.test(self.__init_model)

        #
        #
        self.eliana_log.log('Loading labels file')
        self.test(self.__init_label_file)

        #
        #
        self.__num_classes = 90

        #
        #
        self.eliana_log.log('Loading detection graph')

        try:
            detection_graph = tf.Graph()
        except Exception as e:
            print(e.message, '\n\n' + self.eliana_log.log_error)
        else:
            print(self.eliana_log.log_ok)
        #
        #
        #
        self.eliana_log.log('Annotating objects')
        try:
            with detection_graph.as_default():

                od_graph_def = tf.GraphDef()

                with tf.gfile.GFile(self.__file_ckpt, 'rb') as fid:
                    serialized_graph = fid.read()
                    od_graph_def.ParseFromString(serialized_graph)
                    tf.import_graph_def(od_graph_def, name='')

                with tf.Session(graph=detection_graph) as sess:
                    for image_path in self.__test_images:

                        img = ElianaImage(path=image_path)

                        annotator = Annotator(
                            img,
                            self.__model,
                            self.__file_ckpt,
                            self.__file_label,
                            self.__num_classes,
                            sess,
                            detection_graph
                        )

                        annotator.annotate()

        except Exception:
            print(self.eliana_log.log_error, '\n')
            traceback.print_exc()
            print(
                '\n' +
                str(self.eliana_log.step_counter),
                'out of',
                str(self.eliana_log.steps),
                'steps executed. Exiting...',
            )
            exit()
        else:
            print(self.eliana_log.log_ok)


annotator_unit_test = AnnotatorUnitTest()
annotator_unit_test.run()
input('Press any key to exit...')
