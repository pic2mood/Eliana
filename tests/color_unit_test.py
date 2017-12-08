"""
.. test:: color_unit_test.py
    :platform: Linux
    :synopsis: unit test for color module

.. created:: Nov 15, 2017
.. author:: Raymel Francisco <franciscoraymel@gmail.com>
"""
import os
from tests.eliana_test import ElianaUnitTest

from eliana.lib.color import Color
from eliana.lib.eliana_image import ElianaImage


class ColorUnitTest(ElianaUnitTest):

    def __init__(self):
        ElianaUnitTest.__init__(self)

    def __init_test_images(self):

        self.__dir_test_images = os.path.join(
            self.dir_env_modules,
            'object_detection',
            'test_images'
        )
        self.__test_images = [
            os.path.join(
                self.__dir_test_images, 'image{}.jpg'.format(i)
            )
            for i in range(1, 3)
        ]
        self.__image_size = (12, 8)

        self.__orig_images = []

        for img_path in self.__test_images:
            img_rgb = ElianaImage(path=img_path)
            self.__orig_images.append(img_rgb)

    def __batch_convert_to_hsv(self):

        self.__hsv_converts = []

        for img_rgb in self.__orig_images:
            img_hsv = Color.to_hsv(img_rgb)
            self.__hsv_converts.append(img_hsv)

    def __batch_convert_to_rgb(self):

        self.__rgb_converts = []

        for img_hsv in self.__hsv_converts:
            img_rgb = Color.to_rgb(img_hsv)
            self.__rgb_converts.append(img_rgb)

    def __batch_colorfulness(self):

        for img in self.__orig_images:
            print('\nColorfulness:', Color.colorfulness(img))

    def run(self):

        self.eliana_log.steps = 5

        self.eliana_log.log('Starting unit test for Color module')
        print(self.eliana_log.log_ok)

        self.eliana_log.log('Loading test images')
        self.test(self.__init_test_images)
        #
        #
        self.eliana_log.log('Image conversion from RGB to HSV')
        self.test(self.__batch_convert_to_hsv)
        #
        #
        self.eliana_log.log('Image conversion from HSV to RGB')
        self.test(self.__batch_convert_to_rgb)
        #
        #
        self.eliana_log.log('Image colorfulness test')
        self.test(self.__batch_colorfulness)
