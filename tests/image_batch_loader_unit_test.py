"""
.. test:: image_batch_loader_unit_test.py
    :platform: Linux
    :synopsis: unit test for texture module

.. created:: Dec 8, 2017
.. author:: Raymel Francisco <franciscoraymel@gmail.com>
"""
import os
from tests.eliana_test import ElianaUnitTest
from eliana.lib.image_batch_loader import ImageBatchLoader

import traceback


class ImageBatchLoaderUnitTest(ElianaUnitTest):

    def __init__(self):
        ElianaUnitTest.__init__(self)

    def __init_test_images(self):

        self.__dir_test_images = os.path.join(
            os.getcwd(),
            'training',
            'data',
            'test_images'
        )
        self.__images = ImageBatchLoader(self.__dir_test_images).images

    def __print_some_of_test_images(self):
        for (path, img) in self.__images:
            print(path, img, sep='\n', end='\n\n')

    def run(self):

        self.eliana_log.steps = 3

        self.eliana_log.log('Starting unit test for Image batch loader module')
        print(self.eliana_log.log_ok)

        self.eliana_log.log('Loading test images')
        self.test(self.__init_test_images)

        self.eliana_log.log('Printing test images data')
        self.test(self.__print_some_of_test_images)


image_batch_loader_unit_test = ImageBatchLoaderUnitTest()
image_batch_loader_unit_test.run()
