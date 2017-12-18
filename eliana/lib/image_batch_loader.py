"""
.. module:: image_batch_loader
    :synopsis: batch loader for images

.. moduleauthor:: Raymel Francisco <franciscoraymel@gmail.com>
.. created:: Nov 27, 2017
"""

from eliana.lib.eliana_image import ElianaImage
import os
from glob import glob


class ImageBatchLoader:

    def __init__(self, dir_, limit=None):

        self.dir = dir_

        print('Test images dir:', self.dir)

        self.__imgs = []
        self.dir_glob = sorted(glob(os.path.join(self.dir, '*.jpg')))

        for img_path in self.dir_glob[:limit]:

            img = ElianaImage(img_path)
            self.__imgs.append(img)

            # print(img_path)

    @property
    def images(self):
        return self.__imgs
