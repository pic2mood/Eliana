"""
.. module:: eliana_image
    :platform: Linux
    :synopsis: eliana image container

.. moduleauthor:: Raymel Francisco <franciscoraymel@gmail.com>
"""

import numpy as np
from PIL import Image

from matplotlib import pyplot as plt

class ElianaImage():
    """.. class:: ElianaImage

    Class for Eliana image container.
    """

    def __init__(self, path):

        """.. method:: ElianaImage(path: str)

        ElianaImage class constructor.

        Args:
            path (str): Image path.
        """

        self.__img_pil = Image.open(path)

        (self.__w, self.__h) = self.__img_pil.size

        self.__img_numpy = self.__load_image_into_numpy_array(self.__img_pil)

    @property
    def width(self):
        return self.__w

    @property
    def height(self):
        return self.__h

    @property
    def as_list(self):
        return self.__img_list

    @property
    def as_numpy(self):
        return self.__img_numpy

    @property
    def as_pil(self):
        return self.__img_pil

    def __load_image_into_numpy_array(self, img):

        img = np.array(
            img.getdata()
        ).reshape(
            (self.__h, self.__w, 3)
        ).astype(
            np.uint8
        )

        return img

    @staticmethod
    def show_img(img, use='pil'):
        """ Shows image.

            Args:
                img (np): image in numpy array representation
        """

        def __show_using_plt():

            plt.figure(figsize=(12, 9))
            plt.imshow(img)
            plt.show()

        def __show_using_pil():
            img.show()

        #
        #
        if use == 'pil':
            __show_using_pil()

        elif use == 'plt':
            __show_using_plt()

        else:
            raise ValueError(
                '"{}" is invalid argument. Use "pil" or "plt" only.'
                .format(use)
            )
