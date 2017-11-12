"""
.. module:: image
    :platform: Linux
    :synopsis: eliana image container

.. moduleauthor:: Raymel Francisco <franciscoraymel@gmail.com>
"""

import numpy as np
#from PIL import Image


class ElianaImage():
    """.. class:: ElianaImage

    Class for Eliana image container.
    """

    def __init__(self, img):

        """.. method:: ElianaImage(img: list)

        ElianaImage class constructor.

        Args:
            img (list): Image in list form.
        """

        self.__load_image_into_numpy_array(img)

    def __load_image_into_numpy_array(self, img):

        (im_width, im_height) = img.size

        self.img = np.array(
            img.getdata()
        ).reshape(
            (im_height, im_width, 3)
        ).astype(
            np.uint8
        )
