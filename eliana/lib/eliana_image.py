"""
.. module:: eliana_image
    :platform: Linux
    :synopsis: eliana image container

.. moduleauthor:: Raymel Francisco <franciscoraymel@gmail.com>
"""

import numpy as np
from PIL import Image

from matplotlib import pyplot as plt


class ElianaImageBase:

    def _init_from_np(self, img: np):

        self.__img_numpy = img
        self.__img_pil = Image.fromarray(
            np.uint8(self.__img_numpy)
        )
        (self.__w, self.__h) = self.__img_pil.size

    def _init_from_pil(self, img: Image):

        self.__img_pil = img
        (self.__w, self.__h) = self.__img_pil.size

        self.__img_numpy = np.array(
            self.__img_pil.getdata()
        ).astype(
            np.uint8
        )

    def show(self, use='pil'):
        """ Shows image.

            Args:
                use (str):

                Use 'pil' to use Pillow Image's show().

                Use 'plt' for matplotlib pyplot's imshow() and show().
        """
        if use == 'pil':
            self.__img_pil.show()

        elif use == 'plt':
            plt.figure(figsize=(12, 9))
            plt.imshow(self.__img_numpy)
            plt.pause(0.05)

        else:
            raise ValueError(
                '"{}" is invalid argument. Use "pil" or "plt" only.'
                .format(use)
            )

    @property
    def width(self):
        return self.__w

    @property
    def height(self):
        return self.__h

    @property
    def as_numpy(self):
        return self.__img_numpy

    @property
    def as_pil(self):
        return self.__img_pil

    @property
    def colorfulness(self):
        return self.__colorfulness

    @colorfulness.setter
    def colorfulness(self, colorfulness):
        self.__colorfulness = colorfulness

    @property
    def texture(self):
        return self.__texture

    @texture.setter
    def texture(self, texture):
        self.__texture = texture


class ElianaImage(ElianaImageBase):
    """.. class:: ElianaImage

    Class for Eliana image container.
    """

    # TODO:
    # Add support to Tensor image.
    # Add sync update between image types

    def __init__(self, path: str=None, np: np=None, pil: Image=None):

        """.. method:: ElianaImage(path: str)

        ElianaImage class constructor.

        Args:
            path (str): Image path.
        """

        if path is not None:
            self.__path = path
            self.__init_from_path(path)

        elif np is not None:
            super()._init_from_np(np)

        elif pil is not None:
            super()._init_from_pil(pil)
        else:
            raise ValueError('No argument supplied.')

        self.__objects = []

    def __init_from_path(self, path):

        super()._init_from_pil(Image.open(path))

    @property
    def path(self):
        return self.__path

    # @property
    # def as_list(self):
    #     return self.__img_list

    @property
    def objects(self):
        return self.__objects

    @objects.setter
    def objects(self, objects):
        self.__objects = objects


class ElianaImageObject(ElianaImageBase):

    def __init__(
            self,
            parent: ElianaImage=None,
            cropped: Image=None,
            annotation: str=None,
            tag_id: int=0
    ):
        self.__parent = parent
        self.__annotation = annotation
        self.__tag_id = tag_id

        super()._init_from_pil(cropped)

    @property
    def annotation(self):
        return self.__annotation

    @property
    def tag_id(self):
        return self.__tag_id
