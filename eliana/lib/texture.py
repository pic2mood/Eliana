"""
.. module:: texture
    :synopsis: main module for texture feature extraction in images

.. moduleauthor:: Raymel Francisco <franciscoraymel@gmail.com>
.. created:: Nov 19, 2017
"""
import numpy as np

# from PIL import Image
# from eliana.lib.eliana_image import ElianaImage

from skimage.feature import greycomatrix, greycoprops

import scipy.ndimage


class Texture:

    # def __init__(self, img: ElianaImage):
    #     self.img = img

    def __init__(self, path):
        self.path = path

        self.__convert_img_to_gray()
        self.__greycomatrix()
        self.__round_greycomatrix_result()

    def __convert_img_to_gray(self):

        self.img_gray = scipy.ndimage.imread(self.path, mode='L')

    def __greycomatrix(self):

        self.greycomatrix_result = greycomatrix(

            self.img_gray,

            distances=[1, 2],
            angles=[0],
            levels=256,
            normed=True,
            symmetric=True
        )

    def __round_greycomatrix_result(self):

        self.greycomatrix_result = np.round(
            self.greycomatrix_result, 3
        )

    def get_texture_value(self):

        return greycoprops(self.greycomatrix_result, 'contrast')
