"""
.. module:: texture
    :synopsis: main module for texture feature extraction in images

.. moduleauthor:: Raymel Francisco <franciscoraymel@gmail.com>
.. created:: Nov 19, 2017
"""
import numpy as np
from eliana.lib.eliana_image import ElianaImage

from skimage.feature import greycomatrix, greycoprops
from skimage import io


class Texture:

    # def __init__(self, img: ElianaImage):
    #     self.img = img

    # def __init__(self, path):
    #     self.path = path

        # self.__convert_img_to_gray()
        # self.__greycomatrix()
        # self.__round_greycomatrix_result()

    def texture(img):
        img_gray = np.average(
            img,
            weights=[0.299, 0.587, 0.114],
            axis=2
        ).astype(np.uint8)

        greycomatrix_ = greycomatrix(

            img_gray,

            distances=[1, 2],
            angles=[0],
            levels=256,
            normed=True,
            symmetric=True
        )
        greycomatrix_ = np.round(greycomatrix_, 3)

        texture = greycoprops(greycomatrix_, 'contrast')
        texture = np.mean(texture)

        return texture


    # def __convert_img_to_gray(self):

    #     self.img_gray = np.array(self.img.as_pil.convert('L'))

    # def __greycomatrix(self):

    #     self.greycomatrix_result = greycomatrix(

    #         self.img_gray,

    #         distances=[1, 2],
    #         angles=[0],
    #         levels=256,
    #         normed=True,
    #         symmetric=True
    #     )

    # def __round_greycomatrix_result(self):

    #     self.greycomatrix_result = np.round(
    #         self.greycomatrix_result, 3
    #     )

    # def get_texture_value(self):

    #     return greycoprops(self.greycomatrix_result, 'contrast')

    # def get_texture_mean(self):

    #     return np.mean(self.get_texture_value())
