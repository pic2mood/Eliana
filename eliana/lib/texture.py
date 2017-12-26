"""
.. module:: texture
    :synopsis: main module for texture feature extraction in images

.. moduleauthor:: Raymel Francisco <franciscoraymel@gmail.com>
.. created:: Nov 19, 2017
"""
import numpy as np

from skimage.feature import greycomatrix, greycoprops
from skimage import io


class Texture:

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
