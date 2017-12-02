"""
.. module:: color
    :synopsis: main module for color processing in images

.. moduleauthor:: Raymel Francisco <franciscoraymel@gmail.com>
.. created:: Nov 15, 2017
"""
from PIL import Image
from eliana.lib.eliana_image import ElianaImage
import numpy as np


class Color:

    @staticmethod
    def to_hsv(img: ElianaImage):
        return ElianaImage(pil=img.as_pil.convert('HSV'))

    @staticmethod
    def to_rgb(img: ElianaImage):
        return ElianaImage(pil=img.as_pil.convert('RGB'))

    @staticmethod
    def colorfulness(img: ElianaImage):

        r, g, b = img.as_pil.split()
        # r = ElianaImage(np=r).as_numpy
        # g = ElianaImage(np=g).as_numpy
        # b = ElianaImage(np=b).as_numpy

        r = np.array(r.getdata())
        g = np.array(g.getdata())
        b = np.array(b.getdata())

        # print(r.__class__)

        rg = np.absolute(r - g)
        yb = np.absolute(0.5 * (r + g) - b)

        (rb_mean, rb_std) = (np.mean(rg), np.std(rg))
        (yb_mean, yb_std) = (np.mean(yb), np.std(yb))

        std_root = np.sqrt((rb_std ** 2) + (yb_std ** 2))
        mean_root = np.sqrt((rb_mean ** 2) + (yb_mean ** 2))

        return std_root + (0.3 * mean_root)

