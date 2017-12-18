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

    @staticmethod
    def scaled_colorfulness(colorfulness: float):
        """
        Based on the scale:

        Not colorful - 0
        Slightly colorful - 15
        Moderately colorful - 33
        Averagely colorful - 45
        Quite colorful - 59
        Highly colorful - 82
        Extremely colorful - 109 

        """

        if colorfulness < 15:
            return 0.1

        elif colorfulness >= 15:
            return 0.2

        elif colorfulness >= 33:
            return 0.3

        elif colorfulness >= 45:
            return 0.4

        elif colorfulness >= 59:
            return 0.5

        elif colorfulness >= 82:
            return 0.6

        elif colorfulness >= 109:
            return 0.7
