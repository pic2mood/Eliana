"""
.. module:: color
    :synopsis: main module for color processing in images

.. moduleauthor:: Raymel Francisco <franciscoraymel@gmail.com>
.. created:: Nov 15, 2017
"""
import numpy as np


class Color:

    @staticmethod
    def colorfulness(img: np):

        r, g, b = img[:, :, 2], img[:, :, 1], img[:, :, 1]

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

        colorfulness = int(colorfulness)

        if colorfulness < 15:
            return 0.1

        elif colorfulness in range(15, 33):
            return 0.2

        elif colorfulness in range(33, 45):
            return 0.3

        elif colorfulness in range(45, 59):
            return 0.4

        elif colorfulness in range(59, 82):
            return 0.5

        elif colorfulness in range(82, 109):
            return 0.6

        elif colorfulness >= 109:
            return 0.7
