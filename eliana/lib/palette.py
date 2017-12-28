"""
.. module:: palette
    :synopsis: gets dominant colors in an image

.. moduleauthor:: Raymel Francisco <franciscoraymel@gmail.com>
.. created:: Dec 28, 2017
"""
from eliana.imports import *

from colorthief import ColorThief
from skimage import io


class Palette:

    def dominant_colors(img, colors=2):

        temp_file = os.path.join(
            os.getcwd(),
            'temp.jpg'
        )
        io.imsave(temp_file, img)

        palette = ColorThief(temp_file).get_palette(color_count=colors)
        colors = ()

        for color in palette:
            r, g, b = color
            hex_ = '#{:02x}{:02x}{:02x}'.format(r, g, b)
            int_ = int(hex_[1:], 16)

            colors = colors + (int_,)

        return colors
