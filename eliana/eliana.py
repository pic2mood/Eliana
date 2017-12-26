"""
.. module:: eliana
    :synopsis: main module for eliana package

.. moduleauthor:: Raymel Francisco <franciscoraymel@gmail.com>
.. created:: Nov 24, 2017
"""
import os

from eliana.lib.annotator import Annotator
from eliana.lib.color import Color
from eliana.lib.texture import Texture
from eliana.lib.mlp import MLP

from eliana.utils import interpolate


class Eliana:

    def __init__(self):

        model_path = os.path.join(
            os.getcwd(),
            'training',
            'models',
            'eliana_ann_overall',
            'eliana_ann_overall.pkl'
        )

        self.mlp = MLP()
        self.mlp.load_model(path=model_path)

    def run_overall(self, img):

        color = Color.colorfulness(img)
        color = Color.scaled_colorfulness(color)
        color = interpolate(color)

        texture = Texture.texture(img)
        texture = interpolate(texture, place=0.1)

        self.mlp.run(input_=[color, texture])


    def run_object(self, objects):
        pass
