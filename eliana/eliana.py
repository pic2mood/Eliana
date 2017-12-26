"""
.. module:: eliana
    :synopsis: main module for eliana package

.. moduleauthor:: Raymel Francisco <franciscoraymel@gmail.com>
.. created:: Nov 24, 2017
"""
from eliana.imports import *


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
