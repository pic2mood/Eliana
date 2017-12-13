"""
.. module:: eliana
    :synopsis: main module for eliana package

.. moduleauthor:: Raymel Francisco <franciscoraymel@gmail.com>
.. created:: Nov 24, 2017
"""

from eliana.lib.eliana_image import ElianaImage, ElianaImageObject
from eliana.lib.image_batch_loader import ImageBatchLoader
from eliana.lib.annotator import Annotator
from eliana.lib.color import Color
from eliana.lib.texture import Texture
from eliana.lib.ann import ANN
from eliana.lib.data_loader import DataLoader

import os
import tensorflow as tf


class Eliana:

    def __init__(self):
        self.data_loader = DataLoader()
        self.img = self.data_loader.images()[0]

    def annotate(self):
        (model,
            file_ckpt,
            file_label,
            num_classes,
            sess,
            detection_graph) = self.data_loader.annotator()

        annotator = Annotator(
            self.img,
            model,
            file_ckpt,
            file_label,
            num_classes,
            sess,
            detection_graph
        )

        result = annotator.annotate()
        for set_ in result:

            _, cropped, tag_id, annotation = set_

            self.img.objects.append(ElianaImageObject(
                parent=self.img,
                cropped=cropped,
                annotation=annotation,
                tag_id=tag_id
            ))

    def color(self):
        self.img.colorfulness = Color.colorfulness(self.img)

    def texture(self):
        self.img.texture = Texture(self.img).get_texture_mean()

    def interpolate(self, value):
        return value * 0.001

    def run(self):

        self.annotate()
        self.color()
        self.texture()

        # print('Objects:', self.img.objects)
        print('Objects')
        for object_ in self.img.objects:
            print(
                #"{0:.2f}".format(score * 100),
                #'% ',
                object_.annotation,
                '\nOriginal: ',
                object_.tag_id,
                '\tInterpolated: ',
                self.interpolate(object_.tag_id),
                sep='',
                end='\n\n'
            )
            object_.show(use='pil')

        print(
            'Colorfulness',
            '\nOriginal:',
            self.img.colorfulness,
            '\nInterpolated:',
            self.interpolate(self.img.colorfulness),
            end='\n\n'
        )
        print(
            'Texture or GLCM Contrast',
            '\nOriginal:',
            self.img.texture,
            '\nInterpolated:',
            self.interpolate(self.img.texture)
        )


Eliana().run()
