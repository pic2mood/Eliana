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
from eliana.lib.mlp import MLP
from eliana.lib.data_loader import DataLoader

import os
import tensorflow as tf

from sklearn.neural_network import MLPClassifier
from sklearn.externals import joblib
import numpy as np


class Eliana:

    def __init__(self):
        self.data_loader = DataLoader()
        self.images = self.data_loader.images()
        self.annotator_params = self.data_loader.annotator()

    def annotate(self):
        (model,
            file_ckpt,
            file_label,
            num_classes,
            sess,
            detection_graph) = self.annotator_params

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

    def interpolate(self, value, place=0.01):
        return float(format(value * place, '.3f'))

    def train(self):

        self.training_data = ()
        training_input = []
        training_output = []

        emotions = []

        with open(self.data_loader.training(), 'r') as ma:
            for entry in ma:
                entry = entry.strip()
                if entry != '':
                    name, emotion = *entry.split('|'),
                    name = name.split('/')[-1]
                    # print(emotion, name)
                    emotions.append(emotion)

        emotions_map = {
            'happiness': 0.1,
            'anger': 0.2,
            'surprise': 0.3,
            'disgust': 0.4,
            'sadness': 0.5,
            'fear': 0.6
        }

        for i, img in enumerate(self.data_loader.images()):

            print(i, img.path)

            self.img = img
            self.annotate()
            self.color()
            self.texture()

            training_input.append(
                [
                    self.interpolate(img.colorfulness),
                    self.interpolate(self.img.texture, place=0.1)
                ]
            )

            print(emotions[i])
            print([emotions_map[emotions[i]]])

            training_output.append(
                emotions_map[emotions[i]] / 0.1
            )

        self.training_data = training_input, training_output

        print(training_input)
        print(training_output)

        model_path = os.path.join(
            os.getcwd(),
            'training',
            'models',
            'eliana_ann_overall',
            'eliana_ann_overall.pkl'
        )

        mlp = MLP()
        # mlp.load_model(path=model_path)
        mlp.load_model(path=None)
        mlp.train(training_input, training_output)
        # mlp.save_model(path=model_path)

        for img in self.images:
            print(img.path)

            img.colorfulness = Color.scaled_colorfulness(
                Color.colorfulness(img)
            )
            img.colorfulness = self.interpolate(img.colorfulness)

            img.texture = Texture(img).get_texture_mean()
            img.texture = self.interpolate(img.texture, place=0.1)

            print(mlp.run(input_=[img.colorfulness, img.texture]))


e = Eliana()
e.train()
# e.run()
