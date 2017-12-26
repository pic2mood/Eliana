"""
.. module:: utils
    :synopsis: utility functions module

.. moduleauthor:: Raymel Francisco <franciscoraymel@gmail.com>
.. created:: Dec 26, 2017
"""
import os
import pandas as pd
from PIL import Image
from glob import glob
from skimage import io
from scipy.misc import imresize

from eliana.lib.mlp import MLP
from eliana.lib.annotator import Annotator
from eliana.lib.color import Color
from eliana.lib.texture import Texture


def image_batch_loader(dir_, limit=None):

    print('Test images dir:', dir_)

    dir_glob = sorted(glob(os.path.join(dir_, '*.jpg')))

    for img_path in dir_glob[:limit]:

        img = io.imread(img_path)

        w_base = 300
        w_percent = (w_base / float(img.shape[0]))
        h = int((float(img.shape[1]) * float(w_percent)))
        img = imresize(img, (w_base, h))

        yield img, img_path


def interpolate(value, place=0.01):
    return float(format(value * place, '.3f'))


def show(img):
    Image.fromarray(img).show()


def train(model, dataset):

    df = pd.read_pickle(dataset)

    mlp = MLP()
    mlp.load_model(path=None)

    inputs = df[['Color', 'Texture']]
    outputs = df[['Emotion Value']].as_matrix().ravel()

    print('Trainer:', mlp.train(inputs, outputs))

    mlp.save_model(path=model)


def build_training_data(dir_images, dataset):

    # manual emotion tags
    emotions = []

    dir_et = os.path.join(dir_images, 'manual_annotator.txt')

    with open(dir_et, 'r') as et:
        for entry in et:

            entry = entry.strip()

            if entry != '':
                name, emotion = tuple(entry.split('|'))
                name = name.split('/')[-1]

                emotions.append(emotion)

    emotions_map = {
        'happiness': 1,
        'anger': 2,
        'surprise': 3,
        'disgust': 4,
        'sadness': 5,
        'fear': 6
    }

    # prepare object annotator
    model = 'ssd_mobilenet_v1_coco_11_06_2017'
    file_ckpt = os.path.join(
        os.getcwd(),
        'training',
        'models',
        model,
        'frozen_inference_graph.pb'
    )
    file_label = os.path.join(
        os.getcwd(),
        'training',
        'data',
        'mscoco_label_map.pbtxt'
    )
    annotator = Annotator(
        model=model,
        ckpt=file_ckpt,
        labels=file_label,
        classes=90
    )

    # data building
    data = []
    for i, (img, img_path) in enumerate(
        image_batch_loader(dir_images, limit=None)
    ):

        objects = annotator.annotate(img)

        # print(objects)
        # for o in objects:
        #     show(o[1])

        color = Color.colorfulness(img)
        color = Color.scaled_colorfulness(color)
        color = interpolate(color)

        texture = Texture.texture(img)
        texture = interpolate(texture, place=0.1)

        data.append(
            [
                img_path.split('/')[-1],
                color,
                texture,
                emotions[i],
                emotions_map[emotions[i]],
                objects
            ]
        )

    df = pd.DataFrame(
        data,
        columns=[
            'Image',
            'Color',
            'Texture',
            'Emotion',
            'Emotion Value',
            'Objects'
        ]
    )
    print('Dataset:\n', df)
    df.to_pickle(dataset)
