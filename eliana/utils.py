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
from skimage import io, color
from scipy.misc import imresize
import numpy as np

import collections
from imutils import build_montages
from cv2 import putText, FONT_HERSHEY_SIMPLEX

from eliana import config
from eliana.lib.mlp import MLP


def image_single_loader(img_path):
    img = io.imread(img_path)
    img = color.gray2rgb(img)

    w_base = 300
    w_percent = (w_base / float(img.shape[0]))
    h = int((float(img.shape[1]) * float(w_percent)))

    img = imresize(img, (w_base, h))

    return img


# @log('Loading images...')
def image_batch_loader(dir_, limit=None):

    # logger_.info('Test images dir: ' + dir_)
    print('Test images dir: ' + dir_)

    dir_glob = sorted(glob(os.path.join(dir_, '*.jpg')))

    for img_path in dir_glob[:limit]:

        img = image_single_loader(img_path)
        yield img, img_path


def interpolate(value, place=0.01):
    return float(format(value * place, '.3f'))


def show(img):
    Image.fromarray(img).show()


def montage(images):

    max_cols = 6
    rows = int(len(images) / max_cols)

    return build_montages(
        images,
        (180, 180),
        (rows, max_cols)
    )[0]


def put_text(img, text):

    putText(
        img,
        text,
        (40, 40),
        FONT_HERSHEY_SIMPLEX,
        1.4,
        (0, 255, 0),
        3
    )


# @log('Initializing training...')
def train(trainer, inputs):

    df = pd.read_pickle(trainer['dataset'])

    mlp = MLP()
    mlp.load_model(path=None)

    inputs = df[inputs]
    outputs = df[['Emotion Value']].as_matrix().ravel()

    # logger_.info('Training fitness: ' + str(mlp.train(inputs, outputs)))
    mlp.train(inputs, outputs)
    mlp.save_model(path=trainer['model'])


def build_training_data(
    trainer, emotion_combinations=['happiness', 'sadness', 'fear']
):
    # emotion filtering
    emotions = {}
    for em in emotion_combinations:
        emotions[em] = config.emotions_map[em]

    # dataset building
    data = []
    for emotion_str, emotion_val in emotions.items():
        dir_images = os.path.join(trainer['raw_images_root'], emotion_str)

        for i, (img, img_path) in enumerate(
            image_batch_loader(dir_=dir_images, limit=None)
        ):
            datum = [img_path.split('/')[-1]]
            for _, func in trainer['features'].items():

                feature = func(img)

                # if multiple features in one category
                if isinstance(feature, collections.Sequence):
                    for item in feature:
                        datum.append(item)
                else:
                    datum.append(feature)

            datum.extend([emotion_str, emotion_val])

            data.append(datum)

    # dataset saving
    df = pd.DataFrame(
        data,
        columns=trainer['columns']
    )
    config.logger_.debug('Dataset:\n' + str(df))
    df.to_pickle(trainer['dataset'])
