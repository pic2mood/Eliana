"""
.. module:: utils
    :synopsis: utility functions module

.. moduleauthor:: Raymel Francisco <franciscoraymel@gmail.com>
.. created:: Dec 26, 2017
"""
import pandas as pd
from PIL import Image
from glob import glob
from skimage import io, color
from scipy.misc import imresize
import numpy as np

from eliana.imports import *


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

    logger_.info('Test images dir: ' + dir_)

    dir_glob = sorted(glob(os.path.join(dir_, '*.jpg')))

    for img_path in dir_glob[:limit]:

        img = image_single_loader(img_path)
        yield img, img_path


def interpolate(value, place=0.01):
    return float(format(value * place, '.3f'))


def show(img):
    Image.fromarray(img).show()


@log('Initializing training...')
def train(model, dataset, inputs):

    df = pd.read_pickle(dataset)

    mlp = MLP()
    mlp.load_model(path=None)

    inputs = df[inputs]
    outputs = df[['Emotion Value']].as_matrix().ravel()

    # logger_.info('Training fitness: ' + str(mlp.train(inputs, outputs)))
    mlp.train(inputs, outputs)
    mlp.save_model(path=model)


def build_training_data(
    dir_images,
    dataset,
    tag,
    columns,
    append=False,
    mode='oia'
):

    # prepare object annotator
    annotator = Annotator(
        model=annotator_params['model'],
        ckpt=annotator_params['ckpt'],
        labels=annotator_params['labels'],
        classes=annotator_params['classes']
    )

    # data building
    data = []
    for i, (img, img_path) in enumerate(
        image_batch_loader(dir_=dir_images, limit=None)
    ):

        print(img_path)
        objects = annotator.annotate(img)
        objects.sort(key=lambda obj: obj[1].shape[0] * obj[1].shape[1])

        # print(objects)
        # for obj in objects:
        #     show(obj[1])

        palette_1, palette_2, palette_3 = Palette.dominant_colors(img)

        palette_1 = interpolate(palette_1, place=0.000000001)
        palette_2 = interpolate(palette_2, place=0.000000001)
        palette_3 = interpolate(palette_3, place=0.000000001)

        color = Color.scaled_colorfulness(img)
        color = interpolate(color)

        texture = Texture.texture(img)
        texture = interpolate(texture, place=0.1)

        if mode == 'overall':
            data.append(
                [
                    img_path.split('/')[-1],
                    palette_1,
                    palette_2,
                    palette_3,
                    color,
                    texture,
                    tag,
                    emotions_map[tag],
                    objects
                ]
            )
        elif mode == 'oia':
            data.append(
                [
                    img_path.split('/')[-1],
                    palette_1,
                    palette_2,
                    palette_3,
                    color,
                    texture,
                    0. if not objects else objects[0][2],
                    tag,
                    emotions_map[tag]
                ]
            )

    if append:
        df = pd.read_pickle(dataset)
        df2 = pd.DataFrame(
            data,
            columns=columns
        )
        df = df.append(df2, ignore_index=True)
    else:
        df = pd.DataFrame(
            data,
            columns=columns
        )
    logger_.debug('Dataset:\n' + str(df))
    df.to_pickle(dataset)
