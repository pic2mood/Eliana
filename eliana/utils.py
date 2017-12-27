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

from eliana.imports import *


# @log('Loading images...')
def image_batch_loader(dir_, limit=None):

    logger_.info('Test images dir: ' + dir_)

    dir_glob = sorted(glob(os.path.join(dir_, '*.jpg')))

    for img_path in dir_glob[:limit]:

        img = io.imread(img_path)
        #if color.is_gray(img):
        img = color.gray2rgb(img)

        w_base = 300
        w_percent = (w_base / float(img.shape[0]))
        h = int((float(img.shape[1]) * float(w_percent)))
        img = imresize(img, (w_base, h))

        yield img, img_path


def interpolate(value, place=0.01):
    return float(format(value * place, '.3f'))


def show(img):
    Image.fromarray(img).show()


@log('Initializing training...')
def train(model, dataset):

    df = pd.read_pickle(dataset)

    mlp = MLP()
    mlp.load_model(path=None)

    inputs = df[['Color', 'Texture']]
    outputs = df[['Emotion Value']].as_matrix().ravel()

    # logger_.info('Training fitness: ' + str(mlp.train(inputs, outputs)))
    mlp.train(inputs, outputs)
    mlp.save_model(path=model)


def build_training_data(dir_images, dataset, tag, append=False):

    # manual emotion tags
    # emotions = []

    # dir_et = os.path.join(dir_images, tags)

    # with open(dir_et, 'r') as et:
    #     for entry in et:

    #         entry = entry.strip()

    #         if entry != '':
    #             name, emotion = tuple(entry.split('|'))
    #             name = name.split('/')[-1]

    #             emotions.append(emotion)

    emotions_map = {
        'happiness': 1,
        'anger': 2,
        'surprise': 3,
        'disgust': 4,
        'sadness': 5,
        'fear': 6
    }

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

        # print(img_path)
        objects = annotator.annotate(img)

        # print(objects)
        # for o in objects:
        #     show(o[1])

        color = Color.scaled_colorfulness(img)
        color = interpolate(color)

        texture = Texture.texture(img)
        texture = interpolate(texture, place=0.1)

        data.append(
            [
                img_path.split('/')[-1],
                color,
                texture,
                tag,
                emotions_map[tag],
                objects
            ]
        )

    if append:
        df = pd.read_pickle(dataset)
        df2 = pd.DataFrame(
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
        df = df.append(df2, ignore_index=True)
        # df.append(data)
        # df = pd.concat(df, df2)
        # logger_.debug('Dataset2:\n' + str(df2))
        # logger_.debug('Dataset:\n' + str(df))
    else:
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
    logger_.debug('Dataset:\n' + str(df))
    df.to_pickle(dataset)
