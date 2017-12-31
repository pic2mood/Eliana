"""
.. module:: eliana
    :synopsis: main module for eliana package

.. moduleauthor:: Raymel Francisco <franciscoraymel@gmail.com>
.. created:: Nov 24, 2017
"""
from eliana.imports import *

import cv2
from imutils import build_montages
from skimage import io

import collections

from eliana import config
from eliana.lib.mlp import MLP
from eliana.utils import *


class Eliana:

    def __init__(self, model_path):

        self.mlp = MLP()
        self.mlp.load_model(path=model_path)

    def run_overall(self, img):

        palette_1, palette_2, palette_3 = Palette.dominant_colors(img)

        palette_1 = interpolate(palette_1, place=0.000000001)
        palette_2 = interpolate(palette_2, place=0.000000001)
        palette_3 = interpolate(palette_3, place=0.000000001)

        color = Color.scaled_colorfulness(img)
        color = interpolate(color)

        texture = Texture.texture(img)
        texture = interpolate(texture, place=0.1)

        return self.mlp.run(input_=[
            palette_1,
            palette_2,
            palette_3,
            color,
            texture]
        )

    def run_object(self, img, trainer):

        input_ = []
        for _, func in trainer['features'].items():

            feature = func(img)

            # if multiple features in one category
            if isinstance(feature, collections.Sequence):
                for item in feature:
                    input_.append(item)
            else:
                input_.append(feature)

            # print(input_)

        return self.mlp.run(input_=input_)

        # annotator = Annotator(
        #     model=annotator_params['model'],
        #     ckpt=annotator_params['ckpt'],
        #     labels=annotator_params['labels'],
        #     classes=annotator_params['classes']
        # )

        # objects = annotator.annotate(img)
        # # objects.sort(key=lambda obj: obj[1].shape[0] * obj[1].shape[1])
        # top_object = 0. if not objects else max(objects, key=lambda o: o[1].shape[0] * o[1].shape[1])
        # # print(top_object)


        # palette_1, palette_2, palette_3 = Palette.dominant_colors(img)

        # # palette_1 = interpolate(palette_1, place=0.000000001)
        # # palette_2 = interpolate(palette_2, place=0.000000001)
        # # palette_3 = interpolate(palette_3, place=0.000000001)

        # color = Color.scaled_colorfulness(img)
        # # color = interpolate(color)

        # texture = Texture.texture(img)
        # # texture = interpolate(texture, place=0.1)

        # print([
        #     palette_1,
        #     palette_2,
        #     palette_3,
        #     color,
        #     top_object if top_object == 0 else top_object[2],
        #     texture])

        # return self.mlp.run(input_=[
        #     palette_1,
        #     palette_2,
        #     palette_3,
        #     color,
        #     top_object if top_object == 0 else top_object[2],
        #     texture]
        # )


enna = Eliana(config.trainer_w_oia['model'])

dir_images = os.path.join(
    os.getcwd(),
    config.trainer_w_oia['raw_images_root'],
    'test'
)

to_montage = []

for i, (img, img_path) in enumerate(
    image_batch_loader(dir_=dir_images, limit=None)
):
    print(img_path)

    result = enna.run_object(img, config.trainer_w_oia)

    print('Run:', result)
    cv2.putText(
        img,
        [k for k, v in config.emotions_map.items() if v == result][0],
        (40, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.4,
        (0, 255, 0),
        3
    )

    to_montage.append(img)

montage = build_montages(to_montage, (180, 180), (6, 6))[0]
show(montage)
