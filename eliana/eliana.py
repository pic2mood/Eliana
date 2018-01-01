"""
.. module:: eliana
    :synopsis: main module for eliana package

.. moduleauthor:: Raymel Francisco <franciscoraymel@gmail.com>
.. created:: Nov 24, 2017
"""
from eliana.imports import *

import cv2
from skimage import io

import collections

from eliana import config
from eliana.lib.mlp import MLP
from eliana.utils import *


class Eliana:

    def __init__(self, trainer):

        self.mlp = MLP()
        self.mlp.load_model(path=trainer['model'])
        self.trainer = trainer

    def run(self, img):

        input_ = []
        for _, func in self.trainer['features'].items():

            feature = func(img)

            # if multiple features in one category
            if isinstance(feature, collections.Sequence):
                for item in feature:
                    input_.append(item)
            else:
                input_.append(feature)

        return self.mlp.run(input_=input_)


enna = Eliana(config.trainer_w_oia)

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

    result = enna.run(img)

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

# montage = build_montages(to_montage, (180, 180), (6, 6))[0]
montage = montage(to_montage)
show(montage)
