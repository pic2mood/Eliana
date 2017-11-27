"""
.. module:: image_batch_loader
    :synopsis: batch loader for images

.. moduleauthor:: Raymel Francisco <franciscoraymel@gmail.com>
.. created:: Nov 27, 2017
"""

# from PIL import Image
from eliana.lib.eliana_image import ElianaImage
import os
from glob import glob


class ImageBatchLoader:

    def __init__(self, _dir):

        self.dir = _dir

        print(self.dir)

        self.img_paths = []

        dir_glob = glob(os.path.join(self.dir, '*.jpg'))

        for img in dir_glob:
            print(img)
            self.img_paths.append(img)

        for img in self.img_paths:
            print(img)

    def __init_images(self):
        pass


dir_working = os.getcwd()
training_dir = os.path.join(
    dir_working,
    'training',
    'data',
    'test_images'
)

ImageBatchLoader(training_dir)
