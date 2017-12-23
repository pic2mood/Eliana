"""
.. module:: image_batch_loader
    :synopsis: batch loader for images

.. moduleauthor:: Raymel Francisco <franciscoraymel@gmail.com>
.. created:: Nov 27, 2017
"""

import os
from glob import glob
from skimage import io

from scipy.misc import imresize

from PIL import Image


class ImageBatchLoader:

    def images(dir_, limit=None):

        print('Test images dir:', dir_)

        dir_glob = sorted(glob(os.path.join(dir_, '*.jpg')))

        for img_path in dir_glob[:limit]:

            img = io.imread(img_path)

            w_base = 300
            w_percent = (w_base / float(img.shape[0]))
            h = int((float(img.shape[1]) * float(w_percent)))
            img = imresize(img, (w_base, h))

            # Image.fromarray(img).show()

            yield img, img_path
