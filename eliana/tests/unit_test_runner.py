"""
.. test:: unit_test_runner.py
    :platform: Linux
    :synopsis: unit test runner

.. created:: Dec 8, 2017
.. author:: Raymel Francisco <franciscoraymel@gmail.com>
"""
import os
import unittest
import logging

from eliana.lib.annotator import Annotator
from eliana.lib.color import Color
from eliana.lib.texture import Texture

from eliana.utils import (image_batch_loader, interpolate)


logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

handler = logging.StreamHandler()
formatter = logging.Formatter(
    '\n[%(asctime)s] %(name)s: %(levelname)s: %(message)s'
)
handler.setFormatter(formatter)

logger.addHandler(handler)


dir_images = os.path.join(
    os.getcwd(),
    'training',
    'data',
    'test_images'
)

img = image_batch_loader(dir_=dir_images, limit=1)


class UnitTest(unittest.TestCase):

    def test_annotator(self):

        logger.info('Testing Annotator module...')
        try:
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
            annotator.annotate(img)

        except Exception as e:
            logger.debug(str(e))

        finally:
            logger.info('DONE.')

    def test_color(self):

        logger.info('Testing Color module...')
        try:
            color = Color.colorfulness(img)
            color = Color.scaled_colorfulness(color)
            color = interpolate(color)

        except Exception as e:
            logger.debug(str(e))

        finally:
            logger.info('DONE.')

    def test_texture(self):

        logger.info('Testing Texture module...')
        try:
            texture = Texture.texture(img)
            texture = interpolate(texture, place=0.1)

        except Exception as e:
            logger.debug(str(e))

        finally:
            logger.info('DONE.')


if __name__ == '__main__':
    unittest.main(verbosity=True)
