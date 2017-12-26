"""
.. test:: unit_test_runner.py
    :platform: Linux
    :synopsis: unit test runner

.. created:: Dec 8, 2017
.. author:: Raymel Francisco <franciscoraymel@gmail.com>
"""
import unittest
import traceback

from eliana.imports import *


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
            annotator = Annotator(
                model=annotator_params['model'],
                ckpt=annotator_params['ckpt'],
                labels=annotator_params['labels'],
                classes=annotator_params['classes']
            )
            annotator.annotate(img)

        except Exception as e:
            logger.error(traceback.format_exc())

        finally:
            logger.info('DONE.')

    def test_color(self):

        logger.info('Testing Color module...')
        try:
            color = Color.colorfulness(img)
            color = Color.scaled_colorfulness(color)
            color = interpolate(color)

        except Exception as e:
            logger.error(traceback.format_exc())

        finally:
            logger.info('DONE.')

    def test_texture(self):

        logger.info('Testing Texture module...')
        try:
            texture = Texture.texture(img)
            texture = interpolate(texture, place=0.1)

        except Exception as e:
            logger.error(traceback.format_exc())

        finally:
            logger.info('DONE.')


if __name__ == '__main__':
    unittest.main(verbosity=True)
