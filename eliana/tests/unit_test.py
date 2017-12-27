"""
.. test:: unit_test_runner.py
    :platform: Linux
    :synopsis: unit test runner

.. created:: Dec 8, 2017
.. author:: Raymel Francisco <franciscoraymel@gmail.com>
"""
import unittest

from eliana.imports import *

img = image_batch_loader(dir_=trainer['test_images'], limit=1)


class UnitTest(unittest.TestCase):

    @log('Testing Annotator module...', verbose=True)
    def test_annotator(self):

        annotator = Annotator(
            model=annotator_params['model'],
            ckpt=annotator_params['ckpt'],
            labels=annotator_params['labels'],
            classes=annotator_params['classes']
        )
        annotator.annotate(img)

    @log('Testing Color module...', verbose=True)
    def test_color(self):

        color = Color.scaled_colorfulness(img)
        color = interpolate(color)

    @log('Testing Texture module...', verbose=True)
    def test_texture(self):

        texture = Texture.texture(img)
        texture = interpolate(texture, place=0.1)


if __name__ == '__main__':
    unittest.main(verbosity=True)
