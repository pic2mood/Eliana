"""
.. test:: unit_test_runner.py
    :platform: Linux
    :synopsis: unit test runner

.. created:: Dec 8, 2017
.. author:: Raymel Francisco <franciscoraymel@gmail.com>
"""
import unittest
from tests.image_batch_loader_unit_test import ImageBatchLoaderUnitTest
from tests.annotator_unit_test import AnnotatorUnitTest
from tests.color_unit_test import ColorUnitTest
from tests.texture_unit_test import TextureUnitTest

# unittest.TextTestRunner(unittest.TestLoader().discover('./tests', pattern='*unit_test.py'))

class TesterSub(unittest.TestCase):

    def test_batch_loader(self):
        ImageBatchLoaderUnitTest().run()

    def test_annotator(self):
        AnnotatorUnitTest().run()

    def test_color(self):
        ColorUnitTest().run()

    def test_texture(self):
        TextureUnitTest().run()


if __name__ == '__main__':
    unittest.main(verbosity=True)
