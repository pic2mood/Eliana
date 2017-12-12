"""
.. test:: texture_unit_test.py
    :platform: Linux
    :synopsis: unit test for texture module

.. created:: Nov 19, 2017
.. author:: Raymel Francisco <franciscoraymel@gmail.com>
"""
import os
from tests.eliana_test import ElianaUnitTest
# from tests.unit_test_runner import TesterSub

from eliana.lib.texture import Texture
from eliana.lib.image_batch_loader import ImageBatchLoader


class TextureUnitTest(ElianaUnitTest):

    def __init__(self):
        ElianaUnitTest.__init__(self)

    def __init_test_images(self):

        self.__dir_test_images = os.path.join(
            self.training_data,
            'data',
            'test_images'
        )
        # self.__test_images = [
        #     os.path.join(
        #         self.__dir_test_images, 'image{}.jpg'.format(i)
        #     )
        #     for i in range(1, 3)
        # ]
        # self.__image_size = (12, 8)

        self.__test_images = ImageBatchLoader(
            self.__dir_test_images, limit=2
        ).images

        self.__images = []

        # for img_path in self.__test_images:
        #     img = ElianaImage(path=img_path)
        #     self.__images.append(img)

    def __batch_get_texture(self):

        # for img in self.__images:
        #   texture = Texture(img)

        for img in self.__test_images:

            # texture = Texture(path)
            texture = Texture(img)

            # img_gray = texture.img_gray

            # print('\nShowing converted grayscale image...')
            # Image.fromarray(img_gray).show()

            texture_value = texture.get_texture_value()
            texture_mean = texture.get_texture_mean()

            print('\nGLCM contrast:\n', texture_value,
                  '\nGLCM contrast mean:', texture_mean)

    def run(self):

        self.eliana_log.steps = 3

        self.eliana_log.log('Starting unit test for Texture module')
        print(self.eliana_log.log_ok)

        self.eliana_log.log('Loading test images')
        self.test(self.__init_test_images)
        #
        #
        self.eliana_log.log('Getting texture data')
        self.test(self.__batch_get_texture)


if __name__ == '__main__':
    TextureUnitTest().run()
