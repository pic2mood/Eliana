
# texture implementation in script mode

import os
import numpy as np

#import skimage
from skimage.feature import greycomatrix, greycoprops
from PIL import Image

import scipy.ndimage
from matplotlib import pyplot as plt

from skimage import color, io
from scipy.misc import imshow, toimage
# from lib.image.eliana_image import ElianaImage

import imp
eliana_image = imp.load_source('eliana_image', './eliana/lib/eliana_image.py')

dir_working = os.getcwd()

__dir_env_modules = os.path.join(
    dir_working,
    'env',
    'eliana',
    'lib',
    'python3.6',
    'site-packages'
)

__dir_test_image = os.path.join(
    __dir_env_modules,
    'object_detection',
    'test_images',
    'image1.jpg'
)

# img = eliana_image.ElianaImage(path=__dir_test_image)

# img = np.array(Image.open(__dir_test_image).convert('L'))

# img = eliana_image.ElianaImage(pil=img.as_pil.convert('L'))
# img = img.as_numpy

# print(img)

# img = img.transpose(2, 0, 1).reshape(-1, img.shape[1])

# eliana_image.ElianaImage(np=img).show(use='plt')

img = scipy.ndimage.imread(__dir_test_image, mode='L')

# img = io.imread(__dir_test_image, as_grey=True)

# plt.figure(figsize=(12, 9))
# plt.gray()
# plt.imshow(img)
# plt.show()

# imshow(img)
# toimage(img).show()

# Image.fromarray(img).show()

print(img)

# result = greycomatrix(img, distances=[1], angles=[0], levels=256, normed=True, symmetric=False)
result = greycomatrix(img, [1, 2], [0], 256, normed=True, symmetric=True)

# print(result)

result = np.round(result, 3)
print(result)
contrast = greycoprops(result, 'contrast')
print('GLCM contrast:', contrast)
