
# texture implementation in script mode

import os
import numpy as np

#import skimage
from skimage.feature import greycomatrix, greycoprops

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

img = eliana_image.ElianaImage(__dir_test_image).as_numpy

print(img)

img = img.transpose(2, 0, 1).reshape(-1, img.shape[1])

print(img)

result = greycomatrix(img, [1], [0], normed=True, symmetric=True)
# result = greycomatrix(img, [1, 2], [0], 4, normed=True, symmetric=True)

result = np.round(result, 3)
contrast = greycoprops(result, 'contrast')
print('GLCM contrast:', contrast)
