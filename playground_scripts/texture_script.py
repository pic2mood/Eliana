
# texture implementation in script mode

import os
import numpy as np

#import skimage
from skimage.feature import greycomatrix, greycoprops

#from lib.image.eliana_image import ElianaImage

from lib.image.eliana_image import ElianaImage

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

img = ElianaImage(__dir_test_image).as_numpy
result = greycomatrix(img, [1, 2], [0], 4,
                        normed=True, symmetric=True)

result = np.round(result, 3)
contrast  = greycoprops(result, 'contrast')
print(contrast)
