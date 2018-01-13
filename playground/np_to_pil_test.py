
import os
from PIL import Image

import scipy.ndimage
import numpy as np

img = os.path.join(
    os.getcwd(),
    'training',
    'data',
    'test_images',
    'img1.jpg'
)

img = scipy.ndimage.imread(img)

img = Image.fromarray(np.uint8(img))
img.show()
