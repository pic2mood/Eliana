
# color implementation in script mode

import os
import numpy as np
from PIL import Image

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

img = Image.open(__dir_test_image)
(__w, __h) = img.size
img_np = np.array(
    img.getdata()
).reshape(
    (__h, __w, 3)
).astype(
    np.uint8
)
print(img_np)

print('----------------------------------------------------')

print(list(img.getdata())[0])
print(list(img.getdata())[1])

# img_hsv = img.convert('HSV')
# (__w, __h) = img_hsv.size
# img_hsv_np = np.array(
#     img_hsv.getdata()
# ).reshape(
#     (__h, __w, 3)
# ).astype(
#     np.uint8
# )
# print(img_hsv_np)
