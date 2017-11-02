
import numpy as np
from PIL import Image

class ElianaImage():

	def __init__(self, img):
		self.__load_image_into_numpy_array(img)


	def __load_image_into_numpy_array(self, img):

		(im_width, im_height) = img.size

		self.img = np.array(img.getdata()).reshape((im_height, im_width, 3)).astype(np.uint8)

