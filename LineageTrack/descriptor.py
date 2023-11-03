import mahotas
import cv2 as cv


class ZernikeMoments:
	def __init__(self, radius):
		# store the size of the radius that will be
		# used when computing moments
		self.radius = radius

	def describe(self, image):
		# return the Zernike moments for the image
		return mahotas.features.zernike_moments(image, self.radius)


class HuMoments:
	def __int__(self):
		pass

	def describe(self, image):
		# image = cv.fromarray(image)
		moments = cv.moments(image)
		return cv.HuMoments(moments)
