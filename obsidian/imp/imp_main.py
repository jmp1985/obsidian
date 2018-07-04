'''
Image processing main class for testing and developing
.. automodule imp_main
.. moduleauthor:: Fiona Young
'''
import numpy as np
from skimage import data
from sbtr_bg import Sbtr_bg
import matplotlib.pyplot as plt

def main():
	
	camera = data.camera()
	print(camera.shape)
	board = data.checkerboard()
	sub = Sbtr_bg(board)
	newimage = sub.subtract(camera)	
	print(newimage.shape)
	plt.imshow(newimage, cmap='gray', interpolation='nearest')
	plt.show()
main()	
