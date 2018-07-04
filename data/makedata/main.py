'''
Construct and save bogus diffraction data for obsidian
'''

import matplotlib.pyplot as plt
import numpy as np
from skimage import data 
from skimage.draw import circle_perimeter_aa
from random import randint
from skimage import io as skio

def makeimg(size):
	img = np.ones(size, dtype=np.double)
	cent = int(size/2)
		
	ncircles = randint(1, 50)	
	
	for n in range(ncircles):
		rad = randint(1,500)
		rr, cc, val = circle_perimeter_aa(cent, cent, rad, shape=img.shape)
		img[rr, cc] = val
	
	return img

def main():

	nimages = 5
	size = (500, 500)

	for n in range(nimages):
		makeimg(size)		
	
	plt.imshow(img, cmap='gray', interpolation='nearest')
	plt.show()

main()
