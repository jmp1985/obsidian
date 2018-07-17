'''
main and experimental program for the machine learning stage of obsidian
'''

import os.path
import pickle
import keras
from keras.models import Sequential
import matplotlib.pyplot as plt

###################################
# import data and classifications #
###################################

classes_dir = 'data/realdata/npy_files/tray2/a5/grid'

# 1d mean traces or processed 2d images as input data
#pickle_in = open('obsidian/datadump/profiles.pickle', 'rb')
pickle_in = open('obsidian/datadump/processed.pickle', 'rb')
inputs = pickle.load(pickle_in)

# classifications
pickle_in = open(os.path.join(classes_dir, 'classifications.pickle'), 'rb')
classes = pickle.load(pickle_in)

data = {name : {'x' : inputs[name], 'y' : classes[name]} for name in inputs.keys() }



###################
# display data    #
###################

demo = list(data.values())[0]['x']
plt.imshow(demo)




###########################
#  build neural network   #
###########################


