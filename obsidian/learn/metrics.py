'''
Custom metrics for assessing performance of obsidian protein classifier
'''

import keras.backend as K
from theano.tensor import basic as T
from theano.tensor import nnet, clip

def precision(y_true, y_pred):
  '''
  Returns batch-wise average of precision.
  Precision is a metric of how many selected items are relevant, corresponding
  to sum(true positives)/sum(predicted positives)
  (see: https://en.wikipedia.org/wiki/Confusion_matrix)
  '''
  true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
  predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
  precision = true_positives / (predicted_positives + K.epsilon())
  return precision

def weighted_binary_crossentropy(weight, from_logits=False):
  
  def my_loss(target, output):
    # Not entirely sure what this is for but it was in the Keras backend method
    if from_logits:
      output = nnet.sigmoid(output)
    output = clip(output, K.epsilon(), 1.0 - K.epsilon())
    # Modified log loss equation with weight for target positive 
    return -(weight * target * T.log(output) + (1.0 - target) * T.log(1.0 - output))
  
  return my_loss
  
