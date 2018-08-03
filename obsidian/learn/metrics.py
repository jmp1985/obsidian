'''
Custom metrics for assessing performance of obsidian protein classifier
'''

import keras.backend as K

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
