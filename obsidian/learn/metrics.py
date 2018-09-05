'''Custom metrics for assessing and training performance of obsidian protein classifier
'''

import keras.backend as K
from theano.tensor import basic as T
from theano.tensor import nnet, clip

def precision(y_true, y_pred):
  '''Returns batch-wise average of precision.
  Precision is a metric of how many selected items are relevant, corresponding
  to sum(true positives)/sum(predicted positives)
  (see: https://en.wikipedia.org/wiki/Confusion_matrix)

  :param y_true: True values
  :param y_pred: Predictied output values
  '''
  true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
  predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
  precision = true_positives / (predicted_positives + K.epsilon())
  return precision

def weighted_binary_crossentropy(weight, from_logits=False):
  '''Custom loss function to bias training towards a class.
  
  :param float weight: A value higher than 1 will bias the model towards positive predictions,
                      a value lower than 1 will bias the model towards negative predictions.
  :returns: weighted loss function
  '''
  
  def weighted_loss(target, output):
    # Not entirely sure what this is for but it was in the Keras backend method
    if from_logits:
      output = nnet.sigmoid(output)
    output = clip(output, K.epsilon(), 1.0 - K.epsilon())
    # Modified log loss equation with weight for target positive 
    return -(weight * target * T.log(output) + (1.0 - target) * T.log(1.0 - output))
  
  return weighted_loss
  
