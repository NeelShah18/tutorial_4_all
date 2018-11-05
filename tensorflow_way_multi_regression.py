import numpy as np
import tensorflow as tf

from sklearn import datasets
from tensorflow.python.framework import ops

#Generate session for tensorflow
sess = tf.Session()

#load dataset 
iris = datasets.load_iris()


