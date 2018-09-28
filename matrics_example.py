import numpy as np
import tensorflow as tf
from tensorflow.python.framework import ops

#reset the computation graph to default
ops.reset_default_graph()

#start the session
sess = tf.Session()

#crete metric with diagonal value 1,1,1
identity_metric = tf.diag([1.0, 1.0, 1.0])




#merge all summary in the code
merge = tf.summary.merge_all()

#save computation graph of teh code at folder.
writer = tf.summary.FileWriter("/mnt/theta/github/tutorial_4_all/graphs/matrix_example", graph=sess.graph)