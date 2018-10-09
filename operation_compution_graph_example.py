import tensorflow as tf
from tensorflow.python.framework import ops
import numpy as np

#reseting default graph in tensorboard
ops.reset_default_graph()

#crete session with name sess
sess = tf.Session()
x_vals = np.array([1. ,2. ,3. ,4., 5.])
x_data = tf.placeholder(dtype=tf.float32)
m = tf.constant(3.)

prod = tf.multiply(x_data, m)
for x_val in x_vals:
    print(sess.run(prod, feed_dict={x_data: x_val}))
merged = tf.summary.merge_all(key='summaries')
my_write = tf.summary.FileWriter("/mnt/theta/github/tutorial_4_all/graphs/opration_compution_graph_example", graph=sess.graph)


