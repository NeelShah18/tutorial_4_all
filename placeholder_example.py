import tensorflow as tf
import numpy as np
from tensorflow.python.framework import ops

#reset the graph
ops.reset_default_graph()

#start the session
sess = tf.Session()

#crete placeholder name x
x = tf.placeholder(tf.float32, shape=(4, 4))

#crete matric with random value 4*4, same size as x
rand_array = np.random.rand(4, 4)

#crete same metrix as x with x values name y
y = tf.identity(x)

#feed_dict use to put the x values in computational graph
sess.run(y, feed_dict={x: rand_array})

#merge all summary fot eh code
merged = tf.summary.merge_all()

#save the graph in folder
writer = tf.summary.FileWriter("/mnt/theta/github/tutorial_4_all/graphs/variable_logs", graph=sess.graph)
