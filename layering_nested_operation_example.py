import numpy as np
import tensorflow as tf

from tensorflow.python.framework import ops

ops.reset_default_graph()

sess = tf.Session()

my_array = np.array([[-1., 2., 3., 0., -6.],
					[-2., -3., 4., 5., 1.],
					[4., 3., 0., -1., 3]])

x_vals = np.array([my_array, my_array+1])
x_data = tf.placeholder(tf.float32, shape=(3,5))
m1 = tf.constant([[1.], [-2.], [4.], [-3.], [0.]])
m2 = tf.constant([[2.]])
a1 = tf.constant([[10.]])

prod1 = tf.matmul(x_data, m1)
prod2 = tf.matmul(prod1, m2)
add1 = tf.add(prod2, a1)

for x_val in x_vals:
	sess.run(add1, feed_dict={x_data: x_val})

merged = tf.summary.merge_all(key="summries")
my_writer = tf.summary.FileWriter("/mnt/theta/github/tutorial_4_all/graphs/layering_nasted_operation_example", sess.graph)
