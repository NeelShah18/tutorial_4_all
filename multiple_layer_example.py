import numpy as np
import tensorflow as tf

from tensorflow.python.framework import ops

ops.reset_default_graph()

#start the session
sess = tf.Session()

#crete dummy image
x_shape = [1, 4, 4, 1]
x_val = np.random.uniform(size=x_shape)

#crete data palce holder
x_data = tf.placeholder(tf.float32, shape=x_shape)

#First layer

#Crete filter with 0.25 because we want to average 2*2 window
my_filter = tf.constant(0.25, shape=[2, 2, 1, 1])
#our windows will be 2*2 with stride of 2 for height and weight 
my_strides = [1, 2, 2, 1]
#crete layer that takes spatial moving window and do the average
mov_Avg_layer = tf.nn.conv2d(x_data, my_filter, my_strides, padding="SAME", name="moving_Avg_layer")

#second layer

#define cutomer layer
def custome_layer(input_matrix):
	input_matrix_sqeezed = tf.squeeze(input_matrix)
	A = tf.constant([[1., 2.], [-1., 3.]])
	b = tf.constant(1., shape=[2, 2])
	temp1 = tf.matmul(A, input_matrix_sqeezed)
	temp = tf.add(temp1, b) # Ax+b
	return  (tf.sigmoid(temp))

#Ass custom layer to graph
with tf.name_scope('Custom_layer') as scope:
	custome_layer1 = custome_layer(mov_Avg_layer)

sess.run(mov_Avg_layer, feed_dict={x_data: x_val})
sess.run(custome_layer1, feed_dict={x_data: x_val})

merged = tf.summary.merge_all(key="summeries")
my_writer = tf.summary.FileWriter('/mnt/theta/github/tutorial_4_all/graphs/multiple_layer_example', sess.graph)

