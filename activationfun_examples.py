import tensorflow as tf
import numpy as np
from tensorflow.python.framework import ops

ops.reset_default_graph()

with tf.Session() as sess:
	#RELU funciton example
	print(sess.run(tf.nn.relu([-3.0, 3.0, 0.0])))
	#relu6 function
	print(sess.run(tf.nn.relu6([-3.0 ,3.0 ,0.0])))
	#sigmoid function
	print(sess.run(tf.nn.sigmoid([-3.0 , 3.0, 0.0])))
	#hyper tangent function
	print(sess.run(tf.nn.tanh([-3.0, 3.0, 0.0])))
	#softsign function
	print(sess.run(tf.nn.softsign([-1.0 , 0.0 , 1.0])))
	#softmax fucntion
	print(sess.run(tf.nn.softmax([-1.0, 0.0, 1.0])))
	#exponetioal linear activation function
	print(sess.run(tf.nn.elu([-1.0, 1.0, 0.0])))

