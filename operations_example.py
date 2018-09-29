import tensorflow as tf
import numpy as np
from tensorflow.python.framework import ops

#reset the default graph in tensorboard
ops.reset_default_graph()

#start the tensorflow session
sess = tf.Session()


#integer division
print(sess.run(tf.div(3, 4)))

#true division
print(sess.run(tf.truediv(3, 4)))

#floor division
print(sess.run(tf.floordiv(3, 4)))

#mod dunction
print(sess.run(tf.mod(21.0, 5.0)))

#cross fucntion
print(sess.run(tf.cross([1.0, 0.0, 0.0], [0.0, 1.0, 0.0])))

#sin function
print(sess.run(tf.sin(3.14)))

#cos function
print(sess.run(tf.cos(3.14)))

#tan function
print(sess.run(tf.div(tf.sin(2.0), tf.cos(4.0))))

#coustum openation example equation = 3x^2 -x + 10

def custome_polynomial(x):
	return tf.subtract(3*tf.square(x), x)+ 10

print(sess.run(custome_polynomial(10)))

#solving list of values with custome_polynomial
test_val = range(15)
output = [3*x*x - x + 10 for x in test_val]
print(output)

