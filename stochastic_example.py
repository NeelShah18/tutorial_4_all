import numpy as np
import tensorflow as tf

from tensorflow.python.framework import ops

ops.reset_default_graph()

sess = tf.Session()

x_vals = np.random.normal(1, 0., 100)
y_vals = np.repeat(10., 100)

x_data = tf.placeholder(shape=[1], dtype=tf.float32)
y_target = tf.placeholder(shape=[1], dtype=tf.float32)


#crete A and model
A = tf.Variable(tf.random_normal(shape=[1]))
my_output = tf.multiply(x_data, A)

#loss: l2
loss = tf.square(my_output - y_target)

#optimizaton: here 0.02 is learning rate
my_optimizer = tf.train.GradientDescentOptimizer(0.02)
train_step = my_optimizer.minimize(loss)

#Initialize all variables
init = tf.global_variables_initializer()
sess.run(init)

#Train mdoel
loss_stochastic = []

for i in range(100):
	rand_index = np.random.choice(100)
	x_rand = np.transpose([x_vals[rand_index]])
	y_rand = np.transpose([y_vals[rand_index]])
	sess.run(train_step, feed_dict={x_data: x_rand, y_target: y_rand})
	if (i+1)%5==0:
		print('step #:'+str(i)+'A= '+str(sess.run(A)))
		temp_loss = sess.run(loss, feed_dict={x_data: x_rand, y_target: y_rand})
		print("Loss: "+str(temp_loss))
		loss_stochastic.append(temp_loss)
