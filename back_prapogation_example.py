import numpy as np
import tensorflow as tf

from tensorflow.python.framework import ops

ops.reset_default_graph()

sess = tf.Session()

x_vals = np.random.normal(1, 0.1, 100)
y_vals = np.repeat(10., 100)
x_data = tf.placeholder(shape=[1], dtype=tf.float32)
y_target = tf.placeholder(shape=[1], dtype=tf.float32)
learning_rate = tf.constant(0.02)

A = tf.Variable(tf.random_normal(shape=[1]))

my_output = tf.multiply(x_data, A)

loss = tf.square(my_output - y_target)
tf.summary.scalar('loss', loss)

init = tf.global_variables_initializer()
sess.run(init)

my_opt = tf.train.GradientDescentOptimizer(learning_rate)
train_step = my_opt.minimize(loss)

merge = tf.summary.merge_all()
writer = tf.summary.FileWriter("/mnt/theta/github/tutorial_4_all/graphs/back_prapogation_example", sess.graph)

for i in range(100):
	rand_index = np.random.choice(100)
	rand_x = [x_vals[rand_index]]
	rand_y = [y_vals[rand_index]]
	sess.run(train_step, feed_dict={x_data: rand_x, y_target: rand_y})
	
	if (i+1)%25 == 0:
		print('Step: '+str(i)+'A ='+str(sess.run(A)))
		temp_loss = sess.run(loss, feed_dict={x_data: rand_x, y_target: rand_y})
		print('loss: '+str(temp_loss))
		



