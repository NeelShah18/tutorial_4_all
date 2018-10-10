import tensorflow as tf
import numpy as np

from tensorflow.python.framework import ops

ops.reset_default_graph()

#Start the session
sess = tf.Session()

#For Batch training, we need to declare our batch size. The larger the batch size, the smoother the convergence will be towards the optimal value. But if the batch size is too large, the optimization algorithm may get stuck in a local minimum, where a more stochastic convergence may jump out.
batch_size = 25

#Genrated data and crete palceholder to input the value in graph
x_vals = np.random.normal(1, 0.1, 100)
y_vals = np.repeat(10., 100)
x_data = tf.placeholder(shape=[None, 1], dtype=tf.float32)
y_target = tf.placeholder(shape=[None, 1], dtype=tf.float32)

A = tf.Variable(tf.random_normal(shape=[1,1]))

#model
my_output = tf.matmul(x_data, A)

#loss function : l2
loss = tf.reduce_mean(tf.square(my_output - y_target))

#initlize variables
init = tf.global_variables_initializer()
sess.run(init)

#optimizerx : 0.02 is learning rate
optimizer = tf.train.GradientDescentOptimizer(0.02)
train_step = optimizer.minimize(loss)

loss_batch = []

#run loop
for i in range(100):
	rand_index = np.random.choice(100, size=batch_size)
	rand_x = np.transpose([x_vals[rand_index]])
	rand_y = np.transpose([y_vals[rand_index]])
	sess.run(train_step, feed_dict={x_data: rand_x, y_target: rand_y})
	if(i+1)%5==0:
		print('step #'+str(i+1)+'A: '+str(sess.run(A)))
		temp_loss = sess.run(loss, feed_dict={x_data: rand_x, y_target: rand_y})
		print('Loss: '+str(temp_loss))
		loss_batch.append(temp_loss)
