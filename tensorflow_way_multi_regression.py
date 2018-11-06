import numpy as np
import tensorflow as tf

from sklearn import datasets
from tensorflow.python.framework import ops

#Generate session for tensorflow
sess = tf.Session()

#load dataset 
iris = datasets.load_iris()

#Take X and Y value
x_vals = np.array([x[3] for x in iris.data])
y_vals = np.array([y[0] for y in iris.data])

#Declare basic variable
learning_rate = 0.05
batch_size = 25

#Placeholder for data in model
x_data = tf.placeholder(shape=[None, 1], dtype=tf.float32)
y_target = tf.placeholder(shape=[None, 1], dtype=tf.float32)

#variables
A = tf.Variable(tf.random_normal(shape=[1, 1]))
B = tf.Variable(tf.random_normal(shape=[1, 1]))

#Formula for linear model
model_output = tf.add(tf.matmul(x_data, A), B)
	
#Initilize L2 loss function
loss = tf.reduce_mean(tf.square(y_target - model_output))
init = tf.global_variables_initializer()

#run the session
sess.run(init)

#Optimizer of the model
my_opt = tf.train.GradientDescentOptimizer(learning_rate)
train_step = my_opt.minimize(loss)


#Train model
loss_vec = []
for i in range(100):
	rand_index = np.random.choice(len(x_vals), size=batch_size)
	rand_x = np.tranpose([x_vals[rand_index]])
	rand_y = np.tranpose([y_vals[rand_index]])
	sess.run(train_step, feed_dict={x_data: rand_x, y_target: rand_y})
	temp_loss = sess.run(loss, feed_dict={x_data: rand_x, y_target: rand_y})
	loss_vec.append(temp_loss)
	if(i+1)%25==0:
		print("Step: "+str(i+1)+" A= "+str(sess.run(A))+" ,B= "+str(sess.run(B))+" ,loss: "+str(temp_loss))













