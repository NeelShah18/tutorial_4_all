#binary classification of iris data : batch data selection
import numpy as np
import tensorflow as tf

from tensorflow.python.framework import ops
from sklearn import datasets

#reset default graph in tensorboard
ops.reset_default_graph()

#load iris dataset
iris = datasets.load_iris()

#convert iris data target to binary foramt
binary_target = np.array([1. if x==0 else 0. for x in iris.target])

#as from decription we see 3 and 4th colum has higest corelation so, we choose those two colum for classification of the data.
iris_2d = np.array([[x[2], x[3]] for x in iris.data])

#select batch size
batch_size = 25

#define leanring rate
learning_rate = 0.05

#start the tensorflow session
sess = tf.Session()

#creating and defining palceholder
x1_data = tf.placeholder(shape=[None, 1], dtype=tf.float32)
x2_data = tf.placeholder(shape=[None, 1], dtype=tf.float32)
y_target = tf.placeholder(shape=[None, 1], dtype=tf.float32)

#creating and defining variable
A = tf.Variable(tf.random_normal(shape=[1, 1]))
b = tf.Variable(tf.random_normal(shape=[1, 1]))

'''
model:
A line can be defined as x1=A.x2 + b. To create a linear separator, we would like to see which side of the line the data points fall. There are three cases:

    A point exactly on the line will satisfy: 0 = x1 - (A.x2 + b)
    A point above the line satisfies: 0 = x1 - (A.x2 + b)
    A point below the line satisfies: 0 = x1 - (A.x2 + b)

We will make the output of this model:
x1 - (A.x2 + b)

Then the predictions will be the sign of that output:
Prediction(x1,x2) = sign(x1 - (A.x2 + b))
'''
my_mult = tf.matmul(x2_data, A)
my_add = tf.add(my_mult, b)
my_output = tf.subtract(x1_data, my_add)

#calculate loss
xentropy = tf.nn.sigmoid_cross_entropy_with_logits(logits=my_output, labels=y_target)

#optimizer: with learning
my_opt = tf.train.GradientDescentOptimizer(learning_rate)
train_step = my_opt.minimize(xentropy)

#Initialize variables
init = tf.global_variables_initializer()
sess.run(init)

#start trainign of the model
for i in range(1000):
    rand_index = np.random.choice(len(iris_2d), size=batch_size)
    #rand_x = np.transpose([iris_2d[rand_index]])
    rand_x = iris_2d[rand_index]
    rand_x1 = np.array([[x[0]] for x in rand_x])
    rand_x2 = np.array([[x[1]] for x in rand_x])
    #rand_y = np.transpose([binary_target[rand_index]])
    rand_y = np.array([[y] for y in binary_target[rand_index]])
    sess.run(train_step, feed_dict={x1_data: rand_x1, x2_data: rand_x2, y_target: rand_y})
    if (i+1)%200==0:
        print('Step #' + str(i+1) + ' A = ' + str(sess.run(A)) + ', b = ' + str(sess.run(b)))



