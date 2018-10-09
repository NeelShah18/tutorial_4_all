import numpy as np
import tensorflow as tf

from tensorflow.python.framework import ops

ops.reset_default_graph()

#starting tensorflow session
sess = tf.Session()

#dummy data
x_vals = tf.linspace(-1., 1., 500)
target = tf.constant(0.)

#l2 loss: The L2 loss is one of the most common regression loss functions. Here we show how to create it in TensorFlow and we evaluate it for plotting later.
l2_y_vals = tf.square(target - x_vals)
l2_y_out = sess.run(l2_y_vals)

#L1 loss: An alternative loss function to consider is the L1 loss. This is very similar to L2 except that we take the absolute value of the difference instead of squaring it.
l1_y_vals = tf.abs(target-x_vals)
l1_y_out = sess.run(l1_y_vals)

#pseudo-huber loss:The psuedo-huber loss function is a smooth approximation to the L1 loss as the (predicted - target) values get larger. When the predicted values are close to the target, the pseudo-huber loss behaves similar to the L2 loss.
delta1 = tf.constant(0.25)
phuber_y_vals = tf.multiply(tf.square(delta1), tf.square(1. + tf.square((target - x_vals)/delta1))-1.)
phuber_y_out = sess.run(phuber_y_vals)

#categorial prediction
# Various predicted X values
x_vals = tf.linspace(-3., 5., 500)

# Target of 1.0
target = tf.constant(1.)
targets = tf.fill([500,], 1.)

#hinge loss: The hinge loss is useful for categorical predictions. Here is is the max(0, 1-(pred*actual)).
hinge_y_vals = tf.maximum(0., 1. - tf.multiply(target, x_vals))
hinge_y_out = sess.run(hinge_y_vals)

#cross entropy loss: The cross entropy loss is a very popular way to measure the loss between categorical targets and output model logits. 
cross_y_vals = - tf.multiply(target, tf.log(x_vals)) - tf.multiply((1. - target), tf.log(1. - x_vals))
cross_y_out = sess.run(cross_y_vals)

#sigmoid entropy loss: TensorFlow also has a sigmoid-entropy loss function. This is very similar to the above cross-entropy function except that we take the sigmoid of the predictions in the function.
x_val_input = tf.expand_dims(x_vals, 1)
target_input = tf.expand_dims(targets, 1)
sigmoid_loss = tf.nn.sigmoid_cross_entropy_with_logits(logits=x_val_input,labels=target_input)
sigmoid_loss_out = sess.run(sigmoid_loss)

#spftmax loss: Tensorflow also has a similar function to the sigmoid cross entropy loss function above, but we take the softmax of the actuals and weight the predicted output instead.
weight = tf.constant(0.5)
softmax_loss = tf.nn.weighted_cross_entropy_with_logits(logits=x_vals, targets=targets, pos_weight=weight)
softmax_loss_out = sess.run(softmax_loss)
