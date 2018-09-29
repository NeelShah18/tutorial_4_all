import numpy as np
import tensorflow as tf
from tensorflow.python.framework import ops

#reset the computation graph to default
ops.reset_default_graph()

#start the session
sess = tf.Session()

#crete metric with diagonal value 1,1,1
identity_metric = tf.diag([1.0, 1.0, 1.0])


#crete materic 2*3 with all 5
C = tf.fill([3,3], 5.0)
F = tf.fill([3,3], 3.0)

#converting np.array to tensor matric
D = tf.convert_to_tensor(np.array([[1.0, 2.0, 3.0], [2.0, 3.0, 4.0], [4.0, 5.0, 6.0]]))

#Matric operaitons

#addition and subtraction
sess.run(C+F)
sess.run(C*F)

#matric multiplication
sess.run(tf.matmul(C, F))

#metric transpose
sess.run(tf.transpose(D))

#matric determinant
sess.run(tf.matrix_determinant(D))

#metric inverse
sess.run(tf.matrix_inverse(D))

#eighen values and vectors
sess.run(tf.self_adjoint_eig(D))

#merge all summary in the code
merge = tf.summary.merge_all()

#save computation graph of teh code at folder.
writer = tf.summary.FileWriter("/mnt/theta/github/tutorial_4_all/graphs/matrix_example", graph=sess.graph)