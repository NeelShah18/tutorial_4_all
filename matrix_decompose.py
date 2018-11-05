import tensorflow as tf
import numpy as np

#Creating tensorflow session variable
sess = tf.Session()

#Generating x and y values
x_vals = np.linspace(0, 10, 100)
y_vals = x_vals + np.random.normal(0, 1, 100)

#Generating metrix structure to perform all metrix opearitons
x_val_column = np.transpose(np.matrix(x_vals))
ones_colum = np.transpose(np.matrix(np.repeat(1, 100)))

#We are creatingmetrix like because her :-> multilinear regression equation will be like = B0 +B1.X1 + B2.X2 ...  
'''
A = |1 x1|
	|1 x2|
	|1 x3|
	|1 x4|
'''
A = np.column_stack((x_val_column, ones_colum))
b = np.transpose(np.matrix(y_vals))

#convert metrix into tensor
A_tensor = tf.constant(A)
b_tensor = tf.constant(b)

#Implementing algorithm through maatrix decomposition
tf_A = tf.matmul(tf.transpose(A_tensor), A_tensor)
L = tf.cholesky(tf_A)
tA_B = tf.matmul(tf.transpose(A_tensor), b)
sol = tf.matrix_solve(L, tA_B)
solution = tf.matrix_solve(tf.transpose(L), sol)


#Eveluation of model
solution_evl = sess.run(solution)
slope = solution_evl[0][0]
y_intersept = solution_evl[1][0]

print(slope)
print(y_intersept)
