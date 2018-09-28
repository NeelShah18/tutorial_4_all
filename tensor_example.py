import tensorflow as tf
from tensorflow.python.framework import ops

# Reset graph
ops.reset_default_graph()

# Start a graph session
sess = tf.Session()

# Create variable
my_var = tf.Variable(tf.zeros([1,20]))

# Add summaries to tensorboard
merged = tf.summary.merge_all()

# Initialize graph writer:
writer = tf.summary.FileWriter("/mnt/theta/github/tutorial_4_all/graphs/variable_logs", graph=sess.graph)

# Initialize operation
initialize_op = tf.global_variables_initializer()

# Run initialization of variable
sess.run(initialize_op)

