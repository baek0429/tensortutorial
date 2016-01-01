import numpy as np
import tensorflow as tf

dataX = np.array([[1,2,3],[99,1,3],[1,30,1]])
dataY = np.array([[1],[50],[3]])

print dataX
print dataY

x = tf.placeholder("float",[None,3], name="x")
y = tf.placeholder("float",[None,1], name="y")

W = tf.Variable(tf.zeros([3,1]))
b = tf.Variable(tf.zeros([1]))

y_model = tf.matmul(x,W) + b

entropy = tf.pow(y_model - y,2)

train_step = tf.train.GradientDescentOptimizer(0.01).minimize(entropy)

init = tf.initialize_all_variables()
sess = tf.Session()
sess.run(init)

sess.run(train_step,feed_dict={x:dataX, y:dataY})

print(sess.run(W))
print(sess.run(b))
