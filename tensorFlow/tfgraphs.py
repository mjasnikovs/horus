import tensorflow as tf

n1 = tf.constant(1)
n2 = tf.constant(2)

n3 = n1 + n2

with tf.Session() as sess:
    result = sess.run(n3)

g = tf.Graph()

with g.as_default():
    print(g is tf.get_default_graph())
