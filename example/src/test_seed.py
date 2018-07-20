import tensorflow as tf
from segnet_tf import utils_tensorflow as ut


#tf.set_random_seed(0)

weights = ut.weight_xavier(3, 1, 
                           1, name="W", seed=42)

sess = tf.Session()
sess.run(tf.global_variables_initializer())
w = sess.run(weights)
print(w.flatten())