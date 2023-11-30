import tensorflow as tf

class Dense(tf.Module):
    def __init__(self, in_features, out_features, name=None):
        super().__init__(name=name)
        self.W = tf.Variable(tf.random.normal([in_features, out_features], name='W'), trainable=True)
        self.b = tf.Variable(tf.zeros([out_features]), trainable=True, name='b')
    
    def __call__(self, x):
        return tf.matmul(x, self.W) + self.b