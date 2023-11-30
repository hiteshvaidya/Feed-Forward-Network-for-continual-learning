import tensorflow as tf
from layer import Dense

class Network(tf.Module):
    """
    neural network
    """
    def __init__(self, data_choice):
        """
        Constructor:
        Declares all layers present in the network
        """
        if data_choice == "cifar10":
            self.layer1 = Dense(32*32, 256)
        else:
            self.layer1 = Dense(28*28, 256)
        self.layer2 = Dense(256, 128)
        self.layer3 = Dense(128, 10)

    def __call__(self, x):
        """
        implementation when 

        :param x: _description_
        :type x: _type_
        """
        # Flatten input
        # x = tf.reshape(x, (-1, 32*32))
        h1 = tf.nn.relu(self.layer1(x))
        h2 = tf.nn.relu(self.layer2(h1))
        return self.layer3(h2)
