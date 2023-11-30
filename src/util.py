from tensorflow.keras.utils import to_categorical
# from tensorflow.keras.datasets import cifar10
# from tensorflow.keras.datasets import mnist
# from tensorflow.keras.datasets import fashion_mnist
# from tensorflow.keras.datasets import kmnist
from sklearn.model_selection import train_test_split
import tensorflow_datasets as tfds
import numpy as np

def loadData(choice):
    """
    Load and preprocess data
    """
    if choice == "cifar10":
        # load cifar-10 data
        (x_train, y_train), (x_test, y_test) = tfds.as_numpy(tfds.load('cifar10', split=['train', 'test'], 
        batch_size=-1, as_supervised=True))
        # extract green channel
        x_train = x_train[:, :, :, 1]
        x_test = x_test[:, :, :, 1]
    elif choice == "mnist":
        # load mnist data
        (x_train, y_train), (x_test, y_test) = tfds.as_numpy(tfds.load('mnist', split=['train', 'test'], 
        batch_size=-1, as_supervised=True))
    elif choice == "fashion":
        # load fashion-mnist data
        (x_train, y_train), (x_test, y_test) = tfds.as_numpy(tfds.load('fashion_mnist', split=['train', 'test'], 
        batch_size=-1, as_supervised=True))
    elif choice == "kmnist":
        # load kmnist data
        (x_train, y_train), (x_test, y_test) = tfds.as_numpy(tfds.load('kmnist', split=['train', 'test'], 
        batch_size=-1, as_supervised=True))

    # split train and validation set
    x_train, x_val, y_train, y_val = train_test_split(x_train, y_train,
                                                    test_size=0.2, random_state=42)
    # normalize input images
    x_train = x_train.astype('float32') / 255.0
    x_val = x_val.astype('float32') / 255.0
    x_test = x_test.astype('float32') / 255.0

    # flatten input
    if choice == "cifar10":
        x_train = np.reshape(x_train, [-1, 32*32])
        x_val = np.reshape(x_val, [-1, 32*32])
        x_test = np.reshape(x_test, [-1, 32*32])
    else:
        x_train = np.reshape(x_train, [-1, 28*28])
        x_val = np.reshape(x_val, [-1, 28*28])
        x_test = np.reshape(x_test, [-1, 28*28])

    # One hot encode labels
    y_train = to_categorical(y_train, num_classes=10)
    y_val = to_categorical(y_val, num_classes=10)
    y_test = to_categorical(y_test, num_classes=10)

    return (x_train, y_train), (x_val, y_val), (x_test, y_test)

def getTaskData(y, classNumber, taskSize=1):
    # print("y: ", y[:10, :])
    true_indexes = np.where(y[:, classNumber] == 1)
    classY = np.zeros([y.shape[0], 1])
    classY[true_indexes, :] = 1
    # print("classY: ", classY[:10, :])
    return classY
