import tensorflow as tf
import numpy as np
import os
import pickle as pkl
from tqdm import tqdm
from network import Network
import util
import pandas as pd
import argparse

class Train:
    def __init__(self, data_choice, learning_rate):
        self.model = Network(data_choice)
        self.learning_rate = learning_rate
        self.optimizer = tf.optimizers.Adam(self.learning_rate)

    def train_step(self, inputs, labels):
        with tf.GradientTape() as tape:
            predictions = self.model(inputs)
            loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=labels, logits=predictions))
        gradients = tape.gradient(loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))
        return loss

    def train(self, x_train, y_train, x_val, y_val, num_epochs, batch_size):
        for epoch in range(num_epochs):
            for i in range(0, len(x_train), batch_size):
                x_batch = x_train[i: i+batch_size]
                y_batch = y_train[i: i+batch_size]
                loss = self.train_step(x_batch, y_batch)
            val_accuracy = self.test(x_val, y_val, 'Validation')
    
    def test(self, x_test, y_test, split='Test'):
        test_logits = self.model(x_test)
        test_predictions = tf.nn.softmax(test_logits)
        test_accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(test_predictions, axis=1), tf.argmax(y_test, axis=1)), dtype=tf.float32))
        test_accuracy *= 100
        print(f'{split} accuracy: {test_accuracy:.2f}')
        return test_accuracy
    
    def getModel(self):
        return self.model
    
    def savemodel(self, path):
        tf.saved_model.save(self.model, path)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="command line arguments")
    parser.add_argument('-d', '--data_choice', type=str, 
                        help="mnist/fashion/kmnist/cifar10")
    parser.add_argument("-n", "--n_epochs", type=int, help="number of epochs")
    parser.add_argument("-lr", "--lr", type=float, help="learning rate")
    parser.add_argument("-b", "--batch_size", type=int, help="number of epochs")
    args = parser.parse_args()

    (x_train, y_train), (x_val, y_val), (x_test, y_test) = util.loadData(
                                                            args.data_choice)
    classAccuracies = np.zeros([3, 1])
    column_names = []
    target_path = os.path.join("../data/iid", args.data_choice)
    if not os.path.exists(target_path):
        os.mkdir(target_path)
    tqdm.write("training...")
    train = Train(args.data_choice, args.lr)
    train.train(x_train, y_train, x_val, y_val, args.n_epochs, args.batch_size)
    classAccuracies[0, 0] = train.test(x_train, y_train, "Training")
    classAccuracies[0, 0] = train.test(x_val, y_val, "Validation")
    classAccuracies[0, 0] = train.test(x_test, y_test, "Testing")
    # for index in tqdm(range(10)):
    #     #print("Training for class ", index)
    #     train = Train(args.data_choice, args.lr)
    #     trainY = util.getTaskData( y_train, index)
    #     valY = util.getTaskData(y_val, index)
    #     testY = util.getTaskData(y_test, index)
    #     train.train(x_train, trainY, x_val, valY, args.n_epochs, args.batch_size)
    #     classAccuracies[0, index] = train.test(x_train, trainY, "Training")
    #     classAccuracies[1, index] = train.test(x_val, valY, "Validation")
    #     classAccuracies[2, index] = train.test(x_test, testY, "Testing")
    #     column_names.append('class'+str(index))
    #     train.savemodel(os.path.join(target_path, 
    #                                   "trained_model_class_" + str(index)))
    #     print("--------------------------------------\n")

    column_names.append("average")
    row_labels = ['train', 'validation', 'test']
    # classAccuracies[:, -1] = np.mean(classAccuracies[:, :-1], axis=1)
    df = pd.DataFrame(classAccuracies, columns=column_names, index=row_labels)
    df.to_csv(os.path.join(target_path, "class_wise_accuracies.csv"), 
              index=True, sep=",", float_format="%.2f")
    
    with open(os.path.join(target_path, "parameters.txt"), "w") as fp:
        fp.write("learning rate: " + str(args.lr) + "\n")
        fp.write("data choice: " + str(args.data_choice) + "\n")
        fp.write("n_epochs: " + str(args.n_epochs) + "\n")
        fp.write("batch size: " + str(args.batch_size) + "\n")
        fp.write("layers: [input-256-128-10]\n")