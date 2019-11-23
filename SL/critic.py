import numpy as np
import tensorflow as tf
from tensorflow.keras import Model
from constants import constants
from data.process import get_returns

class Critic(Model):
    '''
    Critic Model
    a. Input: a state (e.g. one phase, 81 * 35 vector)
    b. Output: (7,) vector representing the value for each power.
    '''

    def __init__(self):
        '''
        Initialization for Critic Model

        Args:
        '''
        super(Critic, self).__init__()
        self.crit1 = tf.keras.layers.Dense(32, input_shape=(-1, constants.STATE_SIZE), activation="relu")
        self.crit2 = tf.keras.layers.Dense(constants.NUM_POWERS)
        

    def call(self, states):
        '''
        Call method for critic

        Args:
        states - [batch_sz, state_size] vector where state_size = 81*35
        
        Returns:
        [batch_sz, 7] vector of the values for each power in each state.
        '''
        out1 = self.crit1(states)
        values = self.crit2(out1)
        return values


    def loss(self, predicted_values, returns):
        """
        Args:
        predicted values - [batch_sz, 7] vector of predicted values for each power
        returns - [batch_sz, 7] vector of actual values for each power
        
        Returns:
        MSE(predicted, returns)
        """
        return tf.reduce_mean((predicted_values - returns) ** 2)
    
    def train(self, state_inputs, supply_center_owners, num_epochs):
        """

        :param state_inputs: [bs, game_length, 81, 35] (list of list of 2D [81, 35] arrays)
        :param num_epochs:
        :return:
        """
        train_data = state_inputs # shape: [bs, game_length, (81, 35)]
        train_labels = get_returns(supply_center_owners) # shape: [bs, game_length, num_powers]
        # batch_sz = 100
        # num_batches = len(train_data)/batch_sz
        for batch in train_data:
            data_batch = train_data[batch] # data_batch shape: [game_length, (81, 35)]
            labels_batch = train_labels[batch] # labels_batch shape: [game_length, num_powers]
            with tf.GradientTape() as tape:
                predicted_values = self.call(data_batch)
                loss = self.loss(predicted_values, labels_batch)
            gradients = tape.gradient(loss, self.trainable_variables)
            self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))

            index += 1
            if index % 100 == 0:
                print("\tBatch number %d of %d: Loss is %.3f" % (
                index, num_batches, loss))

