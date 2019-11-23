import numpy as np
import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.losses import sparse_categorical_crossentropy
from tensorflow.keras.optimizers import Adam

from data import process
from SL.encoder.encoder import Encoder
from SL.decoder.decoder import Decoder
from constants.constants import SEASON, UNIT_POWER, ORDER_DICT

class SL_model(Model):
    '''
    The supervised learning model for the Diplomacy game
    '''

    def __init__(self, num_board_blocks, num_order_blocks, power):
        '''
        Initialization for Encoder Model

        Args:
        num_board_blocks - number of blocks for encoding the board state
        num_order_blocks - number of blocks for encoding previous orders
        '''

        super(Model, self).__init__()

        # creating encoder and decoder networks
        self.encoder = Encoder(num_board_blocks, num_order_blocks)
        self.decoder = Decoder(power)

    def call(self, state_inputs, order_inputs, power_season):
        '''
        Function to run the SL model

        Keyword Args:
        state_inputs - the board state inputs
        order_inputs - the previous order inputs
        power_season - the power and season to be used in film

        Returns:
        a probability distribution over valid orders
        '''

        # casting inputs to float32
        state_inputs = tf.cast(state_inputs, tf.float32)
        order_inputs = tf.cast(order_inputs, tf.float32)

        enc_out = self.encoder.call(state_inputs, order_inputs, power_season)
        dec_out = self.decoder.call(state_inputs,enc_out, None)
        return dec_out

    def loss(self, probs, labels):
        '''
        Function to compute the loss of the SL Model

        Keyword Args:
        probs - the probability distribution output over the orders
        labels - the labels representing the actions taken

        Return:
        the crossentropy loss for the actions taken
        '''
        return tf.reduce_sum(sparse_categorical_crossentropy(labels, probs))

if __name__ == "__main__":
    # TODO: rename to train_SL()
    # retrieving data
    state_inputs, prev_order_inputs, prev_orders_game_labels, season_names, supply_center_owners = process.get_data("data/standard_no_press.jsonl")

    # initializing supervised learning model and optimizer
    model = SL_model(16, 16, "AUSTRIA")
    optimizer = Adam(0.001)

    # Looping through each game
    for i in range(len(state_inputs)):
        # Parsing just the season(not year)
        # Not sure about these conversions
        powers_seasons = []

        # extracting seasons and powers for film
        for j in range(len(season_names[i])):
            # print(season_names[i][j][0])
            powers_seasons.append(SEASON[season_names[i][j][0]] + UNIT_POWER["AUSTRIA"])
        # print(powers_seasons)

        # Getting labels for game
        labels = []
        for phase in prev_orders_game_labels[i]:

            order_indices = [ORDER_DICT[order] for order in phase["AUSTRIA"]]
            one_hots = tf.one_hot(order_indices,depth=13042)
            labels.append(one_hots)
        # casting to floats
        powers_seasons = tf.convert_to_tensor(powers_seasons,dtype=tf.float32)
        state_input = tf.convert_to_tensor(state_inputs[i],dtype=tf.float32)
        order_inputs = tf.convert_to_tensor(prev_order_inputs[i], dtype=tf.float32)
        with tf.GradientTape() as tape:
            # applying SL model
            orders, orders_probs = model.call(state_input, order_inputs, powers_seasons)
            print(orders_probs[2][0].shape)
            print(len(labels[2]))
            loss = model.loss(orders_probs, labels)
        # optimizing
        # gradients = tape.gradient(loss, model.trainable_variables)
        # optimizer.apply_gradients(zip(grads, model.trainable_variables))
