import numpy as np
import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.optimizers import Adam

import process
from SL.encoder.encoder import Encoder
from constants.constants import SEASON, UNIT_POWER

class SL_model(Model):
    '''
    The supervised learning model for the Diplomacy game
    '''

    def __init__(self, num_board_blocks, num_order_blocks):
        '''
        Initialization for Encoder Model

        Args:
        num_board_blocks - number of blocks for encoding the board state
        num_order_blocks - number of blocks for encoding previous orders
        '''

        super(Model, self).__init__()
        self.encoder = Encoder(num_board_blocks, num_order_blocks)

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
        
        enc_embedding = self.encoder.call(state_inputs, order_inputs, power_season)
        print(enc_embedding)

if __name__ == "__main__":
    state_inputs, prev_order_inputs, season_names = process.get_data("data/standard_no_press.jsonl")
    model = SL_model(16, 16)
    # Looping through each game

    for i in range(len(state_inputs)):
        # Parsing just the season(not year)
        # Not sure about these conversions

        powers_seasons = []
        for j in range(len(season_names[i])):
            print(season_names[i][j][0])
            powers_seasons.append(SEASON[season_names[i][j][0]] + UNIT_POWER["AUSTRIA"])
        print(powers_seasons)
        powers_seasons = tf.convert_to_tensor(powers_seasons,dtype=tf.float32)
        input = tf.convert_to_tensor(state_inputs[i],dtype=tf.float32)
        model.call(input, prev_order_inputs[i],powers_seasons)
