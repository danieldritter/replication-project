import numpy as np
import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.optimizers import Adam

from encoder.encoder import Encoder 
import process

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
    
    def call(self, state_inputs, order_inputs):
        embedding = self.encoder.call(state_inputs, order_inputs)

if __name__ == "__main__":
    state_inputs, prev_order_inputs, season_names = process.get_data("data/standard_no_press.jsonl")

    model = SL_model(16, 16)
    model.call(state_inputs, prev_order_inputs)