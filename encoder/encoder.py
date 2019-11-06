import numpy as np
import tensorflow as tf
from tf.keras import Model
from tf.keras.optimizers import Adam

class Encoder(Model):
    '''
    Encoder Model
    '''

    def __init__(self, num_board_blocks, num_order_blocks):
        '''
        Initialization for Encoder Model

        Args:
        num_board_blocks - number of blocks for encoding the board state
        num_order_blocks - number of blocks for encoding previous orders
        '''

        super(Model, self).__init__()
        self.num_board_blocks = num_board_blocks
        self.num_order_blocks = num_order_blocks
        self.optimizer = Adam(0.001)

        # creating blocks
        self.board_blocks = []
        self.order_blocks = []

    def call(self, board_inputs, order_inputs):
        '''
        Call method for encoder

        Args:
        board_inputs - the board state embeddings
        order_inputs - the previous order input embeddings

        Returns:
        The concatenated encoded output of the board inputs and the previous order inputs
        '''

        power_season = get_power_season(board_inputs)

        # applying board layers
        board_out = board_inputs
        for i in range(self.num_board_blocks):
            board_out = self.board_blocks[i].call(board_out, power_season)

        # applying board layers
        order_out = order_inputs
        for i in range(self.num_order_blocks):
            order_out = self.order_blocks[i].call(order_out, power_season)

        return (board_out, order_out)

def get_power_season(input):
    '''
    Function to extract the power and season of the input
    '''

    return
