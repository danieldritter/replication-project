import numpy as np
import tensorflow as tf
from tensorflow.keras import Model
from SL.encoder import block

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

        # creating blocks
        self.board_blocks = [block.Block(35) for i in range(self.num_board_blocks)]
        self.order_blocks = [block.Block(40) for i in range(self.num_order_blocks)]

    def call(self, state_inputs, order_inputs, power_season):
        '''
        Call method for encoder

        Args:
        state_inputs - the board state embeddings
        order_inputs - the previous order input embeddings
        power_season - tuple containing current power and season of game (power,season)

        Returns:
        The concatenated encoded output of the board inputs and the previous order inputs
        '''

        # applying board state layers
        board_out = state_inputs
        for i in range(self.num_board_blocks):
            board_out = self.board_blocks[i].call(board_out, power_season)

        # applying previous order layers
        order_out = order_inputs
        for i in range(self.num_order_blocks):
            order_out = self.order_blocks[i].call(order_out, power_season)

        print(board_out)
        print(order_out)

        return tf.concat(board_out, order_out)
