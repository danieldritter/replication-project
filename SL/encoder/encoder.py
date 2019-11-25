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
        self.s_size = 60
        self.o_size = 60

        # creating blocks, need first block to change size of layers for residual connection
        self.board_blocks = [block.FirstBlock(self.s_size)] + [block.Block(self.s_size) for i in range(self.num_board_blocks - 1)]
        self.order_blocks = [block.FirstBlock(self.o_size)] + [block.Block(self.o_size) for i in range(self.num_order_blocks - 1)]

    def call(self, state_inputs, order_inputs, power_season):
        '''
        Call method for encoder

        Args:
        state_inputs - the board state embeddings
        order_inputs - the previous order input embeddings
        power_season - tuple containing current power and season of game (power,season)

        Returns:
        The concatenated encoded output of the board inputs and the previous order inputs (num_phases, 81, 120)
        '''

        # applying board state layers
        board_out = state_inputs
        # print(board_out)
        for i in range(self.num_board_blocks):
            board_out = self.board_blocks[i].call(board_out, power_season)

        # applying previous order layers
        order_out = order_inputs
        # print(order_out)
        for i in range(self.num_order_blocks):
            order_out = self.order_blocks[i].call(order_out, power_season)

        return tf.concat((board_out, order_out), axis=2)
