import tensorflow as tf
from tensorflow.keras import Model

from SL.encoder.encoder import Encoder
from SL.decoder.decoder import Decoder

class AbstractActor(Model):
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

        super(AbstractActor, self).__init__()

        # creating encoder and decoder networks
        self.power = power
        self.encoder = Encoder(num_board_blocks, num_order_blocks)
        self.decoder = Decoder(power)

    def call(self, state_inputs, order_inputs, power_season, season_input, board_dict):
        '''
        Function to run the SL model

        Keyword Args:
        state_inputs - the board state inputs
        order_inputs - the previous order inputs
        power_season - the power and season to be used in film
        season_input - the names of the seasons to be used in creating the mask
        board_dict - the board state dictionary representation

        Returns:
        a probability distribution over valid orders
        '''

        # casting inputs to float32
        state_inputs = tf.cast(state_inputs, tf.float32)
        order_inputs = tf.cast(order_inputs, tf.float32)
        enc_out = self.encoder.call(state_inputs, order_inputs, power_season)

        # extracting positions and masks to use in decoder
        pos_list, masks = self.decoder.create_pos_masks(state_inputs, season_input, board_dict) 
        dec_out = self.decoder.call(state_inputs,enc_out, pos_list, masks)
        return dec_out

    def loss(self, probs, labels):
        '''
        Function to compute the loss of the Actor

        Keyword Args:
        probs - the probability distribution output over the orders
        labels - the labels representing the actions taken

        Return:
        loss for the actions taken
        '''
        raise NotImplementedError("Not implemented in abstract class.")

    def get_orders(self, game, power_names):
        """
        See diplomacy_research.players.player.Player.get_orders
        :param game: Game object
        :param power_names: A list of power names we are playing, or alternatively a single power name.
        :return: One of the following:
                1) If power_name is a string and with_draw == False (or is not set):
                    - A list of orders the power should play
                2) If power_name is a list and with_draw == False (or is not set):
                    - A list of list, which contains orders for each power
                3) If power_name is a string and with_draw == True:
                    - A tuple of 1) the list of orders for the power, 2) a boolean to accept a draw or not
                4) If power_name is a list and with_draw == True:
                    - A list of tuples, each tuple having the list of orders and the draw boolean
        """
        raise NotImplementedError("TODO!")
