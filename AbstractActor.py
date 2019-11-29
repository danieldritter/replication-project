import tensorflow as tf
from tensorflow.keras import Model

from SL.encoder.encoder import Encoder
from SL.decoder.decoder import Decoder
from tensorflow.keras.optimizers import Adam
from diplomacy_research.models.state_space import dict_to_flatten_board_state, dict_to_flatten_prev_orders_state, get_current_season, extract_state_proto
from diplomacy import Game
from constants.constants import INVERSE_ORDER_DICT, INT_SEASON, UNIT_POWER_RL, UNIT_POWER
from data.process import parse_rl_state

class AbstractActor(Model):
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

        super(AbstractActor, self).__init__()

        # creating encoder and decoder networks
        self.encoder = Encoder(num_board_blocks, num_order_blocks)
        self.decoder = Decoder()
        self.optimizer = Adam(0.001)

    def call(self, state_inputs, order_inputs, power_season, season_input, board_dict, power):
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
        pos_list, masks = self.decoder.create_pos_masks(state_inputs, season_input, board_dict, power)
        dec_out = self.decoder.call(state_inputs, enc_out, pos_list, masks)
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
        # num_dummies/tiling is hacky way to get around TF Strided Slice error
        # that occurs when only passing in one state (e.g. batch size of 1)
        num_dummies = 2
        order_history = Game.get_phase_history(game)
        if len(order_history) == 0:
            prev_orders_state = tf.zeros((1, 81, 40), dtype=tf.float32)
        else:
            prev_orders_state = dict_to_flatten_prev_orders_state(order_history[-1], game.map)
            prev_orders_state = tf.reshape(prev_orders_state, (1, 81, 40))
        prev_orders__state_with_dummies = tf.tile(prev_orders_state, [num_dummies, 1, 1])
        board_state = dict_to_flatten_board_state(game.get_state(), game.map)
        board_state = tf.reshape(board_state, (1, 81, 35))
        board_state_with_dummies = tf.tile(board_state, [num_dummies, 1, 1])
        season = get_current_season(extract_state_proto(game))
        state = game.get_state()
        year = state["name"]
        board_dict = parse_rl_state(state)
        orders = []
        order_probs = []
        for power in power_names:
            power_name = UNIT_POWER_RL[power]
            power_season = tf.concat([UNIT_POWER[power_name],INT_SEASON[season]],axis=0)
            power_season = tf.expand_dims(power_season,axis=0)
            power_season_with_dummies = tf.tile(power_season, [num_dummies, 1])
            probs, position_list = self.call(board_state_with_dummies,
                                             prev_orders__state_with_dummies,
                                             power_season_with_dummies,
                                             [year for _ in range(num_dummies)],
                                             [board_dict for _ in range(num_dummies)],
                                             power_name)
            order_ix = tf.argmax(probs,axis=1)
            orders.append(INVERSE_ORDER_DICT[order_ix])
            order_probs.append(probs[order_ix])
        return orders,order_probs
