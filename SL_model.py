import numpy as np
import datetime
import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.losses import categorical_crossentropy
from tensorflow.keras.optimizers import Adam

from data import process
from SL.encoder.encoder import Encoder
from SL.decoder.decoder import Decoder
from constants.constants import SEASON, UNIT_POWER, ORDER_DICT
from AbstractActor import AbstractActor

class SL_model(AbstractActor):
    '''
    The supervised learning Actor for the Diplomacy game
    '''

    # @tf.function
    def loss(self, prev_order_phase_labels, probs, position_lists):
        '''
        Function to compute the loss of the SL Model

        Keyword Args:
        labels - the previous order game labels
        probs - the probability distribution output over the orders
        prev_order_phase_labels - the labels for the phases of the game
        position_lists - the list of positions that a power controlled

        Return:
        the crossentropy loss for the actions taken
        '''

        loss = 0
        for i in range(len(prev_order_phase_labels)):
            phase = prev_order_phase_labels[i]
            if self.power in phase.keys() and phase[self.power] != [] and phase[self.power] != None:
                provinces = [order.split()[1] for order in phase[self.power]]

                # labels for province at specific phase
                order_indices = [ORDER_DICT[order] for order in phase[self.power]]
                one_hots = tf.one_hot(order_indices,depth=13042)

                # predictions for province at specific phase
                # print(probs.shape)
                # predictions = tf.gather_nd(probs,zip([i for j in range(len(provinces))],[position_lists.index(province) for province in provinces]))
                predictions = tf.convert_to_tensor([probs[i][position_lists.index(province)] for province in provinces], dtype=tf.float32)
                loss += tf.reduce_mean(categorical_crossentropy(one_hots, predictions))

        loss /= len(prev_order_phase_labels)
        return loss

if __name__ == "__main__":

    # Set up tracking metrics and logging directory
    train_loss = tf.keras.metrics.Mean('train_loss', dtype=tf.float32)
    current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    train_log_dir = 'data/logs/' + current_time + '/train'
    train_summary_writer = tf.summary.create_file_writer(train_log_dir)
    # TODO: rename to train_SL()
    # retrieving data
    state_inputs, prev_order_inputs, prev_orders_game_labels, season_names, supply_center_owners, board_dict_list = process.get_data("data/standard_no_press.jsonl")
    power = "AUSTRIA"
    # initializing supervised learning model and optimizer
    model = SL_model(16, 16, power)
    optimizer = Adam(0.001)
    # Looping through each game
    for i in range(len(state_inputs)):
        # Parsing just the season(not year)
        # Not sure about these conversions
        powers_seasons = []
        curr_board_dict = board_dict_list[i]
        # extracting seasons and powers for film
        for j in range(len(season_names[i])):
            # print(season_names[i][j][0])
            powers_seasons.append(SEASON[season_names[i][j][0]] + UNIT_POWER[power])
        # print(powers_seasons)

        # casting encoder and decoder inputs to floats
        powers_seasons = tf.convert_to_tensor(powers_seasons,dtype=tf.float32)
        state_input = tf.convert_to_tensor(state_inputs[i],dtype=tf.float32)
        order_inputs = tf.convert_to_tensor(prev_order_inputs[i], dtype=tf.float32)
        season_input = season_names[i]

        with tf.GradientTape() as tape:
            # applying SL model
            orders_probs, position_lists = model.call(state_input, order_inputs, powers_seasons, season_input, curr_board_dict)
            print(orders_probs.shape)
            if orders_probs.shape[0] != 0:
                orders_probs = tf.transpose(orders_probs, perm=[2, 0, 3, 1])
                orders_probs = tf.squeeze(orders_probs)

                # computing loss for probabilities
                game_loss = model.loss(prev_orders_game_labels[i], orders_probs, position_lists)
                print(game_loss)
                # Add to loss tracking and record loss
                train_loss(game_loss)
                with train_summary_writer.as_default():
                    tf.summary.scalar('loss', train_loss.result(), step=i)
                # optimizing
                gradients = tape.gradient(game_loss, model.trainable_variables)
                optimizer.apply_gradients(zip(gradients, model.trainable_variables))
