
from constants.constants import ORDER_VOCABULARY_SIZE, NUM_PLACES, VALID_ORDERS, ORDER_DICT, UNIT_POWER, ORDERING
import numpy as np
import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import Embedding, Dense, LSTMCell
from tensorflow.keras.preprocessing.sequence import pad_sequences
from SL.decoder.mask import masked_softmax, create_mask

# extracting <GO> token from Valid Orders list
GO = VALID_ORDERS[1]

class Decoder(Model):
    '''
    Decoder Model
    '''

    def __init__(self):
        '''
        Initialization for Decoder Model
        Args:
        power - the power that the model represents
        '''
        super(Decoder, self).__init__()

        self.h_dec_size = ORDER_VOCABULARY_SIZE
        self.lstm_size = 200
        self.embedding_size = 80
        self.attention_size = 120

        # embedding matrix for possible orders
        self.embedding = Embedding(self.h_dec_size, self.embedding_size)

        # layers here
        self.lstm = LSTMCell(self.lstm_size, activation=tf.nn.leaky_relu)
        self.dense = Dense(13042, activation=None)
        self.attention_layer = Dense(self.attention_size)

    def create_pos_masks(self, board_states, season_input, board_dict, power):
        '''
        Function to compute position orders and masks

        Args:
        board_states - the state embeddings to parse the orderable locations from
        season_input - the season names
        board_dict - a list of dictionaries representing board states

        Returns:
        the list of possible positions for the power and the masks 3D array
        '''

        num_phases = len(board_states)
        # constructing possible locations
        board_alignment_matrix = get_orderable_locs(board_dict, power)
        # creating set for positions owned
        position_set = set()
        for phase in board_alignment_matrix:
            for position in phase:
                position_set.add(position)

        position_list = list(position_set)
        # computing mask for masked softmax
        masks = np.full((num_phases, len(position_list), ORDER_VOCABULARY_SIZE),-(10**15), dtype=np.float32)
        for i in range(num_phases):
            masks[i] = create_mask(board_states[i],
                                   season_input[i],
                                   [ORDERING[loc] for loc in position_list],
                                   board_dict[i])
        return position_list, tf.convert_to_tensor(masks)

    @tf.function
    def apply_lstm(self, action_embedding, enc_attention, lstm_prev, mask):
        '''
        Function to apply the LSTM layer within graph execution

        Args:
        action_embedding - the embedding of the action taken
        enc_attention - the location attention from the encoder output
        lstm_prev - the previous output of the lstm
        mask - the mask for the current location at all timesteps
        '''

        hidden_state = tf.concat([action_embedding,enc_attention], axis=1)
        hidden_state = tf.expand_dims(hidden_state, axis=1)

        # running lstm and computing logits
        lstm_out, (_, _) = self.lstm(lstm_prev, hidden_state)
        logits = self.dense(lstm_out)
        order_probabilities = masked_softmax(logits, mask)
        return lstm_out, order_probabilities


    def call(self, board_states, h_enc, position_list, masks):
        '''
        Call method for decoder
        Args:
        board_states - the state embeddings to parse the orderable locations from
        h_enc - the output of the encoder, an [81, H_ENC_COLS] array.
            Ordered along the 0th-dim s.t. each province is already adjacent to the next
            (so already topsorted)
        season_input - the season names
        Returns:
        The probability distributions over valid orders and the list for positions that
        have been controlled
        '''

        # LSTM INPUT [lstm previous (200), [previous order embedding (80) concat attention (120)]]

        # getting number of phases in game
        num_phases = h_enc.shape[0]

        # creating initial input for lstm
        go_tokens = tf.convert_to_tensor([ORDER_DICT[GO] for i in range(num_phases)], dtype=tf.float32)
        lstm_prev = tf.concat((self.embedding(go_tokens), tf.zeros((num_phases,self.attention_size))), axis=1)

        # creating inital LSTM hidden state
        action_taken_embedding = tf.zeros((num_phases,self.embedding_size), dtype=tf.float32)
        game_orders_probs = []
        for j in range(len(position_list)):
            position_order_probs = []

            location = position_list[j]
            mask = masks[:, j]

            enc_out = tf.gather(h_enc, location, axis=1)

            # print("ENC_OUT: ", enc_out.shape)

            # Might need different attention thing
            enc_attention = enc_out

            lstm_out, order_probabilities = self.apply_lstm(action_taken_embedding,
                                                            enc_attention,
                                                            lstm_prev,
                                                            mask)

            # TODO: get actual action taken
            # setting outputs for next iteration
            actions_taken = tf.math.argmax(order_probabilities, axis=1)
            actions_taken_embedding = self.embedding(actions_taken)
            lstm_prev = lstm_out
            position_order_probs.append(order_probabilities)

            game_orders_probs.append(position_order_probs)

            # # looping through locations to decode in a phase
            # for location in phase:
            #     enc_out = tf.gather(h_enc, location, axis=1)
            #     print(enc_out.shape)
            #     enc_attention = enc_out[i]
            #     # enc_attention = tf.squeeze(enc_attention)
            #     hidden_state = tf.concat((action_taken_embedding, enc_attention), axis=0)
            #
            #     # calling the LSTM Cell on the inputs
            #     # hidden_state = tf.expand_dims(hidden_state,axis=1)
            #     print(hidden_state.shape)
            #     print(lstm_prev.shape)
            #     lstm_out, (_, _) = self.lstm(lstm_prev, hidden_state)
            #     logits = self.dense(lstm_out)
            #     # order_probabilities = masked_softmax(h_dec, mask) # TODO: implement in mask.py
            #     order_probabilities = tf.nn.softmax(logits)
            #
            #     # TODO: get actual action taken
            #     action_taken = tf.math.argmax(order_probabilities, axis=0)
            #     action_taken_embedding = self.embedding(action_taken)
            #
            #     phase_orders.append(action_taken)
            #
            #     phase_order_probs.append(order_probabilities)
            #     lstm_prev = lstm_out
            # game_orders_probs.append(phase_order_probs)
            # game_orders.append(phase_orders)
        # return tf.convert_to_tensor(game_orders)

        # converting position list back to strings
        position_list = [ORDERING[position] for position in position_list]
        return tf.convert_to_tensor(game_orders_probs, dtype=tf.float32), position_list


def get_orderable_locs(board_dict_list, power):
    '''
    Function to compute the orderable locations based off of a board state embedding
    Keyword Args:
    board_dict_list - a list of board state dictionaries of phases
    power - the power to extract orderable locations from
    Returns
    the board alignment matrix of shape (num_phases, num_locs) which are indices of provinces
    '''

    board_alignment_matrix = []
    for phase in board_dict_list:
        phase_matrix = []
        for i in range(len(ORDERING)):
            province = ORDERING[i]
            # checking if unit type is not none
            if "unit_type" in phase[province]:
                # if it is the same power as our model
                if "unit_power" in phase[province] and phase[province]["unit_power"] == power:
                    phase_matrix.append(i)
        board_alignment_matrix.append(phase_matrix)
    return board_alignment_matrix
