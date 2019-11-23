
from constants.constants import ORDER_VOCABULARY_SIZE, NUM_PLACES, VALID_ORDERS, ORDER_DICT, UNIT_POWER
import numpy as np
import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import Embedding, Dense, LSTMCell
from SL.decoder.mask import masked_softmax

# extracting <GO> token from Valid Orders list
GO = VALID_ORDERS[1]

class Decoder(Model):
    '''
    Decoder Model
    '''

    def __init__(self, power):
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
        self.power = power

        # embedding matrix for possible orders
        self.embedding = Embedding(self.h_dec_size, self.embedding_size)

        # layers here
        self.lstm = LSTMCell(self.lstm_size, activation=tf.nn.leaky_relu)
        self.dense = Dense(13042, activation=None)
        self.attention_layer = Dense(self.attention_size)


    def call(self, board_states, h_enc, mask):
        '''
        Call method for decoder
        Args:
        board_states - the state embeddings to parse the orderable locations from
        h_enc - the output of the encoder, an [81, H_ENC_COLS] array.
            Ordered along the 0th-dim s.t. each province is already adjacent to the next
            (so already topsorted)
        mask - the mask computed for the orderable locations
        Returns:
        The probability distributions over valid orders
        '''

        # LSTM INPUT [lstm previous (200), [previous order embedding (80) concat attention (120)]]

        # getting number of phases in game
        num_phases = h_enc.shape[0]

        board_alignment_matrix = get_orderable_locs(board_states, self.power)
        print(board_alignment_matrix)

        # creating initial input for lstm
        go_tokens = tf.convert_to_tensor([ORDER_DICT[GO] for i in range(num_phases)], dtype=tf.float32)
        lstm_prev = tf.concat((self.embedding(go_tokens), tf.zeros((num_phases, self.attention_size))), axis=1)
        # creating inital LSTM hidden state
        action_taken_embedding = tf.zeros((num_phases, self.embedding_size), dtype=tf.float32)

        # looping through phases of a game
        game_orders = []
        for phase in board_alignment_matrix:

            phase_orders = []

            # looping through locations to decode in a phase
            for location in phase:
                enc_out = tf.gather(h_enc, [location], axis=1)
                enc_attention = self.compute_attention(enc_out)

                hidden_state = tf.concat((action_taken_embedding, enc_attention), axis=1)

                # calling the LSTM Cell on the inputs
                print(lstm_prev.shape)
                print(hidden_state.shape)

                lstm_out, (_, _) = self.lstm(lstm_prev, [hidden_state, hidden_state])
                logits = self.dense(lstm_out)
                
                # order_probabilities = masked_softmax(h_dec, mask) # TODO: implement in mask.py
                order_probabilities = tf.nn.softmax(logits)

                # TODO: get actual action taken
                action_taken = tf.math.argmax(order_probabilities, axis=1)
                action_taken_embedding = self.embedding(action_taken)

                # print("TEST")
                phase_orders.append(action_taken)
                lstm_prev = lstm_out
            game_orders.append(phase_orders)
        return tf.convert_to_tensor(game_orders)

    def compute_attention(self, encoder_output):
        '''
        Function to compute attention
        '''

        return tf.zeros((encoder_output.shape[0], self.attention_size))

def get_orderable_locs(board_state_embedding, power):
    '''
    Function to compute the orderable locations based off of a board state embedding
    Keyword Args:
    board_state_embedding - the board state of the game
    power - the power to extract orderable locations from
    Returns
    the board alignment matrix of shape (num_phases, num_locs) which are indices of provinces
    '''

    board_alignment_matrix = []
    for phase in board_state_embedding:
        phase_matrix = []
        for i in range(len(phase)):

            province = phase[i]
            # checking if unit type is not none
            if province[2] == 0:
                powers = province[3:11]
                print(powers)

                # if it is the same power as our model
                if np.array_equal(UNIT_POWER[power], powers):
                    phase_matrix.append(i)
        board_alignment_matrix.append(phase_matrix)
    return board_alignment_matrix