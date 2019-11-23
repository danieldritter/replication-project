from constants.constants import ORDER_VOCABULARY_SIZE, NUM_PLACES, VALID_ORDERS, ORDER_DICT
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

    def __init__(self):
        '''
        Initialization for Decoder Model

        Args:
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
        self.dense = Dense(13042, activation=tf.nn.softmax)


    def call(self, state_embedding, h_enc, mask):
        '''
        Call method for decoder

        Args:
        state_embedding - the state embeddings to parse the orderable locations from
        h_enc - the output of the encoder, an [81, H_ENC_COLS] array.
            Ordered along the 0th-dim s.t. each province is already adjacent to the next
            (so already topsorted)
        mask - the mask computed for the orderable locations

        Returns:
        The concatenated encoded output of the board inputs and the previous order inputs
        '''

        orderable_locs = get_orderable_locs(state_embedding)
        orderable_locs_indices = list(map(lambda province: [ORDER_DICT[order] for order in province], orderable_locs))
        print(orderable_locs_embedding)

        attention = compute_attention(h_enc)

        # getting number of phases in game
        num_phases = h_enc.shape[0]

        # accumulating taken orders
        go_token_embedding = tf.cast_to_tensor([ORDER_DICT[GO] for i in range(num_phases)], dtype=tf.float32)
        initial_attention = self.compute_attention(go_token_embedding) 
        previous_order_embedding = tf.concat((go_token_embedding, initial_attention), axis=1)

        for location in orderable_locs_indices:
            # province is h^i^t_enc in paper,
            # valid_orders = masked_softmax(h_dec, mask) # TODO: implement in mask.py
            valid_orders = h_dec

            # make sure to append previous order with province (province is h^i^t_enc in paper)
            # also record orders output by each loop through (i.e. for each province)
            location_embedding = self.embedding(location)
            lstm_input = tf.concat((location_embedding, attention), axis=1)

            # calling the LSTM Cell on the inputs
            lstm_out, (_, _) = self.lstm(lstm_input, previous_order_embedding)
            order_probabilities = self.dense(lstm_out)

            # get actual action taken
            

            # print("TEST")
            orders_list.append(valid_orders)

        return orders_list

    def compute_attention(self, encoder_output):
        '''
        Function to compute attention
        '''

        return tf.zeros((encoder_output.shape[0], self.attention_size))
