import tensorflow as tf
from tensorflow.keras.layers import Layer, LSTM, Dense

class FiLM(Layer):
    '''
    Class representing a film normaliziation layer
    '''

    def __init__(self):
        '''
        Constructor for the film layer
        '''

        super(FiLM,self).__init__()
        self.lstm1 = LSTM(64, activation="sigmoid")
        
        # dense layers for outputting gamma and beta tensors
        self.d1 = Dense(1)
        self.d2 = Dense(1)

    def call(self, lstm_input, inputs_state):
        '''
        Call function for FiLM layer
        
        Keyword Args:
        lstm_input - the input to the lstm
        inputs-state - the input of hidden states

        Returns:
        two tensors representing gamma values and beta values
        '''
        # Maybe decide on specific activation functions here
        # Not sure on shape here

        out = self.lstm1(lstm_input, initial_state=self.lstm1.get_initial_state(inputs_state))

        gamma = self.d1(out)
        beta = self.d2(out)
        return gamma, beta
