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
        self.lstm1 = LSTM(64)
        
        # dense layers for outputting gamma and beta tensors
        self.d1 = Dense(1)
        self.d2 = Dense(1)

    def call(self,inputs):
        '''
        Call function for FiLM layer
        
        Keyword Args:
        the inputs to the model which are season names and power names

        Returns:
        two tensors representing gamma values and beta values
        '''
        # Maybe decide on specific activation functions here
        # Not sure on shape here

        lstm_input = tf.reshape(inputs,(inputs.shape[0],inputs.shape[1],1))
        out = self.lstm1(lstm_input,initial_state=self.lstm1.get_initial_state(inputs))

        gamma = self.d1(out)
        beta = self.d2(out)
        return gamma, beta
