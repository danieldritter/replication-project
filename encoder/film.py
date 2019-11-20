import tensorflow as tf
from tensorflow.keras.layers import Layer, LSTM, Dense

class FiLM(Layer):

    def __init__(self):
        super(FiLM,self).__init__()
        self.lstm1 = LSTM(256)
        self.d2 = Dense(2)

    def call(self,inputs):
        # Maybe decide on specific activation functions here
        # Not sure on shape here
        lstm_input = tf.reshape(inputs,(inputs.shape[0],inputs.shape[1],1))
        out = self.lstm1(lstm_input,initial_state=self.lstm1.get_initial_state(inputs))
        print(self.d2(out))
        gamma, beta = self.d2(out)
        print(gamma, beta)
        return gamma, beta
