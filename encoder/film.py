import tensorflow as tf

class FiLM(tf.keras.layers.Layer):

    def __init__(self):
        super(FiLM,self).__init__()
        self.lstm1 = tf.keras.layers.LSTM(256)
        self.d2 = tf.keras.layer.Dense(2)

    def call(self,inputs):
        # Maybe decide on specific activation functions here
        out = self.lstm1(self.input,intial_state=self.lstm1.get_initial_state(inputs))
        gamma, beta = self.d2(out)
        return gamma, beta
