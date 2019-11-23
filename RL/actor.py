import numpy as np
import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.losses import sparse_categorical_crossentropy
from tensorflow.keras.optimizers import Adam

from data import process
from SL.encoder.encoder import Encoder
from SL.decoder.decoder import Decoder
from constants.constants import SEASON, UNIT_POWER, ORDER_DICT
from AbstractActor import AbstractActor


class ActorRL(AbstractActor):
    '''
    The RL Actor for the Diplomacy game
    '''
    def loss(self, probs, labels):
        '''
        Function to compute the loss of the RL Model

        Keyword Args:
        probs - the probability distribution output over the orders
        labels - the labels representing the actions taken

        Return:
        the policy gradient loss thing (no batching)
        '''
        raise NotImplementedError("TODO!")

if __name__ == "__main__":
    pass