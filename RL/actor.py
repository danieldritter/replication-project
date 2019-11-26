import numpy as np
import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.losses import sparse_categorical_crossentropy
from tensorflow.keras.optimizers import Adam

from data import process
from SL.encoder.encoder import Encoder
from SL.decoder.decoder import Decoder
from constants.constants import SEASON, UNIT_POWER, ORDER_DICT, GAMMA
from AbstractActor import AbstractActor
from RL.reward import advantage


class ActorRL(AbstractActor):
    '''
    The RL Actor for the Diplomacy game
    '''
    def loss(self, action_probs, values, returns, n_step=15, gamma=GAMMA):
        '''
        Function to compute the loss of the RL Model

        Keyword Args:
        action_probs - np array for the probability of taking the action that was taken
        values: game_length 1D array of values for each time step for a given power.
        returns: game_length 1D array of returns from each time step to the end of the episode.
        for a given power.
        n_step: number of steps before return, n = 15
        gamma: gamma

        Return:
        the policy gradient loss thing (no batching)
        '''
        assert len(action_probs) == len(values)
        assert len(action_probs) == len(returns)
        adv = advantage(values, returns, n_step, gamma)
        return -np.sum(np.log(action_probs) * adv)

if __name__ == "__main__":
    loss_test()
