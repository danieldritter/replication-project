import numpy as np
import tensorflow as tf
from tensorflow.keras import Model
from AbstractCritic import AbstractCritic
from constants.constants import GAMMA


class CriticRL(AbstractCritic):
    '''
    TODO:
    Critic Model
    a. Input: a state (e.g. one phase, 81 * 35 vector)
    b. Output: (num_powers,) vector representing the value for each power.
    '''
    def loss(self, values, returns, n_step=15, gamma=GAMMA):
        """
        Args:
        values: game_length 1D array of values for each time step for a given power.
        returns: game_length 1D array of returns from each time step to the end of the episode.
        for a given power.
        n_step: number of steps before return, n = 15

        Returns:
        sum(MSE(returns + V(t + n), V(t)))
        """
        g15 = gamma ** n_step
        loss = []
        for i in range(len(returns) - n_step):
            step_returns = returns[i] - (returns[i + n_step] * g15)
            loss += (values[i + n_step] + step_returns - values[i]) ** 2
        for i in range(len(returns) - n_step, len(returns)):
            loss += (returns[i] - values[i]) ** 2
        return loss





    