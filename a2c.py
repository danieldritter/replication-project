import os
import numpy as np
import tensorflow as tf

from diplomacy import Game
from diplomacy_research.models import state_space
from tornado import gen
from RL.reward import Reward, get_returns
from RL.actor import ActorRL
from RL.critic import CriticRL


# Killing optional CPU driver warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
class A2C:
    def __init__(self):
        """
        Initialize Actor, Critic model.
        """
        self.actor = ActorRL()
        self.critic = CriticRL()

    @gen.coroutine
    def generate_trajectory(self, player1, player_others):
        game = Game()
        powers = list(game.powers)
        np.random.shuffle(powers)
        power1 = powers[0]
        powers_others = powers[1:]

        action_probs = []
        orders = []
        values = []

        supply_centers = [{power1: game.get_centers(power1)}]
        while not game.is_game_done:
            action_prob, order = player1.get_orders(game, power1)
            orders_others = yield {
                power_name: player_others.get_orders(game, power_name) for
                power_name in powers_others}

            board = state_space.dict_to_flatten_board_state(game.current_state(), game.map)
            state_value = self.critic.call(board)

            game.set_orders(power1, order)
            for power_name, power_orders in orders_others.items():
                game.set_orders(power_name, power_orders)
            game.process()

            # Collect data
            supply_centers.append({power1: game.get_centers(power1)})
            action_probs.append(action_prob)
            orders.append(order)
            values.append(state_value)

            # local_rewards.append(reward_class.get_local_reward(power1))
            # global_rewards.append(0 if not game.is_game_done else reward_class.get_terminal_reward(power1))

        returns = get_returns([supply_centers]) # put in list to match shape of [bs, game_length, dict}
        return action_probs, returns, values

    def train(self, player1, player_others, num_episodes):
        """
        Training loop for A2C. Generates a complete trajectory for one episode,
        and then updates both the actor and critic networks using that trajectory

        :param player1: class that is your player (RL Agent)
        :param player_others: class that is other players
        :param num_episodes: number of episodes to train the networks for
        :returns: None
        """
        total_rewards = []
        final_average = []
        for eps in range(num_episodes):
            with tf.GradientTape(persistent=True) as tape:
                action_probs, returns, values = self.generate_trajectory(player1, player_others)
                actor_loss = self.actor.loss_function(
                    action_probs, values, returns)  # + .05 * calc_entropy(policy[0])
                critic_loss = self.critic.loss_function(
                    values, returns)

            actor_grad = tape.gradient(
                actor_loss, self.actor.trainable_variables)
            critic_grad = tape.gradient(
                critic_loss, self.critic.trainable_variables)
            self.optimizer.apply_gradients(
                zip(actor_grad, self.actor.trainable_variables))
            self.optimizer.apply_gradients(
                zip(critic_grad, self.critic.trainable_variables))

        return total_rewards