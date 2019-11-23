import os
import numpy as np
import tensorflow as tf

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

    def train(self, num_episodes):
        """
        Training loop for A2C. Generates a complete trajectory for one episode,
        and then updates both the actor and critic networks using that trajectory

        while loop that interfaces with the engine, combines actor/critic loss, optimizes gradients etc.

        :param num_episodes: number of episodes to train the networks for
        :returns: None
        TODO(Shamay): how do you play a game?
        """
        total_rewards = []
        final_average = []
        for eps in range(num_episodes):
            st = self.env.reset()
            done = False
            tot_rwd = 0.0
            while not done:
                with tf.GradientTape(persistent=True) as tape:
                    policy = self.actor(tf.expand_dims(st, axis=0))
                    action = np.random.choice(
                        self.env.action_space.n, p=np.squeeze(policy))
                    nst, rwd, done, _ = self.env.step(action)
                    state_value = self.critic(tf.expand_dims(st, axis=0))
                    next_state_value = 0 if done else self.critic(
                        tf.expand_dims(nst, axis=0))

                    actor_loss = self.actor.loss_function(
                        rwd, next_state_value, state_value, policy[0][action], self.gamma) #+ .05 * calc_entropy(policy[0])
                    critic_loss = self.critic.loss_function(
                        rwd, next_state_value, state_value, self.gamma)
                    tot_rwd += rwd
                actor_grad = tape.gradient(
                    actor_loss, self.actor.trainable_variables)
                critic_grad = tape.gradient(
                    critic_loss, self.critic.trainable_variables)
                self.optimizer.apply_gradients(
                    zip(actor_grad, self.actor.trainable_variables))
                self.optimizer.apply_gradients(
                    zip(critic_grad, self.critic.trainable_variables))
                st = nst
            total_rewards.append(tot_rwd)
            print(
                f"Episode #: {eps} "
                f"Total reward: {tot_rwd})")
            if eps > num_episodes - 100:
                final_average.append(tot_rwd)
        print(np.mean(final_average))
        return total_rewards