import os

import gym
import numpy as np
import tensorflow as tf

# Killing optional CPU driver warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

class Actor(tf.keras.Model):

    def __init__(self, state_size, hidden_size, num_actions):
        """
        The Actor class that inherits from tf.keras.Model.

        When actor's forward pass is called, it will output the
        probabilities of taking each action in a given state.

        :param state_size: number of parameters that define the state.
        :param hidden_size: hyperparameter for fully connected layer for computing action probabilities.
        :param num_actions: number of actions in an environment.
        """

        super(Actor, self).__init__()

        self.num_actions = num_actions
        self.W1 = tf.Variable(
            tf.random.normal([state_size, hidden_size], stddev=0.01))
        self.b1 = tf.Variable(tf.random.normal([hidden_size], stddev=0.01))
        self.W2 = tf.Variable(
            tf.random.normal([hidden_size, num_actions], stddev=0.01))
        self.b2 = tf.Variable(tf.random.normal([num_actions], stddev=0.01))

    @tf.function
    def call(self, state):
        """
        Performs the forward pass on a batch of states for the Actor,
        returning a probability distribution across actions for each state as a
        [episode_length, num_actions] dimensioned array.

        :param state: An [episode_length, state_size] dimensioned array
        representing history of states of an episode
        :return: probability distribution across actions for each state
        as a [episode_length, num_actions] dimensioned array.
        """
        logits = tf.nn.relu(tf.matmul(state, self.W1) + self.b1)
        return tf.nn.softmax(tf.matmul(logits, self.W2) + self.b2)

    @tf.function
    def loss_function(self, reward, next_state_value, state_value, action_prob, gamma):
        """
        Calculates the Actor model's loss.

        :param discounted_rewards: history of rewards throughout a complete
        episode (represented as an [episode_length] array)
        :param state_values: history of estimated state values from the critic
        throughout a complete episode (represented as an [episode_length] array)
        :param chosen_actor_probs: history of action probabilities for the
        chosen action of each step in a complete episode (represented as an
        [episode_length] array)
        :return: actor loss, a scalar
        """
        advantage = reward + gamma * next_state_value - state_value
        actor_loss = - \
            tf.reduce_sum(tf.math.log(action_prob) *
                          tf.stop_gradient(advantage))

        return actor_loss


class Critic(tf.keras.Model):

    def __init__(self, state_size, hidden_size):
        """
        The Critic class that inherits from tf.keras.Model.

        When Critic's forward pass is called, it will output its estimated value
        for a given state.

        :param state_size: number of parameters that define the state.
        :param hidden_size: hyperparameter for fully connected layer for
        computing state values.
        """
        super(Critic, self).__init__()

        # Initialize Weights and Biases for Critic Network
        self.W1 = tf.Variable(tf.random.normal(
            [state_size, hidden_size], stddev=.01))
        self.b1 = tf.Variable(tf.random.normal([hidden_size], stddev=.01))
        self.W2 = tf.Variable(tf.random.normal(
            [hidden_size, 1], stddev=.01))
        self.b2 = tf.Variable(tf.random.normal([1], stddev=.01))

    @tf.function
    def call(self, state):
        """
        Performs the forward pass on a batch of states for the Critic,
        returning a [episode_length] dimensioned array of estimated values
        (one value for each state)
        :param state: An [episode_length, state_size] dimensioned array
        representing history of states of an episode
        :return: estimated values of each inputted state as an [episode_length]
        dimensioned array.
        """
        logits = tf.nn.relu(tf.matmul(state, self.W1) + self.b1)
        value = tf.squeeze(tf.matmul(logits, self.W2) + self.b2)
        return value

    @tf.function
    def loss_function(self, reward, next_state_value, state_value, gamma):
        """
        Calculates the Critic model's loss.

        :param discounted_rewards: history of rewards throughout a complete
        episode (represented as an [episode_length] array)
        :param state_values: history of estimated state values from the critic
        throughout a complete episode (represented as an [episode_length] array)
        :return: critic loss, a scalar
        """
        advantage = reward + gamma * next_state_value - state_value
        return tf.reduce_sum(tf.square(advantage))


def calc_entropy(policy):
    entropy = 0.0
    for i in range(len(policy)):
        entropy -= policy[i] * tf.math.log(policy[i])
    return entropy


class A2C:
    def __init__(self, env):
        state_size = env.observation_space.shape[0]
        num_actions = env.action_space.n
        hidden_size = 32

        self.actor = Actor(state_size, hidden_size, num_actions)
        self.critic = Critic(state_size, hidden_size)
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
        self.env = env
        self.gamma = .99

    def train(self, num_episodes):
        """
        Training loop for A2C. Generates a complete trajectory for one episode, and then updates both the
        actor and critic networks using that trajectory

        :param num_episodes: number of episodes to train the networks for

        :returns: None
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



def main():
    env = gym.make("CartPole-v1")
    agent = A2C(env)
    agent.train(1000)


if __name__ == '__main__':
    main()