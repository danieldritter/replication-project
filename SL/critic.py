import numpy as np
import tensorflow as tf
from tensorflow.keras import Model
from constants.constants import UNIT_POWER, STATE_SIZE, NUM_POWERS
from data.process import get_returns
from AbstractCritic import AbstractCritic
import datetime

class CriticSL(AbstractCritic):
    '''
    Critic Model
    a. Input: a state (e.g. one phase, 81 * 35 vector)
    b. Output: (7,) vector representing the value for each power.
    '''
    def loss(self, predicted_values, returns):
        """
        Args:
        predicted values - [batch_sz, 7] vector of predicted values for each power
        returns - [batch_sz, 7] vector of actual values for each power
        
        Returns:
        MSE(predicted, returns)
        """
        return tf.reduce_mean((predicted_values - returns) ** 2)

    def process_data(self, state_inputs, supply_center_owners):
        """

        :param state_inputs: [bs, game_length, 81, 35] (list of list of 2D [81, 35] arrays)
        :param supply_center_owners: List of supply center owners for each state
        in each game in batch (bs, game_length, {powers: centers}).
        :return: [bs, (NUM_POWERS, num_powers)]
        """
        power_inputs = []
        for batch in supply_center_owners:
            init_owners = batch[0]
            powers = list(init_owners.keys())
            power_mask = np.zeros(7, dtype=np.float32)
            for power in powers:
                power_mask[np.argmax(UNIT_POWER[power])] = 1.
            # power_one_hot = np.argmax(np.array([UNIT_POWER[power] for power in powers], dtype=np.float32), axis=)
            power_inputs.append(power_mask)

        state_inputs_unrolled = [np.array([state.ravel() for state in game]) for game in state_inputs]
        train_data = (state_inputs_unrolled, power_inputs)
        train_labels = get_returns(supply_center_owners)
        return train_data, train_labels

    def train(self, state_inputs, supply_center_owners, num_epochs=1):
        """

        :param state_inputs: [bs, game_length, 81, 35] (list of list of 2D [81, 35] arrays)
        :param supply_center_owners: [bs, game_length, {power: [centers]}] (list of list of dicts)
        :param num_epochs: number of epoch
        :return:
        """

        # Set up tracking metrics and logging directory
        train_loss = tf.keras.metrics.Mean("train_critic_sl_loss", dtype=tf.float32)
        current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        train_log_dir = 'data/logs/critic_sl/' + current_time + '/train'
        train_model_dir = './models/critic_sl/checkpoints/checkpoint'
        train_summary_writer = tf.summary.create_file_writer(train_log_dir)

        train_data, train_labels = self.process_data(state_inputs, supply_center_owners)
        train_data_states = train_data[0] # shape: [bs, game_length, 81 * 35]
        train_data_powers = train_data[1] # shape: [bs, game_length, num_powers]
        num_batches = len(train_data_states)
        for batch_num in range(num_batches):
            states_batch = train_data_states[batch_num] # states_batch shape: [game_length, (81, 35)]
            powers_batch = train_data_powers[batch_num] # shape: [7, num_powers]
            labels_batch = train_labels[batch_num] # labels_batch shape: [game_length, num_powers]

            # shuffle within minibatch
            tf.random.shuffle(states_batch, seed=1)
            tf.random.shuffle(powers_batch, seed=1)
            tf.random.shuffle(labels_batch, seed=1)

            with tf.GradientTape() as tape:
                predicted_values = self.call(states_batch) # shape: [game_length, 7]

                # mask out any powers not playing
                predicted_values_masked = tf.boolean_mask(predicted_values, powers_batch, axis=1)
                loss = self.loss(predicted_values_masked, labels_batch)
                # Add to loss tracking and record loss
                train_loss(loss)
                with train_summary_writer.as_default():
                    tf.summary.scalar('loss', train_loss.result(), step=batch_num)

            gradients = tape.gradient(loss, self.trainable_variables)
            self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))

            if batch_num % 1 == 0:
                print("\tBatch number %d of %d: Loss is %.3f" % (
                batch_num, num_batches, loss))
        self.save_weights(train_model_dir)
