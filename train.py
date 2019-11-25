from SL_model import SL_model
from SL.critic import CriticSL
from RL.critic import CriticRL
from data.process import get_data
import tensorflow as tf
import numpy as np
from a2c import A2C


def copy_sl_to_rl(critic_sl, critic_rl, train_data):
    critic_sl(train_data[0][0])
    critic_rl(train_data[0][0])
    critic_rl.set_weights(critic_sl.get_weights())

def train():
    """
    1. Process data.
    2. Train actor supervised in SL model
    3. Train critic supervised
    4. Train RL agent as a function of actor and critic weights.
    """
    # actor_sl = SL_model(num_board_blocks, num_order_blocks)
    critic_sl = CriticSL()
    critic_sl2 = CriticSL()
    critic_rl = CriticRL()

    state_inputs, prev_order_inputs, prev_orders_game_labels, season_names, supply_center_owners, board_dict_list = get_data("data/standard_no_press.jsonl")
    train_data, train_labels = critic_sl.process_data(state_inputs, supply_center_owners)
    critic_sl.train(state_inputs, supply_center_owners)
    copy_sl_to_rl(critic_sl, critic_rl, train_data)

    # DOESN'T WORK FOR SOME REASON https://stackoverflow.com/questions/55719047/is-loading-in-eager-tensorflow-broken-right-now
    # checkpoint = tf.train.Checkpoint(model=critic_sl2)
    # status = checkpoint.restore(tf.train.latest_checkpoint("./models/critic_sl/checkpoints/"))
    # sl_weights = critic_sl.get_weights()
    # new_weights = critic_sl2.get_weights()
    # print(critic_sl2.get_weights())

if __name__ == "__main__":
    train()
