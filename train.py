from SL_model import SL_model
from SL.critic import CriticSL
from RL.critic import CriticRL
from data.process import get_data
import tensorflow as tf
import numpy as np
import pickle
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
    actor_sl = SL_model(num_board_blocks, num_order_blocks)
    critic_sl = CriticSL()
    critic_rl = CriticRL()

    state_inputs, prev_order_inputs, prev_orders_game_labels, season_names, supply_center_owners, board_dict_list = get_data("data/standard_no_press.jsonl")
    train_data, train_labels = critic_sl.process_data(state_inputs, supply_center_owners)
    critic_sl.train(state_inputs, supply_center_owners)
    weights_file = open("critic_weights.pickle","wb+")
    pickle.dump(critic_sl.get_weights(),weights_file)
    weights_file.close()

    new_weights_file = open("critic_weights.pickle","rb")
    new_weights = pickle.load(new_weights_file)
    critic_rl.set_weights(new_weights)

if __name__ == "__main__":
    train()
