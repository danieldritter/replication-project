from SL_model import SL_model
from SL.critic import Critic
from data.process import get_data
from a2c import A2C

def train():
    """
    1. Process data.
    2. Train actor supervised in SL model
    3. Train critic supervised
    4. Train RL agent as a function of actor and critic weights.
    """
    # actor_sl = SL_model(num_board_blocks, num_order_blocks)
    critic_sl = Critic()
    state_inputs, prev_order_inputs, season_names, supply_center_owners = get_data("data/standard_no_press.jsonl")
    critic_sl.train(state_inputs, supply_center_owners)

if __name__ == "__main__":
    train()