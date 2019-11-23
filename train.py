from SL_model import SL_model
from SL.critic import Critic
from a2c import A2C

def train():
    """
    1. Process data.
    2. Train actor supervised in SL model
    3. Train critic supervised
    4. Train RL agent as a function of actor and critic weights.
    """
    actor_sl = SL_model(num_board_blocks, num_order_blocks)
    critic_sl = Critic()