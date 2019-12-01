from SL_model import SL_model
from SL.critic import CriticSL
from RL.critic import CriticRL
from RL.actor import ActorRL
from data.process import get_data
import pickle
from a2c import A2C

def set_rl_weights(new_weights, rl_model, train_data):
    rl_model(train_data[0])
    rl_model.set_weights(new_weights)

def train():
    """
    1. Process data.
    2. Train actor supervised in SL model
    3. Train critic supervised
    4. Train RL agent as a function of actor and critic weights.
    """
    # process data
    state_inputs, prev_order_inputs, prev_orders_game_labels, season_names, \
    supply_center_owners, board_dict_list = get_data("data/standard_no_press.jsonl", num_games=1)

    # train SL actor
    print("Training SL actor")
    actor_sl = SL_model(num_board_blocks=16, num_order_blocks=16)
    actor_sl.train(state_inputs, prev_order_inputs, prev_orders_game_labels, season_names, board_dict_list)

    # save actor weights
    print("Saving SL actor weights")
    weights_file = open("actor_weights.pickle", "wb+")
    pickle.dump(actor_sl.get_weights(), weights_file)
    weights_file.close()

    # train SL critic
    print("Training SL critic")
    critic_sl = CriticSL()
    critic_sl.train(state_inputs, supply_center_owners)

    # save critic weights
    print("Saving SL critic weights")
    weights_file = open("critic_weights.pickle","wb+")
    pickle.dump(critic_sl.get_weights(), weights_file)
    weights_file.close()

    # load actor, critic weights from SL
    print("Loading actor, critic weights ready for RL training")
    ### LOADING ACTOR DOESN'T WORK BECAUSE YOU NEED TO CALL IT ON SOMETHING FIRST ###
    ## see https://stackoverflow.com/questions/55719047/is-loading-in-eager-tensorflow-broken-right-now
    new_weights_file = open("sl_weights_50_chunks.pickle", "rb")
    new_weights_actor = pickle.load(new_weights_file)
    weights_file.close()

    actor_rl = ActorRL(num_board_blocks=16, num_order_blocks=16)
    # actor_rl.call(state_inputs[0], prev_order_inputs[0], season_names[0],board_dict_list[0],"AUSTRIA")

    ##########################################################################
    new_weights_file = open("critic_weights.pickle","rb")
    new_weights = pickle.load(new_weights_file)
    weights_file.close()

    critic_rl = CriticRL()
    train_data = critic_sl.process_data(state_inputs, supply_center_owners)[0][0] # needed so that critic_rl knows input shapes or something
    set_rl_weights(new_weights, critic_rl, train_data)

    # Train RL A2C
    print("Training A2C")
    a2c = A2C(actor_rl, critic_rl)
    a2c.train(num_episodes=1)
    actor_rl.set_weights(new_weights_actor)
    a2c.train(num_episodes=1)

    # save actor/critic RL weights
    print("Saving RL actor/critic weights")
    weights_file = open("critic_rl_weights.pickle", "wb+")
    pickle.dump(critic_rl.get_weights(), weights_file)
    weights_file.close()

    weights_file = open("actor_rl_weights.pickle", "wb+")
    pickle.dump(actor_rl.get_weights(), weights_file)
    weights_file.close()
    print("Done!")

if __name__ == "__main__":
    train()
