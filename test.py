import random
from diplomacy import Game
from tornado import gen
from RL.reward import Reward
from diplomacy.utils.export import to_saved_game_format
import SL_model
import json
import pickle

# importing from research
from diplomacy_research.models import state_space
from diplomacy_research.players.random_player import RandomPlayer
from diplomacy_research.players.rule_based_player import RuleBasedPlayer
from diplomacy_research.players.rule_based_player import ModelBasedPlayer
from diplomacy_research.utils.cluster import start_io_loop, stop_io_loop

# grabbing adjacency matrix
adj_matrix = state_space.get_adjacency_matrix("standard")
# print(adj_matrix)

# grabbing ordering of provinces
ordering = state_space.STANDARD_TOPO_LOCS
# print(ordering)

# province types
coasts = ["BUL/EC", "BUL/SC", "SPA/NC", "SPA/SC", "STP/NC", "STP/SC"]
water = ["ADR", "AEG", "BAL", "BAR", "BLA", "EAS", "ENG", "BOT",
         "GOL", "HEL", "ION", "IRI", "MID", "NAT", "NTH", "NRG",
         "SKA", "TYN", "WES"]

def test_game():
    # creating multiple agents
    # Basic test of 7 random action agents
    test_game = Game()
    reward_class = Reward(test_game)
    supply_centers_dist = test_game.get_centers()
    while not test_game.is_game_done:
        # Getting the list of possible orders for all locations
        possible_orders = test_game.get_all_possible_orders()
        # print(possible_orders)

        # For each power, randomly sampling a valid order
        for power_name, power in test_game.powers.items():
            # print(power_name, power)
            power_orders = [random.choice(possible_orders[loc]) for loc in test_game.get_orderable_locations(power_name)
                            if possible_orders[loc]]
            test_game.set_orders(power_name, power_orders)


        # Messages can be sent locally with game.add_message
        # e.g. game.add_message(Message(sender='FRANCE',
        #                               recipient='ENGLAND',
        #                               message='This is a message',
        #                               phase=self.get_current_phase(),
        #                               time_sent=int(time.time())))

        # Processing the game to move to the next phase
        test_game.process()
        print(test_game.phase)
        print(reward_class.get_local_reward_all_powers())
        input()

    print(reward_class.get_terminal_reward_all_powers())

    print(test_game.outcome)

# Testing function based on diplomacy_research repo example
@gen.coroutine
def main():
    state_inputs, prev_order_inputs, prev_orders_game_labels, season_names, supply_center_owners, board_dict_list = process.get_data("data/standard_no_press.jsonl", num_games=100)
    weights_file = open("sl_weights_50_chunks.pickle", "rb+")
    weights = pickle.load(weights_file)
    sl_model = SL_model(16, 16)
    set_sl_weights(weights, sl_model, state_inputs, prev_order_inputs, prev_orders_game_labels, season_names, board_dict_list)

    """ Plays a local game with 7 bots """
    player1 = RandomPlayer() # Use main player here x1
    # player1 = sl_model (Use when get_orders is ready)
    player2 = RandomPlayer() # Use other player here x6

    game = Game()
    reward_class = Reward(game)
    supply_centers_dist = game.get_centers()

    # For randomly choosing the power of the special player
    powers = list(game.powers)
    random.shuffle(powers)
    powers1 = powers[0:1]
    powers2 = powers[1:7]

    # Playing game
    while not game.is_game_done:
        orders1 = yield {power_name: player1.get_orders(game, power_name) for power_name in powers1}
        orders2 = yield {power_name: player2.get_orders(game, power_name) for power_name in powers2}

        for power_name, power_orders in orders1.items():
            game.set_orders(power_name, power_orders)
        for power_name, power_orders in orders2.items():
            game.set_orders(power_name, power_orders)
        game.process()
        print(reward_class.get_local_reward_all_powers())
        input()
    print(reward_class.get_terminal_reward_all_powers())

    print(game.outcome)

    # Calculating support
    phase_history = game.get_phase_history()
    support_count, x_support_count, eff_x_support_count = 0, 0, 0
    for phase in phase_history:
        for order_index in range(len(phase.orders[powers1[0]])):
            order_split = phase.orders[powers1[0]][order_index].split()
            if 'S' in order_split:
                support_count += 1
                s_loc = order_split.index('S')
                supported = order_split[s_loc+1] + " " + order_split[s_loc+2]
                if supported not in phase.state['units'][powers1[0]]:
                    x_support_count += 1

    print("X-Support Ratio: " + str(x_support_count / support_count))


    # Saving to disk
    with open('game.json', 'w') as file:
        file.write(json.dumps(to_saved_game_format(game)))
    # stop_io_loop()

if __name__ == '__main__':
    main()
    # start_io_loop(main)
