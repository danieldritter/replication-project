import random
from diplomacy import Game
from RL.reward import Reward
from diplomacy.utils.export import to_saved_game_format

# importing from research
from diplomacy_research.models import state_space

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

# creating multiple agents

game = Game()
reward_class = Reward(game)
supply_centers_dist = game.get_centers()
while not game.is_game_done:

    # Getting the list of possible orders for all locations
    possible_orders = game.get_all_possible_orders()
    # print(possible_orders)

    # For each power, randomly sampling a valid order
    for power_name, power in game.powers.items():
        # print(power_name, power)
        power_orders = [random.choice(possible_orders[loc]) for loc in game.get_orderable_locations(power_name)
                        if possible_orders[loc]]
        game.set_orders(power_name, power_orders)


    # Messages can be sent locally with game.add_message
    # e.g. game.add_message(Message(sender='FRANCE',
    #                               recipient='ENGLAND',
    #                               message='This is a message',
    #                               phase=self.get_current_phase(),
    #                               time_sent=int(time.time())))

    # Processing the game to move to the next phase
    game.process()
    print(reward_class.get_local_reward_all_powers())
    # input()

print(reward_class.get_terminal_reward_all_powers())

print(game.outcome)