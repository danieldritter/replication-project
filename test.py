import random
from diplomacy import Game
from RL.reward import Reward
from diplomacy.utils.export import to_saved_game_format

# importing from research
from diplomacy_research.models import state_space
from diplomacy_research.players.random_player import RandomPlayer
from diplomacy_research.players.rule_based_player import RuleBasedPlayer
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
    print(game.phase)
    print(reward_class.get_local_reward_all_powers())
    input()

print(reward_class.get_terminal_reward_all_powers())

print(game.outcome)

def main():
    """ Plays a local game with 7 bots """
    player1 = DipNetSLPlayer() # Use main player here x1
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

    print(reward_class.get_terminal_reward_all_powers())

    print(game.outcome)

    # Saving to disk
    with open('game.json', 'w') as file:
        file.write(json.dumps(to_saved_game_format(game)))
    stop_io_loop()

if __name__ == '__main__':
    start_io_loop(main)
