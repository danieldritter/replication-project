# As a reward function, we use the average of
# (1) a local reward function (+1/-1 when a supply center is gained or lost (updated every phase and not just in Winter)), and
# (2) a terminal reward function (for a solo victory, the winner gets 34 points; for a draw, the 34 points are divided
# proportionally to the number of supply centers).
import numpy as np
from constants.constants import NUM_POWERS, GAMMA
from data.process import get_data

class Reward():
    def __init__(self, game):
        self.game = game
        self.prev_supply_centers_dist = game.get_centers()

    def get_local_reward(self, power_name):
        old_owned_centers = self.prev_supply_centers_dist[power_name]
        new_owned_centers = self.game.get_centers(power_name)
        reward = len(new_owned_centers) - len(old_owned_centers)
        self.prev_supply_centers_dist[power_name] = self.game.get_centers(power_name)
        return reward

    def get_local_reward_all_powers(self):
        reward_all_powers = dict()
        for power_name in self.game.powers.keys():
            reward_all_powers[power_name] = self.get_local_reward(power_name)

        return reward_all_powers

    def get_terminal_reward(self, power_name):
        return len(self.game.get_centers(power_name))

    def get_terminal_reward_all_powers(self):
        return {power_name: self.get_terminal_reward(power_name)
                for power_name in self.game.get_centers()}

def get_average_reward(supply_center_owners_per_game):
    """
    Returns the average reward for each time step.

    :param states: List of states in a game (game_length, 81, 35).
    :param supply_center_owners_per_game: List of supply center owners for each state in a game (game_length, {powers: centers}).
    :return: (game_length, num_powers) nparray of average rewards, where game_length = len(states), num_powers =
    received for each power in each state
    """
    game_length = len(supply_center_owners_per_game)
    num_powers = len(supply_center_owners_per_game[0].values())
    local_rewards = np.zeros((game_length, num_powers))
    terminal_rewards = np.zeros((game_length, num_powers))
    for phase in range(game_length):
        supply_center_owners_per_phase = supply_center_owners_per_game[
            phase]
        supply_center_values = list(
            supply_center_owners_per_phase.values())
        supply_center_counts = [len(supply_center_value) for
                                supply_center_value in
                                supply_center_values]
        local_rewards[phase] = np.array(supply_center_counts)
        # if phase > 0:
        #     local_rewards[phase] -= local_rewards[phase - 1]
        if phase == game_length - 1:
            terminal_rewards[phase] = np.array(supply_center_counts)

    for phase in range(game_length - 1, 0, -1):
        local_rewards[phase] -= local_rewards[phase - 1]

    local_rewards = local_rewards[1:]
    terminal_rewards = terminal_rewards[1:]

    return (local_rewards + terminal_rewards) / 2

def get_returns(supply_center_owners, gamma=0.99):
    """
    TODO
    Using the reward function, create labels for the value of each state (phase) in each game for each player.
    a. Apply the reward function to all states in a game
    b. Run the discount function to get the values for each state (cumulative disc reward)
    c. Add that to the dictionary board_dict_list in process.py
        i. key: "value"
           value: {country: value(country) for all countries}

    :param supply_center_owners: List of supply center owners for each state in each game in batch (bs, game_length, {powers: centers}).
    :return [bs, game_length, 7] array of returns from each state (list of nparray of returns (with length game_length))
    """
    batch_returns = []
    for supply_center_owners_per_game in supply_center_owners:
        game_rewards = get_average_reward(
            supply_center_owners_per_game)  # shape: (game_length - 1, num_powers)
        game_returns = np.zeros_like(game_rewards - 1)
        game_returns[-1] = game_rewards[-1]
        for i in range(len(game_returns) - 2, -1, -1):
            game_returns[i] = game_rewards[i] + gamma * game_returns[i + 1]

        batch_returns.append(game_returns)

    return batch_returns

def advantage(values, returns, n_step=15, gamma=GAMMA):
    """
    Args:
    values: game_length 1D array of values for each time step for a given power.
    returns: game_length 1D array of returns from each time step to the end of the episode.
    for a given power.
    n_step: number of steps before return, n = 15

    Returns:
    1D np array of returns(t) + V(t + n) - V(t) for each t
    """
    g_n_step = gamma ** n_step
    loss_list = []
    for i in range(len(returns) - n_step):
        step_returns = returns[i] - (returns[i + n_step] * g_n_step)
        loss_list.append(values[i + n_step] + step_returns - values[i])
    for i in range(len(returns) - n_step, len(returns)):
        loss_list.append((returns[i] - values[i]))
    return loss_list

if __name__ == "__main__":
    state_inputs, prev_order_inputs, prev_orders_game_labels, season_names, supply_center_owners, board_dict_list = get_data("../data/standard_no_press.jsonl")
    get_returns(supply_center_owners)