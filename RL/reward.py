# As a reward function, we use the average of
# (1) a local reward function (+1/-1 when a supply center is gained or lost (updated every phase and not just in Winter)), and
# (2) a terminal reward function (for a solo victory, the winner gets 34 points; for a draw, the 34 points are divided
# proportionally to the number of supply centers).

class Reward():
    def __init__(self, game):
        # TODO: how to "average" the local and terminal rewards?
        self.game = game
        self.prev_supply_centers_dist = game.get_centers()

    def get_local_reward(self, power_name):
        old_owned_centers = self.prev_supply_centers_dist[power_name]
        new_owned_centers = self.game.get_centers(power_name)
        reward = len(new_owned_centers) - len(old_owned_centers)
        print(
            f"Old {power_name}: {old_owned_centers}, New {power_name}: {new_owned_centers}")
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