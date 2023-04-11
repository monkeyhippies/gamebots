from typing import Union

import numpy as np

from gamebots.components.states import InfoState, GameState
from gamebots.components.actions import Action


class Strategy:
    def update_strategy(self):
        pass

    def act(self, info_state: InfoState):
        pass


class RegretStrategy(Strategy):
    """
                            update                                -> act
    (accumulate_regret -> update_strategy -> accumulate_strategy) -> act
    """
    def __init__(self, num_info_states, num_actions):
        self.num_info_states = num_info_states
        self.num_actions = num_actions

        self.strategy = np.zeros((num_info_states, num_actions))
        self.cumulative_regret = np.zeros((num_info_states, num_actions))
        self.cumulative_strategy = np.zeros((num_info_states, num_actions))

        # Initialize with dummy zero regrets
        self.update(regret=np.zeros_like(self.strategy))

    def act(self, info_state: InfoState) -> Action:
        action = Action(np.random.choice(self.num_actions, p=self.strategy[info_state]))
        return action

    def update(self, regret):
        self.accumulate_regret(regret)
        self.update_strategy()
        self.accumulate_strategy()

    def accumulate_regret(self, regret):
        self.cumulative_regret += regret

    def use_average_strategy(self):
        self.strategy = self.cumulative_strategy / self.cumulative_strategy.sum(axis=1, keepdims=True)
        self.normalize_strategy()

    def update_strategy(self):
        # Ignore actions with negative regret
        self.strategy = np.clip(self.cumulative_regret, a_min=0, a_max=None, out=self.strategy)
        self.normalize_strategy()

    def normalize_strategy(self):
        strategy_sums = self.strategy.sum(axis=1)
        nonzero_sums = strategy_sums[strategy_sums != 0][:, None]
        if nonzero_sums.size != 0:
            self.strategy[strategy_sums != 0] /= nonzero_sums
        # Use uniform strategy if all zeros
        self.strategy[strategy_sums == 0] = 1.0 / self.num_actions

    def accumulate_strategy(self):
        self.cumulative_strategy += self.strategy
