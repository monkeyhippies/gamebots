from typing import Union

import numpy as np


class Action(int):
    pass


class State(int):
    is_end = False

    def __new__(cls, val=None, is_end: bool = False, *args, **kwargs):
        object = super().__new__(cls, val, *args, **kwargs)
        object.is_end = is_end
        return object


class GameState(State):
    pass


class InfoState(State):
    pass


class Strategy:
    def update_strategy(self):
        pass

    def act(self, info_state: InfoState):
        pass


class RegretMatchingStrategy:
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
        self.strategy = self.cumulative_strategy / self.cumulative_strategy.sum(axis=1)
        self.normalize_strategy()

    def update_strategy(self):
        # Ignore actions with negative regret
        self.strategy = np.clip(self.cumulative_regret, a_min=0, a_max=None, out=self.strategy)
        self.normalize_strategy()

    def normalize_strategy(self):
        strategy_sums = self.strategy.sum(axis=1)
        nonzero_sums = strategy_sums[strategy_sums != 0]
        if nonzero_sums.size != 0:
            self.strategy[strategy_sums != 0] /= nonzero_sums
        # Use uniform strategy if all zeros
        self.strategy[strategy_sums == 0] = 1.0 / self.num_actions

    def accumulate_strategy(self):
        self.cumulative_strategy += self.strategy


class Player:
    def __init__(self, strategy: Strategy):
        self.strategy = strategy


    def play(self, info_state: InfoState) -> Action:
        return self.strategy.act(info_state)

    def feedback(self, regret):
        self.strategy.update(regret)


class Game:
    def __init__(self, *args, **kwargs):
        assert hasattr(self, "players")

    def game2info_states(self, state: GameState) -> list[InfoState]:
        pass

    def get_state(self) -> GameState:
        pass

    def update_state(self, state: GameState, actions: list[Action]) -> GameState:
        pass

    def play(self, state: GameState) -> list[Action]:
        pass

    def process_feedbacks(
        self,
        prev_state: GameState,
        player_actions: list[Action],
        next_state: GameState,
    ) -> None:
        feedbacks = self.get_feedbacks(
            prev_state=prev_state,
            player_actions=player_actions,
            next_state=next_state
        )
        for player, feedback in zip(self.players, feedbacks):
            player.feedback(feedback)

    def get_feedbacks(
        self,
        prev_state: GameState,
        player_actions: list[Action],
        next_state: GameState,
    ) -> list[np.array]:
        """
        """
        pass

    def step(self, state: GameState) -> (list[Action], GameState):
        player_actions = self.play(state)
        next_state = self.update_state(state, player_actions)
        return player_actions, next_state

    def step_until_end(self):
        game_state = self.get_state()
        while not game_state.is_end:
            player_actions, next_game_state = self.step(game_state)
            yield game_state, player_actions, next_game_state
            game_state = next_game_state

    def is_end(self) -> bool:
        return self.get_state().is_end

    def restart(self):
        pass


class RPSGame(Game):
    def __init__(self):
        self.rock, self.paper, self.scissors = Action(0), Action(1), Action(2)
        self.actions = [self.rock, self.paper, self.scissors]
        self.states = [GameState(0), GameState(1, is_end=True)]
        self.state_idx = 0
        self.player1 = Player(strategy=RegretMatchingStrategy(
            num_info_states=1,
            num_actions=len(self.actions)
        ))
        self.player2 = Player(strategy=RegretMatchingStrategy(
            num_info_states=1,
            num_actions=len(self.actions)
        ))
        self.players = [self.player1, self.player2]

        self.rewards_lookup = np.array([
            [0, 0, 0],
            [0, 1, -1], 
            [0, 2, 1],
            [1, 0, 1],
            [1, 1, 0], 
            [1, 2, -1],
            [2, 0, -1],
            [2, 1, 1], 
            [2, 2, 0],
        ])

    def game2info_states(self, state: GameState) -> list[InfoState]:
        # game state is info state
        return [InfoState(state, is_end=state.is_end) for _ in self.players]

    def play(self, state):
        actions = []
        for player, info_state in zip(
            self.players, self.game2info_states(state)
        ):
            action = player.play(info_state)
            actions.append(action)
        return actions

    def update_state(self, state: GameState, actions: list[Action]):
        # Only 1 info state
        self.state_idx = int(not state)
        return self.states[self.state_idx]

    def get_state(self):
        # There's only 1 game state
        return self.states[self.state_idx]

    def restart(self):
        self.state_idx = 0

    def get_feedbacks(
        self,
        prev_state: GameState,
        player_actions: list[Action],
        next_state: GameState,
    ) -> tuple[np.array]:
        p1_action, p2_action = player_actions
        # Calculate p1 rewards for all possible actions with p2 action fixed
        p1_rewards = self.rewards_lookup[self.rewards_lookup[:, 1] == p2_action][:, -1]

        # Calculate p2 rewards for all possible actions, p1 action fixed
        p2_rewards = -self.rewards_lookup[self.rewards_lookup[:, 0] == p1_action][:, -1]

        p1_regrets = p1_rewards - p1_rewards[p1_action]
        p2_regrets = p2_rewards - p2_rewards[p2_action]

        return p1_regrets, p2_regrets


game = RPSGame()
for _ in range(20000):
    for game_state, player_actions, next_state in game.step_until_end():
        game.process_feedbacks(game_state, player_actions, next_state)
    game.restart()
print(game.players[0].strategy.cumulative_strategy)
print(game.players[1].strategy.cumulative_strategy)
