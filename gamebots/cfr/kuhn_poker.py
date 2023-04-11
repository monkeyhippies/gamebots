from typing import List

from toolz import partial
import numpy as np

from gamebots.components.strategies import RegretStrategy
from gamebots.components.states import InfoState

kuhn_strategy = RegretStrategy(num_info_states=12, num_actions=2)

"""
Entire game state determined by:
    1. players' cards
    2. game node
"""


class KuhnNode:
    num_cards = 3
    num_players = 2

    def __init__(
        self,
        node_id: int,
        active_player: int,
        strategy: RegretStrategy,
        pot_amount: np.array,
        is_end: bool = False,
        winning_player: int = None,
    ):
        self.node_id = node_id
        self.active_player = active_player

        self.pot_amount = pot_amount
        self.strategy = strategy

        self.is_end = is_end
        self.winning_player = winning_player
        self.next_nodes: List[KuhnNode] = []

    def get_info_state(self, player_cards: np.array) -> InfoState:
        return self.node_id * self.num_cards + player_cards[self.active_player]

    def accumulate_game_regret(
        self,
        player_cards: np.array,
        reach_probs: np.array,
        immediate_regrets: np.array, # immediate_regrets shape (num_info_states, num_actions)
    ) -> np.array:
        """
        Recursively adds info-state's regret from current game
        to each node's self.immediate_regret

        Returns: np.array of expected rewards from this node
        """
        if self.is_end:
            return self.get_rewards(player_cards)

        info_state = self.get_info_state(player_cards)

        # probability of reaching this state, excluding active player's actions
        reach_prob = np.prod(reach_probs[: self.active_player]) * \
            np.prod(reach_probs[self.active_player + 1:])

        active_player_reach_prob = reach_probs[self.active_player]
        # NOT NEEDED? reach_probs[self.active_player] *= self.strategy.strategy[info_state][action]
        # regrets shape (num_actions, )
        active_player_regret = np.zeros(len(self.next_nodes))
        expected_rewards = None
        # NOTE this assumes actions are sequential integers based on order of self.next_nodes
        # An alternative is for each node to save what action index lead to it?
        for action, node in enumerate(self.next_nodes):
            action_prob = self.strategy.strategy[info_state][action]
            reach_probs[self.active_player] = active_player_reach_prob * action_prob
            # NOTE the expected_reward_if_action excludes probability of picking this action
            # expected_rewards_if_action shape=(num_players, )
            expected_rewards_if_action = node.accumulate_game_regret(player_cards, reach_probs, immediate_regrets)

            # update expected rewards of all players
            if expected_rewards is None:
                expected_rewards = action_prob * expected_rewards_if_action
            else:
                expected_rewards += action_prob * expected_rewards_if_action

            # calculate regret of active player
            active_player_regret[action] = expected_rewards_if_action[self.active_player]

        # reset reach_prob to original value
        reach_probs[self.active_player] = active_player_reach_prob

        # subtract the expected reward if using current strategy
        # shape (num_actions, )
        active_player_regret -= expected_rewards[self.active_player]
        active_player_regret *= reach_prob
        immediate_regrets[info_state] += active_player_regret
        return expected_rewards

    def get_rewards(self, player_cards: np.array):
        assert self.is_end
        if self.winning_player is not None:
            winner = self.winning_player
        else:
            winner = np.argmax(player_cards)

        rewards = np.empty(self.num_players)
        rewards[:] = -self.pot_amount
        rewards[winner] += self.pot_amount.sum()

        return rewards

    def play(self, player_cards: np.array):
        if self.is_end:
            return self.get_rewards(player_cards)
        else:
            action = self.strategy.act(self.get_info_state(player_cards))
            return self.next_nodes[action].play(player_cards)
jack = 0
queen = 1
king = 2
pot_1_1 = np.array([1.0, 1.0])
pot_2_1 = np.array([2.0, 1.0])
pot_1_2 = np.array([1.0, 2.0])
pot_2_2 = np.array([2.0, 2.0])

# info states are cross product of (node_id and card of active player) 
num_info_states = 12
num_players = 2

# Each player has 1 card
strategy = RegretStrategy(num_info_states=num_info_states, num_actions=2)

Node = partial(
    KuhnNode,
    strategy=strategy,
)

game_start = Node(node_id=0, active_player=0, pot_amount=pot_1_1)
p1_checked = Node(node_id=1, active_player=1, pot_amount=pot_1_1)
p1_bet = Node(node_id=2, active_player=1, pot_amount=pot_2_1)
p2_bet = Node(node_id=3, active_player=0, pot_amount=pot_1_2)
p2_checked = Node(node_id=4, active_player=0, pot_amount=pot_1_1, is_end=True)
p2_folded = Node(node_id=5, active_player=0, pot_amount=pot_2_1, is_end=True, winning_player=0)
p2_called = Node(node_id=6, active_player=0, pot_amount=pot_2_2, is_end=True)
p1_folded = Node(node_id=7, active_player=1, pot_amount=pot_1_2, is_end=True, winning_player=1)
p1_called = Node(node_id=8, active_player=1, pot_amount=pot_2_2, is_end=True)

game_start.next_nodes.extend([p1_checked, p1_bet])
p1_checked.next_nodes.extend([p2_checked, p2_bet])
p1_bet.next_nodes.extend([p2_folded, p2_called])
p2_bet.next_nodes.extend([p1_folded, p1_called])

reach_probs = np.ones(KuhnNode.num_players)
immediate_regrets = np.zeros((num_info_states, num_players))
for step in range(200_000):
    player_cards = np.random.choice(KuhnNode.num_cards, KuhnNode.num_players, replace=False)
    game_start.accumulate_game_regret(
        player_cards=player_cards,
        reach_probs=reach_probs,
        immediate_regrets=immediate_regrets,
    )
    strategy.update(immediate_regrets)
    
    # Reset game
    immediate_regrets.fill(0.0)
    reach_probs.fill(1.0)

strategy.use_average_strategy()
rewards = np.zeros(num_players)
num_games = 50_000
for step in range(num_games):
    player_cards = np.random.choice(KuhnNode.num_cards, KuhnNode.num_players, replace=False)
    rewards += game_start.play(player_cards)
print(rewards / num_games)
print(strategy.strategy * 100)

# player 2 strategy
np.isclose = partial(np.isclose, atol=.02)
assert all(np.isclose(strategy.strategy[5], np.array([0, 1]))) # Node, card: 1K
assert all(np.isclose(strategy.strategy[8], np.array([0, 1]))) # Node, card: 2K
assert all(np.isclose(strategy.strategy[4], np.array([1, 0]))) # Node, card: 1Q
assert all(np.isclose(strategy.strategy[7], np.array([2/3, 1/3]))) # Node, card: 2Q
assert all(np.isclose(strategy.strategy[3], np.array([2/3, 1/3]))) # Node, card: 1J
assert all(np.isclose(strategy.strategy[6], np.array([1, 0]))) # Node, card: 2J
