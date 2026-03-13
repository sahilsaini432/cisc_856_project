import math
import numpy as np


class UCTStrategy:
    """UCT with reward normalization to [0, 1] based on observed min/max rewards in the tree."""

    def __init__(self, exploration_constant):
        self.exploration_constant = exploration_constant
        self.min_value = float("inf")
        self.max_value = float("-inf")

    def reset(self):
        self.min_value = float("inf")
        self.max_value = float("-inf")

    def update(self, reward):
        self.min_value = min(self.min_value, reward)
        self.max_value = max(self.max_value, reward)

    def score(self, node) -> float:
        if node.visits == 0:
            return float("inf")
        q = node.value / node.visits
        value_range = self.max_value - self.min_value
        normalized_q = (q - self.min_value) / value_range if value_range > 1e-8 else 0.5
        return normalized_q + self.exploration_constant * math.sqrt(math.log(node.parent.visits) / node.visits)

    def best_child(self, node):
        scores = [self.score(child) for child in node.children]
        max_score = max(scores)
        best_children = [child for child, s in zip(node.children, scores) if s == max_score]
        return np.random.choice(best_children)


class UCB1Strategy:
    """UCB1 — Q(v) + C * sqrt(ln(N_parent) / N_v). No reward normalization."""

    def __init__(self, exploration_constant):
        self.exploration_constant = exploration_constant

    def reset(self):
        pass

    def update(self, _):
        pass

    def score(self, node) -> float:
        if node.visits == 0:
            return float("inf")
        q = node.value / node.visits
        return q + self.exploration_constant * math.sqrt(math.log(node.parent.visits) / node.visits)

    def best_child(self, node):
        scores = [self.score(child) for child in node.children]
        max_score = max(scores)
        best_children = [child for child, s in zip(node.children, scores) if s == max_score]
        return np.random.choice(best_children)


class PUCTStrategy:
    """PUCT — Q(v) + C * P(a) * sqrt(N_parent) / (1 + N_v). Used in AlphaGo/AlphaZero.
    Requires nodes to have a `prior` attribute."""

    def __init__(self, exploration_constant):
        self.exploration_constant = exploration_constant

    def reset(self):
        pass

    def update(self, _):
        pass

    def score(self, node) -> float:
        q = node.value / node.visits if node.visits > 0 else 0.0
        return q + self.exploration_constant * node.prior * math.sqrt(node.parent.visits) / (1 + node.visits)

    def best_child(self, node):
        scores = [self.score(child) for child in node.children]
        max_score = max(scores)
        best_children = [child for child, s in zip(node.children, scores) if s == max_score]
        return np.random.choice(best_children)
