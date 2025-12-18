"""
Research Experiment: Dead-end handling strategies in MCTS.

Compares three strategies for handling dead-ends during MCTS rollout:
- penalty: Return a fixed penalty value
- restart: Restart the rollout from the beginning
- partial: Return partial reward based on progress
"""

from math import log, sqrt
from random import choice, random, seed
from statistics import mean, stdev

from benchmarks import get_sequence
from hp_model import (
    DIRECTIONS,
    Conformation,
    get_hh_contacts,
    get_valid_moves,
    is_valid,
)


def mcts_with_strategy(
    sequence,
    max_evaluations,
    dead_end_strategy="penalty",
    penalty_value=-50,
    random_seed=None,
):
    """
    Run MCTS with different dead-end handling strategies.

    Strategies:
    - 'penalty': Return penalty value for dead-ends (default)
    - 'restart': Restart rollout from beginning when hitting dead-end
    - 'partial': Return partial reward based on how far we got
    """
    if random_seed is not None:
        seed(random_seed)

    class Node:
        def __init__(self, state, parent=None):
            self.state = state
            self.parent = parent
            self.children = {}
            self.visits = 0
            self.total_reward = 0.0
            self.untried = get_valid_moves(state)
            self.is_terminal = state.is_complete() or len(self.untried) == 0

        def ucb(self, c=1.0):
            if self.visits == 0:
                return float("inf")
            if self.parent is None:
                return self.total_reward / self.visits
            return (self.total_reward / self.visits) + c * sqrt(
                log(self.parent.visits) / self.visits
            )

        def select_child(self):
            return max(self.children.values(), key=lambda n: n.ucb())

        def expand(self):
            if not self.untried or self.is_terminal:
                return None
            action = choice(self.untried)
            self.untried.remove(action)
            new_coords = self.state.coords.copy()
            dx, dy, dz = DIRECTIONS[action]
            lx, ly, lz = new_coords[-1]
            new_coords.append((lx + dx, ly + dy, lz + dz))
            new_state = Conformation(sequence, coords=new_coords)
            child = Node(new_state, self)
            self.children[action] = child
            return child

        def backprop(self, reward):
            node = self
            while node:
                node.visits += 1
                node.total_reward += reward
                node = node.parent

    def greedy_rollout(conf):
        """Greedy rollout with specified dead-end handling."""
        current = conf.copy()
        max_restarts = 3 if dead_end_strategy == "restart" else 0
        restarts = 0

        while not current.is_complete():
            valid = get_valid_moves(current)

            if not valid:
                # Dead-end reached
                if dead_end_strategy == "restart" and restarts < max_restarts:
                    current = Conformation(sequence, coords=[(0, 0, 0)])
                    restarts += 1
                    continue
                elif dead_end_strategy == "partial":
                    progress = len(current.coords) / len(sequence)
                    return penalty_value * (1 - progress), False
                else:
                    return penalty_value, False

            # Greedy move selection for H residues
            next_idx = len(current.coords)
            if sequence[next_idx] == "H":
                scored = []
                occupied = set(current.coords)
                cx, cy, cz = current.coords[-1]
                for move in valid:
                    dx, dy, dz = DIRECTIONS[move]
                    nx, ny, nz = cx + dx, cy + dy, cz + dz
                    h_count = 0
                    for ddx, ddy, ddz in DIRECTIONS:
                        neighbor = (nx + ddx, ny + ddy, nz + ddz)
                        if neighbor in occupied:
                            try:
                                idx = current.coords.index(neighbor)
                                if sequence[idx] == "H":
                                    h_count += 1
                            except:
                                pass
                    scored.append((h_count, move))
                scored.sort(reverse=True)
                if scored[0][0] > 0 and random() < 0.8:
                    move = scored[0][1]
                else:
                    move = choice(valid)
            else:
                move = choice(valid)

            dx, dy, dz = DIRECTIONS[move]
            lx, ly, lz = current.coords[-1]
            current.coords.append((lx + dx, ly + dy, lz + dz))

        if is_valid(current):
            return get_hh_contacts(current), True
        return penalty_value, False

    # Initialize
    root = Node(Conformation(sequence, coords=[(0, 0, 0)]))
    root.untried = [0]  # Fix first move

    best_reward = float("-inf")
    eval_count = 0

    while eval_count < max_evaluations:
        # Selection
        node = root
        while not node.is_terminal and not node.untried and node.children:
            node = node.select_child()

        # Expansion
        if not node.is_terminal and node.untried:
            node = node.expand()

        if node is None:
            continue

        # Rollout
        reward, success = greedy_rollout(node.state)
        eval_count += 1

        if success and reward > best_reward:
            best_reward = reward

        # Backpropagation
        node.backprop(reward)

    return best_reward, eval_count


def run_experiment(sequence_name, max_evals=50000, num_runs=20):
    """Compare different dead-end handling strategies."""
    sequence = get_sequence(sequence_name)
    strategies = ["penalty", "restart", "partial"]

    print(f"\n{'='*73}")
    print("EXPERIMENT: Dead-End Handling Strategies")
    print(f"Sequence: {sequence_name} (length {len(sequence)})")
    print(f"Evaluations: {max_evals}, Runs: {num_runs}")
    print(f"{'='*73}\n")

    results = {}

    for strategy in strategies:
        rewards = []
        for run in range(num_runs):
            reward, _ = mcts_with_strategy(
                sequence, max_evals, dead_end_strategy=strategy, random_seed=run
            )
            if reward > float("-inf"):
                rewards.append(reward)

        if rewards:
            results[strategy] = {
                "mean": mean(rewards),
                "std": stdev(rewards) if len(rewards) > 1 else 0,
                "max": max(rewards),
                "success_rate": len(rewards) / num_runs,
            }
        else:
            results[strategy] = {"mean": 0, "std": 0, "max": 0, "success_rate": 0}

        print(
            f"{strategy.upper():10} | Mean: {results[strategy]['mean']:.2f} "
            f"± {results[strategy]['std']:.2f} | Max: {results[strategy]['max']} "
            f"| Success: {results[strategy]['success_rate']*100:.0f}%"
        )

    best = max(results.keys(), key=lambda s: results[s]["mean"])
    print(f"\nBest strategy: {best.upper()}")

    return results


if __name__ == "__main__":
    # Default experiment settings
    run_experiment(sequence_name="S1_20", max_evals=50000, num_runs=20)
