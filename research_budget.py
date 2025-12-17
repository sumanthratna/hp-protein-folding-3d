"""
Research Experiment: Equal compute budget comparison.

Compares GA vs MCTS performance at different compute budgets
to understand how each algorithm scales with more evaluations.
"""

from statistics import mean, stdev

from benchmarks import get_sequence
from ga_solver import GASolver
from mcts_solver import MCTSSolver


def run_experiment(sequence_name, num_runs=20):
    """Compare GA vs MCTS at different compute budgets."""
    sequence = get_sequence(sequence_name)

    budgets = [10000, 25000, 50000, 100000]

    print(f"\n{'='*70}")
    print("EXPERIMENT: Equal Compute Budget Scaling")
    print(f"Sequence: {sequence_name} (length {len(sequence)})")
    print(f"Runs per budget: {num_runs}")
    print(f"{'='*70}\n")

    results = {"ga": {}, "mcts": {}}

    for budget in budgets:
        # GA runs
        ga_rewards = []
        for run in range(num_runs):
            solver = GASolver(
                sequence=sequence, max_evaluations=budget, random_seed=run
            )
            solver.solve()
            if solver.best_individual:
                ga_rewards.append(solver.best_individual.fitness)

        # MCTS runs
        mcts_rewards = []
        for run in range(num_runs):
            solver = MCTSSolver(
                sequence=sequence, max_evaluations=budget, random_seed=run + 1000
            )
            solver.solve()
            if solver.best_reward > float("-inf"):
                mcts_rewards.append(solver.best_reward)

        results["ga"][budget] = {
            "mean": mean(ga_rewards) if ga_rewards else 0,
            "std": stdev(ga_rewards) if len(ga_rewards) > 1 else 0,
        }
        results["mcts"][budget] = {
            "mean": mean(mcts_rewards) if mcts_rewards else 0,
            "std": stdev(mcts_rewards) if len(mcts_rewards) > 1 else 0,
        }

        ga_mean = results["ga"][budget]["mean"]
        mcts_mean = results["mcts"][budget]["mean"]
        winner = (
            "MCTS" if mcts_mean > ga_mean else "GA" if ga_mean > mcts_mean else "TIE"
        )

        print(
            f"Budget={budget:6d} | GA: {ga_mean:.2f} | MCTS: {mcts_mean:.2f} "
            f"| Winner: {winner}"
        )

    return results
