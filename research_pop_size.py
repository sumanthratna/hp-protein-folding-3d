"""
Research Experiment: GA population size vs solution quality trade-off.

Tests different GA population sizes with the same total evaluation budget
to find the optimal balance between population diversity and generations.
"""

from statistics import mean, stdev

from benchmarks import get_sequence
from ga_solver import GASolver


def run_experiment(sequence_name, total_budget=100000, num_runs=20):
    """Test different GA population sizes with same total evaluation budget."""
    sequence = get_sequence(sequence_name)

    # Different population sizes
    # With fixed budget, larger pop = fewer generations
    pop_sizes = [50, 100, 200, 400]

    print(f"\n{'='*70}")
    print("EXPERIMENT: GA Population Size Trade-off")
    print(f"Sequence: {sequence_name} (length {len(sequence)})")
    print(f"Total budget: {total_budget}, Runs: {num_runs}")
    print(f"{'='*70}\n")

    results = {}

    for pop_size in pop_sizes:
        rewards = []
        generations = []

        for run in range(num_runs):
            solver = GASolver(
                sequence=sequence,
                population_size=pop_size,
                max_evaluations=total_budget,
                random_seed=run,
            )
            solver.solve()

            if solver.best_individual:
                rewards.append(solver.best_individual.fitness)
                generations.append(solver.generation)

        if rewards:
            results[pop_size] = {
                "mean": mean(rewards),
                "std": stdev(rewards) if len(rewards) > 1 else 0,
                "max": max(rewards),
                "avg_generations": mean(generations),
            }

        print(
            f"Pop={pop_size:3d} | Mean: {results[pop_size]['mean']:.2f} "
            f"± {results[pop_size]['std']:.2f} | Max: {results[pop_size]['max']} "
            f"| Gens: {results[pop_size]['avg_generations']:.0f}"
        )

    best = max(results.keys(), key=lambda p: results[p]["mean"])
    print(f"\nBest population size: {best}")

    return results
