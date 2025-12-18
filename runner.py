"""
Experiment runner for comparing algorithms on HP protein folding.

Supports GA, MCTS, and DQN algorithms.
Orchestrates fair comparison with same evaluation budget for all algorithms.
"""

from statistics import mean, median, stdev
from time import perf_counter

from benchmarks import get_best_known, get_optimal_energy, get_sequence, is_verified
from dqn_solver import DQNSolver
from ga_solver import GASolver
from mcts_solver import MCTSSolver


def run_ga_experiment(
    sequence: str,
    max_evaluations: int = 100000,
    population_size: int = 100,
    crossover_rate: float = 0.8,
    mutation_rate: float = 0.05,
    random_seed: int | None = None,
    target_fitness: int | None = None,
) -> dict:
    """
    Run a single GA experiment.

    Returns:
        Dictionary with results: best_fitness, best_energy, evaluations_used, etc.
    """
    solver = GASolver(
        sequence=sequence,
        population_size=population_size,
        crossover_rate=crossover_rate,
        mutation_rate=mutation_rate,
        max_evaluations=max_evaluations,
        random_seed=random_seed,
    )

    solver.solve(target_fitness=target_fitness)
    stats = solver.get_statistics()

    return {
        "best_fitness": stats["best_fitness"],
        "best_energy": stats["best_energy"],
        "evaluations_used": stats["evaluations_used"],
        "evaluations_to_best": stats["evaluations_to_best"],
        "generation": stats["generation"],
        "best_fitness_history": stats["best_fitness_history"],
        "best_conformation": (
            solver.best_individual.conformation if solver.best_individual else None
        ),
    }


def run_mcts_experiment(
    sequence: str,
    max_evaluations: int = 100000,
    exploration_constant: float = 1.414,  # sqrt(2)
    random_seed: int | None = None,
    target_reward: float | None = None,
) -> dict:
    """
    Run a single MCTS experiment.

    Returns:
        Dictionary with results: best_reward, best_energy, evaluations_used, etc.
    """
    solver = MCTSSolver(
        sequence=sequence,
        exploration_constant=exploration_constant,
        max_evaluations=max_evaluations,
        random_seed=random_seed,
    )

    solver.solve(target_reward=target_reward)
    stats = solver.get_statistics()

    return {
        "best_reward": stats["best_reward"],
        "best_energy": stats["best_energy"],
        "evaluations_used": stats["evaluations_used"],
        "evaluations_to_best": stats["evaluations_to_best"],
        "best_reward_history": stats["best_reward_history"],
        "best_conformation": solver.best_conformation,
    }


def run_dqn_experiment(
    sequence: str,
    max_evaluations: int = 100000,
    learning_rate: float = 0.0005,
    epsilon_start: float = 1.0,
    epsilon_end: float = 0.15,  # Higher for more exploration
    epsilon_decay_evals: int | None = None,  # Default: 60% of max_evaluations
    replay_buffer_size: int = 50000,
    batch_size: int = 64,
    target_update_freq: int = 500,
    gamma: float = 0.99,
    grid_size: int = 20,
    training_sequences: list[str] | None = None,
    random_seed: int | None = None,
    target_contacts: int | None = None,
    use_double_dqn: bool = True,
    greedy_exploration_prob: float = 0.3,  # Lower for more random exploration
) -> dict:
    """
    Run a single DQN experiment.

    Returns:
        Dictionary with results: best_contacts, best_energy, evaluations_used, etc.
    """
    # Default epsilon decay to half of max_evaluations
    if epsilon_decay_evals is None:
        epsilon_decay_evals = int(max_evaluations * 0.6)  # 60% of budget for decay

    solver = DQNSolver(
        sequence=sequence,
        max_evaluations=max_evaluations,
        learning_rate=learning_rate,
        epsilon_start=epsilon_start,
        epsilon_end=epsilon_end,
        epsilon_decay_evals=epsilon_decay_evals,
        replay_buffer_size=replay_buffer_size,
        batch_size=batch_size,
        target_update_freq=target_update_freq,
        gamma=gamma,
        grid_size=grid_size,
        training_sequences=training_sequences,
        random_seed=random_seed,
        use_double_dqn=use_double_dqn,
        greedy_exploration_prob=greedy_exploration_prob,
    )

    solver.solve(target_contacts=target_contacts)
    stats = solver.get_statistics()

    return {
        "best_contacts": stats["best_contacts"],
        "best_reward": stats["best_reward"],
        "best_energy": stats["best_energy"],
        "evaluations_used": stats["evaluations_used"],
        "evaluations_to_best": stats["evaluations_to_best"],
        "episodes": stats["episodes"],
        "best_contacts_history": stats["best_contacts_history"],
        "best_conformation": solver.best_conformation,
    }


def run_comparison(
    sequence_name: str,
    algorithm: str,  # "ga", "mcts", or "dqn"
    max_evaluations: int = 100000,
    num_runs: int = 30,
    base_seed: int = 0,
    **kwargs,
) -> dict:
    """
    Run multiple trials of an algorithm on a sequence.

    Args:
        sequence_name: Name of benchmark sequence
        algorithm: "ga", "mcts", or "dqn"
        max_evaluations: Maximum evaluations per run
        num_runs: Number of independent runs
        base_seed: Base seed for random number generation
        **kwargs: Additional algorithm-specific parameters

    Returns:
        Dictionary with aggregated statistics
    """
    sequence = get_sequence(sequence_name)
    optimal_contacts = get_optimal_energy(sequence_name)

    results = []
    success_count = 0
    wall_times = []

    for run in range(num_runs):
        seed = base_seed + run if base_seed is not None else None
        target = optimal_contacts if optimal_contacts is not None else None

        # Time each run
        start_time = perf_counter()

        if algorithm == "ga":
            result = run_ga_experiment(
                sequence=sequence,
                max_evaluations=max_evaluations,
                random_seed=seed,
                target_fitness=target,
                **kwargs,
            )
            best_contacts = result["best_fitness"]
        elif algorithm == "mcts":
            result = run_mcts_experiment(
                sequence=sequence,
                max_evaluations=max_evaluations,
                random_seed=seed,
                target_reward=float(target) if target is not None else None,
                **kwargs,
            )
            best_contacts = (
                int(result["best_reward"])
                if result["best_reward"] > float("-inf")
                else 0
            )
        elif algorithm == "dqn":
            result = run_dqn_experiment(
                sequence=sequence,
                max_evaluations=max_evaluations,
                random_seed=seed,
                target_contacts=target,
            )
            best_contacts = result["best_contacts"]
        else:
            raise ValueError(f"Unknown algorithm: {algorithm}")

        end_time = perf_counter()
        wall_time = end_time - start_time
        result["wall_time"] = wall_time
        wall_times.append(wall_time)

        results.append(result)

        # Check if optimal was found
        if optimal_contacts is not None and best_contacts >= optimal_contacts:
            success_count += 1

    # Aggregate statistics
    best_energies = [r["best_energy"] for r in results]
    best_contacts_list = []
    for r in results:
        if algorithm == "ga":
            best_contacts_list.append(r["best_fitness"])
        elif algorithm == "mcts":
            contacts = int(r["best_reward"]) if r["best_reward"] > float("-inf") else 0
            best_contacts_list.append(contacts)
        elif algorithm == "dqn":
            best_contacts_list.append(r["best_contacts"])

    evaluations_used = [r["evaluations_used"] for r in results]
    evaluations_to_best = [
        r["evaluations_to_best"]
        for r in results
        if r["evaluations_to_best"] is not None
    ]

    # Track best found across all runs
    max_contacts_found = max(best_contacts_list)

    return {
        "sequence_name": sequence_name,
        "sequence_length": len(sequence),
        "algorithm": algorithm,
        "num_runs": num_runs,
        "is_verified": is_verified(sequence_name),
        "optimal_contacts": optimal_contacts,  # None for unverified
        "best_known": get_best_known(sequence_name),
        "max_contacts_found": max_contacts_found,
        "success_rate": (
            success_count / num_runs if optimal_contacts is not None else None
        ),
        "mean_best_contacts": mean(best_contacts_list),
        "std_best_contacts": (
            stdev(best_contacts_list) if len(best_contacts_list) > 1 else 0.0
        ),
        "mean_best_energy": mean(best_energies),
        "std_best_energy": (stdev(best_energies) if len(best_energies) > 1 else 0.0),
        "mean_evaluations_used": mean(evaluations_used),
        "mean_evaluations_to_best": (
            mean(evaluations_to_best) if evaluations_to_best else None
        ),
        "median_evaluations_to_best": (
            median(evaluations_to_best) if evaluations_to_best else None
        ),
        "mean_wall_time": mean(wall_times),
        "std_wall_time": stdev(wall_times) if len(wall_times) > 1 else 0.0,
        "total_wall_time": sum(wall_times),
        "all_results": results,
    }


def compare_algorithms(
    sequence_name: str,
    max_evaluations: int = 100000,
    num_runs: int = 30,
    base_seed: int = 0,
    ga_params: dict | None = None,
    mcts_params: dict | None = None,
) -> dict:
    """
    Compare GA and MCTS on the same sequence.

    Returns:
        Dictionary with comparison results
    """
    ga_params = ga_params or {}
    mcts_params = mcts_params or {}

    ga_results = run_comparison(
        sequence_name=sequence_name,
        algorithm="ga",
        max_evaluations=max_evaluations,
        num_runs=num_runs,
        base_seed=base_seed,
        **ga_params,
    )

    mcts_results = run_comparison(
        sequence_name=sequence_name,
        algorithm="mcts",
        max_evaluations=max_evaluations,
        num_runs=num_runs,
        base_seed=base_seed + 1000,  # Different seed for MCTS
        **mcts_params,
    )

    return {
        "sequence_name": sequence_name,
        "ga": ga_results,
        "mcts": mcts_results,
        "comparison": {
            "ga_better_contacts": ga_results["mean_best_contacts"]
            > mcts_results["mean_best_contacts"],
            "mcts_better_contacts": mcts_results["mean_best_contacts"]
            > ga_results["mean_best_contacts"],
            "ga_better_success": (ga_results["success_rate"] or 0)
            > (mcts_results["success_rate"] or 0),
            "mcts_better_success": (mcts_results["success_rate"] or 0)
            > (ga_results["success_rate"] or 0),
            "contact_difference": ga_results["mean_best_contacts"]
            - mcts_results["mean_best_contacts"],
        },
    }


def run_multi_algorithm_comparison(
    sequence_name: str,
    algorithms: list[str],
    max_evaluations: int = 100000,
    num_runs: int = 30,
    base_seed: int = 0,
) -> dict:
    """
    Compare multiple algorithms on the same sequence.

    Args:
        sequence_name: Name of benchmark sequence
        algorithms: list of algorithm names (e.g., ['ga', 'mcts'])
        max_evaluations: Maximum evaluations per run
        num_runs: Number of independent runs per algorithm
        base_seed: Base seed for random number generation

    Returns:
        Dictionary with results for each algorithm and comparison metrics
    """
    results = {}

    for i, algo in enumerate(algorithms):
        # Use different seed offsets for each algorithm
        seed = base_seed + (i * 1000)
        results[algo] = run_comparison(
            sequence_name=sequence_name,
            algorithm=algo,
            max_evaluations=max_evaluations,
            num_runs=num_runs,
            base_seed=seed,
        )

    # Find best algorithm by mean contacts
    best_algo = max(algorithms, key=lambda a: results[a]["mean_best_contacts"])
    best_contacts = results[best_algo]["mean_best_contacts"]

    # Find most efficient algorithm
    most_efficient = min(algorithms, key=lambda a: results[a]["mean_evaluations_used"])

    return {
        "sequence_name": sequence_name,
        "algorithms": algorithms,
        "results": results,
        "summary": {
            "best_algorithm": best_algo,
            "best_mean_contacts": best_contacts,
            "most_efficient": most_efficient,
        },
    }


def print_multi_comparison_summary(comparison: dict):
    """Print a summary comparing multiple algorithms."""
    results = comparison["results"]
    algorithms = comparison["algorithms"]
    summary = comparison["summary"]

    # Get sequence info from first algorithm's results
    first_algo = algorithms[0]
    is_verified_seq = results[first_algo].get("is_verified", False)
    optimal = results[first_algo].get("optimal_contacts")
    seq_length = results[first_algo].get("sequence_length", "?")

    print(f"\n{'='*70}")
    print(f"Comparison for {comparison['sequence_name']} (length {seq_length})")
    if is_verified_seq:
        print(f"[VERIFIED BENCHMARK - Optimal: {optimal} contacts]")
    else:
        print(f"[UNVERIFIED - Relative comparison only]")
    print(f"{'='*70}")

    # Print results for each algorithm
    for algo in algorithms:
        r = results[algo]
        print(f"\n{algo.upper()} Results:")
        print(
            f"  Mean best contacts: {r['mean_best_contacts']:.2f} ± {r['std_best_contacts']:.2f}"
        )
        print(f"  Max found: {r['max_contacts_found']}")
        if is_verified_seq and optimal:
            pct = r["mean_best_contacts"] / optimal * 100
            print(f"  % of optimal: {pct:.1f}%")
            if r["success_rate"] is not None:
                print(f"  Success rate (found optimal): {r['success_rate']*100:.1f}%")
        print(f"  Mean evaluations: {r['mean_evaluations_used']:.0f}")
        print(
            f"  Avg wall time: {r['mean_wall_time']:.2f}s ± {r['std_wall_time']:.2f}s"
        )

    # Print comparison summary
    print(f"\n{'─'*70}")
    print("Summary:")
    print(
        f"  Best solution quality: {summary['best_algorithm'].upper()} ({summary['best_mean_contacts']:.2f} contacts)"
    )
    print(f"  Most efficient: {summary['most_efficient'].upper()}")

    # Best overall found
    best_overall = max(r["max_contacts_found"] for r in results.values())
    print(f"\nBest found overall: {best_overall} contacts")
    print(f"{'='*70}\n")


def print_results_summary(results: dict):
    """Print a summary of experiment results."""
    print(f"\n{'='*60}")
    print(
        f"Results for {results['sequence_name']} (length {results['sequence_length']})"
    )
    print(f"Algorithm: {results['algorithm'].upper()}")
    print(f"{'='*60}")
    print(f"Number of runs: {results['num_runs']}")
    if results["optimal_contacts"] is not None:
        print(f"Optimal H-H contacts: {results['optimal_contacts']}")
        print(f"Success rate: {results['success_rate']*100:.1f}%")
    print(f"\nBest H-H contacts found:")
    print(
        f"  Mean: {results['mean_best_contacts']:.2f} ± {results['std_best_contacts']:.2f}"
    )
    print(f"\nBest energy found:")
    print(
        f"  Mean: {results['mean_best_energy']:.2f} ± {results['std_best_energy']:.2f}"
    )
    print(f"\nEvaluations:")
    print(f"  Mean used: {results['mean_evaluations_used']:.0f}")
    if results["mean_evaluations_to_best"] is not None:
        print(f"  Mean to best: {results['mean_evaluations_to_best']:.0f}")
        print(f"  Median to best: {results['median_evaluations_to_best']:.0f}")
    print(f"\nWall Time:")
    print(
        f"  Avg per run: {results['mean_wall_time']:.2f}s ± {results['std_wall_time']:.2f}s"
    )
    print(f"  Total: {results['total_wall_time']:.2f}s")
    print(f"{'='*60}\n")


def print_comparison_summary(comparison: dict):
    """Print a summary comparing GA and MCTS."""
    ga = comparison["ga"]
    mcts = comparison["mcts"]
    comp = comparison["comparison"]
    is_verified_seq = ga.get("is_verified", False)
    optimal = ga["optimal_contacts"]  # Only set for verified sequences

    print(f"\n{'='*70}")
    print(
        f"Comparison for {comparison['sequence_name']} (length {ga['sequence_length']})"
    )
    if is_verified_seq:
        print(f"[VERIFIED BENCHMARK - Optimal: {optimal} contacts]")
    else:
        print(f"[UNVERIFIED - Relative comparison only]")
    print(f"{'='*70}")

    # Best found across both algorithms
    best_overall = max(ga["max_contacts_found"], mcts["max_contacts_found"])

    print(f"\nGA Results:")
    print(
        f"  Mean best contacts: {ga['mean_best_contacts']:.2f} ± {ga['std_best_contacts']:.2f}"
    )
    print(f"  Max found: {ga['max_contacts_found']}")
    if is_verified_seq and optimal:
        ga_pct = ga["mean_best_contacts"] / optimal * 100
        print(f"  % of optimal: {ga_pct:.1f}%")
        print(f"  Success rate (found optimal): {ga['success_rate']*100:.1f}%")
    print(f"  Mean evaluations: {ga['mean_evaluations_used']:.0f}")
    print(f"  Avg wall time: {ga['mean_wall_time']:.2f}s ± {ga['std_wall_time']:.2f}s")

    print(f"\nMCTS Results:")
    print(
        f"  Mean best contacts: {mcts['mean_best_contacts']:.2f} ± {mcts['std_best_contacts']:.2f}"
    )
    print(f"  Max found: {mcts['max_contacts_found']}")
    if is_verified_seq and optimal:
        mcts_pct = mcts["mean_best_contacts"] / optimal * 100
        print(f"  % of optimal: {mcts_pct:.1f}%")
        print(f"  Success rate (found optimal): {mcts['success_rate']*100:.1f}%")
    print(f"  Mean evaluations: {mcts['mean_evaluations_used']:.0f}")
    print(
        f"  Avg wall time: {mcts['mean_wall_time']:.2f}s ± {mcts['std_wall_time']:.2f}s"
    )

    print(f"\nRelative Comparison:")
    diff = comp["contact_difference"]
    if abs(diff) < 0.1:
        print(f"  Both algorithms found similar solutions")
    elif diff > 0:
        print(f"  GA found better solutions (by {diff:.2f} contacts)")
    else:
        print(f"  MCTS found better solutions (by {-diff:.2f} contacts)")

    # Efficiency comparison
    ga_evals = ga["mean_evaluations_used"]
    mcts_evals = mcts["mean_evaluations_used"]
    if ga_evals < mcts_evals:
        print(
            f"  GA was more efficient ({ga_evals:.0f} vs {mcts_evals:.0f} evaluations)"
        )
    elif mcts_evals < ga_evals:
        print(
            f"  MCTS was more efficient ({mcts_evals:.0f} vs {ga_evals:.0f} evaluations)"
        )

    if (
        is_verified_seq
        and ga["success_rate"] is not None
        and mcts["success_rate"] is not None
    ):
        if comp["ga_better_success"]:
            print(
                f"  GA had higher success rate ({ga['success_rate']*100:.1f}% vs {mcts['success_rate']*100:.1f}%)"
            )
        elif comp["mcts_better_success"]:
            print(
                f"  MCTS had higher success rate ({mcts['success_rate']*100:.1f}% vs {ga['success_rate']*100:.1f}%)"
            )

    print(f"\nBest found overall: {best_overall} contacts")
    print(f"{'='*70}\n")
