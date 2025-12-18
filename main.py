#!/usr/bin/env python3

import argparse
import os
import sys

from benchmarks import get_unverified_sequence_names, get_verified_sequence_names
from runner import (
    print_multi_comparison_summary,
    run_comparison,
    run_multi_algorithm_comparison,
)
from visualize import plot_conformation_3d


AVAILABLE_ALGORITHMS = {
    "ga": "Genetic Algorithm",
    "mcts": "Monte Carlo Tree Search",
    "dqn": "Deep Q-Network",
}


def parse_algorithms(algo_string: str) -> list:
    """Parse comma-separated algorithm names and validate them."""
    if not algo_string:
        return list(AVAILABLE_ALGORITHMS.keys())  # Default: all algorithms

    algorithms = [a.strip().lower() for a in algo_string.split(",")]

    # Validate each algorithm
    invalid = [a for a in algorithms if a not in AVAILABLE_ALGORITHMS]
    if invalid:
        valid_list = ", ".join(AVAILABLE_ALGORITHMS.keys())
        print(f"Error: Unknown algorithm(s): {', '.join(invalid)}")
        print(f"Available algorithms: {valid_list}")
        sys.exit(1)

    return algorithms


def main():
    parser = argparse.ArgumentParser(
        description="3D HP Protein Folding: Algorithm Comparison",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Compare all algorithms on a sequence
  python main.py -s S1_20
  
  # Run only GA
  python main.py -s S1_20 -a ga
  
  # Compare GA and MCTS with custom parameters
  python main.py -s UM1_27 -a ga,mcts -e 100000 -n 30
  
  # List available sequences and algorithms
  python main.py --list-sequences
  python main.py --list-algorithms
        """,
    )

    parser.add_argument(
        "--sequence",
        "-s",
        type=str,
        help="Sequence name from benchmarks (e.g., S1_20, UM1_27)",
    )
    parser.add_argument(
        "--algorithms",
        "-a",
        type=str,
        default=None,
        metavar="ALG1,ALG2,...",
        help=f'Comma-separated list of algorithms to run. Available: {", ".join(AVAILABLE_ALGORITHMS.keys())}. Default: all',
    )
    parser.add_argument(
        "--max-evaluations",
        "-e",
        type=int,
        default=10000,
        help="Maximum number of evaluations per run (default: 10000)",
    )
    parser.add_argument(
        "--num-runs",
        "-n",
        type=int,
        default=5,
        help="Number of independent runs per algorithm (default: 5)",
    )
    parser.add_argument(
        "--list-sequences",
        action="store_true",
        help="List all available benchmark sequences",
    )
    parser.add_argument(
        "--list-algorithms", action="store_true", help="List all available algorithms"
    )
    parser.add_argument(
        "--plot", action="store_true", help="Generate visualization plots"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Base random seed for reproducibility (default: 0)",
    )
    parser.add_argument(
        "--pretrain",
        action="store_true",
        help="Pre-train DQN on diverse sequences before fine-tuning (improves DQN performance)",
    )
    parser.add_argument(
        "--pretrain-evals",
        type=int,
        default=100000,
        help="Number of evaluations for DQN pre-training phase (default: 100000)",
    )

    args = parser.parse_args()

    # Handle listing options
    if args.list_algorithms:
        print("Available algorithms:")
        for key, name in AVAILABLE_ALGORITHMS.items():
            print(f"  {key:8} - {name}")
        return

    if args.list_sequences:
        print("Available benchmark sequences:\n")
        print("VERIFIED (exact optimal known):")
        for name in sorted(get_verified_sequence_names()):
            print(f"  {name}")
        print("\nUNVERIFIED (relative comparison only):")
        for name in sorted(get_unverified_sequence_names()):
            print(f"  {name}")
        return

    if not args.sequence:
        print("Error: --sequence is required (use --list-sequences to see options)")
        sys.exit(1)

    # Parse and validate algorithms
    algorithms = parse_algorithms(args.algorithms)

    # Print header
    print(f"\n{'='*70}")
    print("3D HP Protein Folding: Computational Intelligence Comparison")
    print(f"{'='*70}")
    print(f"\nResearch Question:")
    print("How do different algorithms compare in solution quality and computational")
    print("efficiency for 3D HP protein folding across sequences of varying lengths?")
    print(f"\n{'='*70}\n")

    # Ensure results directory exists
    os.makedirs("results", exist_ok=True)

    print(f"Sequence: {args.sequence}")
    print(f"Algorithms: {', '.join(algorithms)}")
    print(f"Max evaluations: {args.max_evaluations}")
    print(f"Number of runs: {args.num_runs}")
    print(f"Random seed: {args.seed}")
    if "dqn" in algorithms and args.pretrain:
        print(f"DQN pre-training: {args.pretrain_evals} evaluations")
    print()

    # Build extra kwargs for DQN pre-training
    dqn_kwargs = {}
    if args.pretrain:
        dqn_kwargs["pretrain"] = True
        dqn_kwargs["pretrain_evaluations"] = args.pretrain_evals

    if len(algorithms) == 1:
        # Single algorithm run
        algo = algorithms[0]
        print(f"Running {AVAILABLE_ALGORITHMS[algo]}...\n")

        results = run_comparison(
            sequence_name=args.sequence,
            algorithm=algo,
            max_evaluations=args.max_evaluations,
            num_runs=args.num_runs,
            base_seed=args.seed,
            **(dqn_kwargs if algo == "dqn" else {}),
        )

        from runner import print_results_summary

        print_results_summary(results)

        # Plot best conformation
        if (
            args.plot
            and results["all_results"]
            and results["all_results"][0].get("best_conformation")
        ):
            try:
                best_conf = results["all_results"][0]["best_conformation"]
                plot_conformation_3d(
                    best_conf,
                    title=f"{algo.upper()} Best: {args.sequence}",
                    show=False,
                    save_path=f"results/{algo}_best_{args.sequence}.png",
                )
                print(f"Saved plot to results/{algo}_best_{args.sequence}.png")
            except Exception as e:
                print(f"Warning: Could not generate plot: {e}")

    else:
        # Multi-algorithm comparison (2+)
        comparison = run_multi_algorithm_comparison(
            sequence_name=args.sequence,
            algorithms=algorithms,
            max_evaluations=args.max_evaluations,
            num_runs=args.num_runs,
            base_seed=args.seed,
            dqn_kwargs=dqn_kwargs if "dqn" in algorithms else None,
        )

        print_multi_comparison_summary(comparison)

        if args.plot:
            # Plot best conformations for each algorithm
            for algo in algorithms:
                try:
                    results = comparison["results"].get(algo, {})
                    if results.get("all_results") and results["all_results"][0].get(
                        "best_conformation"
                    ):
                        best_conf = results["all_results"][0]["best_conformation"]
                        plot_conformation_3d(
                            best_conf,
                            title=f"{algo.upper()} Best: {args.sequence}",
                            show=False,
                            save_path=f"results/{algo}_best_{args.sequence}.png",
                        )
                        print(
                            f"Saved {algo.upper()} best conformation to results/{algo}_best_{args.sequence}.png"
                        )
                except Exception as e:
                    print(f"Warning: Could not generate {algo} plot: {e}")


if __name__ == "__main__":
    main()
