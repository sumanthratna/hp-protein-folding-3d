# 3D HP Protein Folding: GA vs MCTS Comparison

**CPSC 5740 Final Project**  
**Author:** Sumanth Ratna
**Date:** December 2024

---

## Problem Description

The **3D HP (Hydrophobic-Polar) protein folding model** is a simplified representation of protein folding where:

- A protein is a sequence of **H** (hydrophobic) and **P** (polar) monomers
- The protein folds on a **3D cubic lattice** as a self-avoiding walk
- The goal is to **maximize non-sequential H-H contacts** (minimize energy)

This is an **NP-hard combinatorial optimization problem**. The search space grows exponentially with sequence length—each monomer can move in 5 directions (excluding backtracking), yielding O(5^n) possible conformations.

### Energy Function

```
E(conformation) = -1 × (number of non-sequential H-H contacts)
```

A contact occurs when two H monomers are adjacent on the lattice but not consecutive in the sequence.

---

## Research Questions

**Primary Question:** How do Genetic Algorithm (GA) and Monte Carlo Tree Search (MCTS) compare in solution quality and computational efficiency for 3D HP protein folding?

**Additional Questions:**
- **Q2:** How do different dead-end handling strategies affect MCTS performance?
- **Q3:** What is the optimal population size vs. generations trade-off for GA?
- **Q4:** How does performance scale with increasing compute budget (10k–100k evaluations)?

---

## Algorithms Implemented

### Genetic Algorithm (GA)

| Component | Implementation |
|-----------|----------------|
| **Encoding** | Sequence of absolute direction moves (±X, ±Y, ±Z) |
| **Initialization** | 30% greedy (prioritizes H-H contacts), 70% random valid walks |
| **Crossover** | Multi-point crossover with validity repair |
| **Mutation** | Segment regrowth + single-point mutations |
| **Selection** | Tournament selection (size 5) with elitism |
| **Local Search** | Hill climbing on best individuals |

### Monte Carlo Tree Search (MCTS)

| Component | Implementation |
|-----------|----------------|
| **Selection** | UCT with exploration constant C=0.5 |
| **Expansion** | Smart ordering (moves creating H-H contacts first) |
| **Simulation** | Greedy rollout (80% bias toward H-H contacts) |
| **Backpropagation** | Max reward tracking |
| **Enhancements** | Multiple rollouts per expansion, periodic local search |

---

## Quick Start

```bash
# 1. Build (creates virtual environment and installs dependencies)
./build.sh

# 2. Run test demo (~1 minute)
./test.sh

# 3. Run custom experiments
source venv/bin/activate
python3 main.py -s UM1_27 -a ga,mcts -e 50000 -n 5 --plot
```

---

## Project Structure

```
├── hp_model.py              # Core HP model: conformations, energy calculation
├── ga_solver.py             # Genetic Algorithm implementation
├── mcts_solver.py           # Monte Carlo Tree Search implementation
├── benchmarks.py            # Benchmark sequences with known optima
├── runner.py                # Experiment runner and statistics
├── visualize.py             # 3D visualization and comparison plots
├── main.py                  # Command-line interface (single entry point)
├── research_dead_end.py     # Experiment: MCTS dead-end handling strategies
├── research_pop_size.py     # Experiment: GA population size trade-off
├── research_budget.py       # Experiment: GA vs MCTS budget scaling
├── build.sh                 # Build script (creates venv, installs deps)
├── test.sh                  # Test script with demo and results summary
├── requirements.txt         # Python dependencies
└── results/                 # Generated outputs (plots saved here with --plot)
```

---

## Benchmark Sequences

### Verified Benchmarks (Unger & Moult 27-mers)

These have **proven optimal** contact counts:

| Name   | Sequence                      | Length | Optimal Contacts |
|--------|-------------------------------|--------|------------------|
| UM1_27 | `HHHHHHHHHHHHHPPPHHHHHHHHHHH` | 27     | 28               |
| UM2_27 | `HHHHHHHHHHHHPPPHHHHHHHHHHHH` | 27     | 28               |

### Unverified Benchmarks (Chen & Huang)

Used for relative comparison only:

| Name  | Length | Description |
|-------|--------|-------------|
| S1_20 | 20     | Short sequence |
| S2_24 | 24     | Medium sequence |
| S3_25 | 25     | Medium sequence |

---

## Usage

### Command-Line Interface

```bash
# List available sequences
python3 main.py --list-sequences

# List available algorithms
python3 main.py --list-algorithms

# Run comparison
python3 main.py -s <sequence> -a <algorithms> -e <evaluations> -n <runs> [--plot]
```

### CLI Options

| Flag | Description | Default |
|------|-------------|---------|
| `-s, --sequence` | Benchmark sequence name | Required |
| `-a, --algorithms` | Comma-separated list (e.g., `ga,mcts`) | All |
| `-e, --max-evaluations` | Max fitness evaluations per run | 10000 |
| `-n, --num-runs` | Number of independent runs | 5 |
| `--plot` | Generate visualization plots | False |

### Examples

```bash
# Quick comparison on short sequence
python3 main.py -s S1_20 -a ga,mcts -e 5000 -n 3

# Full experiment on verified benchmark with plots
python3 main.py -s UM1_27 -a ga,mcts -e 100000 -n 10 --plot

# Single algorithm run
python3 main.py -s UM2_27 -a mcts -e 50000 -n 5
```

---

## Key Results

### Performance Comparison (100k evaluations, 10 runs)

| Sequence | Optimal | GA Mean | GA Best | MCTS Mean | MCTS Best | GA Time | MCTS Time | Winner |
|----------|---------|---------|---------|-----------|-----------|---------|-----------|--------|
| UM1_27   | 28      | 22.00   | 23      | 22.97     | 23        | ~4.2s   | ~5.3s     | **MCTS** |
| UM2_27   | 28      | 23.23   | 24      | 24.00     | 24        | ~4.1s   | ~5.2s     | **MCTS** |

### Key Findings

1. **MCTS consistently outperforms GA** on all tested sequences
2. **MCTS achieves 82–86% of optimal** vs GA's 79–83%
3. **GA is ~20% faster** per evaluation but produces worse solutions
4. **Sequential decision-making** (MCTS) better suits the chain-growth nature of protein folding
5. **Dead-end handling** has minimal impact when using greedy policies
6. **GA benefits from larger populations** (200–400) but cannot match MCTS quality

---

## Output Metrics

The experiment runner reports:

- **Mean best contacts** ± std deviation
- **Max contacts found** across all runs
- **% of optimal** (for verified benchmarks)
- **Success rate** (% of runs finding optimal)
- **Mean evaluations used**
- **Average wall time** per run

---

## Visualization

When run with `--plot`, the tool generates:

1. **3D conformation plots** showing the best fold found by each algorithm
2. **Comparison bar charts** showing mean contacts with error bars

Example:
```bash
python3 main.py -s UM1_27 -a ga,mcts -e 50000 -n 5 --plot
# Generates: results/ga_best_UM1_27.png, results/mcts_best_UM1_27.png, results/comparison_UM1_27.png
```

---

## Reproducing Full Results

```bash
# Activate environment
source venv/bin/activate

# Run full comparison on all benchmarks
for seq in S1_20 S2_24 S3_25 UM1_27 UM2_27; do
  python3 main.py -s $seq -a ga,mcts -e 100000 -n 10 --plot
done

# Run additional research experiments (Q2-Q4)
python3 main.py --research dead-end -s S1_20 -e 50000 -n 20
python3 main.py --research pop-size -s S1_20 -e 100000 -n 20
python3 main.py --research budget-scaling -s S1_20 -n 20
```

---

## Dependencies

- Python 3.10+ (tested with 3.12)
- NumPy ≥ 1.21.0
- Matplotlib ≥ 3.5.0

Install via:
```bash
./build.sh
# or manually:
python3 -m venv venv && source venv/bin/activate && pip install -r requirements.txt
```

---

## References

1. Unger, R., & Moult, J. (1993). *Genetic algorithms for protein folding simulations.* Journal of Molecular Biology, 231(1), 75-81.

2. Chen, M., & Huang, W. Q. (2005). *A branch and bound algorithm for the protein folding problem in the HP lattice model.* Genomics, Proteomics & Bioinformatics, 3(4), 225-230.

3. Browne, C., et al. (2012). *A Survey of Monte Carlo Tree Search Methods.* IEEE Transactions on Computational Intelligence and AI in Games, 4(1), 1-43.

4. Dill, K. A. (1985). *Theory for the folding and stability of globular proteins.* Biochemistry, 24(6), 1501-1509.

