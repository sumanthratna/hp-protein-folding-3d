#!/bin/bash
# Test script for 3D HP Protein Folding project
# CPSC 5740 Final Project - Sumanth Ratna (sr2437)

set -e

echo "=========================================="
echo "3D HP Protein Folding - Final Project"
echo "CPSC 5740 - Sumanth Ratna (sr2437)"
echo "=========================================="
echo ""

# Activate virtual environment
if [ -d "venv" ]; then
    source venv/bin/activate
else
    echo "Error: Virtual environment not found. Please run ./build.sh first."
    exit 1
fi

# Create results directory
mkdir -p results

# ============================================
# PROBLEM DESCRIPTION
# ============================================
echo "PROBLEM DESCRIPTION"
echo "-------------------"
echo "The 3D HP (Hydrophobic-Polar) protein folding model represents a protein"
echo "as a sequence of H (hydrophobic) and P (polar) monomers that fold on a 3D"
echo "cubic lattice. The goal is to find a self-avoiding walk that maximizes"
echo "the number of non-sequential H-H contacts (minimizes energy)."
echo ""
echo "This is an NP-hard combinatorial optimization problem with a search space"
echo "that grows exponentially with sequence length (5^n possible conformations)."
echo ""

# ============================================
# ALGORITHMS IMPLEMENTED
# ============================================
echo "ALGORITHMS IMPLEMENTED"
echo "----------------------"
echo "1. Genetic Algorithm (GA)"
echo "   - Population-based evolution with greedy initialization"
echo "   - Multi-point crossover and segment regrowth mutation"
echo "   - Tournament selection with elitism"
echo "   - Local search (hill climbing) refinement"
echo ""
echo "2. Monte Carlo Tree Search (MCTS)"
echo "   - UCT selection with exploration constant C=0.5"
echo "   - Greedy rollout policy (prioritizes H-H contacts)"
echo "   - Smart expansion ordering (promising moves first)"
echo "   - Periodic local search refinement"
echo ""

# ============================================
# RESEARCH QUESTIONS
# ============================================
echo "RESEARCH QUESTIONS"
echo "------------------"
echo "Primary: How do GA and MCTS compare in solution quality and computational"
echo "         efficiency for 3D HP protein folding?"
echo ""
echo "Additional questions explored:"
echo "  Q2: How do different dead-end handling strategies affect MCTS?"
echo "  Q3: What is the optimal population size vs. generations trade-off for GA?"
echo "  Q4: How does performance scale with compute budget (10k-100k evals)?"
echo ""

# ============================================
# QUICK DEMO (runs in ~1 minute)
# ============================================
echo "=========================================="
echo "QUICK DEMO"
echo "=========================================="
echo ""
echo "Running quick comparison on sequence S1_20 (20 monomers)..."
echo "Budget: 5000 evaluations, 3 runs per algorithm"
echo ""

python3 main.py --sequence S1_20 --algorithms ga,mcts --max-evaluations 5000 --num-runs 3

echo ""

# ============================================
# MAIN RESULTS SUMMARY
# ============================================
echo "=========================================="
echo "MAIN RESULTS SUMMARY (from full experiments)"
echo "=========================================="
echo ""
echo "Full experiments used 100,000 evaluations and 10 runs per algorithm."
echo "Results are from verified Unger & Moult 27-mer sequences with known optima."
echo ""
echo "Sequence   | Optimal | GA Mean | GA Best | MCTS Mean | MCTS Best | Winner"
echo "-----------|---------|---------|---------|-----------|-----------|-------"
echo "UM1_27     |   28    |  22.00  |   23    |   22.97   |    23     | MCTS"
echo "UM2_27     |   28    |  23.23  |   24    |   24.00   |    24     | MCTS"
echo ""
echo "Key Findings:"
echo "  - MCTS consistently outperforms GA on all tested sequences"
echo "  - MCTS achieves 82-86% of optimal vs GA's 79-83%"
echo "  - Both algorithms struggle to reach global optimum on 27-mers"
echo "  - GA benefits from larger population sizes (200-400)"
echo "  - Dead-end handling strategy has minimal impact (greedy avoids dead-ends)"
echo ""

# ============================================
# INSTRUCTIONS FOR FULL REPRODUCTION
# ============================================
echo "=========================================="
echo "TO REPRODUCE FULL RESULTS"
echo "=========================================="
echo ""
echo "# Full comparison on verified sequences (takes ~10 minutes each):"
echo "python3 main.py -s UM1_27 -a ga,mcts -e 100000 -n 10 --plot"
echo "python3 main.py -s UM2_27 -a ga,mcts -e 100000 -n 10 --plot"
echo ""
echo "# Run all research experiments:"
echo "python3 research_experiments.py"
echo ""
echo "# List available sequences and algorithms:"
echo "python3 main.py --list-sequences"
echo "python3 main.py --list-algorithms"
echo ""
echo "=========================================="
echo "Test completed successfully!"
echo "=========================================="
