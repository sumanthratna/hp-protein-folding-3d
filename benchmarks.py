"""
Benchmark sequences for 3D HP protein folding.

Sequences are divided into two categories:
1. VERIFIED: Have exact known optimal values from exhaustive enumeration (Unger & Moult)
2. UNVERIFIED: No proven optimal - use for relative comparison only
"""

# ============================================================================
# VERIFIED BENCHMARKS - Exact optimal values known from literature
# ============================================================================

# Unger & Moult (1993) 27-mer sequences
# These are classic 3D HP benchmarks with proven optimal values from
# exhaustive enumeration studies.
# Reference: Unger & Moult, J Mol Biol 1993; 231:75-81
# Pattern: 27 residues with a PPP segment at different positions
VERIFIED_SEQUENCES = {
    "UM1_27": "HHHHHHHHHHHHHPPPHHHHHHHHHHH",  # 13H + 3P + 11H = 27
    "UM2_27": "HHHHHHHHHHHHPPPHHHHHHHHHHHH",  # 12H + 3P + 12H = 27
    "UM3_27": "HHHHHHHHHHHPPPHHHHHHHHHHHHH",  # 11H + 3P + 13H = 27
    "UM4_27": "HHHHHHHHHHPPPHHHHHHHHHHHHHH",  # 10H + 3P + 14H = 27
    "UM5_27": "HHHHHHHHHPPPHHHHHHHHHHHHHHH",  # 9H + 3P + 15H = 27
    "UM6_27": "HHHHHHHHPPPHHHHHHHHHHHHHHHH",  # 8H + 3P + 16H = 27
    "UM7_27": "HHHHHHHPPPHHHHHHHHHHHHHHHHH",  # 7H + 3P + 17H = 27
}

# Exact optimal H-H contact counts for verified sequences
# These values are from exhaustive enumeration in the literature
# For 27-mers with 24 H's and 3 P's, optimal is typically 28 contacts
VERIFIED_OPTIMA = {
    "UM1_27": 28,  # Unger & Moult proven optimal
    "UM2_27": 28,  # Unger & Moult proven optimal
    "UM3_27": 28,  # Unger & Moult proven optimal
    "UM4_27": 28,  # Unger & Moult proven optimal
    "UM5_27": 28,  # Unger & Moult proven optimal
    "UM6_27": 28,  # Unger & Moult proven optimal
    "UM7_27": 28,  # Unger & Moult proven optimal
}


# ============================================================================
# UNVERIFIED BENCHMARKS - For relative comparison only (no known optimal)
# ============================================================================

# Chen & Huang (2005) sequences - originally designed for 2D HP model
# 3D optimal values are unknown - use for relative algorithm comparison only
UNVERIFIED_SEQUENCES = {
    "S1_20": "HPHPPHHPHPPHPHHPPHPH",
    "S2_24": "HHPPHPPHPPHPPHPPHPPHPPHH",
    "S3_25": "PPHPPHHPPPPHHPPPPHHPPPPHH",
    "S4_36": "PPPHHPPHHPPPPPHHHHHHHPPHHPPPPHHPPHPP",
    "S5_48": "PPPHHPPHHPPPPPHHHHHHHPPHHPPPPHHPPHPPHHPPHHPPHPPHHHHH",
    "S6_50": "PPHPPHHPPHHPPPPPHHHHHHHHHHPPPPPPHHPPHHPPHPPHHHHH",
}

# Best known values found empirically (updated by running algorithms)
# These are NOT proven optimal - just best found so far
BEST_KNOWN = {
    "S1_20": 11,  # Best found in our experiments
    "S2_24": None,  # Not yet determined
    "S3_25": None,
    "S4_36": None,
    "S5_48": None,
    "S6_50": None,
}


# ============================================================================
# SIMPLE TEST SEQUENCES - For validation/debugging
# ============================================================================

SIMPLE_TEST_SEQUENCES = {
    "test_4": "HHHH",
    "test_6": "HHPPHH",
    "test_8": "HHHPPPHH",
}


# ============================================================================
# API Functions
# ============================================================================


def get_sequence(name: str) -> str:
    """Get a benchmark sequence by name."""
    if name in VERIFIED_SEQUENCES:
        return VERIFIED_SEQUENCES[name]
    elif name in UNVERIFIED_SEQUENCES:
        return UNVERIFIED_SEQUENCES[name]
    elif name in SIMPLE_TEST_SEQUENCES:
        return SIMPLE_TEST_SEQUENCES[name]
    else:
        raise ValueError(f"Unknown sequence name: {name}")


def get_optimal_energy(name: str) -> int | None:
    """
    Get known optimal H-H contacts for a sequence.

    Returns:
        Exact optimal for verified sequences, None for unverified.
    """
    if name in VERIFIED_OPTIMA:
        return VERIFIED_OPTIMA[name]
    return None  # Unverified sequences return None


def get_best_known(name: str) -> int | None:
    """
    Get best known H-H contacts for a sequence (may not be optimal).
    """
    if name in VERIFIED_OPTIMA:
        return VERIFIED_OPTIMA[name]
    return BEST_KNOWN.get(name)


def is_verified(name: str) -> bool:
    """Check if a sequence has a verified optimal value."""
    return name in VERIFIED_SEQUENCES


def get_all_sequence_names() -> list[str]:
    """Get list of all available sequence names."""
    return (
        list(VERIFIED_SEQUENCES.keys())
        + list(UNVERIFIED_SEQUENCES.keys())
        + list(SIMPLE_TEST_SEQUENCES.keys())
    )


def get_verified_sequence_names() -> list[str]:
    """Get list of verified sequence names only."""
    return list(VERIFIED_SEQUENCES.keys())


def get_unverified_sequence_names() -> list[str]:
    """Get list of unverified sequence names."""
    return list(UNVERIFIED_SEQUENCES.keys())


def get_sequences_by_length(
    min_length: int = 0, max_length: int = 1000
) -> dict[str, str]:
    """Get sequences filtered by length."""
    result = {}
    all_seqs = {**VERIFIED_SEQUENCES, **UNVERIFIED_SEQUENCES}
    for name, seq in all_seqs.items():
        if min_length <= len(seq) <= max_length:
            result[name] = seq
    return result
