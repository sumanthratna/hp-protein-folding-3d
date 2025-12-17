"""
Genetic Algorithm solver for 3D HP protein folding.

Optimized GA with:
- Local search (hill climbing) after mutations
- Segment regrowth mutation
- Repair mechanism for invalid conformations
- Greedy initialization for some individuals
- Adaptive mutation rate
"""

from random import choice, randint, random, sample, seed, shuffle

from hp_model import (
    DIRECTIONS,
    Conformation,
    get_hh_contacts,
    is_valid,
    random_valid_walk,
)


class GAIndividual:
    """Represents an individual in the GA population."""

    def __init__(self, conformation: Conformation, fitness: int | None = None):
        self.conformation = conformation
        self.fitness = fitness if fitness is not None else self._compute_fitness()

    def _compute_fitness(self) -> int:
        """Compute fitness as number of H-H contacts (higher is better)."""
        if not self.conformation.is_complete():
            return 0

        if not is_valid(self.conformation):
            return 0  # Invalid conformations get fitness 0

        return get_hh_contacts(self.conformation)

    def copy(self) -> "GAIndividual":
        """Create a deep copy."""
        return GAIndividual(conformation=self.conformation.copy(), fitness=self.fitness)


def greedy_valid_walk(sequence: str, max_attempts: int = 100) -> Conformation | None:
    """
    Generate a greedy self-avoiding walk that tries to place H's near other H's.
    """
    n = len(sequence)

    for _ in range(max_attempts):
        coords = [(0, 0, 0)]
        moves = []
        occupied = {(0, 0, 0)}

        for i in range(n - 1):
            x, y, z = coords[-1]
            valid_dirs = []

            for move_idx, (dx, dy, dz) in enumerate(DIRECTIONS):
                next_pos = (x + dx, y + dy, z + dz)
                if next_pos not in occupied:
                    valid_dirs.append((move_idx, next_pos))

            if not valid_dirs:
                break  # Dead end

            # If next monomer is H, prefer positions adjacent to existing H's
            if sequence[i + 1] == "H":
                scored_moves = []
                for move_idx, next_pos in valid_dirs:
                    # Count adjacent H's at this position
                    h_neighbors = 0
                    nx, ny, nz = next_pos
                    for dx, dy, dz in DIRECTIONS:
                        neighbor = (nx + dx, ny + dy, nz + dz)
                        if neighbor in occupied:
                            # Check if it's an H (and not consecutive)
                            try:
                                neighbor_idx = coords.index(neighbor)
                                if sequence[neighbor_idx] == "H" and neighbor_idx < i:
                                    h_neighbors += 1
                            except ValueError:
                                pass
                    scored_moves.append((h_neighbors, move_idx, next_pos))

                # Sort by H neighbors (descending) and pick best with some randomness
                scored_moves.sort(reverse=True)
                # Pick from top choices with probability weighting
                if scored_moves[0][0] > 0 and random() < 0.7:
                    # 70% chance to pick a position with H neighbors
                    best_score = scored_moves[0][0]
                    best_moves = [(m, p) for s, m, p in scored_moves if s == best_score]
                    move_idx, next_pos = choice(best_moves)
                else:
                    move_idx, next_pos = choice(valid_dirs)
            else:
                move_idx, next_pos = choice(valid_dirs)

            coords.append(next_pos)
            occupied.add(next_pos)
            moves.append(move_idx)

        if len(coords) == n:
            return Conformation(sequence=sequence, moves=moves, coords=coords)

    return None


def regrow_segment(
    conf: Conformation, start_pos: int, max_attempts: int = 50
) -> Conformation | None:
    """
    Regrow the chain from position start_pos to the end.
    Returns a new valid conformation or None if failed.
    """
    if start_pos <= 0 or start_pos >= conf.n:
        return None

    sequence = conf.sequence
    n = conf.n

    for _ in range(max_attempts):
        # Keep the first part of the chain
        new_coords = conf.coords[:start_pos]
        new_moves = conf.moves[: start_pos - 1] if start_pos > 1 else []
        occupied = set(new_coords)

        # Regrow the rest
        for i in range(start_pos - 1, n - 1):
            x, y, z = new_coords[-1]
            valid_dirs = []

            for move_idx, (dx, dy, dz) in enumerate(DIRECTIONS):
                next_pos = (x + dx, y + dy, z + dz)
                if next_pos not in occupied:
                    valid_dirs.append(move_idx)

            if not valid_dirs:
                break  # Dead end

            move = choice(valid_dirs)
            dx, dy, dz = DIRECTIONS[move]
            next_pos = (x + dx, y + dy, z + dz)

            new_coords.append(next_pos)
            occupied.add(next_pos)
            new_moves.append(move)

        if len(new_coords) == n:
            return Conformation(sequence=sequence, moves=new_moves, coords=new_coords)

    return None


def local_search(
    conf: Conformation, max_iterations: int = 10
) -> tuple[Conformation, int]:
    """
    Perform local search (hill climbing) to improve a conformation.
    Returns improved conformation and number of evaluations used.
    """
    if not is_valid(conf):
        return conf, 0

    current = conf
    current_fitness = get_hh_contacts(current)
    evals = 1

    for _ in range(max_iterations):
        improved = False

        # Try regrow from random positions
        positions_to_try = list(range(max(1, conf.n // 2), conf.n))
        shuffle(positions_to_try)

        for start_pos in positions_to_try[:5]:  # Try up to 5 positions
            new_conf = regrow_segment(current, start_pos)
            if new_conf and is_valid(new_conf):
                evals += 1
                new_fitness = get_hh_contacts(new_conf)
                if new_fitness > current_fitness:
                    current = new_conf
                    current_fitness = new_fitness
                    improved = True
                    break

        if not improved:
            break

    return current, evals


class GASolver:
    """Optimized Genetic Algorithm solver for HP protein folding."""

    def __init__(
        self,
        sequence: str,
        population_size: int = 100,
        crossover_rate: float = 0.8,
        mutation_rate: float = 0.15,  # Higher base mutation rate
        tournament_size: int = 5,  # Larger tournament for more selection pressure
        num_crossover_points: int = 2,
        max_evaluations: int = 100000,
        local_search_prob: float = 0.3,  # Probability of local search
        random_seed: int | None = None,
    ):
        self.sequence = sequence
        self.n = len(sequence)
        self.population_size = population_size
        self.crossover_rate = crossover_rate
        self.mutation_rate = mutation_rate
        self.base_mutation_rate = mutation_rate
        self.tournament_size = tournament_size
        self.num_crossover_points = num_crossover_points
        self.max_evaluations = max_evaluations
        self.local_search_prob = local_search_prob

        if random_seed is not None:
            seed(random_seed)

        self.population: list[GAIndividual] = []
        self.best_individual: GAIndividual | None = None
        self.evaluation_count = 0
        self.generation = 0
        self.best_fitness_history: list[int] = []
        self.evaluations_to_best: int | None = None
        self.generations_without_improvement = 0

    def initialize_population(self):
        """Initialize population with mix of random and greedy walks."""
        self.population = []

        # 30% greedy initialization, 70% random
        num_greedy = self.population_size // 3

        for i in range(self.population_size):
            if i < num_greedy:
                conf = greedy_valid_walk(self.sequence)
            else:
                conf = random_valid_walk(self.sequence)

            if conf is None:
                conf = random_valid_walk(self.sequence, max_attempts=2000)

            if conf is None:
                moves = [randint(0, 5) for _ in range(self.n - 1)]
                conf = Conformation(sequence=self.sequence, moves=moves)

            individual = GAIndividual(conf)
            self.population.append(individual)
            self.evaluation_count += 1

        self._update_best()

    def _update_best(self):
        """Update the best individual found so far."""
        current_best = max(self.population, key=lambda ind: ind.fitness)
        if (
            self.best_individual is None
            or current_best.fitness > self.best_individual.fitness
        ):
            self.best_individual = current_best.copy()
            self.evaluations_to_best = self.evaluation_count
            self.generations_without_improvement = 0
        else:
            self.generations_without_improvement += 1

    def tournament_selection(self) -> GAIndividual:
        """Select an individual using tournament selection."""
        tournament = sample(
            self.population, min(self.tournament_size, len(self.population))
        )
        return max(tournament, key=lambda ind: ind.fitness)

    def crossover(
        self, parent1: GAIndividual, parent2: GAIndividual
    ) -> tuple[GAIndividual, GAIndividual]:
        """Perform multi-point crossover with repair."""
        moves1 = parent1.conformation.moves.copy()
        moves2 = parent2.conformation.moves.copy()

        if len(moves1) != len(moves2) or len(moves1) == 0:
            return parent1.copy(), parent2.copy()

        n = len(moves1)
        if self.num_crossover_points == 1:
            points = [randint(1, n - 1)]
        else:
            point1 = randint(1, max(2, n - 2))
            point2 = randint(point1 + 1, n)
            points = sorted([point1, point2])

        child1_moves = moves1.copy()
        child2_moves = moves2.copy()

        swap = False
        last_point = 0
        for point in points:
            if swap:
                child1_moves[last_point:point] = moves2[last_point:point]
                child2_moves[last_point:point] = moves1[last_point:point]
            swap = not swap
            last_point = point

        if swap:
            child1_moves[last_point:] = moves2[last_point:]
            child2_moves[last_point:] = moves1[last_point:]

        child1_conf = Conformation(sequence=self.sequence, moves=child1_moves)
        child2_conf = Conformation(sequence=self.sequence, moves=child2_moves)

        # Try to repair invalid offspring
        child1_conf = self._repair_if_needed(child1_conf)
        child2_conf = self._repair_if_needed(child2_conf)

        child1 = GAIndividual(child1_conf)
        child2 = GAIndividual(child2_conf)

        self.evaluation_count += 2

        return child1, child2

    def _repair_if_needed(self, conf: Conformation) -> Conformation:
        """Try to repair an invalid conformation."""
        if is_valid(conf):
            return conf

        # Find first collision point
        seen = set()
        collision_idx = -1
        for i, pos in enumerate(conf.coords):
            if pos in seen:
                collision_idx = i
                break
            seen.add(pos)

        if collision_idx > 0:
            # Try to regrow from before the collision
            start = max(1, collision_idx - 2)
            repaired = regrow_segment(conf, start)
            if repaired and is_valid(repaired):
                return repaired

        return conf  # Return original if repair failed

    def mutate(self, individual: GAIndividual) -> GAIndividual:
        """Mutate using segment regrowth or point mutation."""
        if len(individual.conformation.moves) == 0:
            return individual.copy()

        # 60% segment regrowth, 40% point mutation
        if random() < 0.6:
            # Segment regrowth mutation
            start_pos = randint(max(1, self.n // 2), self.n - 1)
            new_conf = regrow_segment(individual.conformation, start_pos)
            if new_conf and is_valid(new_conf):
                self.evaluation_count += 1
                return GAIndividual(new_conf)

        # Point mutation
        mutated_moves = individual.conformation.moves.copy()
        pos = randint(0, len(mutated_moves) - 1)

        old_move = mutated_moves[pos]
        new_move = randint(0, 5)
        while new_move == old_move:
            new_move = randint(0, 5)

        mutated_moves[pos] = new_move
        mutated_conf = Conformation(sequence=self.sequence, moves=mutated_moves)
        mutated_conf = self._repair_if_needed(mutated_conf)

        self.evaluation_count += 1
        return GAIndividual(mutated_conf)

    def _adaptive_mutation_rate(self):
        """Adjust mutation rate based on progress."""
        if self.generations_without_improvement > 20:
            # Increase mutation when stuck
            self.mutation_rate = min(0.4, self.base_mutation_rate * 2)
        elif self.generations_without_improvement > 50:
            # Even higher mutation
            self.mutation_rate = min(0.5, self.base_mutation_rate * 3)
        else:
            self.mutation_rate = self.base_mutation_rate

    def evolve_generation(self):
        """Evolve one generation with local search."""
        new_population = []

        # Elitism: keep top 2 individuals
        sorted_pop = sorted(self.population, key=lambda ind: ind.fitness, reverse=True)
        for i in range(min(2, len(sorted_pop))):
            new_population.append(sorted_pop[i].copy())

        # Adaptive mutation
        self._adaptive_mutation_rate()

        while len(new_population) < self.population_size:
            parent1 = self.tournament_selection()
            parent2 = self.tournament_selection()

            if random() < self.crossover_rate:
                child1, child2 = self.crossover(parent1, parent2)
            else:
                child1 = parent1.copy()
                child2 = parent2.copy()

            if random() < self.mutation_rate:
                child1 = self.mutate(child1)
            if random() < self.mutation_rate:
                child2 = self.mutate(child2)

            # Optional local search
            if random() < self.local_search_prob and child1.fitness > 0:
                improved_conf, evals = local_search(
                    child1.conformation, max_iterations=5
                )
                self.evaluation_count += evals
                if is_valid(improved_conf):
                    child1 = GAIndividual(improved_conf)

            new_population.append(child1)
            if len(new_population) < self.population_size:
                new_population.append(child2)

        self.population = new_population[: self.population_size]
        self._update_best()
        self.generation += 1
        self.best_fitness_history.append(
            self.best_individual.fitness if self.best_individual else 0
        )

    def solve(self, target_fitness: int | None = None) -> GAIndividual:
        """Run GA until termination criteria are met."""
        self.initialize_population()

        while self.evaluation_count < self.max_evaluations:
            if target_fitness is not None and self.best_individual is not None:
                if self.best_individual.fitness >= target_fitness:
                    break

            self.evolve_generation()

            if self.evaluation_count >= self.max_evaluations:
                break

        return self.best_individual if self.best_individual else self.population[0]

    def get_statistics(self) -> dict:
        """Get statistics about the run."""
        return {
            "best_fitness": self.best_individual.fitness if self.best_individual else 0,
            "best_energy": -self.best_individual.fitness if self.best_individual else 0,
            "evaluations_used": self.evaluation_count,
            "evaluations_to_best": self.evaluations_to_best,
            "generation": self.generation,
            "best_fitness_history": self.best_fitness_history.copy(),
        }
