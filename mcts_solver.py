"""
Monte Carlo Tree Search solver for 3D HP protein folding.

Optimized MCTS with:
- Greedy rollout policy (places H's near other H's)
- Multiple rollouts per expansion for better estimates
- Smart expansion ordering
- Local search on best results
"""

from math import log, sqrt
from random import choice, random, seed, shuffle

from hp_model import (
    DIRECTIONS,
    Conformation,
    get_hh_contacts,
    get_valid_moves,
    is_valid,
)


class MCTSNode:
    """Node in the MCTS search tree."""

    def __init__(
        self,
        state: Conformation,
        parent: "MCTSNode | None" = None,
        action: int | None = None,
    ):
        self.state = state
        self.parent = parent
        self.action = action
        self.children: dict[int, "MCTSNode"] = {}
        self.visits = 0
        self.total_reward = 0.0
        self.max_reward = float("-inf")  # Track best reward through this node
        self.untried_actions: list[int] = get_valid_moves(state)
        self.is_terminal = state.is_complete() or len(self.untried_actions) == 0

    def is_fully_expanded(self) -> bool:
        return len(self.untried_actions) == 0

    def get_average_reward(self) -> float:
        if self.visits == 0:
            return 0.0
        return self.total_reward / self.visits

    def ucb_score(self, exploration_constant: float, use_max: bool = False) -> float:
        if self.visits == 0:
            return float("inf")

        if self.parent is None:
            return self.get_average_reward()

        # Option to use max reward instead of average for more exploitation
        if use_max and self.max_reward > float("-inf"):
            exploitation = 0.7 * self.get_average_reward() + 0.3 * self.max_reward
        else:
            exploitation = self.get_average_reward()

        exploration = exploration_constant * sqrt(log(self.parent.visits) / self.visits)
        return exploitation + exploration

    def select_child(self, exploration_constant: float) -> "MCTSNode":
        if not self.children:
            return None

        best_child = None
        best_score = float("-inf")

        for child in self.children.values():
            score = child.ucb_score(exploration_constant, use_max=True)
            if score > best_score:
                best_score = score
                best_child = child

        return best_child

    def expand_smart(self, sequence: str) -> "MCTSNode | None":
        """Expand with smart action ordering - prefer moves that place H near other H's."""
        if self.is_fully_expanded() or self.is_terminal:
            return None

        # Score untried actions
        scored_actions = []
        current_pos = self.state.coords[-1]
        next_idx = len(self.state.coords)
        occupied = set(self.state.coords)

        for action in self.untried_actions:
            dx, dy, dz = DIRECTIONS[action]
            new_pos = (current_pos[0] + dx, current_pos[1] + dy, current_pos[2] + dz)

            score = 0
            # If next monomer is H, prefer positions near existing H's
            if next_idx < len(sequence) and sequence[next_idx] == "H":
                nx, ny, nz = new_pos
                for ddx, ddy, ddz in DIRECTIONS:
                    neighbor = (nx + ddx, ny + ddy, nz + ddz)
                    if neighbor in occupied:
                        try:
                            neighbor_idx = self.state.coords.index(neighbor)
                            if sequence[neighbor_idx] == "H":
                                score += 1
                        except ValueError:
                            pass

            scored_actions.append((score, action))

        # Sort by score (descending) and pick best with some randomness
        scored_actions.sort(reverse=True)

        # 70% chance to pick highest scored, 30% random
        if scored_actions[0][0] > 0 and random() < 0.7:
            best_score = scored_actions[0][0]
            best_actions = [a for s, a in scored_actions if s == best_score]
            action = choice(best_actions)
        else:
            action = choice(self.untried_actions)

        self.untried_actions.remove(action)

        # Create new state
        new_coords = self.state.coords.copy()
        dx, dy, dz = DIRECTIONS[action]
        new_pos = (current_pos[0] + dx, current_pos[1] + dy, current_pos[2] + dz)
        new_coords.append(new_pos)

        new_state = Conformation(sequence=sequence, coords=new_coords)
        child = MCTSNode(new_state, parent=self, action=action)
        self.children[action] = child

        return child

    def backpropagate(self, reward: float):
        node = self
        while node is not None:
            node.visits += 1
            node.total_reward += reward
            if reward > node.max_reward:
                node.max_reward = reward
            node = node.parent


def greedy_rollout(
    conf: Conformation, sequence: str
) -> tuple[Conformation | None, bool]:
    """
    Perform a greedy rollout that prefers placing H's near other H's.
    Returns (final_conf, is_complete).
    """
    current_conf = conf.copy()

    while not current_conf.is_complete():
        valid_moves = get_valid_moves(current_conf)

        if not valid_moves:
            return current_conf, False  # Dead end

        next_idx = len(current_conf.coords)
        current_pos = current_conf.coords[-1]
        occupied = set(current_conf.coords)

        # Score moves based on potential H-H contacts
        if sequence[next_idx] == "H":
            scored_moves = []
            for move in valid_moves:
                dx, dy, dz = DIRECTIONS[move]
                new_pos = (
                    current_pos[0] + dx,
                    current_pos[1] + dy,
                    current_pos[2] + dz,
                )

                h_neighbors = 0
                nx, ny, nz = new_pos
                for ddx, ddy, ddz in DIRECTIONS:
                    neighbor = (nx + ddx, ny + ddy, nz + ddz)
                    if neighbor in occupied:
                        try:
                            neighbor_idx = current_conf.coords.index(neighbor)
                            # Count H neighbors (non-consecutive)
                            if (
                                sequence[neighbor_idx] == "H"
                                and abs(neighbor_idx - next_idx) > 1
                            ):
                                h_neighbors += 1
                        except ValueError:
                            pass

                scored_moves.append((h_neighbors, move))

            scored_moves.sort(reverse=True)

            # 80% greedy, 20% random for diversity
            if scored_moves[0][0] > 0 and random() < 0.8:
                best_score = scored_moves[0][0]
                best_moves = [m for s, m in scored_moves if s == best_score]
                move = choice(best_moves)
            else:
                move = choice(valid_moves)
        else:
            move = choice(valid_moves)

        dx, dy, dz = DIRECTIONS[move]
        new_pos = (current_pos[0] + dx, current_pos[1] + dy, current_pos[2] + dz)
        current_conf.coords.append(new_pos)

    return current_conf, True


def local_search_mcts(
    conf: Conformation, sequence: str, max_iterations: int = 5
) -> Conformation:
    """Apply local search to improve a conformation."""
    if not is_valid(conf):
        return conf

    current = conf
    current_fitness = get_hh_contacts(current)

    for _ in range(max_iterations):
        improved = False
        n = len(current.coords)

        # Try regrowing from random positions in the second half
        positions = list(range(max(1, n * 2 // 3), n))
        shuffle(positions)

        for start_pos in positions[:3]:
            # Keep first part, regrow rest
            new_coords = current.coords[:start_pos]
            occupied = set(new_coords)

            success = True
            for i in range(start_pos, n):
                if not new_coords:
                    break

                last_pos = new_coords[-1]
                valid_dirs = []

                for move_idx, (dx, dy, dz) in enumerate(DIRECTIONS):
                    next_pos = (last_pos[0] + dx, last_pos[1] + dy, last_pos[2] + dz)
                    if next_pos not in occupied:
                        valid_dirs.append((move_idx, next_pos))

                if not valid_dirs:
                    success = False
                    break

                # Greedy selection for H
                if sequence[i] == "H":
                    scored = []
                    for move_idx, next_pos in valid_dirs:
                        h_count = 0
                        nx, ny, nz = next_pos
                        for ddx, ddy, ddz in DIRECTIONS:
                            neighbor = (nx + ddx, ny + ddy, nz + ddz)
                            if neighbor in occupied:
                                try:
                                    neighbor_idx = new_coords.index(neighbor)
                                    if sequence[neighbor_idx] == "H":
                                        h_count += 1
                                except ValueError:
                                    pass
                        scored.append((h_count, move_idx, next_pos))
                    scored.sort(reverse=True)
                    _, _, next_pos = scored[0] if scored[0][0] > 0 else choice(scored)
                else:
                    _, next_pos = choice(valid_dirs)

                new_coords.append(next_pos)
                occupied.add(next_pos)

            if success and len(new_coords) == n:
                new_conf = Conformation(sequence=sequence, coords=new_coords)
                if is_valid(new_conf):
                    new_fitness = get_hh_contacts(new_conf)
                    if new_fitness > current_fitness:
                        current = new_conf
                        current_fitness = new_fitness
                        improved = True
                        break

        if not improved:
            break

    return current


class MCTSSolver:
    """Optimized Monte Carlo Tree Search solver for HP protein folding."""

    def __init__(
        self,
        sequence: str,
        exploration_constant: float = 1.0,  # Lower C for more exploitation
        max_evaluations: int = 100000,
        dead_end_penalty: float = -50.0,
        num_rollouts_per_expand: int = 3,  # Multiple rollouts for better estimates
        local_search_freq: int = 1000,  # Apply local search every N evaluations
        random_seed: int | None = None,
    ):
        self.sequence = sequence
        self.n = len(sequence)
        self.exploration_constant = exploration_constant
        self.max_evaluations = max_evaluations
        self.dead_end_penalty = dead_end_penalty
        self.num_rollouts_per_expand = num_rollouts_per_expand
        self.local_search_freq = local_search_freq

        if random_seed is not None:
            seed(random_seed)

        root_state = Conformation(sequence=sequence, coords=[(0, 0, 0)])
        self.root = MCTSNode(root_state)

        # Fix first move to reduce symmetry
        if len(self.root.untried_actions) > 0:
            self.root.untried_actions = [0] if 0 in self.root.untried_actions else []

        self.evaluation_count = 0
        self.best_conformation: Conformation | None = None
        self.best_reward = float("-inf")
        self.best_reward_history: list[float] = []
        self.evaluations_to_best: int | None = None

    def rollout(self, node: MCTSNode) -> float:
        """Perform a greedy rollout from the given node."""
        final_conf, is_complete = greedy_rollout(node.state, self.sequence)

        if not is_complete:
            return self.dead_end_penalty

        if is_valid(final_conf):
            reward = get_hh_contacts(final_conf)

            if reward > self.best_reward:
                self.best_reward = reward
                self.best_conformation = final_conf.copy()
                self.evaluations_to_best = self.evaluation_count

            return float(reward)
        else:
            return self.dead_end_penalty

    def select(self, node: MCTSNode) -> MCTSNode:
        """Select a node using UCT."""
        while not node.is_terminal and (
            node.is_fully_expanded() or len(node.children) > 0
        ):
            if not node.is_fully_expanded():
                return node

            child = node.select_child(self.exploration_constant)
            if child is None:
                break
            node = child

        return node

    def simulate(self):
        """Perform one MCTS simulation with multiple rollouts."""
        node = self.select(self.root)

        if not node.is_terminal and not node.is_fully_expanded():
            node = node.expand_smart(self.sequence)

        if node is None:
            return

        # Multiple rollouts for better value estimate
        total_reward = 0.0
        for _ in range(self.num_rollouts_per_expand):
            reward = self.rollout(node)
            total_reward += reward
            self.evaluation_count += 1
            self.best_reward_history.append(self.best_reward)

        avg_reward = total_reward / self.num_rollouts_per_expand
        node.backpropagate(avg_reward)

    def solve(self, target_reward: float | None = None) -> Conformation:
        """Run MCTS until termination criteria are met."""
        last_local_search = 0

        while self.evaluation_count < self.max_evaluations:
            if target_reward is not None and self.best_reward >= target_reward:
                break

            self.simulate()

            # Periodic local search on best solution
            if (
                self.evaluation_count - last_local_search >= self.local_search_freq
                and self.best_conformation is not None
            ):
                improved = local_search_mcts(
                    self.best_conformation, self.sequence, max_iterations=10
                )
                if is_valid(improved):
                    new_reward = get_hh_contacts(improved)
                    if new_reward > self.best_reward:
                        self.best_reward = new_reward
                        self.best_conformation = improved
                        self.evaluations_to_best = self.evaluation_count
                last_local_search = self.evaluation_count

        # Final local search
        if self.best_conformation is not None:
            improved = local_search_mcts(
                self.best_conformation, self.sequence, max_iterations=20
            )
            if is_valid(improved):
                new_reward = get_hh_contacts(improved)
                if new_reward > self.best_reward:
                    self.best_reward = new_reward
                    self.best_conformation = improved

        if self.best_conformation is not None:
            return self.best_conformation

        return self.root.state

    def get_statistics(self) -> dict:
        """Get statistics about the run."""
        return {
            "best_reward": self.best_reward,
            "best_energy": (
                -int(self.best_reward) if self.best_reward > float("-inf") else 0
            ),
            "evaluations_used": self.evaluation_count,
            "evaluations_to_best": self.evaluations_to_best,
            "best_reward_history": self.best_reward_history.copy(),
        }
