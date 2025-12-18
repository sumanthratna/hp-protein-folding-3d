"""
Deep Q-Network (DQN) solver for 3D HP protein folding.

Enhanced DQN with:
- Double DQN to reduce overestimation
- Better reward shaping with potential-based rewards
- Domain-guided exploration (bias toward H-H contacts)
- Huber loss for robust training
- Slower epsilon decay based on evaluations
"""

from collections import deque
from random import choice, random, seed

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from hp_model import (
    DIRECTIONS,
    Conformation,
    get_hh_contacts,
    get_valid_moves,
    is_valid,
)


def generate_training_sequences(
    num_sequences: int = 50,
    min_length: int = 10,
    max_length: int = 25,
    h_ratio_range: tuple[float, float] = (0.3, 0.6),
    random_seed: int | None = None,
) -> list[str]:
    """
    Generate diverse training sequences for pre-training.

    Args:
        num_sequences: Number of sequences to generate
        min_length: Minimum sequence length
        max_length: Maximum sequence length
        h_ratio_range: Range of H monomer ratios (min, max)
        random_seed: Random seed for reproducibility

    Returns:
        List of HP sequences
    """
    if random_seed is not None:
        np.random.seed(random_seed)

    sequences = []
    for _ in range(num_sequences):
        length = np.random.randint(min_length, max_length + 1)
        h_ratio = np.random.uniform(*h_ratio_range)
        num_h = int(length * h_ratio)

        # Create sequence with specified H ratio
        seq_list = ["H"] * num_h + ["P"] * (length - num_h)
        np.random.shuffle(seq_list)
        sequences.append("".join(seq_list))

    return sequences


class DQNNetwork(nn.Module):
    """3D CNN network for DQN that processes protein conformations."""

    def __init__(self, grid_size: int = 20, num_channels: int = 2):
        super(DQNNetwork, self).__init__()
        self.grid_size = grid_size
        self.num_channels = num_channels

        # 3D Convolutional layers with pooling
        self.conv1 = nn.Conv3d(num_channels, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm3d(32)
        self.pool1 = nn.MaxPool3d(kernel_size=2, stride=2)

        self.conv2 = nn.Conv3d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm3d(64)
        self.pool2 = nn.MaxPool3d(kernel_size=2, stride=2)

        self.conv3 = nn.Conv3d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm3d(128)

        # Adaptive pooling to fixed size
        self.adaptive_pool = nn.AdaptiveAvgPool3d((2, 2, 2))

        # After adaptive pooling: 128 * 2 * 2 * 2 = 1024 values
        conv_output_size = 128 * 2 * 2 * 2

        # Larger FC layers with dropout
        self.fc1 = nn.Linear(conv_output_size, 256)
        self.dropout1 = nn.Dropout(0.2)
        self.fc2 = nn.Linear(256, 128)
        self.dropout2 = nn.Dropout(0.2)
        self.fc3 = nn.Linear(128, 6)  # 6 actions

        self.relu = nn.ReLU()

    def forward(self, x):
        """Forward pass through the network."""
        x = self.pool1(self.relu(self.bn1(self.conv1(x))))
        x = self.pool2(self.relu(self.bn2(self.conv2(x))))
        x = self.relu(self.bn3(self.conv3(x)))
        x = self.adaptive_pool(x)
        x = x.view(x.size(0), -1)
        x = self.dropout1(self.relu(self.fc1(x)))
        x = self.dropout2(self.relu(self.fc2(x)))
        x = self.fc3(x)
        return x


def encode_state(partial_conf: Conformation, grid_size: int = 20) -> np.ndarray:
    """
    Encode a partial conformation as a 3D grid.

    Returns:
        numpy array of shape (2, grid_size, grid_size, grid_size)
        Channel 0: H monomers
        Channel 1: P monomers
    """
    if len(partial_conf.coords) == 0:
        return np.zeros((2, grid_size, grid_size, grid_size), dtype=np.float32)

    coords = np.array(partial_conf.coords)
    min_coords = coords.min(axis=0)
    max_coords = coords.max(axis=0)
    center = (min_coords + max_coords) / 2
    grid_center = grid_size // 2

    grid = np.zeros((2, grid_size, grid_size, grid_size), dtype=np.float32)

    for i, (x, y, z) in enumerate(partial_conf.coords):
        gx = int(x - center[0] + grid_center)
        gy = int(y - center[1] + grid_center)
        gz = int(z - center[2] + grid_center)

        if 0 <= gx < grid_size and 0 <= gy < grid_size and 0 <= gz < grid_size:
            if partial_conf.sequence[i] == "H":
                grid[0, gx, gy, gz] = 1.0
            else:
                grid[1, gx, gy, gz] = 1.0

    return grid


class ReplayBuffer:
    """Experience replay buffer with priority sampling option."""

    def __init__(self, capacity: int = 100000):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        """Add a transition to the buffer."""
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size: int):
        """Sample a batch of transitions."""
        indices = np.random.choice(len(self.buffer), batch_size, replace=False)
        return [self.buffer[idx] for idx in indices]

    def __len__(self):
        return len(self.buffer)


class DQNSolver:
    """Enhanced Deep Q-Network solver for HP protein folding."""

    def __init__(
        self,
        sequence: str,
        max_evaluations: int = 100000,
        random_seed: int | None = None,
        learning_rate: float = 0.0005,  # Lower LR for stability
        epsilon_start: float = 1.0,
        epsilon_end: float = 0.15,  # Higher final epsilon for more exploration
        epsilon_decay_evals: int = 30000,  # Faster decay to spend more time at epsilon_end
        replay_buffer_size: int = 50000,
        batch_size: int = 64,  # Larger batch
        target_update_freq: int = 500,  # More frequent updates
        gamma: float = 0.99,
        grid_size: int = 20,
        training_sequences: list[str] | None = None,
        device: str | None = None,
        use_double_dqn: bool = True,  # Enable Double DQN
        greedy_exploration_prob: float = 0.3,  # Lower prob for more random exploration
        train_every_n_steps: int = 4,  # Train less frequently for speed
    ):
        self.sequence = sequence
        self.n = len(sequence)
        self.max_evaluations = max_evaluations
        self.learning_rate = learning_rate
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay_evals = epsilon_decay_evals
        self.batch_size = batch_size
        self.target_update_freq = target_update_freq
        self.gamma = gamma
        self.grid_size = grid_size
        self.training_sequences = training_sequences or [sequence]
        self.use_double_dqn = use_double_dqn
        self.greedy_exploration_prob = greedy_exploration_prob
        self.train_every_n_steps = train_every_n_steps
        self.env_step_count = 0  # Track env steps for training frequency

        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        # Enable TF32 for faster float32 matmul
        if self.device.type == "cuda":
            torch.set_float32_matmul_precision("high")

        if random_seed is not None:
            seed(random_seed)
            torch.manual_seed(random_seed)
            np.random.seed(random_seed)

        # Networks
        self.q_network = DQNNetwork(grid_size=grid_size).to(self.device)
        self.target_network = DQNNetwork(grid_size=grid_size).to(self.device)
        self.target_network.load_state_dict(self.q_network.state_dict())
        self.target_network.eval()

        # Compile networks for faster execution (PyTorch 2.0+)
        if hasattr(torch, "compile") and self.device.type == "cuda":
            try:
                self.q_network = torch.compile(self.q_network, mode="reduce-overhead")
                self.target_network = torch.compile(
                    self.target_network, mode="reduce-overhead"
                )
            except Exception:
                pass  # Fall back to eager mode if compilation fails

        # Optimizer with weight decay
        self.optimizer = optim.Adam(
            self.q_network.parameters(), lr=learning_rate, weight_decay=1e-5
        )

        # Replay buffer
        self.replay_buffer = ReplayBuffer(capacity=replay_buffer_size)

        # Training state
        self.episode_count = 0
        self.step_count = 0
        self.evaluation_count = 0
        self.best_conformation: Conformation | None = None
        self.best_contacts = 0
        self.best_contacts_history: list[int] = []
        self.evaluations_to_best: int | None = None

    def _score_move(
        self, partial_conf: Conformation, move: int, next_monomer_type: str
    ) -> int:
        """Score a move based on potential H-H contacts (domain knowledge)."""
        if next_monomer_type != "H":
            return 0

        current_pos = partial_conf.coords[-1]
        dx, dy, dz = DIRECTIONS[move]
        new_pos = (current_pos[0] + dx, current_pos[1] + dy, current_pos[2] + dz)

        # Build position->index map for O(1) lookup
        pos_to_idx = {pos: idx for idx, pos in enumerate(partial_conf.coords)}

        # Count adjacent H positions (non-consecutive)
        h_neighbors = 0
        chain_len = len(partial_conf.coords)
        for ddx, ddy, ddz in DIRECTIONS:
            neighbor = (new_pos[0] + ddx, new_pos[1] + ddy, new_pos[2] + ddz)
            idx = pos_to_idx.get(neighbor)
            if (
                idx is not None
                and partial_conf.sequence[idx] == "H"
                and idx < chain_len - 1
            ):
                h_neighbors += 1
        return h_neighbors

    def _get_action(
        self, state: torch.Tensor, partial_conf: Conformation, epsilon: float
    ) -> int:
        """Select action using epsilon-greedy with domain-guided exploration."""
        valid_moves = get_valid_moves(partial_conf)

        if not valid_moves:
            return -1  # No valid moves (dead end)

        if random() < epsilon:
            # Exploration: use domain knowledge to bias random selection
            next_idx = len(partial_conf.coords)
            next_type = (
                partial_conf.sequence[next_idx]
                if next_idx < len(partial_conf.sequence)
                else "P"
            )

            # With some probability, use greedy exploration (prefer H-H contacts)
            if random() < self.greedy_exploration_prob and next_type == "H":
                scored_moves = [
                    (self._score_move(partial_conf, m, next_type), m)
                    for m in valid_moves
                ]
                scored_moves.sort(reverse=True)

                # Pick from best moves with probability
                if scored_moves[0][0] > 0:
                    best_score = scored_moves[0][0]
                    best_moves = [m for s, m in scored_moves if s == best_score]
                    return choice(best_moves)

            return choice(valid_moves)
        else:
            # Greedy action from Q-network
            self.q_network.eval()
            with torch.no_grad():
                q_values = self.q_network(state).squeeze(0)
                masked_q = torch.full((6,), float("-inf"), device=self.device)
                valid_tensor = torch.tensor(valid_moves, device=self.device)
                masked_q[valid_tensor] = q_values[valid_tensor]
            self.q_network.train()
            return int(torch.argmax(masked_q).item())

    def _count_contacts(self, partial_conf: Conformation) -> int:
        """Count H-H contacts in a partial conformation."""
        if len(partial_conf.coords) < 2:
            return 0

        # Build position->index map for O(1) lookup
        pos_to_idx = {pos: idx for idx, pos in enumerate(partial_conf.coords)}
        contacts = 0

        for i, pos in enumerate(partial_conf.coords):
            if partial_conf.sequence[i] != "H":
                continue
            x, y, z = pos
            for dx, dy, dz in DIRECTIONS:
                neighbor = (x + dx, y + dy, z + dz)
                j = pos_to_idx.get(neighbor)
                if (
                    j is not None
                    and j > i
                    and abs(j - i) > 1
                    and partial_conf.sequence[j] == "H"
                ):
                    contacts += 1
        return contacts

    def _compute_reward(
        self,
        conf: Conformation,
        prev_conf: Conformation,
        done: bool,
        is_dead_end: bool,
    ) -> float:
        """Compute reward with potential-based shaping."""
        if is_dead_end:
            return -50.0  # Penalty for dead ends

        # Count contacts before and after
        prev_contacts = self._count_contacts(prev_conf)
        curr_contacts = self._count_contacts(conf)
        new_contacts = curr_contacts - prev_contacts

        # Immediate reward for new contacts
        reward = float(new_contacts) * 2.0  # Bonus for each new contact

        # Small step cost
        reward -= 0.01

        if done:
            if is_valid(conf):
                # Final bonus: total contacts
                total = get_hh_contacts(conf)
                reward += float(total) * 1.0  # Extra reward for final contacts
            else:
                reward -= 50.0  # Invalid final conformation

        return reward

    def _step(
        self, partial_conf: Conformation, action: int
    ) -> tuple[Conformation, bool, bool]:
        """Execute an action and return next state."""
        if action == -1 or len(partial_conf.coords) >= self.n:
            return partial_conf, True, True

        valid_moves = get_valid_moves(partial_conf)
        if action not in valid_moves:
            return partial_conf, True, True

        current_pos = partial_conf.coords[-1]
        dx, dy, dz = DIRECTIONS[action]
        new_pos = (current_pos[0] + dx, current_pos[1] + dy, current_pos[2] + dz)
        new_coords = partial_conf.coords + [new_pos]
        new_conf = Conformation(sequence=partial_conf.sequence, coords=new_coords)

        done = new_conf.is_complete()
        return new_conf, done, False

    def _train_step(self):
        """Perform one training step with Double DQN."""
        if len(self.replay_buffer) < self.batch_size:
            return

        batch = self.replay_buffer.sample(self.batch_size)

        # Efficient batch unpacking using numpy stacking then single tensor conversion
        states_np = np.stack([s for s, a, r, ns, d in batch])
        actions_np = np.array([a for s, a, r, ns, d in batch], dtype=np.int64)
        rewards_np = np.array([r for s, a, r, ns, d in batch], dtype=np.float32)
        dones_np = np.array([d for s, a, r, ns, d in batch], dtype=np.bool_)

        # Handle None next_states
        zero_state = np.zeros(
            (2, self.grid_size, self.grid_size, self.grid_size), dtype=np.float32
        )
        next_states_np = np.stack(
            [ns if ns is not None else zero_state for s, a, r, ns, d in batch]
        )

        # Single transfer to GPU
        states = torch.from_numpy(states_np).to(self.device)
        actions = torch.from_numpy(actions_np).to(self.device)
        rewards = torch.from_numpy(rewards_np).to(self.device)
        next_states = torch.from_numpy(next_states_np).to(self.device)
        dones = torch.from_numpy(dones_np).to(self.device)

        # Current Q values
        q_values = self.q_network(states)
        q_value = q_values.gather(1, actions.unsqueeze(1)).squeeze(1)

        # Next Q values with Double DQN
        with torch.no_grad():
            if self.use_double_dqn:
                # Double DQN: use Q-network to select action, target to evaluate
                next_q_online = self.q_network(next_states)
                next_actions = next_q_online.argmax(1, keepdim=True)
                next_q_target = self.target_network(next_states)
                next_q_value = next_q_target.gather(1, next_actions).squeeze(1)
            else:
                # Standard DQN
                next_q_values = self.target_network(next_states)
                next_q_value = next_q_values.max(1)[0]

            target_q_value = rewards + (self.gamma * next_q_value * ~dones)

        # Huber loss (more robust than MSE)
        loss = nn.SmoothL1Loss()(q_value, target_q_value)

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), max_norm=1.0)
        self.optimizer.step()

        self.step_count += 1

        # Update target network
        if self.step_count % self.target_update_freq == 0:
            self.target_network.load_state_dict(self.q_network.state_dict())

    def _get_epsilon(self) -> float:
        """Get current epsilon based on evaluation count."""
        if self.evaluation_count >= self.epsilon_decay_evals:
            return self.epsilon_end
        progress = self.evaluation_count / self.epsilon_decay_evals
        return self.epsilon_start - progress * (self.epsilon_start - self.epsilon_end)

    def _run_episode(self, training: bool = True) -> tuple[Conformation | None, int]:
        """Run one episode of folding."""
        episode_sequence = choice(self.training_sequences)
        conf = Conformation(sequence=episode_sequence, coords=[(0, 0, 0)])

        while not conf.is_complete():
            # Encode state once and reuse
            state_np = encode_state(conf, grid_size=self.grid_size)
            state = torch.from_numpy(state_np).unsqueeze(0).to(self.device)
            epsilon = self._get_epsilon() if training else 0.0  # Greedy for eval
            action = self._get_action(state, conf, epsilon)

            prev_conf = conf
            next_conf, done, is_dead_end = self._step(conf, action)

            if training:
                reward = self._compute_reward(next_conf, prev_conf, done, is_dead_end)
                next_state_np = (
                    encode_state(next_conf, grid_size=self.grid_size)
                    if not is_dead_end
                    else None
                )
                self.replay_buffer.push(state_np, action, reward, next_state_np, done)
                self.env_step_count += 1

                # Train less frequently for speed
                if (
                    len(self.replay_buffer) >= self.batch_size
                    and self.env_step_count % self.train_every_n_steps == 0
                ):
                    self._train_step()

            conf = next_conf
            self.evaluation_count += 1

            if done or is_dead_end:
                break
            if self.evaluation_count >= self.max_evaluations:
                break

        # Evaluate final conformation
        if conf.is_complete() and is_valid(conf):
            contacts = get_hh_contacts(conf)
            if contacts > self.best_contacts:
                self.best_contacts = contacts
                self.best_conformation = conf.copy()
                self.evaluations_to_best = self.evaluation_count
            return conf, contacts
        return None, 0

    def solve(self, target_contacts: int | None = None) -> Conformation | None:
        """Train and solve using DQN."""
        # Training phase
        while self.evaluation_count < self.max_evaluations:
            if target_contacts is not None and self.best_contacts >= target_contacts:
                break

            self._run_episode(training=True)
            self.episode_count += 1
            self.best_contacts_history.append(self.best_contacts)

            if self.evaluation_count >= self.max_evaluations:
                break

        # Extended evaluation phase: multiple greedy runs
        eval_budget = min(self.n * 100, self.max_evaluations // 5)  # 20% for eval
        eval_start = self.evaluation_count

        while self.evaluation_count - eval_start < eval_budget:
            final_conf, contacts = self._run_episode(training=False)
            if final_conf and contacts > self.best_contacts:
                self.best_contacts = contacts
                self.best_conformation = final_conf.copy()
                self.evaluations_to_best = self.evaluation_count

        return self.best_conformation

    def get_statistics(self) -> dict:
        """Get statistics about the run."""
        return {
            "best_contacts": self.best_contacts,
            "best_reward": float(self.best_contacts),
            "best_energy": -self.best_contacts,
            "evaluations_used": self.evaluation_count,
            "evaluations_to_best": self.evaluations_to_best,
            "episodes": self.episode_count,
            "best_contacts_history": self.best_contacts_history.copy(),
        }

    def save_model(self, path: str):
        """Save model weights to file."""
        # Handle compiled models
        q_net = self.q_network
        if hasattr(q_net, "_orig_mod"):
            q_net = q_net._orig_mod

        torch.save(
            {
                "q_network": q_net.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "episode_count": self.episode_count,
                "step_count": self.step_count,
            },
            path,
        )

    def load_model(self, path: str):
        """Load model weights from file."""
        checkpoint = torch.load(path, map_location=self.device, weights_only=True)

        # Handle compiled models
        q_net = self.q_network
        target_net = self.target_network
        if hasattr(q_net, "_orig_mod"):
            q_net = q_net._orig_mod
            target_net = target_net._orig_mod

        q_net.load_state_dict(checkpoint["q_network"])
        target_net.load_state_dict(checkpoint["q_network"])
        self.optimizer.load_state_dict(checkpoint["optimizer"])

    def pretrain(
        self,
        training_sequences: list[str],
        pretrain_evaluations: int = 100000,
        verbose: bool = True,
    ):
        """
        Pre-train on multiple sequences before fine-tuning on target.

        Args:
            training_sequences: List of sequences to train on
            pretrain_evaluations: Number of evaluations for pre-training
            verbose: Print progress updates
        """
        self.training_sequences = training_sequences
        original_max_evals = self.max_evaluations
        self.max_evaluations = pretrain_evaluations

        if verbose:
            print(f"Pre-training on {len(training_sequences)} sequences...")
            print(
                f"  Sequence lengths: {min(len(s) for s in training_sequences)}-{max(len(s) for s in training_sequences)}"
            )
            print(f"  Pre-train evaluations: {pretrain_evaluations}")

        # Reset counters for pre-training
        self.evaluation_count = 0
        self.episode_count = 0
        self.best_contacts = 0
        self.best_conformation = None

        # Pre-training loop
        last_report = 0
        while self.evaluation_count < pretrain_evaluations:
            self._run_episode(training=True)
            self.episode_count += 1

            # Progress reporting
            if verbose and self.evaluation_count - last_report >= 10000:
                epsilon = self._get_epsilon()
                print(
                    f"  Pretrain progress: {self.evaluation_count}/{pretrain_evaluations} "
                    f"(ε={epsilon:.3f}, episodes={self.episode_count})"
                )
                last_report = self.evaluation_count

        if verbose:
            print(f"  Pre-training complete. Episodes: {self.episode_count}")

        # Reset for fine-tuning on target sequence
        self.training_sequences = [self.sequence]
        self.max_evaluations = original_max_evals
        self.evaluation_count = 0
        self.episode_count = 0
        self.best_contacts = 0
        self.best_conformation = None
        self.best_contacts_history = []
        self.evaluations_to_best = None

        # Keep learned weights but reset epsilon for fine-tuning exploration
        # (epsilon will naturally start high again based on evaluation_count=0)
