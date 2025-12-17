"""
Core HP (Hydrophobic-Polar) protein folding model for 3D cubic lattice.
"""

from random import choice


DIRECTIONS = [(1, 0, 0), (0, 1, 0), (-1, 0, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1)]


class Conformation:
    def __init__(
        self,
        sequence: str,
        moves: list[int] | None = None,
        coords: list[tuple[int, int, int]] | None = None,
    ):
        self.sequence = sequence
        self.n = len(sequence)
        if coords is not None:
            self.coords = coords
            self.moves = self._coords_to_moves(coords) if moves is None else moves
        elif moves is not None:
            self.moves = moves
            self.coords = self._moves_to_coords(moves)
        else:
            self.moves = []
            self.coords = [(0, 0, 0)]

    def _moves_to_coords(self, moves):
        coords = [(0, 0, 0)]
        for move in moves:
            dx, dy, dz = DIRECTIONS[move]
            lx, ly, lz = coords[-1]
            coords.append((lx + dx, ly + dy, lz + dz))
        return coords

    def _coords_to_moves(self, coords):
        moves = []
        for i in range(len(coords) - 1):
            x1, y1, z1 = coords[i]
            x2, y2, z2 = coords[i + 1]
            moves.append(DIRECTIONS.index((x2 - x1, y2 - y1, z2 - z1)))
        return moves

    def is_complete(self):
        return len(self.coords) == self.n

    def get_occupied_set(self):
        return set(self.coords)

    def copy(self):
        return Conformation(
            self.sequence, self.moves.copy() if self.moves else None, self.coords.copy()
        )


def is_valid(conf):
    return len(conf.coords) == len(set(conf.coords))


def compute_energy(conf):
    if not conf.is_complete():
        raise ValueError("Incomplete")
    if not is_valid(conf):
        return 1000
    occupied, contacts, h_pos = set(conf.coords), 0, []
    for i, (m, p) in enumerate(zip(conf.sequence, conf.coords)):
        if m == "H":
            h_pos.append((i, p))
    for i, (idx1, pos1) in enumerate(h_pos):
        x1, y1, z1 = pos1
        for dx, dy, dz in DIRECTIONS:
            neighbor = (x1 + dx, y1 + dy, z1 + dz)
            if neighbor in occupied:
                try:
                    idx2 = conf.coords.index(neighbor)
                    if (
                        conf.sequence[idx2] == "H"
                        and idx2 != idx1
                        and abs(idx2 - idx1) > 1
                        and idx2 > idx1
                    ):
                        contacts += 1
                except:
                    pass
    return -contacts


def get_valid_moves(partial_conf):
    if partial_conf.is_complete():
        return []
    occupied = set(partial_conf.coords)
    x, y, z = partial_conf.coords[-1]
    return [
        i
        for i, (dx, dy, dz) in enumerate(DIRECTIONS)
        if (x + dx, y + dy, z + dz) not in occupied
    ]


def random_valid_walk(sequence, max_attempts=1000):
    n = len(sequence)
    for _ in range(max_attempts):
        coords, moves, occupied = [(0, 0, 0)], [], {(0, 0, 0)}
        for i in range(n - 1):
            x, y, z = coords[-1]
            valid = [
                j
                for j, (dx, dy, dz) in enumerate(DIRECTIONS)
                if (x + dx, y + dy, z + dz) not in occupied
            ]
            if not valid:
                break
            move = choice(valid)
            dx, dy, dz = DIRECTIONS[move]
            nxt = (x + dx, y + dy, z + dz)
            coords.append(nxt)
            occupied.add(nxt)
            moves.append(move)
        if len(coords) == n:
            conf = Conformation(sequence, moves, coords)
            if is_valid(conf):
                return conf
    return None


def get_hh_contacts(conf):
    return -compute_energy(conf)
