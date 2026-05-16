"""Nine Men's Morris board constants.

Positions 0-23 are arranged as three concentric rings, numbered clockwise
from the top-left corner of each ring (outer ring first):

    0 -------- 1 -------- 2
    |           |           |
    |   8 ---- 9 ---- 10   |
    |   |       |       |   |
    |   |  16--17--18   |   |
    |   |   |       |   |   |
    7--15--23       19--11-- 3
    |   |   |       |   |   |
    |   |  22--21--20   |   |
    |   |               |   |
    |  14----13----12   |
    |                   |
    6 -------- 5 -------- 4

Action encoding (used by rules.py and the policy head):
  0-23  : place (placement phase) or capture (must-capture sub-turn)
  24-599: move/fly — index = 24 + src * 24 + dst
"""

from __future__ import annotations

from typing import Final

NUM_POSITIONS: Final[int] = 24
NUM_PIECES_PER_PLAYER: Final[int] = 9

NUM_PLACE_CAPTURE_ACTIONS: Final[int] = NUM_POSITIONS
ACTION_SPACE_SIZE: Final[int] = NUM_PLACE_CAPTURE_ACTIONS + NUM_POSITIONS * NUM_POSITIONS  # 600

# Outer-ring midpoints (1,3,5,7) have 3 neighbours.
# Middle-ring midpoints (9,11,13,15) have 4 neighbours (connected to both outer and inner).
# Inner-ring midpoints (17,19,21,23) have 3 neighbours.
# Corner positions have 2 neighbours.
ADJACENCY: Final[list[list[int]]] = [
    [1, 7],  # 0  outer TL
    [0, 2, 9],  # 1  outer TM
    [1, 3],  # 2  outer TR
    [2, 4, 11],  # 3  outer MR
    [3, 5],  # 4  outer BR
    [4, 6, 13],  # 5  outer BM
    [5, 7],  # 6  outer BL
    [6, 0, 15],  # 7  outer ML
    [9, 15],  # 8  middle TL
    [8, 10, 1, 17],  # 9  middle TM
    [9, 11],  # 10 middle TR
    [10, 12, 3, 19],  # 11 middle MR
    [11, 13],  # 12 middle BR
    [12, 14, 5, 21],  # 13 middle BM
    [13, 15],  # 14 middle BL
    [14, 8, 7, 23],  # 15 middle ML
    [17, 23],  # 16 inner TL
    [16, 18, 9],  # 17 inner TM
    [17, 19],  # 18 inner TR
    [18, 20, 11],  # 19 inner MR
    [19, 21],  # 20 inner BR
    [20, 22, 13],  # 21 inner BM
    [21, 23],  # 22 inner BL
    [22, 16, 15],  # 23 inner ML
]

MILLS: Final[list[tuple[int, int, int]]] = [
    # Outer ring sides
    (0, 1, 2),
    (2, 3, 4),
    (4, 5, 6),
    (6, 7, 0),
    # Middle ring sides
    (8, 9, 10),
    (10, 11, 12),
    (12, 13, 14),
    (14, 15, 8),
    # Inner ring sides
    (16, 17, 18),
    (18, 19, 20),
    (20, 21, 22),
    (22, 23, 16),
    # Spokes connecting all three rings
    (1, 9, 17),
    (3, 11, 19),
    (5, 13, 21),
    (7, 15, 23),
]

# Pre-computed: for each position, the mills that contain it.
MILLS_BY_POSITION: Final[list[list[tuple[int, int, int]]]] = [
    [m for m in MILLS if i in m] for i in range(NUM_POSITIONS)
]
