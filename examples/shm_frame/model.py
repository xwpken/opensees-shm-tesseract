"""Geometry and discretization of the static steel frame."""

from __future__ import annotations

N_SUBELEMENTS = 4

NODE_COORDINATES = {
    1: (0.0, 0.0), 2: (5.0, 0.0), 3: (10.0, 0.0), 4: (15.0, 0.0),
    5: (0.0, 4.0), 6: (5.0, 4.0), 7: (10.0, 4.0), 8: (15.0, 4.0),
    9: (0.0, 8.0), 10: (5.0, 8.0), 11: (10.0, 8.0), 12: (15.0, 8.0),
    13: (0.0, 12.0), 14: (5.0, 12.0), 15: (10.0, 12.0), 16: (15.0, 12.0),
}

MEMBERS = (
    (1, 5), (5, 9), (9, 13),
    (2, 6), (6, 10), (10, 14),
    (3, 7), (7, 11), (11, 15),
    (4, 8), (8, 12), (12, 16),
    (5, 6), (9, 10), (13, 14),
    (6, 7), (10, 11), (14, 15),
    (7, 8), (11, 12), (15, 16),
)


def member_nodes(member_index: int) -> tuple[int, ...]:
    """Return the end and internal node tags of a subdivided member."""
    node_i, node_j = MEMBERS[member_index]
    first_internal = 17 + member_index * (N_SUBELEMENTS - 1)
    return (node_i, first_internal, first_internal + 1, first_internal + 2, node_j)
