"""Utility functions for testing GPlusTree invariants."""

from gplus_trees.gplus_tree import (
    GPlusTree,
    Stats
)

TREE_FLAGS = (
    "is_heap",
    "is_search_tree",
    "internal_has_replicas",
    "internal_packed",
    "linked_leaf_nodes",
    "all_leaf_values_present",
    "leaf_keys_in_order",
)

def assert_tree_invariants_tc(tc, t: GPlusTree, stats: Stats) -> None:
    """TestCase version: use inside unittest.TestCase methods."""
    for flag in TREE_FLAGS:
        tc.assertTrue(
            getattr(stats, flag),
            f"Invariant failed: {flag} is False"
        )

    if not t.is_empty():
        tc.assertGreater(
            stats.item_count, 0,
            f"Invariant failed: item_count={stats.item_count} ≤ 0 for non-empty tree"
        )
        tc.assertGreater(
            stats.item_slot_count, 0,
            f"Invariant failed: item_slot_count={stats.item_slot_count} ≤ 0 for non-empty tree"
        )
        tc.assertGreater(
            stats.gnode_count, 0,
            f"Invariant failed: gnode_count={stats.gnode_count} ≤ 0 for non-empty tree"
        )
        tc.assertGreater(
            stats.gnode_height, 0,
            f"Invariant failed: gnode_height={stats.gnode_height} ≤ 0 for non-empty tree"
        )
        tc.assertGreater(
            stats.rank, 0,
            f"Invariant failed: rank={stats.rank} ≤ 0 for non-empty tree"
        )
        tc.assertIsNotNone(
            stats.least_item,
            "Invariant failed: least_item is None for non-empty tree"
        )
        tc.assertIsNotNone(
            stats.greatest_item,
            "Invariant failed: greatest_item is None for non-empty tree"
        )

class InvariantError(Exception):
    """Raised when a GPlusTree invariant is violated."""
    pass

def assert_tree_invariants_raise(t: GPlusTree, stats: Stats) -> None:
    """Check all invariants, raising on the first failure."""
    # flag‐based invariants
    for flag in TREE_FLAGS:
        if not getattr(stats, flag):
            raise InvariantError(
                f"Invariant failed: {flag} is False\n{t.print_structure()}"
            )

    # non‐empty‐tree invariants
    if not t.is_empty():
        if stats.item_count <= 0:
            raise InvariantError(
                f"Invariant failed: item_count={stats.item_count} ≤ 0 for non-empty tree"
            )
        if stats.item_slot_count <= 0:
            raise InvariantError(
                f"Invariant failed: item_slot_count={stats.item_slot_count} ≤ 0 for non-empty tree"
            )
        if stats.gnode_count <= 0:
            raise InvariantError(
                f"Invariant failed: gnode_count={stats.gnode_count} ≤ 0 for non-empty tree"
            )
        if stats.gnode_height <= 0:
            raise InvariantError(
                f"Invariant failed: gnode_height={stats.gnode_height} ≤ 0 for non-empty tree"
            )
        if stats.rank <= 0:
            raise InvariantError(
                f"Invariant failed: rank={stats.rank} ≤ 0 for non-empty tree"
            )
        if stats.least_item is None:
            raise InvariantError(
                "Invariant failed: least_item is None for non-empty tree"
            )
        if stats.greatest_item is None:
            raise InvariantError(
                "Invariant failed: greatest_item is None for non-empty tree"
            )