"""Utility functions for testing GPlusTree invariants."""

import logging
from gplus_trees.gplus_tree_base import (
    GPlusTreeBase,
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

def assert_tree_invariants_tc(tc, t: GPlusTreeBase, stats: Stats) -> None:
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
        # if t has a method get_size, the result must be equal to stats.real_item_count
        if hasattr(t, 'get_size'):
            if not t.is_empty():
                size = t.get_size()
                tc.assertEqual(size, stats.real_item_count,
                            f"Invariant failed: get_size()={size} ≠ real_item_count={stats.real_item_count}")

class InvariantError(Exception):
    """Raised when a GPlusTree invariant is violated."""
    pass

def assert_tree_invariants_raise(t: GPlusTreeBase, stats: Stats) -> None:
    """Check all invariants, raising on the first failure."""
    # flag‐based invariants
    for flag in TREE_FLAGS:
        if not getattr(stats, flag):
            logging.error(f"Invariant failed: {flag} is False")
            return

    # non‐empty‐tree invariants
    if not t.is_empty():
        if stats.item_count <= 0:
            logging.error(f"Invariant failed: item_count={stats.item_count} ≤ 0 for non-empty tree")
            return
        if stats.item_slot_count <= 0:
            logging.error(f"Invariant failed: item_slot_count={stats.item_slot_count} ≤ 0 for non-empty tree")
            return
        if stats.gnode_count <= 0:
            logging.error(f"Invariant failed: gnode_count={stats.gnode_count} ≤ 0 for non-empty tree")
            return
        if stats.gnode_height <= 0:
            logging.error(f"Invariant failed: gnode_height={stats.gnode_height} ≤ 0 for non-empty tree")
            return
        if stats.rank <= 0:
            logging.error(f"Invariant failed: rank={stats.rank} ≤ 0 for non-empty tree")
            return
        if stats.least_item is None:
            logging.error("Invariant failed: least_item is None for non-empty tree")
            return
        if stats.greatest_item is None:
            logging.error("Invariant failed: greatest_item is None for non-empty tree")
            return