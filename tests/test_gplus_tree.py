
# -*- coding: utf-8 -*-
# ------------------------------------------------------------------------------
#
#   Copyright 2025 Jannik Hehemann
#
#   Licensed under the Apache License, Version 2.0 (the "License");
#   you may not use this file except in compliance with the License.
#   You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
#   Unless required by applicable law or agreed to in writing, software
#   distributed under the License is distributed on an "AS IS" BASIS,
#   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#   See the License for the specific language governing permissions and
#   limitations under the License.
#
# ------------------------------------------------------------------------------
"""Tests for jhehemann/customs/k-list abstract data structure."""
# pylint: skip-file

import unittest
import random
import json
import os
import statistics
import datetime
import random
import math

from pprint import pprint
from dataclasses import asdict

from packages.jhehemann.customs.gtree.gplus_tree import (
    GPlusTree,
    GPlusNode,
    gtree_stats_,
    collect_leaf_keys,
)
from packages.jhehemann.customs.gtree.base import Item
from stats_gplus_tree import (
    check_leaf_keys_and_values,
)

BASE_TIMESTAMP = datetime.datetime(2021, 1, 1, tzinfo=datetime.timezone.utc)
DUMMY_KEY = "0" * 64
DUMMY_ITEM = Item(DUMMY_KEY, None, None)

class TreeTestCase(unittest.TestCase):

    def setUp(self):
        self.tree = GPlusTree()

    def tearDown(self):
        # nothing to do if no tree or it’s empty
        if not getattr(self, 'tree', None) or self.tree.is_empty():
            return

        stats = gtree_stats_(self.tree, {})

        # --- core invariants ---
        self.assertTrue(stats.is_search_tree, "Search-tree invariant violated")
        self.assertTrue(stats.is_heap, "Heap invariant violated")
        self.assertTrue(stats.linked_leaf_nodes, "Not all leaf nodes linked")
        self.assertTrue(
            stats.internal_has_replicas,
            "Internal nodes do not contain replicas only"
        )
        self.assertTrue(
            stats.internal_packed,
            "Internal nodes with single entry not collapsed"
        )
        self.assertTrue(
            stats.all_leaf_values_present,
            "Leaf nodes do not contain full values"
        )
        self.assertTrue(
            stats.leaf_keys_in_order,
            "Leaf nodes do not contain sorted keys"
        )

        # --- optional invariants ---
        expected_item_count = getattr(self, 'expected_item_count', None)
        if expected_item_count is not None:
            self.assertEqual(
                stats.item_count, expected_item_count,
                f"Item count {stats.item_count} does not match expected {expected_item_count}\nTree structure:\n{self.tree.print_structure()}"
            )

        expected_root_rank = getattr(self, 'expected_root_rank', None)
        if expected_root_rank is not None:
            self.assertEqual(
                self.tree.node.rank, expected_root_rank,
                f"Root rank {self.tree.node.rank} does not match expected {expected_root_rank}"
            )

        expected_keys = getattr(self, 'expected_keys', None)
        if expected_keys is not None:
            keys = collect_leaf_keys(self.tree)
            self.assertEqual(
                sorted(keys), sorted(expected_keys),
                f"Leaf keys {keys} do not match expected {expected_keys}"
            )

        expected_gnode_count = getattr(self, 'expected_gnode_count', None)
        if expected_gnode_count is not None:
            self.assertEqual(
                stats.gnode_count, expected_gnode_count,
                f"GNode count {stats.gnode_count} does not match expected {expected_gnode_count}"
            )

        # Leaf invariants
        expected_keys = getattr(self, 'expected_leaf_keys', None)
        keys, presence_ok, have_values, order_ok = (
            check_leaf_keys_and_values(self.tree, expected_keys)
        )

        # Values and ordering must always hold
        self.assertTrue(have_values, "Leaf items must have non-None values")
        self.assertTrue(order_ok, "Leaf keys must be in sorted order")

        # If expected_leaf_keys was set, also enforce presence
        if expected_keys is not None:
            self.assertTrue(
                presence_ok,
                f"Leaf keys {keys} do not match expected {expected_keys}"
            )

    def _replica_repr(self, key):
        """Create a replica item with the given key."""
        return Item(key, None, None)
        
    def _assert_min_then_next(self, node, min, next):
        """Check that the minimum item is the expected key, and the next entry is also correct."""
        result = node.set.get_min()
        min_entry, next_entry = result.found_entry, result.next_entry
        self.assertIsNotNone(min_entry, f"{min.key} entry missing")
        self.assertEqual(min_entry.item.key, min.key, 
                         f"Minimum item should be {min.key}")
        self.assertEqual(
            min_entry.item.value, min.value, 
            f"Minimum item {min.key} should have value {min.value}"
        )
        self.assertTrue(min_entry.left_subtree.is_empty(),
                        f"Minimum item {min.key} should have no left subtree")
        
        if next is not None:
            self.assertIsNotNone(next_entry, "Next entry missing")
            self.assertEqual(next_entry.item.key, next.key,
                             f"Next item should be {next.key}")
            self.assertEqual(
                next_entry.item.value, next.value,
                f"Next item {next.key} should have value {next.value}"
            )
        else:
            self.assertIsNone(next_entry, f"Next entry should be None {next_entry}")
            
        return min_entry, next_entry

class TestInsertEmptyTree(TreeTestCase):            
    def test_root_and_leaf_for_rank_1_insert(self):
        key, rank = "a", 1
        item = Item(key, ord(key))
        self.tree.insert(item, rank)
        root = self.tree.node

        # Basic sanity
        self.expected_gnode_count = 1
        self.assertFalse(self.tree.is_empty(), "Tree should not be empty")
        self.assertFalse(root.set.is_empty(), "Root set must not be empty")
        self.assertEqual(root.rank, rank, f"Root rank should be {rank}")

        # Check root/leaf entries: dummy (no value) then item (with value)
        _, real = self._assert_min_then_next(root, DUMMY_ITEM, item)
        self.assertTrue(real.left_subtree.is_empty(),
                        "Leaf entry should have no left subtree")

        # Right subtree & next‑pointer
        self.assertTrue(root.right_subtree.is_empty(),
                        "Root should have no right subtree")
        self.assertIsNone(root.next, "Root should have no next leaf")

    def test_insert_rank_gt1_creates_internal_and_two_leaves(self):
        key, rank = "x", 3
        item = Item(key, ord(key))
        self.tree.insert(item, rank)

        root = self.tree.node

        with self.subTest("root"):
            self.assertFalse(self.tree.is_empty(), "Tree should not be empty")
            self.assertFalse(root.set.is_empty(), "Root set must not be empty")
            self.assertEqual(root.rank, rank, f"Root rank should be {rank}")
            self.assertIsNone(root.next, "Root should have no next")

            # Check root entries: dummy then replica (no values)
            _, replica = self._assert_min_then_next(
                root, DUMMY_ITEM, self._replica_repr(key)
            )
            
        with self.subTest("left leaf"):
            left_leaf = replica.left_subtree
            self.assertIsNotNone(left_leaf, "Left leaf missing")
            self.assertFalse(left_leaf.is_empty(),
                             "Left leaf must not be empty")
            self.assertEqual(left_leaf.node.rank, 1, 
                             "Left leaf rank should be 1")

            # Left leaf must contain only the dummy (no real items)
            self._assert_min_then_next(
                left_leaf.node, min=DUMMY_ITEM, next=None
            )
            
            # Right subtree & next‑pointer
            self.assertTrue(left_leaf.node.right_subtree.is_empty(),
                            "Left leaf should have no right subtree")
            self.assertIs(left_leaf.node.next, root.right_subtree,
                          "Left leaf next should point to right leaf")

        with self.subTest("right leaf"):
            right_leaf = root.right_subtree
            self.assertIsNotNone(right_leaf, "Right leaf missing")
            self.assertFalse(right_leaf.is_empty(), 
                             "Right leaf must not be empty")
            self.assertEqual(right_leaf.node.rank, 1, 
                             "Right leaf rank should be 1")

            # Right leaf must contain only the full representation of the item
            self._assert_min_then_next(
                right_leaf.node, min=item, next=None
            )
            
            # Right subtree & next‑pointer
            self.assertTrue(right_leaf.node.right_subtree.is_empty(),
                            "Right leaf should have no right subtree")
            self.assertIsNone(right_leaf.node.next,
                              "Right leaf should have no next pointer")


class TestInsertNonEmptyTree(TreeTestCase):
    def test_insert_multiple_increasing_keys_rank_1(self):
        keys = ["a", "b", "c", "d"]
        ranks = [1, 1, 1, 1]
        for i, k in enumerate(keys):
            calculated_rank = ranks[i]
            self.tree.insert(Item(k, ord(k)), rank=calculated_rank)
        self.expected_root_rank = max(ranks)
        self.expected_gnode_count = 1   # 1 root/leaf
        self.expected_item_count = 5    # currently incl. replicas & dummys
        self.expected_leaf_keys = ["a", "b", "c", "d"]

    def test_insert_multiple_decreasing_keys_rank_1(self):
        keys = ["d", "c", "b", "a"]
        ranks = [1, 1, 1, 1]
        for i, k in enumerate(keys):
            calculated_rank = ranks[i]
            self.tree.insert(Item(k, ord(k)), rank=calculated_rank)
        self.expected_root_rank = max(ranks)
        self.expected_gnode_count = 1   # 1 root/leaf
        self.expected_item_count = 5    # currently incl. replicas & dummys
        self.expected_leaf_keys = ["a", "b", "c", "d"]
        
    def test_insert_multiple_middle_keys_rank_1(self):
        keys = ["a", "e", "b", "d", "c"]
        ranks = [1, 1, 1, 1, 1]
        for i, k in enumerate(keys):
            calculated_rank = ranks[i]
            self.tree.insert(Item(k, ord(k)), rank=calculated_rank)
        self.expected_root_rank = max(ranks)
        self.expected_gnode_count = 1   # 1 root/leaf
        self.expected_item_count = 6   # currently incl. replicas & dummys
        self.expected_leaf_keys = ["a", "b", "c", "d", "e"]
        
    def test_insert_higher_rank_creates_root(self):
        keys = ["a", "z"]
        ranks = [1, 2]
        for i, k in enumerate(keys):
            calculated_rank = ranks[i]
            self.tree.insert(Item(k, ord(k)), rank=calculated_rank)
        self.expected_root_rank = max(ranks)
        self.expected_gnode_count = 3   # 1 root, 2 leaves
        self.expected_item_count = 5    # currently incl. replicas & dummys
        self.expected_leaf_keys = ["a", "z"]

    def test_insert_increasing_keys_and_ranks(self):
        keys = ["a", "b", "c"]
        ranks = [1, 2, 3]
        for i, k in enumerate(keys):
            calculated_rank = ranks[i]
            self.tree.insert(Item(k, ord(k)), rank=calculated_rank)
        self.expected_root_rank = max(ranks)
        self.expected_gnode_count = 5   # 1 root, 1 internal, 3 leaves
        self.expected_item_count = 8    # currently incl. replicas & dummys
        self.expected_leaf_keys = ["a", "b", "c"]

    def test_insert_decreasing_keys_increasing_ranks(self):
        keys = ["c", "b", "a"]
        ranks = [1, 2, 3]
        for i, k in enumerate(keys):
            calculated_rank = ranks[i]
            self.tree.insert(Item(k, ord(k)), rank=calculated_rank)
        self.expected_root_rank = max(ranks)
        self.expected_gnode_count = 5   # 1 root, 1 internal, 3 leaves
        self.expected_item_count = 8    # currently incl. replicas & dummys
        self.expected_leaf_keys = ["a", "b", "c"]

    def test_insert_non_existing_internal_rank_creates_node(self):
        keys = ["a", "c", "b"]
        ranks = [1, 3, 2]
        for i, k in enumerate(keys):
            calculated_rank = ranks[i]
            self.tree.insert(Item(k, ord(k)), rank=calculated_rank)
        self.expected_root_rank = max(ranks)
        self.expected_gnode_count = 5   # 1 root, 1 internal, 3 leaves
        self.expected_item_count = 8    # currently incl. replicas & dummys
        self.expected_leaf_keys = ["a", "b", "c"]

        # Check items
        result = self.tree.node.set.retrieve("c")   # root
        found = result.found_entry
        self.assertTrue(found is not None, "'c' should be in root")
        left = found.left_subtree   # new node (tree)
        self.assertEqual(left.node.rank, 2, "New node rank should be 2")
        result = left.node.set.retrieve("b")
        found = result.found_entry
        self.assertTrue(found is not None, "'b' should be in new node")
        
    def test_insert_decreasing_keys_and_ranks(self):
        keys = ["c", "b", "a"]
        ranks = [3, 2, 1]
        for i, k in enumerate(keys):
            calculated_rank = ranks[i]
            self.tree.insert(Item(k, ord(k)), rank=calculated_rank)
        self.expected_root_rank = max(ranks)
        self.expected_gnode_count = 5   # 1 root, 1 internal, 3 leaves
        self.expected_item_count = 8    # currently incl. replicas & dummys
        self.expected_leaf_keys = ["a", "b", "c"]
        
    def test_insert_increasing_keys_decreasing_ranks(self):
        keys = ["a", "b", "c"]
        ranks = [3, 2, 1]
        for i, k in enumerate(keys):
            calculated_rank = ranks[i]
            self.tree.insert(Item(k, ord(k)), rank=calculated_rank)
        self.expected_root_rank = max(ranks)
        self.expected_gnode_count = 5   # 1 root, 1 internal, 3 leaves
        self.expected_item_count = 8    # currently incl. replicas & dummys
        self.expected_leaf_keys = ["a", "b", "c"]
        
    def test_insert_multiple_increasing_keys_rank_gt_1(self):
        keys = ["a", "b", "c", "d"]
        ranks = [3, 3, 3, 3]
        for i, k in enumerate(keys):
            calculated_rank = ranks[i]
            self.tree.insert(Item(k, ord(k)), rank=calculated_rank)
        self.expected_root_rank = max(ranks)
        self.expected_gnode_count = 6   # 1 root, 5 leaves 
        self.expected_item_count = 10   # currently incl. replicas & dummys
        self.expected_leaf_keys = ["a", "b", "c", "d"]

    def test_insert_multiple_decreasing_keys_rank_gt_1(self):
        keys = ["d", "c", "b", "a"]
        ranks = [3, 3, 3, 3]
        for i, k in enumerate(keys):
            calculated_rank = ranks[i]
            self.tree.insert(Item(k, ord(k)), rank=calculated_rank)
        self.expected_root_rank = max(ranks)
        self.expected_gnode_count = 6   # 1 root, 5 leaves 
        self.expected_item_count = 10   # currently incl. replicas & dummys
        self.expected_leaf_keys = ["a", "b", "c", "d"]  
    
    def test_insert_middle_key_rank_gt_1_splits_child(self):
        keys = ["a", "c", "b"]
        ranks = [1, 1, 2]
        for i, k in enumerate(keys):
            calculated_rank = ranks[i]
            self.tree.insert(Item(k, ord(k)), rank=calculated_rank)
        self.expected_root_rank = max(ranks)
        self.expected_gnode_count = 3   # 1 root, 2 leaves
        self.expected_item_count = 6    # currently incl. replicas & dummys
        self.expected_leaf_keys = ["a", "b", "c"]

        result = self.tree.node.set.retrieve("c")
        found_entry = result.found_entry
        self.assertTrue(found_entry is not None, "Item 'c' should be present in root")
        self.assertEqual(found_entry.left_subtree.node.rank, 1, "New node rank should be 2")
        result = found_entry.left_subtree.node.set.retrieve("b")
        found_entry = result.found_entry
        self.assertTrue(found_entry is not None, "Item 'b' should be present in new node")
        
    def test_insert_duplicate_key_updates_value(self):
        keys = ["a", "a"]
        values = [99, 100]
        ranks = [1, 1]
        for i, k in enumerate(keys):
            calculated_rank = ranks[i]
            self.tree.insert(Item(k, values[i]), ranks[i])
        self.expected_root_rank = max(ranks)
        self.expected_gnode_count = 1
        self.expected_item_count = 2    # currently incl. replicas & dummys
        self.expected_leaf_keys = ["a"]
        # Check items
        result = self.tree.node.set.retrieve("a")
        found_entry = result.found_entry
        self.assertTrue(found_entry is not None, "'a' should be in root")
        self.assertEqual(found_entry.item.value, 100,
                         "'a' value should be updated to 100")




if __name__ == "__main__":
    unittest.main()