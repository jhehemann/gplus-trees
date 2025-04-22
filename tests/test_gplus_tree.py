
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

from typing import Tuple, Optional, List

from packages.jhehemann.customs.gtree.gplus_tree import (
    GPlusTree,
    GPlusNode,
    gtree_stats_,
    collect_leaf_keys,
)
from packages.jhehemann.customs.gtree.base import (
    Item,
    Entry,
)
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
        """Create a replica item with given key, no value and no timestamp."""
        return Item(key, None, None)
        
    # def _assert_min_then_next(self, node, min, next):
    #     """Check that the minimum item is the expected key, and the next entry is also correct."""
    #     result = node.set.get_min()
    #     min_entry, next_entry = result.found_entry, result.next_entry
    #     self.assertIsNotNone(min_entry, f"{min.key} entry missing")
    #     self.assertEqual(min_entry.item.key, min.key, 
    #                      f"Minimum item should be {min.key}")
    #     self.assertEqual(
    #         min_entry.item.value, min.value, 
    #         f"Minimum item {min.key} should have value {min.value}"
    #     )
    #     self.assertTrue(min_entry.left_subtree.is_empty(),
    #                     f"Minimum item {min.key} should have no left subtree")
        
    #     if next is not None:
    #         self.assertIsNotNone(next_entry, "Next entry missing")
    #         self.assertEqual(next_entry.item.key, next.key,
    #                          f"Next item should be {next.key}")
    #         self.assertEqual(
    #             next_entry.item.value, next.value,
    #             f"Next item {next.key} should have value {next.value}"
    #         )
    #     else:
    #         self.assertIsNone(next_entry, f"Next entry should be None {next_entry}")
            
    #     return min_entry, next_entry
    
    def _assert_internal_node_properties(
            self, node: GPlusNode, items: List[Item], rank: int
        )-> Tuple[Optional[Entry], Optional[Entry]]:
        """
        Verify that `node` is an internal node with the expected rank containing exactly `items`in order, with the first item’s left subtree
        being empty and the next pointer being None.
        
        Returns:
            (min_entry, next_entry): the first two entries in `node.set`
                (next_entry is None if there's only one).
        """
        # must exist and be an internal node
        self.assertIsNotNone(node, "Node should not be None")
        self.assertEqual(node.rank, rank, f"Node rank should be {rank}")
        self.assertIsNone(node.next,
                          "Internal node should have no next pointer")

        # correct number of items
        actual_len = node.set.item_count()
        expected_len = len(items)
        self.assertEqual(
            actual_len, expected_len,
            f"Expected {expected_len} entries in node.set, found {actual_len}"
        )

        # verify each entry’s key, value=0 and empty left subtree for min item
        for i, (entry, expected_item) in enumerate(zip(node.set, items)):
            self.assertEqual(
                entry.item.key, expected_item.key,
                f"Entry #{i} key mismatch: expected {expected_item.key}, got {entry.item.key}"
            )
            self.assertEqual(
                entry.item.value, expected_item.value,
                f"Entry #{i} value for key {entry.item.key} should be None"
            )
            if i == 0:
                self.assertIsNotNone(entry.left_subtree,
                                     "Use empty trees; never None.")
                self.assertTrue(
                    entry.left_subtree.is_empty(),
                    "The first (min) entry’s left subtree should be empty"
                )
            else:
                self.assertFalse(
                    entry.left_subtree.is_empty(),
                    f"Entry #{i} ({expected_item.key}) should have non-empty "
                    f"left_subtree"
                )

        # collect and return the first two entries
        entries = list(node.set)
        min_entry = entries[0]
        next_entry = entries[1] if len(entries) > 1 else None
        return min_entry, next_entry


    def _assert_leaf_node_properties(
            self, node: GPlusNode, items: List[Item]
        ) -> Tuple[Optional[Entry], Optional[Entry]]:
        """
        Verify that `node` is a rank 1 leaf containing exactly `items` in order,
        and that all its subtrees are empty.

        Returns:
            (min_entry, next_entry): the first two entries in `node.set`
                (next_entry is None if there's only one).
        """
        # must exist and be a leaf
        self.assertIsNotNone(node, "Node should not be None")
        self.assertEqual(node.rank, 1, f"Leaf node rank should be 1")
        
        # correct number of items
        actual_len   = node.set.item_count()
        expected_len = len(items)
        self.assertEqual(
            actual_len, expected_len,
            f"Leaf node has {actual_len} items; expected {expected_len}"
        )

        # no children at a leaf
        self.assertIsNotNone(node.right_subtree, "Use empty trees; never None.")
        self.assertTrue(
            node.right_subtree.is_empty(),
            "Leaf node's right_subtree should be empty"
        )

        # verify each entry’s key/value and empty left subtree
        for i, (entry, expected) in enumerate(zip(node.set, items)):
            self.assertEqual(
                entry.item.key, expected.key,
                f"Entry #{i} key: expected {expected.key}, got {entry.item.key}"
            )
            self.assertEqual(
                entry.item.value, expected.value,
                f"Entry #{i} ({expected.key}) value: expected "
                f"{expected.value!r}, "
                f"got {entry.item.value!r}"
            )
            self.assertIsNotNone(entry.left_subtree,
                                 "Use empty trees; never None.")

            self.assertTrue(
                entry.left_subtree.is_empty(),
                f"Entry #{i} ({expected.key}) should have empty left_subtree"
            )

        # collect and return the first two entries
        entries = list(node.set)
        min_entry = entries[0]
        next_entry = entries[1] if len(entries) > 1 else None
        return min_entry, next_entry

class TestInsertInTree(TreeTestCase):
    def tearDown(self):
        self.assertFalse(self.tree.is_empty(), "Tree should not be empty")
        self.assertFalse(self.tree.node.set.is_empty(),
                         "Root set must not be empty")
        
        super().tearDown()

class TestInsertInEmptyTree(TestInsertInTree):    
    def test_insert_rank_1_creates_single_node(self):
        key, rank = "a", 1
        item = Item(key, ord(key))
        self.tree.insert(item, rank)
        root = self.tree.node      
        
        self._assert_leaf_node_properties(
            root,
            [DUMMY_ITEM, item]
        )

        leaf_iter = self.tree.iter_leaf_nodes()
        first_leaf = next(leaf_iter)
        self.assertIsNotNone(first_leaf, "Leaf node missing")
        self.assertIs(first_leaf, root, "Leaf node should be root")
        self.assertIsNone(next(leaf_iter, None),
                          "There should be only one leaf node")
        self.assertIsNone(root.next, "Root should have no next pointer")
        
    def test_insert_rank_gt_1_creates_root_and_two_leaves(self):
        key, rank = "x", 3
        item = Item(key, ord(key))
        self.tree.insert(item, rank)
        root = self.tree.node

        with self.subTest("root"):
            _, replica = self._assert_internal_node_properties(
                root,
                [DUMMY_ITEM, self._replica_repr(key)],
                rank
            )
            
        with self.subTest("left leaf"):
            t_left_leaf = replica.left_subtree
            self._assert_leaf_node_properties(
                t_left_leaf.node, [DUMMY_ITEM]
            )
            self.assertIs(t_left_leaf.node.next, root.right_subtree,
                          "Left leaf should point to right leaf")

        with self.subTest("right leaf"):
            t_right_leaf = root.right_subtree
            self._assert_leaf_node_properties(
                t_right_leaf.node, [item]
            )
            self.assertIsNone(t_right_leaf.node.next,
                              "Right leaf should have no next pointer")


class TestInsertInNonEmptyTreeLeaf(TestInsertInTree):
    def setUp(self):
        super().setUp()
        # Create a tree with two items
        keys = ["b", "d"]
        ranks = [1, 1]
        self.item_map = { k: (Item(k, ord(k))) for k in keys}
        self.rank_map = { key: rank for key, rank in zip(keys, ranks) }  
        for k in keys:
            item, rank = self.item_map[k], self.rank_map[k]
            self.tree.insert(item, rank)
        self.expected_root_rank = rank
        self.expected_gnode_count = 1
        self.expected_item_count = 4    # currently incl. replicas & dummys

    def tearDown(self):
        self.assertIsNone(self.tree.node.next,
                          "Leaf should have no next pointer")

    def test_insert_lowest_key(self):
        key, rank = "a", 1
        item = Item(key, ord(key))
        self.tree.insert(item, rank)

        self._assert_leaf_node_properties(
            self.tree.node,
            [DUMMY_ITEM, item, self.item_map["b"], self.item_map["d"]]
        )
        self.expected_leaf_keys = ["a", "b", "d"]

    def test_insert_highest_key(self):
        key, rank = "e", 1
        item = Item(key, ord(key))
        self.tree.insert(item, rank)

        self._assert_leaf_node_properties(
            self.tree.node,
            [DUMMY_ITEM, self.item_map["b"], self.item_map["d"], item]
        )
        self.expected_leaf_keys = ["b", "d", "e"]

    def test_insert_middle_key(self):
        key, rank = "c", 1
        item = Item(key, ord(key))
        self.tree.insert(item, rank)

        self._assert_leaf_node_properties(
            self.tree.node,
            [DUMMY_ITEM, self.item_map["b"], item, self.item_map["d"]]
        )
        self.expected_leaf_keys = ["b", "c", "d"]
        
class TestInsertInNonEmptyTreeGTMaxRankCreatesRoot(TestInsertInTree):
    def setUp(self):
        super().setUp()
        # Create a tree with 1 node and two full items
        keys = ["b", "d"]
        ranks = [1, 1]
        self.item_map = { k: (Item(k, ord(k))) for k in keys}
        self.rank_map = { key: rank for key, rank in zip(keys, ranks) }
        for k in keys:
            item, rank = self.item_map[k], self.rank_map[k]
            self.tree.insert(item, rank)
        
        self.insert_rank = 3
        self.expected_root_rank = self.insert_rank
        self.expected_gnode_count = 3
        self.expected_item_count = 6    # currently incl. replicas & dummys

    def test_insert_lowest_key(self):
        key, rank = "a", self.insert_rank
        item = Item(key, ord(key))
        self.tree.insert(item, rank)

        with self.subTest("root"):
            _, replica = self._assert_internal_node_properties(
                self.tree.node,
                [DUMMY_ITEM, self._replica_repr(key)],
                rank
            )
        with self.subTest("left leaf"):
            t_left_leaf = replica.left_subtree
            self._assert_leaf_node_properties(
                t_left_leaf.node, [DUMMY_ITEM]
            )
            self.assertIs(t_left_leaf.node.next, self.tree.node.right_subtree,
                          "Left leaf should point to right leaf")
        with self.subTest("right leaf"):
            t_right_leaf = self.tree.node.right_subtree
            self._assert_leaf_node_properties(
                t_right_leaf.node,
                [item, self.item_map["b"], self.item_map["d"]]
            )
            self.assertIsNone(t_right_leaf.node.next,
                              "Right leaf should have no next pointer")

        self.expected_leaf_keys = ["a", "b", "d"]

    def test_insert_highest_key(self):
        key, rank = "e", self.insert_rank
        item = Item(key, ord(key))
        self.tree.insert(item, rank)

        with self.subTest("root"):
            _, replica = self._assert_internal_node_properties(
                self.tree.node,
                [DUMMY_ITEM, self._replica_repr(key)],
                rank
            )
        with self.subTest("left leaf"):
            t_left_leaf = replica.left_subtree
            self._assert_leaf_node_properties(
                t_left_leaf.node,
                [DUMMY_ITEM, self.item_map["b"], self.item_map["d"]]
            )
            self.assertIs(t_left_leaf.node.next, self.tree.node.right_subtree,
                          "Left leaf should point to right leaf")
        with self.subTest("right leaf"):
            t_right_leaf = self.tree.node.right_subtree
            self._assert_leaf_node_properties(
                t_right_leaf.node,
                [item]
            )
            self.assertIsNone(t_right_leaf.node.next,
                              "Right leaf should have no next pointer")

        self.expected_leaf_keys = ["b", "d", "e"]
    
    def test_insert_middle_key(self):
        key, rank = "c", self.insert_rank
        item = Item(key, ord(key))
        self.tree.insert(item, rank)

        with self.subTest("root"):
            _, replica = self._assert_internal_node_properties(
                self.tree.node,
                [DUMMY_ITEM, self._replica_repr(key)],
                rank
            )
        with self.subTest("left leaf"):
            t_left_leaf = replica.left_subtree
            self._assert_leaf_node_properties(
                t_left_leaf.node,
                [DUMMY_ITEM, self.item_map["b"]]
            )
            self.assertIs(t_left_leaf.node.next, self.tree.node.right_subtree,
                          "Left leaf should point to right leaf")
        with self.subTest("right leaf"):
            t_right_leaf = self.tree.node.right_subtree
            # self.assertIsNotNone(t_right_leaf, "Use empty trees; never None.")
            self._assert_leaf_node_properties(
                t_right_leaf.node,
                [item, self.item_map["d"]]
            )
            self.assertIsNone(t_right_leaf.node.next,
                              "Right leaf should have no next pointer")

        self.expected_leaf_keys = ["b", "c", "d"]
        
class TestInsertInNonEmptyTreeRankGT1(TestInsertInTree):
    def setUp(self):
        super().setUp()
        # Create a tree with two items
        keys = ["b", "d", "f"]
        ranks = [1, 3, 1]
        self.item_map = { k: (Item(k, ord(k))) for k in keys}
        self.rank_map = { key: rank for key, rank in zip(keys, ranks) }
        for k in keys:
            item, rank = self.item_map[k], self.rank_map[k]
            self.tree.insert(item, rank)
        
        self.insert_rank = 3
        self.expected_root_rank = self.insert_rank
        self.expected_gnode_count = 4
        self.expected_item_count = 8    # currently incl. replicas & dummys


    def test_insert_lowest_key_splits_leaf(self):
        key, rank = "a", self.insert_rank
        item = Item(key, ord(key))
        self.tree.insert(item, rank)
        root = self.tree.node
        root_entries = list(root.set)

        with self.subTest("root"):
            self._assert_internal_node_properties(
                root,
                [DUMMY_ITEM, self._replica_repr(key), self._replica_repr("d")],
                rank
            )
        with self.subTest("leaf 1"):
            leaf_1 = root_entries[1].left_subtree.node
            self._assert_leaf_node_properties(leaf_1, [DUMMY_ITEM])
            self.assertIs(leaf_1.next, root_entries[2].left_subtree,
                          "Leaf 1 should point to leaf 2 tree")
        
        with self.subTest("leaf 2"):
            leaf_2 = root_entries[2].left_subtree.node
            self._assert_leaf_node_properties(
                leaf_2,
                [item, self.item_map["b"]]
            )
            self.assertIs(leaf_2.next, root.right_subtree, 
                          "Leaf 2 should point to leaf 3 tree")
        
        with self.subTest("leaf 3"):
            leaf_3 = root.right_subtree.node
            self._assert_leaf_node_properties(
                leaf_3,
                [self.item_map["d"], self.item_map["f"]]
            )
            self.assertIsNone(leaf_3.next,
                              "Leaf 3 should have no next pointer")
            
        self.expected_leaf_keys = ["a", "b", "d", "f"]

    def test_insert_lowest_key_no_leaf_split(self):
        key, rank = "c", self.insert_rank
        item = Item(key, ord(key))
        self.tree.insert(item, rank)
        root = self.tree.node
        root_entries = list(root.set)

        with self.subTest("root"):
            self._assert_internal_node_properties(
                root,
                [DUMMY_ITEM, self._replica_repr("c"), self._replica_repr("d")],
                rank
            )

        with self.subTest("leaf 1"):
            leaf_1 = root_entries[1].left_subtree.node
            self._assert_leaf_node_properties(
                leaf_1,
                [DUMMY_ITEM, self.item_map["b"]]
            )
            self.assertIs(leaf_1.next, root_entries[2].left_subtree,
                          "Leaf 1 should point to leaf 2 tree")
            
        with self.subTest("leaf 2"):
            leaf_2 = root_entries[2].left_subtree.node
            self._assert_leaf_node_properties(
                leaf_2,
                [item]
            )
            self.assertIs(leaf_2.next, root.right_subtree,
                          "Leaf 2 should point to leaf 3 tree")
            
        with self.subTest("leaf 3"):
            leaf_3 = root.right_subtree.node
            self._assert_leaf_node_properties(
                leaf_3,
                [self.item_map["d"], self.item_map["f"]]
            )
            self.assertIsNone(leaf_3.next,
                              "Leaf 3 should have no next pointer")
            
        self.expected_leaf_keys = ["b", "c", "d", "f"]
            
    def test_insert_highest_key_splits_leaf(self):
        key, rank = "e", self.insert_rank
        item = Item(key, ord(key))
        self.tree.insert(item, rank)
        root = self.tree.node
        root_entries = list(root.set)

        with self.subTest("root"):
            self._assert_internal_node_properties(
                root,
                [DUMMY_ITEM, self._replica_repr("d"), self._replica_repr(key)],
                rank
            )
        
        with self.subTest("leaf 1"):
            leaf_1 = root_entries[1].left_subtree.node
            self._assert_leaf_node_properties(
                leaf_1,
                [DUMMY_ITEM, self.item_map["b"]]
            )
            self.assertIs(leaf_1.next, root_entries[2].left_subtree,
                          "Leaf 1 should point to leaf 2 tree")
            
        with self.subTest("leaf 2"):
            leaf_2 = root_entries[2].left_subtree.node
            self._assert_leaf_node_properties(
                leaf_2,
                [self.item_map["d"]]
            )
            self.assertIs(leaf_2.next, root.right_subtree,
                          "Leaf 2 should point to leaf 3 tree")
            
        with self.subTest("leaf 3"):
            leaf_3 = root.right_subtree.node
            self._assert_leaf_node_properties(
                leaf_3,
                [item, self.item_map["f"]]
            )
            self.assertIsNone(leaf_3.next,
                              "Leaf 3 should have no next pointer")
            
        self.expected_leaf_keys = ["b", "d", "e", "f"]

    def test_insert_highest_key_no_leaf_split(self):
        key, rank = "g", self.insert_rank
        item = Item(key, ord(key))
        self.tree.insert(item, rank)
        root = self.tree.node
        root_entries = list(root.set)
        
        with self.subTest("root"):
            self._assert_internal_node_properties(
                root,
                [DUMMY_ITEM, self._replica_repr("d"), self._replica_repr(key)],
                rank
            )
        
        with self.subTest("leaf 1"):
            leaf_1 = root_entries[1].left_subtree.node
            self._assert_leaf_node_properties(
                leaf_1,
                [DUMMY_ITEM, self.item_map["b"]]
            )
            self.assertIs(leaf_1.next, root_entries[2].left_subtree,
                          "Leaf 1 should point to leaf 2 tree")
        
        with self.subTest("leaf 2"):
            leaf_2 = root_entries[2].left_subtree.node
            self._assert_leaf_node_properties(
                leaf_2,
                [self.item_map["d"], self.item_map["f"]]
            )
            self.assertIs(leaf_2.next, root.right_subtree,
                          "Leaf 2 should point to leaf 3 tree")
            
        with self.subTest("leaf 3"):
            leaf_3 = root.right_subtree.node
            self._assert_leaf_node_properties(
                leaf_3,
                [item]
            )
            self.assertIsNone(leaf_3.next,
                              "Leaf 3 should have no next pointer")
        
        self.expected_leaf_keys = ["b", "d", "f", "g"]

class TestInsertInNonEmptyTreeCollapsedLayerCreatesInternal(TestInsertInTree):
    def setUp(self):
        super().setUp()
        # Create a tree with two items
        keys = ["b", "d"]
        ranks = [1, 3, 1]
        self.item_map = { k: (Item(k, ord(k))) for k in keys}
        self.rank_map = { key: rank for key, rank in zip(keys, ranks) }
        for k in keys:
            item, rank = self.item_map[k], self.rank_map[k]
            self.tree.insert(item, rank)
        
        self.insert_rank = 2
        self.expected_root_rank = max(ranks)
        self.expected_gnode_count = 5
        self.expected_item_count = 8    # currently incl. replicas & dummys

    def test_insert_lowest_key(self):
        key, rank = "a", self.insert_rank
        item = Item(key, ord(key))
        self.tree.insert(item, rank)
        root = self.tree.node

        with self.subTest("root"):
            _, r_replica = self._assert_internal_node_properties(
                root,
                [DUMMY_ITEM, self._replica_repr("d")],
                3
            )
        
        with self.subTest("created internal"):
            new_internal = r_replica.left_subtree.node
            _, i_replica = self._assert_internal_node_properties(
                new_internal,
                [DUMMY_ITEM, self._replica_repr(key)],
                self.insert_rank
            )
        
        with self.subTest("leaf 1"):
            leaf_1 = i_replica.left_subtree.node
            self._assert_leaf_node_properties(
                leaf_1,
                [DUMMY_ITEM]
            )
            self.assertIs(leaf_1.next, new_internal.right_subtree,
                          "Leaf 1 should point to leaf 2 tree")

        with self.subTest("leaf 2"):
            leaf_2 = new_internal.right_subtree.node
            self._assert_leaf_node_properties(
                leaf_2,
                [item, self.item_map["b"]]
            )
            self.assertIs(leaf_2.next, root.right_subtree,
                          "Leaf 2 should point to leaf 3 tree")
        
        with self.subTest("leaf 3"):
            leaf_3 = root.right_subtree.node
            self._assert_leaf_node_properties(
                leaf_3,
                [self.item_map["d"]]
            )
            self.assertIsNone(leaf_3.next,
                              "Leaf 3 should have no next pointer")

        self.expected_leaf_keys = ["a", "b", "d"]

    def test_insert_highest_key(self):
        key, rank = "e", self.insert_rank
        item = Item(key, ord(key))
        self.tree.insert(item, rank)
        root = self.tree.node
        with self.subTest("root"):
            _, r_replica = self._assert_internal_node_properties(
                root,
                [DUMMY_ITEM, self._replica_repr("d")],
                3
            )
        with self.subTest("created internal"):
            new_internal = root.right_subtree.node
            _, i_replica = self._assert_internal_node_properties(
                new_internal,
                [self._replica_repr("d"), self._replica_repr(key)],
                self.insert_rank
            )
        
        with self.subTest("leaf 1"):
            leaf_1 = r_replica.left_subtree.node
            self._assert_leaf_node_properties(
                leaf_1,
                [DUMMY_ITEM, self.item_map["b"]]
            )
            self.assertIs(leaf_1.next, i_replica.left_subtree,
                          "Leaf 1 should point to leaf 2 tree")
            
        with self.subTest("leaf 2"):
            leaf_2 = i_replica.left_subtree.node
            self._assert_leaf_node_properties(
                leaf_2,
                [self.item_map["d"]]
            )
            self.assertIs(leaf_2.next, new_internal.right_subtree,
                          "Leaf 2 should point to leaf 3 tree")
            
        with self.subTest("leaf 3"):
            leaf_3 = new_internal.right_subtree.node
            self._assert_leaf_node_properties(
                leaf_3,
                [item]
            )
            self.assertIsNone(leaf_3.next,
                              "Leaf 3 should have no next pointer")
        
        self.expected_leaf_keys = ["b", "d", "e"]

class TestInsertInNonEmptyTreeRankGT2LowestKey(TestInsertInTree):
    def setUp(self):
        super().setUp()
        # Create a tree with two items
        keys = ["b", "d", "f", "h"]
        ranks = [1, 2, 2, 3]
        self.item_map = { k: (Item(k, ord(k))) for k in keys}
        self.rank_map = { key: rank for key, rank in zip(keys, ranks) }
        for k in keys:
            item, rank = self.item_map[k], self.rank_map[k]
            self.tree.insert(item, rank)
        
        self.insert_rank = 3
        self.expected_root_rank = max(ranks)

    def test_insert_lowest_key_splits_child_lowest_collapses_left_split(self):
        key, rank = "a", self.insert_rank
        item = Item(key, ord(key))
        self.tree.insert(item, rank)
        root = self.tree.node
        root_entries = list(root.set)

        self.expected_gnode_count = 7
        self.expected_item_count = 12    # currently incl. replicas & dummys
        
        with self.subTest("root"):
            self._assert_internal_node_properties(
                root,
                [DUMMY_ITEM, self._replica_repr(key), self._replica_repr("h")],
                3
            )
        
        with self.subTest("child right split (internal)"):
            c_right = root_entries[2].left_subtree.node
            self._assert_internal_node_properties(
                c_right,
                [
                    self._replica_repr(key),
                    self._replica_repr("d"),
                    self._replica_repr("f")
                ],
                2
            )
            c_right_entries = list(c_right.set)
        
        # child left split with single Dummy should collapse
        with self.subTest("leaf 1"):
            leaf_1 = root_entries[1].left_subtree.node
            self._assert_leaf_node_properties(
                leaf_1,
                [DUMMY_ITEM]
            )
            self.assertIs(leaf_1.next, c_right_entries[1].left_subtree,
                          "Leaf 1 should point to leaf 2 tree")
            
        with self.subTest("leaf 2"):
            leaf_2 = c_right_entries[1].left_subtree.node
            self._assert_leaf_node_properties(
                leaf_2,
                [item, self.item_map["b"]]
            )
            self.assertIs(leaf_2.next, c_right_entries[2].left_subtree,
                          "Leaf 2 should point to leaf 3 tree")
            
        with self.subTest("leaf 3"):
            leaf_3 = c_right_entries[2].left_subtree.node
            self._assert_leaf_node_properties(
                leaf_3,
                [self.item_map["d"]]
            )
            self.assertIs(leaf_3.next, c_right.right_subtree,
                          "Leaf 3 should point to leaf 4 tree")
            
        with self.subTest("leaf 4"):
            leaf_4 = c_right.right_subtree.node
            self._assert_leaf_node_properties(
                leaf_4,
                [self.item_map["f"]]
            )
            self.assertIs(leaf_4.next, root.right_subtree,
                          "Leaf 4 should point to leaf 5 tree")
        
        with self.subTest("leaf 5"):
            leaf_4 = root.right_subtree.node
            self._assert_leaf_node_properties(
                leaf_4,
                [self.item_map["h"]]
            )
            self.assertIsNone(leaf_4.next,
                              "Leaf 4 should have no next pointer")
            
        self.expected_leaf_keys = ["a", "b", "d", "f", "h"]

    def test_insert_lowest_key_splits_child_middle(self):
        key, rank = "e", self.insert_rank
        item = Item(key, ord(key))
        self.tree.insert(item, rank)
        root = self.tree.node
        root_entries = list(root.set)

        self.expected_gnode_count = 8
        self.expected_item_count = 13    # currently incl. replicas & dummys

        with self.subTest("root"):
            self._assert_internal_node_properties(
                root,
                [DUMMY_ITEM, self._replica_repr(key), self._replica_repr("h")],
                3
            )

        with self.subTest("child left split (internal)"):
            c_left = root_entries[1].left_subtree.node
            self._assert_internal_node_properties(
                c_left,
                [DUMMY_ITEM, self._replica_repr("d")],
                2
            )
            c_left_entries = list(c_left.set)

        with self.subTest("child right split (internal)"):
            c_right = root_entries[2].left_subtree.node
            self._assert_internal_node_properties(
                c_right,
                [self._replica_repr(key), self._replica_repr("f")],
                2
            )
            c_right_entries = list(c_right.set)

        with self.subTest("leaf 1"):
            leaf_1 = c_left_entries[1].left_subtree.node
            self._assert_leaf_node_properties(
                leaf_1,
                [DUMMY_ITEM, self.item_map["b"]]
            )
            self.assertIs(leaf_1.next, c_left.right_subtree,
                          "Leaf 1 should point to leaf 2 tree")
            
        with self.subTest("leaf 2"):
            leaf_2 = c_left.right_subtree.node
            self._assert_leaf_node_properties(
                leaf_2,
                [self.item_map["d"]]
            )
            self.assertIs(leaf_2.next, c_right_entries[1].left_subtree,
                          "Leaf 2 should point to leaf 3 tree")
        
        with self.subTest("leaf 3"):
            leaf_3 = c_right_entries[1].left_subtree.node
            self._assert_leaf_node_properties(
                leaf_3,
                [item]
            )
            self.assertIs(leaf_3.next, c_right.right_subtree,
                          "Leaf 3 should point to leaf 4 tree")
            
        with self.subTest("leaf 4"):
            leaf_4 = c_right.right_subtree.node
            self._assert_leaf_node_properties(
                leaf_4,
                [self.item_map["f"]]
            )
            self.assertIs(leaf_4.next, root.right_subtree,
                          "Leaf 4 should point to leaf 5 tree")
        
        with self.subTest("leaf 5"):
            leaf_5 = root.right_subtree.node
            self._assert_leaf_node_properties(
                leaf_5,
                [self.item_map["h"]]
            )
            self.assertIsNone(leaf_5.next,
                              "Leaf 5 should have no next pointer")
            
        self.expected_leaf_keys = ["b", "d", "e", "f", "h"]

    def test_insert_lowest_key_splits_child_highest(self):
        key, rank = "g", self.insert_rank
        item = Item(key, ord(key))
        self.tree.insert(item, rank)
        root = self.tree.node
        root_entries = list(root.set)

        self.expected_gnode_count = 7
        self.expected_item_count = 12    # currently incl. replicas & dummys

        with self.subTest("root"):
            self._assert_internal_node_properties(
                root,
                [DUMMY_ITEM,  self._replica_repr(key), self._replica_repr("h")],
                3
            )

        with self.subTest("child left split (internal)"):
            c_left = root_entries[1].left_subtree.node
            self._assert_internal_node_properties(
                c_left,
                [DUMMY_ITEM, self._replica_repr("d"), self._replica_repr("f")],
                2
            )
            c_left_entries = list(c_left.set)

        with self.subTest("leaf 1"):
            leaf_1 = c_left_entries[1].left_subtree.node
            self._assert_leaf_node_properties(
                leaf_1,
                [DUMMY_ITEM, self.item_map["b"]]
            )
            self.assertIs(leaf_1.next, c_left_entries[2].left_subtree,
                          "Leaf 1 should point to leaf 2 tree")
            
        with self.subTest("leaf 2"):
            leaf_2 = c_left_entries[2].left_subtree.node
            self._assert_leaf_node_properties(
                leaf_2,
                [self.item_map["d"]]
            )
            self.assertIs(leaf_2.next, c_left.right_subtree,
                          "Leaf 2 should point to leaf 3 tree")
        
        with self.subTest("leaf 3"):
            leaf_3 = c_left.right_subtree.node
            self._assert_leaf_node_properties(
                leaf_3,
                [self.item_map["f"]]
            )
             # child right split with single replica should collapse
            self.assertIs(leaf_3.next, root_entries[2].left_subtree,
                          "Leaf 3 should point to leaf 4 tree")
            
        with self.subTest("leaf 4"):
            leaf_4 = root_entries[2].left_subtree.node
            self._assert_leaf_node_properties(
                leaf_4,
                [item]
            )
            self.assertIs(leaf_4.next, root.right_subtree,
                          "Leaf 4 should point to leaf 5 tree")
        
        with self.subTest("leaf 5"):
            leaf_5 = root.right_subtree.node
            self._assert_leaf_node_properties(
                leaf_5,
                [self.item_map["h"]]
            )
            self.assertIsNone(leaf_5.next,
                              "Leaf 5 should have no next pointer")
        
        self.expected_leaf_keys = ["b", "d", "f", "g", "h"]

    
class TestInsertInNonEmptyTreeRankGT2HighestKey(TestInsertInTree):
    def setUp(self):
        super().setUp()
        keys = ["a", "c", "e"]
        ranks = [3, 2, 2]
        self.item_map = { k: (Item(k, ord(k))) for k in keys}
        self.rank_map = { key: rank for key, rank in zip(keys, ranks) }
        for k in keys:
            item, rank = self.item_map[k], self.rank_map[k]
            self.tree.insert(item, rank)
        
        self.insert_rank = 3
        self.expected_root_rank = max(ranks)

    def test_insert_highest_key_splits_child_lowest_collapses_left_split(self):
        # print("\nSelf.tree before insert:\n", self.tree.print_structure())
        key, rank = "b", self.insert_rank
        item = Item(key, ord(key))
        self.tree.insert(item, rank)
        # print("\n\nSelf.tree after insert:\n", self.tree.print_structure())
        root = self.tree.node
        root_entries = list(root.set)

        self.expected_gnode_count = 7
        self.expected_item_count = 11    # currently incl. replicas & dummys
        
        with self.subTest("root"):
            self._assert_internal_node_properties(
                root,
                [DUMMY_ITEM, self._replica_repr("a"), self._replica_repr(key)],
                3
            )
        
        with self.subTest("child right split (internal)"):
            c_right = root.right_subtree.node
            self._assert_internal_node_properties(
                c_right,
                [
                    self._replica_repr(key),
                    self._replica_repr("c"),
                    self._replica_repr("e")
                ],
                2
            )
            c_right_entries = list(c_right.set)
        
        # child left split with single replica should collapse
        with self.subTest("leaf 1"):
            leaf_1 = root_entries[1].left_subtree.node
            self._assert_leaf_node_properties(
                leaf_1,
                [DUMMY_ITEM]
            )
            self.assertIs(leaf_1.next, root_entries[2].left_subtree,
                          "Leaf 1 should point to leaf 2 tree")
            
        with self.subTest("leaf 2"):
            leaf_2 = root_entries[2].left_subtree.node
            self._assert_leaf_node_properties(
                leaf_2,
                [self.item_map["a"]]
            )
            self.assertIs(leaf_2.next, c_right_entries[1].left_subtree,
                            "Leaf 2 should point to leaf 3 tree")
        
        with self.subTest("leaf 3"):
            leaf_3 = c_right_entries[1].left_subtree.node
            self._assert_leaf_node_properties(
                leaf_3,
                [item]
            )
            self.assertIs(leaf_3.next, c_right_entries[2].left_subtree,
                          "Leaf 3 should point to leaf 4 tree")
        
        with self.subTest("leaf 4"):
            leaf_4 = c_right_entries[2].left_subtree.node
            self._assert_leaf_node_properties(
                leaf_4,
                [self.item_map["c"]]
            )
            self.assertIs(leaf_4.next, c_right.right_subtree,
                          "Leaf 4 should point to leaf 5 tree")
        
        with self.subTest("leaf 5"):
            leaf_5 = c_right.right_subtree.node
            self._assert_leaf_node_properties(
                leaf_5,
                [self.item_map["e"]]
            )
            self.assertIsNone(leaf_5.next,
                              "Leaf 5 should have no next pointer")
        
        self.expected_leaf_keys = ["a", "b", "c", "e"]

    def test_insert_highest_key_splits_child_middle(self):
        key, rank = "d", self.insert_rank
        item = Item(key, ord(key))
        self.tree.insert(item, rank)
        root = self.tree.node
        root_entries = list(root.set)

        self.expected_gnode_count = 8
        self.expected_item_count = 12    # currently incl. replicas & dummys

        with self.subTest("root"):
            self._assert_internal_node_properties(
                root,
                [DUMMY_ITEM, self._replica_repr("a"), self._replica_repr(key)],
                3
            )

        with self.subTest("child left split (internal)"):
            c_left = root_entries[2].left_subtree.node
            self._assert_internal_node_properties(
                c_left,
                [self._replica_repr("a"), self._replica_repr("c")],
                2
            )
            c_left_entries = list(c_left.set)

        with self.subTest("child right split (internal)"):
            c_right = root.right_subtree.node
            self._assert_internal_node_properties(
                c_right,
                [self._replica_repr(key), self._replica_repr("e")],
                2
            )
            c_right_entries = list(c_right.set)

        with self.subTest("leaf 1"):
            leaf_1 = root_entries[1].left_subtree.node
            self._assert_leaf_node_properties(
                leaf_1,
                [DUMMY_ITEM]
            )
            self.assertIs(leaf_1.next, c_left_entries[1].left_subtree,
                          "Leaf 1 should point to leaf 2 tree")
            
        with self.subTest("leaf 2"):
            leaf_2 = c_left_entries[1].left_subtree.node
            self._assert_leaf_node_properties(
                leaf_2,
                [self.item_map["a"]]
            )
            self.assertIs(leaf_2.next, c_left.right_subtree,
                          "Leaf 2 should point to leaf 3 tree")
        
        with self.subTest("leaf 3"):
            leaf_3 = c_left.right_subtree.node
            self._assert_leaf_node_properties(
                leaf_3,
                [self.item_map["c"]]
            )
            self.assertIs(leaf_3.next, c_right_entries[1].left_subtree,
                          "Leaf 3 should point to leaf 4 tree")
            
        with self.subTest("leaf 4"):
            leaf_4 = c_right_entries[1].left_subtree.node
            self._assert_leaf_node_properties(
                leaf_4,
                [item]
            )
            self.assertIs(leaf_4.next, c_right.right_subtree,
                          "Leaf 4 should point to leaf 5 tree")
            
        with self.subTest("leaf 5"):
            leaf_5 = c_right.right_subtree.node
            self._assert_leaf_node_properties(
                leaf_5,
                [self.item_map["e"]]
            )
            self.assertIsNone(leaf_5.next,
                              "Leaf 5 should have no next pointer")
            
        self.expected_leaf_keys = ["a", "c", "d", "e"]

    def test_insert_highest_key_splits_child_highest_collapses_right_split(self):
        key, rank = "f", self.insert_rank
        item = Item(key, ord(key))
        self.tree.insert(item, rank)
        root = self.tree.node
        root_entries = list(root.set)
        self.expected_gnode_count = 7
        self.expected_item_count = 11    # currently incl. replicas & dummys
        
        with self.subTest("root"):
            self._assert_internal_node_properties(
                root,
                [DUMMY_ITEM, self._replica_repr("a"), self._replica_repr(key)],
                3
            )
        with self.subTest("child left split (internal)"):
            c_left = root_entries[2].left_subtree.node
            self._assert_internal_node_properties(
                c_left,
                [
                    self._replica_repr("a"),
                    self._replica_repr("c"),
                    self._replica_repr("e")
                ],
                2
            )
            c_left_entries = list(c_left.set)
        
        with self.subTest("leaf 1"):
            leaf_1 = root_entries[1].left_subtree.node
            self._assert_leaf_node_properties(
                leaf_1,
                [DUMMY_ITEM]
            )
            self.assertIs(leaf_1.next, c_left_entries[1].left_subtree,
                          "Leaf 1 should point to leaf 2 tree")
        
        with self.subTest("leaf 2"):
            leaf_2 = c_left_entries[1].left_subtree.node
            self._assert_leaf_node_properties(
                leaf_2,
                [self.item_map["a"]]
            )
            self.assertIs(leaf_2.next, c_left_entries[2].left_subtree,
                          "Leaf 2 should point to leaf 3 tree")
        
        with self.subTest("leaf 3"):
            leaf_3 = c_left_entries[2].left_subtree.node
            self._assert_leaf_node_properties(
                leaf_3,
                [self.item_map["c"]]
            )
            self.assertIs(leaf_3.next, c_left.right_subtree,
                          "Leaf 3 should point to leaf 4 tree")
        
        with self.subTest("leaf 4"):
            leaf_4 = c_left.right_subtree.node
            self._assert_leaf_node_properties(
                leaf_4,
                [self.item_map["e"]]
            )
            self.assertIs(leaf_4.next, root.right_subtree,
                          "Leaf 4 should point to leaf 5 tree")
        
        # child right split with single replica should collapse
        with self.subTest("leaf 5"):
            leaf_5 = root.right_subtree.node
            self._assert_leaf_node_properties(
                leaf_5,
                [item]
            )
            self.assertIsNone(leaf_5.next,
                              "Leaf 5 should have no next pointer")
        
        self.expected_leaf_keys = ["a", "c", "e", "f"]
        
    
class TestInsertNonemptyTreeHighCollapsingNodesRightLeft(TestInsertInTree):
    def setUp(self):
        super().setUp()
        keys = ["a", "b", "d", "e"]
        ranks = [4, 3, 1, 2]
        self.item_map = { k: (Item(k, ord(k))) for k in keys}
        self.rank_map = { key: rank for key, rank in zip(keys, ranks) }
        for k in keys:
            item, rank = self.item_map[k], self.rank_map[k]
            self.tree.insert(item, rank)
        
        self.insert_rank = 4
        self.expected_root_rank = max(ranks)

    def test_insert_high_collapse_rank_3_right_rank_2_left_split_leaf(self):
        # print(f"\n\n\nSelf before insert: {self.tree.print_structure()}")
        key, rank = "c", self.insert_rank
        item = Item(key, ord(key))
        self.tree.insert(item, rank)
        # print(f"\n\n\nSelf after insert: {self.tree.print_structure()}")
        root = self.tree.node
        root_entries = list(root.set)

        self.expected_gnode_count = 8
        self.expected_item_count = 13    # currently incl. replicas & dummys
        
        with self.subTest("root"):
            self._assert_internal_node_properties(
                root,
                [
                    DUMMY_ITEM,
                    self._replica_repr("a"), 
                    self._replica_repr(key)
                ],
                4
            )
        
        with self.subTest("child rank 3 (left split of right collapsed)"):
            c_left = root_entries[2].left_subtree.node
            self._assert_internal_node_properties(
                c_left,
                [self._replica_repr("a"), self._replica_repr("b")],
                3
            )
            c_left_entries = list(c_left.set)
        
        with self.subTest("child rank 2 (right split of left collapsed)"):
            c_right = root.right_subtree.node
            self._assert_internal_node_properties(
                c_right,
                [self._replica_repr(key), self._replica_repr("e")],
                2
            )
            c_right_entries = list(c_right.set)
        
        with self.subTest("leaf 1"):
            leaf_1 = root_entries[1].left_subtree.node
            self._assert_leaf_node_properties(
                leaf_1,
                [DUMMY_ITEM]
            )
            self.assertIs(leaf_1.next, c_left_entries[1].left_subtree,
                          "Leaf 1 should point to leaf 2 tree")
            
        with self.subTest("leaf 2"):
            leaf_2 = c_left_entries[1].left_subtree.node
            self._assert_leaf_node_properties(
                leaf_2,
                [self.item_map["a"]]
            )
            self.assertIs(leaf_2.next, c_left.right_subtree,
                          "Leaf 2 should point to leaf 3 tree")
            
        with self.subTest("leaf 3"):
            leaf_3 = c_left.right_subtree.node
            self._assert_leaf_node_properties(
                leaf_3,
                [self.item_map["b"]]
            )
            self.assertIs(leaf_3.next, c_right_entries[1].left_subtree,
                          "Leaf 3 should point to leaf 4 tree")
            
        with self.subTest("leaf 4"):
            leaf_4 = c_right_entries[1].left_subtree.node
            self._assert_leaf_node_properties(
                leaf_4,
                [item, self.item_map["d"]]
            )
            self.assertIs(leaf_4.next, c_right.right_subtree,
                          "Leaf 4 should point to leaf 5 tree")
            
        with self.subTest("leaf 5"):
            leaf_5 = c_right.right_subtree.node
            self._assert_leaf_node_properties(
                leaf_5,
                [self.item_map["e"]]
            )
            self.assertIsNone(leaf_5.next,
                              "Leaf 5 should have no next pointer")
            
        self.expected_leaf_keys = ["a", "b", "c", "d", "e"]

    
class TestInsertNonemptyTreeMidCollapsingNodesRightLeft(TestInsertInTree):
    def setUp(self):
        super().setUp()
        keys = ["a", "b", "d", "e", "f"]
        ranks = [4, 3, 1, 2, 4]
        self.item_map = { k: (Item(k, ord(k))) for k in keys}
        self.rank_map = { key: rank for key, rank in zip(keys, ranks) }
        for k in keys:
            item, rank = self.item_map[k], self.rank_map[k]
            self.tree.insert(item, rank)
        
        self.insert_rank = 4
        self.expected_root_rank = max(ranks)

    def test_insert_mid_collapse_rank_3_right_rank_2_left_split_leaf(self):
        
        # print(f"\n\n\nSelf before insert: {self.tree.print_structure()}")
        key, rank = "c", self.insert_rank
        item = Item(key, ord(key))
        self.tree.insert(item, rank)
        # print(f"\n\n\nSelf after insert: {self.tree.print_structure()}")
        root = self.tree.node
        root_entries = list(root.set)

        self.expected_gnode_count = 9
        self.expected_item_count = 15    # currently incl. replicas & dummys
        
        with self.subTest("root"):
            self._assert_internal_node_properties(
                root,
                [
                    DUMMY_ITEM,
                    self._replica_repr("a"), 
                    self._replica_repr(key),
                    self._replica_repr("f")
                ],
                4
            )
        
        with self.subTest("child rank 3 (left split of right collapsed)"):
            c_left = root_entries[2].left_subtree.node
            self._assert_internal_node_properties(
                c_left,
                [self._replica_repr("a"), self._replica_repr("b")],
                3
            )
            c_left_entries = list(c_left.set)
        
        with self.subTest("child rank 2 (right split of left collapsed)"):
            c_right = root_entries[3].left_subtree.node
            self._assert_internal_node_properties(
                c_right,
                [self._replica_repr(key), self._replica_repr("e")],
                2
            )
            c_right_entries = list(c_right.set)
        
        with self.subTest("leaf 1"):
            leaf_1 = root_entries[1].left_subtree.node
            self._assert_leaf_node_properties(
                leaf_1,
                [DUMMY_ITEM]
            )
            self.assertIs(leaf_1.next, c_left_entries[1].left_subtree,
                          "Leaf 1 should point to leaf 2 tree")
            
        with self.subTest("leaf 2"):
            leaf_2 = c_left_entries[1].left_subtree.node
            self._assert_leaf_node_properties(
                leaf_2,
                [self.item_map["a"]]
            )
            self.assertIs(leaf_2.next, c_left.right_subtree,
                          "Leaf 2 should point to leaf 3 tree")
            
        with self.subTest("leaf 3"):
            leaf_3 = c_left.right_subtree.node
            self._assert_leaf_node_properties(
                leaf_3,
                [self.item_map["b"]]
            )
            self.assertIs(leaf_3.next, c_right_entries[1].left_subtree,
                          "Leaf 3 should point to leaf 4 tree")
            
        with self.subTest("leaf 4"):
            leaf_4 = c_right_entries[1].left_subtree.node
            self._assert_leaf_node_properties(
                leaf_4,
                [item, self.item_map["d"]]
            )
            self.assertIs(leaf_4.next, c_right.right_subtree,
                          "Leaf 4 should point to leaf 5 tree")
            
        with self.subTest("leaf 5"):
            leaf_5 = c_right.right_subtree.node
            self._assert_leaf_node_properties(
                leaf_5,
                [self.item_map["e"]]
            )
            self.assertIs(leaf_5.next, root.right_subtree,
                          "Leaf 5 should point to leaf 6 tree")
            
        with self.subTest("leaf 6"):
            leaf_6 = root.right_subtree.node
            self._assert_leaf_node_properties(
                leaf_6,
                [self.item_map["f"]]
            )
            self.assertIsNone(leaf_6.next,
                              "Leaf 6 should have no next pointer")
            
        self.expected_leaf_keys = ["a", "b", "c", "d", "e", "f"]


class TestInsertInNonEmptyTree(TestInsertInTree):
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

    def test_insert_internal_rank_2_left_creates_node(self):
        keys = ["a", "c", "b"]
        ranks = [1, 3, 2]
        item_map = {k: (Item(k, ord(k)), r) for k, r in zip(keys, ranks)}
        replica_map = {k: Item(k, None) for k in keys}
        for k in keys:
            item, rank = item_map[k]
            self.tree.insert(item, rank=rank)

        self.expected_root_rank = max(ranks)
        self.expected_gnode_count = 5   # 1 root, 1 internal, 3 leaves
        self.expected_item_count = 8    # currently incl. replicas & dummys
        self.expected_leaf_keys = ["a", "b", "c"]

        # print(self.tree.print_structure())

        # Check root
        exp_items = [DUMMY_ITEM, replica_map["c"]]
        self._assert_internal_node_properties(
            self.tree.node, exp_items, rank=item_map["c"][1]
        )

        # Check root's right subtree
        leaf_3 = self.tree.node.right_subtree
        items = [item_map["c"][0]]
        self._assert_leaf_node_properties(leaf_3.node, items)
        
        # Check root item's left subtree
        _, next_root = self.tree.node.set.get_min()
        internal_l = next_root.left_subtree
        items = [DUMMY_ITEM, replica_map["b"]]
        self._assert_internal_node_properties(
            internal_l.node, items, rank=item_map["b"][1]
        )
        
        # Check internal node replica's left subtree
        _, next_internal_l = internal_l.node.set.get_min()
        leaf_1 = next_internal_l.left_subtree
        items = [DUMMY_ITEM, item_map["a"][0]]
        self._assert_leaf_node_properties(leaf_1.node, items)
        
        # Check internal node's right subtree
        leaf_2 = internal_l.node.right_subtree
        items = [item_map["b"][0]]
        self._assert_leaf_node_properties(leaf_2.node, items)
        
        # Check next pointer
        self.assertIsNone(self.tree.node.next,
                          "Root should have no next pointer")
        self.assertIsNone(internal_l.node.next,
                          "Internal node should have no next pointer")
        self.assertIs(leaf_1.node.next, leaf_2, 
                      "Leaf 1 should point to leaf 2")
        self.assertIs(leaf_2.node.next, leaf_3,
                      "Leaf 2 should point to leaf 3")
        self.assertIsNone(leaf_3.node.next,
                          "Leaf 3 next pointer should be None")

    def test_insert_internal_rank_2_right_creates_node(self):
        keys = ["c", "a", "b"]
        ranks = [1, 3, 2]
        item_map = {k: (Item(k, ord(k)), r) for k, r in zip(keys, ranks)}
        replica_map = {k: Item(k, None) for k in keys}
        for k in keys:
            item, rank = item_map[k]
            self.tree.insert(item, rank=rank)

        self.expected_root_rank = max(ranks)
        self.expected_gnode_count = 5   # 1 root, 1 internal, 3 leaves
        self.expected_item_count = 8    # currently incl. replicas & dummys
        self.expected_leaf_keys = ["a", "b", "c"]

        # Check root
        exp_items = [DUMMY_ITEM, replica_map["a"]]
        self._assert_internal_node_properties(
            self.tree.node, exp_items, rank=item_map["a"][1]
        )
        
        # Check root item's left subtree
        _, next_root = self.tree.node.set.get_min()
        leaf_1 = next_root.left_subtree
        items = [DUMMY_ITEM]
        self._assert_leaf_node_properties(leaf_1.node, items)
        
        # Check root's right subtree
        internal_r = self.tree.node.right_subtree
        items = [replica_map["a"], replica_map["b"]]
        self._assert_internal_node_properties(
            internal_r.node, items, rank=item_map["b"][1]
        )
        
        # Check internal node replica's left subtree
        _, next_internal_r = internal_r.node.set.get_min()
        leaf_2 = next_internal_r.left_subtree
        items = [item_map["a"][0]]
        self._assert_leaf_node_properties(leaf_2.node, items)
        
        # Check internal node's right subtree
        leaf_3 = internal_r.node.right_subtree
        items = [item_map["b"][0], item_map["c"][0]]
        self._assert_leaf_node_properties(leaf_3.node, items)
        
        # Check next pointer
        self.assertIsNone(self.tree.node.next,
                          "Root should have no next pointer")
        self.assertIsNone(internal_r.node.next,
                          "Internal node should have no next pointer")
        self.assertIs(leaf_1.node.next, leaf_2, 
                      "Leaf 1 should point to leaf 2")
        self.assertIs(leaf_2.node.next, leaf_3,
                      "Leaf 2 should point to leaf 3")
        self.assertIsNone(leaf_3.node.next,
                          "Leaf 3 next pointer should be None")
        
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
    
    def test_insert_middle_key_rank_2_splits_leaf(self):
        keys = ["a", "c", "b"]
        ranks = [1, 1, 2]
        for i, k in enumerate(keys):
            calculated_rank = ranks[i]
            self.tree.insert(Item(k, ord(k)), rank=calculated_rank)
        self.expected_root_rank = max(ranks)
        self.expected_gnode_count = 3   # 1 root, 2 leaves
        self.expected_item_count = 6    # currently incl. replicas & dummys
        self.expected_leaf_keys = ["a", "b", "c"]

        result = self.tree.node.set.retrieve("b")
        found_entry = result.found_entry
        self.assertTrue(found_entry is not None, "Item 'b' should be present in root")
        self.assertEqual(found_entry.left_subtree.node.rank, 1, "New node rank should be 2")
        result = found_entry.left_subtree.node.set.retrieve("a")
        found_entry = result.found_entry
        self.assertTrue(found_entry is not None, "Item 'a' should be present in new node")
        
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

    def test_insert_packed_tree(self):
        keys = ["a", "b", "c"]
        ranks = [2, 2, 2]
        for i, k in enumerate(keys):
            calculated_rank = ranks[i]
            self.tree.insert(Item(k, ord(k)), rank=calculated_rank)
        self.expected_root_rank = max(ranks)
        self.expected_gnode_count = 5   # 1 root, 2 leaves
        self.expected_item_count = 8    # currently incl. replicas & dummys

if __name__ == "__main__":
    unittest.main()