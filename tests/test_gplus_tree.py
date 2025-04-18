
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
)
from packages.jhehemann.customs.gtree.base import Item
from stats_gplus_tree import (
    check_leaf_keys_and_values
)

BASE_TIMESTAMP = datetime.datetime(2021, 1, 1, tzinfo=datetime.timezone.utc)
DUMMY_KEY = "0" * 64

def geometric(p):
    u = random.random()
    return math.ceil(math.log(1 - u) / math.log(1 - p))

class TestGPlusTreeInsert(unittest.TestCase):
    # def setUp(self):
    #     self.tree = GPlusTree()

    def setUp(self):
        # whenever a test creates self.tree, register the cleanup
        self.tree = GPlusTree()
        self.expected_gnode_count = None
        self.expected_item_count = None
        self.expected_leaf_keys = None
        self.addCleanup(self._assert_tree_invariants)

    def _assert_tree_invariants(self):
        # only run if the test actually set up self.tree
        if not hasattr(self, 'tree') or self.tree.is_empty():
            # print("Tree is empty or not set up.")
            return

        # Empty‐tree check
        if self.tree.is_empty():
            self.assertIsNone(self.tree.node, "Node should be None in empty tree.")
            # print("Tree is empty.")
            return
        else:
            self.assertIsNotNone(self.tree.node, "Node should not be None in non-empty tree.")
            # print("Tree is not empty. Structure:")
            # print(self.tree.print_structure())

        # Structural invariants
        stats = gtree_stats_(self.tree, {})
        # print("\n\nTree stats:")
        # pprint(asdict(stats))
        self.assertTrue(stats.is_search_tree, "Tree violates search‑tree invariant")
        self.assertTrue(stats.is_heap, "Tree violates heap invariant")
        self.assertTrue(stats.linked_leaf_nodes, "Leaf nodes are not linked correctly")
        self.assertTrue(stats.internal_has_replicas, "Internal nodes do not contain replicas only")
        self.assertTrue(stats.internal_packed, "Internal nodes are not collapsed correctly")
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
        self.assertTrue(have_values, "Leaf items must have non‑None values")
        self.assertTrue(order_ok, "Leaf keys must be in sorted order")

        # If expected_leaf_keys was set, also enforce presence
        if expected_keys is not None:
            # print(f"Expected keys: {expected_keys}")
            # print(f"Actual keys: {keys}")
            self.assertTrue(
                presence_ok,
                f"Leaf keys {keys} do not match expected {expected_keys}"
            )

        expected_item_count = getattr(self, 'expected_item_count', None)
        if expected_item_count is not None:
            self.assertEqual(
                stats.item_count, expected_item_count,
                f"Item count {stats.item_count} does not match expected {expected_item_count}"
            )

    def test_insert_into_empty_tree_rank_1(self):
        item = Item("a", 1)
        self.tree.insert(item, rank=1)
        self.assertIsNotNone(self.tree.node, "Root should not be None")
        self.assertEqual(self.tree.node.rank, 1, "Root rank should be 1")
        self.expected_item_count = 2    # currently incl. replicas & dummys
        self.expected_leaf_keys = ["a"]
        self.expected_gnode_count = 1
        result = self.tree.node.set.retrieve("0" * 64)
        self.assertTrue(result.found_entry is not None, "Dummy item should be present in the set")
        self.assertEqual(result.next_entry.item, item, "Item should be found in the tree.\n")

    def test_insert_into_empty_tree_rank_gt_1(self):
        item = Item("a", 10, BASE_TIMESTAMP)
        self.tree.insert(item, rank=3)
        self.assertIsNotNone(self.tree.node, f"Root should not be None")
        self.assertEqual(self.tree.node.rank, 3, f"Root rank should be 3")
        self.expected_item_count = 4    # currently incl. replicas & dummys
        self.expected_leaf_keys = ["a"]
        self.expected_gnode_count = 3   # 2 leaf nodes + 1 root
        self.assertEqual(self.tree.node.set.item_count(), 2, f"Replica count in root should be 2 (1 dummy + 1 item)")      

    def test_insert_multiple_increasing_keys_rank_1(self):
        keys = ["a", "b", "c", "d"]
        ranks = [1, 1, 1, 1]
        for i, k in enumerate(keys):
            calculated_rank = ranks[i]
            self.tree.insert(Item(k, ord(k)), rank=calculated_rank)
        self.expected_item_count = 5    # currently incl. replicas & dummys
        self.expected_leaf_keys = ["a", "b", "c", "d"]
        self.expected_gnode_count = 1

    def test_insert_multiple_decreasing_keys_rank_1(self):
        keys = ["d", "c", "b", "a"]
        ranks = [1, 1, 1, 1]
        for i, k in enumerate(keys):
            calculated_rank = ranks[i]
            self.tree.insert(Item(k, ord(k)), rank=calculated_rank)
        self.expected_item_count = 5    # currently incl. replicas & dummys
        self.expected_leaf_keys = ["a", "b", "c", "d"]
        self.expected_gnode_count = 1

    def test_insert_multiple_random_keys_rank_1(self):
        keys = ["g", "e", "f", "d", "i", "h", "b", "a", "c"]
        ranks = [1, 1, 1, 1, 1, 1, 1, 1, 1]
        for i, k in enumerate(keys):
            calculated_rank = ranks[i]
            self.tree.insert(Item(k, ord(k)), rank=calculated_rank)
        self.expected_item_count = 10   # currently incl. replicas & dummys
        self.expected_leaf_keys = ["a", "b", "c", "d", "e", "f", "g", "h", "i"]
        self.expected_gnode_count = 1

    def test_insert_higher_rank_creates_root(self):
        keys = ["a", "z"]
        ranks = [1, 2]
        for i, k in enumerate(keys):
            calculated_rank = ranks[i]
            self.tree.insert(Item(k, ord(k)), rank=calculated_rank)
        self.assertEqual(self.tree.node.rank, 2, "Root rank should be 2.\n" + str(self.tree.print_structure()))
        self.expected_item_count = 5    # currently incl. replicas & dummys
        self.expected_leaf_keys = ["a", "z"]
        self.expected_gnode_count = 3   # 2 leaf nodes + 1 root

    def test_insert_increasing_keys_and_ranks(self):
        keys = ["a", "b", "c"]
        ranks = [1, 2, 3]
        for i, k in enumerate(keys):
            calculated_rank = ranks[i]
            self.tree.insert(Item(k, ord(k)), rank=calculated_rank)
        self.assertEqual(self.tree.node.rank, max(ranks), "Root rank should be 2.\n" + str(self.tree.print_structure()))
        self.expected_item_count = 8    # currently incl. replicas & dummys
        self.expected_leaf_keys = ["a", "b", "c"]
        self.expected_gnode_count = 5   # 2 leaf nodes + 1 root

    def test_insert_decreasing_keys_increasing_ranks(self):
        keys = ["c", "b", "a"]
        ranks = [1, 2, 3]
        for i, k in enumerate(keys):
            calculated_rank = ranks[i]
            self.tree.insert(Item(k, ord(k)), rank=calculated_rank)
        # print("\n\nTree structure:")
        # print(self.tree.print_structure())
        self.assertEqual(self.tree.node.rank, max(ranks), "Root rank should be 2.\n" + str(self.tree.print_structure()))
        self.expected_item_count = 8    # currently incl. replicas & dummys
        self.expected_leaf_keys = ["a", "b", "c"]
        self.expected_gnode_count = 5   # 2 leaf nodes + 1 root

    def test_insert_non_existing_internal_rank_creates_node(self):
        keys = ["a", "c", "b"]
        ranks = [1, 3, 2]
        for i, k in enumerate(keys):
            calculated_rank = ranks[i]
            self.tree.insert(Item(k, ord(k)), rank=calculated_rank)
        self.assertEqual(self.tree.node.rank, 3, "Root rank should be 2.\n" + str(self.tree.print_structure()))
        result = self.tree.node.set.retrieve("c")
        found_entry = result.found_entry
        self.assertTrue(found_entry is not None, "Item 'c' should be present in root")
        self.assertEqual(found_entry.left_subtree.node.rank, 2, "New node rank should be 2")
        result = found_entry.left_subtree.node.set.retrieve("b")
        found_entry = result.found_entry
        self.assertTrue(found_entry is not None, "Item 'b' should be present in new node")
        self.expected_item_count = 8    # currently incl. replicas & dummys
        self.expected_leaf_keys = ["a", "b", "c"]
        self.expected_gnode_count = 5   # 2 leaf nodes + 1 root
    
    def test_insert_decreasing_keys_and_ranks(self):
        keys = ["c", "b", "a"]
        ranks = [3, 2, 1]
        for i, k in enumerate(keys):
            calculated_rank = ranks[i]
            self.tree.insert(Item(k, ord(k)), rank=calculated_rank)
        # print("\n\nTree structure:")
        # print(self.tree.print_structure())
        self.assertEqual(self.tree.node.rank, max(ranks), "Root rank should be 2.\n" + str(self.tree.print_structure()))
        self.expected_item_count = 8    # currently incl. replicas & dummys
        self.expected_leaf_keys = ["a", "b", "c"]
        self.expected_gnode_count = 5   # 2 leaf nodes + 1 root

    def test_insert_increasing_keys_decreasing_ranks(self):
        keys = ["a", "b", "c"]
        ranks = [3, 2, 1]
        for i, k in enumerate(keys):
            calculated_rank = ranks[i]
            self.tree.insert(Item(k, ord(k)), rank=calculated_rank)
        # print("\n\nTree structure:")
        # print(self.tree.print_structure())
        self.assertEqual(self.tree.node.rank, max(ranks), "Root rank should be 2.\n" + str(self.tree.print_structure()))
        self.expected_item_count = 8    # currently incl. replicas & dummys
        self.expected_leaf_keys = ["a", "b", "c"]
        self.expected_gnode_count = 5   # 2 leaf nodes + 1 root

        # print("\n\nTree structure:")
        # print(self.tree.print_structure())


    # def test_insert_rasafdasdfasdf_keys_increasing_ranks(self):
    #     keys = ["d", "c", "a", "b"]
    #     ranks = [1, 2, 4, 3]
    #     for i, k in enumerate(keys):
    #         calculated_rank = ranks[i]
    #         self.tree.insert(Item(k, ord(k)), rank=calculated_rank)
    #     print("\n\nTree structure:")
    #     print(self.tree.print_structure())
    #     self.assertEqual(self.tree.node.rank, max(ranks), "Root rank should be 2.\n" + str(self.tree.print_structure()))
    #     self.expected_item_count = 9    # currently incl. replicas & dummys
    #     self.expected_leaf_keys = ["a", "b", "c", "d"]
    #     self.expected_gnode_count = 5   # 2 leaf nodes + 1 root
    
    
    
    
    
    # def test_insert_higher_rank_gt_1_creates_root_collapsed(self):
    #     keys = ["a", "z"]
    #     ranks = [1, 5]
    #     for i, k in enumerate(keys):
    #         calculated_rank = ranks[i]
    #         self.tree.insert(Item(k, ord(k)), rank=calculated_rank)
    #     self.assertEqual(self.tree.node.rank, 5, "Root rank should be 5.\n" + str(self.tree.print_structure()))
    #     self.expected_item_count = 5    # currently incl. replicas & dummys
    #     self.expected_leaf_keys = ["a", "z"]
    #     self.expected_gnode_count = 3   # 2 leaf nodes + 1 root

    

    # def test_insert_duplicate_key_updates_value(self):
    #     item1 = Item("x", 5, BASE_TIMESTAMP)
    #     item2 = Item("x", 99, BASE_TIMESTAMP)
    #     self.tree.insert(item1, rank=1)
    #     self.tree.insert(item2, rank=1)
    #     #stats = gtree_stats_(self.tree, {})
    #     result = self.tree.node.set.retrieve("x")
    #     self.assertEqual(result.found_entry.item.value, 99, f"Duplicate key should update value.\n{str(self.tree.print_structure())}")


    # def test_leaf_and_internal_insertions(self):
    #     self.assertTrue(self.tree.insert(Item("m", 50), rank=2))
    #     self.assertTrue(self.tree.insert(Item("n", 53), rank=2))
    #     self.assertTrue(self.tree.insert(Item("a", 1), rank=1))
    #     self.assertTrue(self.tree.insert(Item("z", 100), rank=1))
    #     # Only leaf nodes should contain full values
    #     # Internal nodes only contain keys with None values
    #     for entry in self.tree.node.set:
    #         self.assertIsNone(entry.item.value)

    # def test_insert_structure_consistency(self):
    #     keys = ["b", "a", "c", "d", "e", "f", "g", "h"]
    #     ranks = [1, 3, 2, 4, 2, 1, 2, 1]
    #     for i, k in enumerate(keys):
    #         calculated_rank = ranks[i]
    #         self.assertTrue(self.tree.insert(Item(k, ord(k)), rank=calculated_rank))
    #     self.assertEqual(self.tree.node.rank, max(ranks), f"Root rank should be {max(ranks)}.\n{str(self.tree.print_structure())}")

    # def test_leaf_linkage(self):
    #     # Insert a known sequence of items
    #     keys = ["a", "b", "c", "d", "e", "f", "g", "h"]
    #     ranks = [1, 3, 2, 4, 2, 1, 2, 1]
    #     for i, k in enumerate(keys):
    #         self.tree.insert(Item(k, ord(k)), rank=ranks[i])

    #     stats = gtree_stats_(self.tree, {})
    #     # print("\nTree stats:")
    #     # pprint(asdict(stats))
    #     # print("\nTree structure:")
    #     # print(self.tree.print_structure())

    #     # Gather leaf nodes using .next traversal
    #     leaf_nodes_by_next = []
    #     leaf = self.tree
    #     while not leaf.is_empty() and leaf.node.rank > 1:            
    #         result = leaf.node.set.get_min()
    #         if result.next_entry is not None:
    #             leaf = result.next_entry.left_subtree
    #         else:
    #             leaf = leaf.node.right_subtree

    #     while leaf is not None and not leaf.is_empty():
    #         leaf_nodes_by_next.append(leaf.node)
    #         leaf = leaf.node.next

    #     # Gather leaf nodes using logical method (e.g. iter_leaf_nodes)
    #     leaf_nodes_by_iterator = list(self.tree.iter_leaf_nodes())

    #     # Check same number of leaves
    #     self.assertEqual(
    #         len(leaf_nodes_by_next), len(leaf_nodes_by_iterator),
    #         "Leaf count mismatch between .next traversal and iterator"
    #     )

    #     # Check they are the same objects
    #     for i, (n1, n2) in enumerate(zip(leaf_nodes_by_next, leaf_nodes_by_iterator)):
    #         self.assertIs(
    #             n1, n2,
    #             f"Leaf node {i} differs between .next traversal and logical traversal"
    #         )

    # def test_dummy_element_in_layers(self):
    #     """(6) A G⁺-Tree includes a dummy element ⊥ (key="0" *64) as the first item in each layer."""
    #     # Insert a few items with different ranks so that multiple layers form.
    #     for key, rank in [
    #         ("a", 2),
    #         ("b", 1),
    #         ("c", 4),
    #         ("d", 2),
    #         ("e", 1),
    #         ]:
    #         self.tree.insert(Item(key, ord(key)), rank=rank)
    #     # Descend the leftmost tree nodes from the root and check first item.
    #     current = self.tree
    #     while not current.is_empty():
    #         if current.node.set.is_empty():
    #             raise RuntimeError("Expected non-empty set for a non-empty GPlusTree during iteration. \nCurrent tree:\n", current.print_structure())
    #             # current = current.node.right_subtree
    #         else:
    #             #print(f"Current node: {current.print_structure()}")
    #             result = current.node.set.get_min()
    #             self.assertIsNotNone(result.found_entry, "First entry should not be None")
    #             self.assertIsNotNone(result.found_entry.item, "First Item  should not be None")
    #             self.assertEqual(result.found_entry.item.key, DUMMY_KEY,
    #                                 f"First entry is not the dummy element.Current tree:\n{current.print_structure()}")
    #             if result.next_entry is not None:
    #                 current = result.next_entry.left_subtree
    #             else:
    #                 current = current.node.right_subtree

    # def test_leaf_stores_full_representation(self):
    #     """(2) Leaf layer stores the full representation of each item."""
    #     # Insert several items at rank 1; these should end up in the leaf layer.
    #     keys = ["d", "a", "c", "b", "e"]
    #     for key in keys:
    #         self.tree.insert(Item(key, ord(key)), rank=1)
    #     # In the leaves, check that each non-dummy item has a non-None value.
    #     for leaf in self.tree.iter_leaf_nodes():
    #         # print(f"Leaf: {leaf.set.print_structure()}")
    #         for entry in leaf.set:
    #             # Skip dummy items.
    #             if entry.item.key == DUMMY_KEY:
    #                 continue
    #             self.assertIsNotNone(entry.item.value,
    #                                  f"Leaf entry for key {entry.item.key} does not have full representation")
    #             self.assertEqual(entry.item.value, ord(entry.item.key),
    #                              f"Leaf entry for key {entry.item.key} has an incorrect value")

    # def test_internal_nodes_store_replicas(self):
    #     """(1) A G⁺-Tree stores replicas of selected item’s keys in internal nodes.
    #        (7) The item rank defines the maximum layer at which an item is stored.
    #     """
    #     # Insert items with rank > 1 so that internal nodes must store replicas.
    #     for key, rank in [
    #         ("a", 2),
    #         ("b", 1),
    #         ("c", 4),
    #         ("d", 2),
    #         ("e", 1),
    #         ]:
    #         self.tree.insert(Item(key, ord(key)), rank=rank)
    #     # For an internal node (if tree rank > 1) check that non-dummy entries are replicas (i.e. value is None).
    #     if self.tree.node.rank > 1:
    #         for entry in self.tree.node.set:
    #             if entry.item.key == DUMMY_KEY:
    #                 continue
    #             self.assertIsNone(entry.item.value,
    #                               f"Internal node entry for key {entry.item.key} should be a replica (value None)")
    #     # Additionally, in the leaf layer, the item with key "x" (for example) should appear fully.
    #     full_found = False
    #     for leaf in self.tree.iter_leaf_nodes():
    #         # print(f"Leaf: {leaf.set.print_structure()}")
    #         for entry in leaf.set:
    #             if entry.item.key == "c" and entry.item.value is not None:
    #                 full_found = True
    #     self.assertTrue(full_found, "Item 'c' full representation not found in a leaf layer. Self.tree:\n" +
    #                      str(self.tree.print_structure()))

    # def test_total_order_by_key(self):
    #     """(4) Items in a G⁺-Tree follow a total order by their key."""
    #     # Insert keys in random order.
    #     keys = ["g", "e", "f", "d", "i", "h", "b", "a", "c"]
    #     for key in keys:
    #         self.tree.insert(Item(key, ord(key)), rank=1)
    #     # Traverse the leaves and collect non-dummy keys.
    #     collected = []
    #     for leaf in self.tree.iter_leaf_nodes():
    #         for entry in leaf.set:
    #             if entry.item.key == DUMMY_KEY:
    #                 continue
    #             collected.append(entry.item.key)
    #     expected = sorted(collected)
    #     self.assertEqual(collected, expected,
    #                      f"Leaf keys are not in sorted order. Collected: {collected}, Expected: {expected}")

if __name__ == "__main__":
    unittest.main()