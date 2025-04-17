
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

BASE_TIMESTAMP = datetime.datetime(2021, 1, 1, tzinfo=datetime.timezone.utc)
DUMMY_KEY = "0" * 64

def geometric(p):
    u = random.random()
    return math.ceil(math.log(1 - u) / math.log(1 - p))

class TestGPlusTreeInsert(unittest.TestCase):
    def setUp(self):
        self.tree = GPlusTree()

    def test_insert_into_empty_tree_rank_1(self):
        item = Item("a", 1, BASE_TIMESTAMP)
        success = self.tree.insert(item, rank=1)
        stats = gtree_stats_(self.tree, {})
        self.assertTrue(success, "Insert should be successful.\n" + str(self.tree.print_structure()))
        self.assertIsNotNone(self.tree.node, "Node should not be None after insertion.\n" + str(self.tree.print_structure()))
        self.assertEqual(self.tree.node.rank, 1, "Root rank should be 1.\n" + str(self.tree.print_structure()))
        self.assertEqual(stats.item_count, 2, "Item count in the tree should be 2 (1 dummy + 1 item)\n" + str(self.tree.print_structure()))
        result = self.tree.node.set.retrieve("a")
        self.assertEqual(result.found_entry.item, item, "Item should be found in the tree.\n" + str(self.tree.print_structure()))
        result = self.tree.node.set.retrieve("0" * 64)
        self.assertTrue(result.found_entry is not None, "Dummy item should be present in the set.\n" + str(self.tree.print_structure()))

    def test_insert_into_empty_tree_rank_gt_1(self):
        item = Item("a", 10, BASE_TIMESTAMP)
        success = self.tree.insert(item, rank=3)
        stats = gtree_stats_(self.tree, {})
        self.assertTrue(success, f"Insert should be successful.\n{str(self.tree.print_structure())}")
        self.assertIsNotNone(self.tree.node, f"Node should not be None after insertion.\n{str(self.tree.print_structure())}")
        self.assertEqual(self.tree.node.rank, 3, f"Root rank should be 3, but got {self.tree.node.rank}\n{str(self.tree.print_structure())}")
        self.assertEqual(stats.item_count, 4, f"Item count in the tree should be 4 (1 dummy + 1 item + 1 replica for each as layer two should be collapsed)\n{str(self.tree.print_structure())}")
        self.assertEqual(self.tree.node.set.item_count(), 2, f"Item count in root should be 2 (1 dummy + 1 item)\n{str(self.tree.print_structure())}")
        result = self.tree.node.set.retrieve("0" * 64)
        self.assertEqual(result.next_entry.left_subtree.node.next, self.tree.node.right_subtree, f"Leaves should be linked.\n{str(self.tree.print_structure())}")


    def test_insert_multiple_increasing_keys(self):
        for key in ["a", "b", "c", "d"]:
            self.assertTrue(self.tree.insert(Item(key, ord(key), timestamp=BASE_TIMESTAMP), rank=1))
        for key in ["a", "b", "c", "d"]:
            result = self.tree.node.set.retrieve(key)
            self.assertEqual(result.found_entry.item.key, key, f"Key {key} not found in tree.\n{str(self.tree.print_structure())}")

    def test_insert_multiple_decreasing_keys(self):
        for key in reversed(["a", "b", "c", "d"]):
            self.assertTrue(self.tree.insert(Item(key, ord(key), timestamp=BASE_TIMESTAMP), rank=1))
        for key in ["a", "b", "c", "d"]:
            result = self.tree.node.set.retrieve(key)
            self.assertEqual(result.found_entry.item.key, key, f"Key {key} not found in tree.\n{str(self.tree.print_structure())}")

    def test_insert_duplicate_key_updates_value(self):
        item1 = Item("x", 5, BASE_TIMESTAMP)
        item2 = Item("x", 99, BASE_TIMESTAMP)
        self.tree.insert(item1, rank=1)
        self.tree.insert(item2, rank=1)
        #stats = gtree_stats_(self.tree, {})
        result = self.tree.node.set.retrieve("x")
        self.assertEqual(result.found_entry.item.value, 99, f"Duplicate key should update value.\n{str(self.tree.print_structure())}")

    def test_insert_higher_rank_creates_root(self):
        item_low = Item("a", 1)
        item_high = Item("z", 100)
        self.tree.insert(item_low, rank=1)
        self.tree.insert(item_high, rank=3)
        self.assertEqual(self.tree.node.rank, 3, "Root rank should be 3.\n" + str(self.tree.print_structure()))

    def test_leaf_and_internal_insertions(self):
        self.assertTrue(self.tree.insert(Item("m", 50), rank=2))
        self.assertTrue(self.tree.insert(Item("n", 53), rank=2))
        self.assertTrue(self.tree.insert(Item("a", 1), rank=1))
        self.assertTrue(self.tree.insert(Item("z", 100), rank=1))
        # Only leaf nodes should contain full values
        # Internal nodes only contain keys with None values
        for entry in self.tree.node.set:
            self.assertIsNone(entry.item.value)

    def test_insert_structure_consistency(self):
        keys = ["b", "a", "c", "d", "e", "f", "g", "h"]
        ranks = [1, 3, 2, 4, 2, 1, 2, 1]
        for i, k in enumerate(keys):
            calculated_rank = ranks[i]
            self.assertTrue(self.tree.insert(Item(k, ord(k)), rank=calculated_rank))
        self.assertEqual(self.tree.node.rank, max(ranks), f"Root rank should be {max(ranks)}.\n{str(self.tree.print_structure())}")

    def test_leaf_linkage(self):
        # Insert a known sequence of items
        keys = ["a", "b", "c", "d", "e", "f", "g", "h"]
        ranks = [1, 3, 2, 4, 2, 1, 2, 1]
        for i, k in enumerate(keys):
            self.tree.insert(Item(k, ord(k)), rank=ranks[i])

        stats = gtree_stats_(self.tree, {})
        # print("\nTree stats:")
        # pprint(asdict(stats))
        # print("\nTree structure:")
        # print(self.tree.print_structure())

        # Gather leaf nodes using .next traversal
        leaf_nodes_by_next = []
        leaf = self.tree
        while not leaf.is_empty() and leaf.node.rank > 1:            
            result = leaf.node.set.get_min()
            if result.next_entry is not None:
                leaf = result.next_entry.left_subtree
            else:
                leaf = leaf.node.right_subtree

        while leaf is not None and not leaf.is_empty():
            leaf_nodes_by_next.append(leaf.node)
            leaf = leaf.node.next

        # Gather leaf nodes using logical method (e.g. iter_leaf_nodes)
        leaf_nodes_by_iterator = list(self.tree.iter_leaf_nodes())

        # Check same number of leaves
        self.assertEqual(
            len(leaf_nodes_by_next), len(leaf_nodes_by_iterator),
            "Leaf count mismatch between .next traversal and iterator"
        )

        # Check they are the same objects
        for i, (n1, n2) in enumerate(zip(leaf_nodes_by_next, leaf_nodes_by_iterator)):
            self.assertIs(
                n1, n2,
                f"Leaf node {i} differs between .next traversal and logical traversal"
            )

    def test_dummy_element_in_layers(self):
        """(6) A G⁺-Tree includes a dummy element ⊥ (key="0" *64) as the first item in each layer."""
        # Insert a few items with different ranks so that multiple layers form.
        for key, rank in [
            ("a", 2),
            ("b", 1),
            ("c", 4),
            ("d", 2),
            ("e", 1),
            ]:
            self.tree.insert(Item(key, ord(key)), rank=rank)
        # Descend the leftmost tree nodes from the root and check first item.
        current = self.tree
        while not current.is_empty():
            if current.node.set.is_empty():
                raise RuntimeError("Expected non-empty set for a non-empty GPlusTree during iteration. \nCurrent tree:\n", current.print_structure())
                # current = current.node.right_subtree
            else:
                #print(f"Current node: {current.print_structure()}")
                result = current.node.set.get_min()
                self.assertIsNotNone(result.found_entry, "First entry should not be None")
                self.assertIsNotNone(result.found_entry.item, "First Item  should not be None")
                self.assertEqual(result.found_entry.item.key, DUMMY_KEY,
                                    f"First entry is not the dummy element.Current tree:\n{current.print_structure()}")
                if result.next_entry is not None:
                    current = result.next_entry.left_subtree
                else:
                    current = current.node.right_subtree

    def test_leaf_stores_full_representation(self):
        """(2) Leaf layer stores the full representation of each item."""
        # Insert several items at rank 1; these should end up in the leaf layer.
        keys = ["d", "a", "c", "b", "e"]
        for key in keys:
            self.tree.insert(Item(key, ord(key)), rank=1)
        # In the leaves, check that each non-dummy item has a non-None value.
        for leaf in self.tree.iter_leaf_nodes():
            # print(f"Leaf: {leaf.set.print_structure()}")
            for entry in leaf.set:
                # Skip dummy items.
                if entry.item.key == DUMMY_KEY:
                    continue
                self.assertIsNotNone(entry.item.value,
                                     f"Leaf entry for key {entry.item.key} does not have full representation")
                self.assertEqual(entry.item.value, ord(entry.item.key),
                                 f"Leaf entry for key {entry.item.key} has an incorrect value")

    def test_internal_nodes_store_replicas(self):
        """(1) A G⁺-Tree stores replicas of selected item’s keys in internal nodes.
           (7) The item rank defines the maximum layer at which an item is stored.
        """
        # Insert items with rank > 1 so that internal nodes must store replicas.
        for key, rank in [
            ("a", 2),
            ("b", 1),
            ("c", 4),
            ("d", 2),
            ("e", 1),
            ]:
            self.tree.insert(Item(key, ord(key)), rank=rank)
        # For an internal node (if tree rank > 1) check that non-dummy entries are replicas (i.e. value is None).
        if self.tree.node.rank > 1:
            for entry in self.tree.node.set:
                if entry.item.key == DUMMY_KEY:
                    continue
                self.assertIsNone(entry.item.value,
                                  f"Internal node entry for key {entry.item.key} should be a replica (value None)")
        # Additionally, in the leaf layer, the item with key "x" (for example) should appear fully.
        full_found = False
        for leaf in self.tree.iter_leaf_nodes():
            # print(f"Leaf: {leaf.set.print_structure()}")
            for entry in leaf.set:
                if entry.item.key == "c" and entry.item.value is not None:
                    full_found = True
        self.assertTrue(full_found, "Item 'c' full representation not found in a leaf layer. Self.tree:\n" +
                         str(self.tree.print_structure()))

    def test_total_order_by_key(self):
        """(4) Items in a G⁺-Tree follow a total order by their key."""
        # Insert keys in random order.
        keys = ["g", "e", "f", "d", "i", "h", "b", "a", "c"]
        for key in keys:
            self.tree.insert(Item(key, ord(key)), rank=1)
        # Traverse the leaves and collect non-dummy keys.
        collected = []
        for leaf in self.tree.iter_leaf_nodes():
            for entry in leaf.set:
                if entry.item.key == DUMMY_KEY:
                    continue
                collected.append(entry.item.key)
        expected = sorted(collected)
        self.assertEqual(collected, expected,
                         f"Leaf keys are not in sorted order. Collected: {collected}, Expected: {expected}")

if __name__ == "__main__":
    unittest.main()