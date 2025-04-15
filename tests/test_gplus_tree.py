
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
        found, _ = self.tree.node.set.retrieve("a")
        self.assertEqual(found, item, "Item should be found in the tree.\n" + str(self.tree.print_structure()))
        dummy, _ = self.tree.node.set.retrieve("0" * 64)
        self.assertTrue(dummy is not None, "Dummy item should be present in the set.\n" + str(self.tree.print_structure()))

    def test_insert_into_empty_tree_rank_gt_1(self):
        item = Item("a", 10, BASE_TIMESTAMP)
        success = self.tree.insert(item, rank=3)
        stats = gtree_stats_(self.tree, {})
        self.assertTrue(success, f"Insert should be successful.\n{str(self.tree.print_structure())}")
        self.assertIsNotNone(self.tree.node, f"Node should not be None after insertion.\n{str(self.tree.print_structure())}")
        self.assertEqual(self.tree.node.rank, 3, f"Root rank should be 3, but got {self.tree.node.rank}\n{str(self.tree.print_structure())}")
        self.assertEqual(stats.item_count, 4, f"Item count in the tree should be 4 (1 dummy + 1 item + 1 replica for each as layer two should be collapsed)\n{str(self.tree.print_structure())}")
        self.assertEqual(self.tree.node.set.item_count(), 2, f"Item count in root should be 2 (1 dummy + 1 item)\n{str(self.tree.print_structure())}")
        _, next_entry = self.tree.node.set.retrieve("0" * 64)
        self.assertEqual(next_entry[1].node.next, self.tree.node.right_subtree, f"Leaves should be linked.\n{str(self.tree.print_structure())}")


    def test_insert_multiple_increasing_keys(self):
        for key in ["a", "b", "c", "d"]:
            self.assertTrue(self.tree.insert(Item(key, ord(key), timestamp=BASE_TIMESTAMP), rank=1))
        for key in ["a", "b", "c", "d"]:
            found, _ = self.tree.node.set.retrieve(key)
            self.assertEqual(found.key, key, f"Key {key} not found in tree.\n{str(self.tree.print_structure())}")

    def test_insert_multiple_decreasing_keys(self):
        for key in reversed(["a", "b", "c", "d"]):
            self.assertTrue(self.tree.insert(Item(key, ord(key), timestamp=BASE_TIMESTAMP), rank=1))
        for key in ["a", "b", "c", "d"]:
            found, _ = self.tree.node.set.retrieve(key)
            self.assertEqual(found.key, key, f"Key {key} not found in tree.\n{str(self.tree.print_structure())}")

    def test_insert_duplicate_key_updates_value(self):
        item1 = Item("x", 5, BASE_TIMESTAMP)
        item2 = Item("x", 99, BASE_TIMESTAMP)
        self.tree.insert(item1, rank=1)
        self.tree.insert(item2, rank=1)
        #stats = gtree_stats_(self.tree, {})
        found, _ = self.tree.node.set.retrieve("x")
        self.assertEqual(found.value, 99, f"Duplicate key should update value.\n{str(self.tree.print_structure())}")

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
            item = entry[0]
            self.assertIsNone(item.value)

    def test_insert_structure_consistency(self):
        keys = ["b", "a", "c", "d", "e", "f", "g", "h"]
        ranks = [1, 3, 2, 4, 2, 1, 2, 1]
        for i, k in enumerate(keys):
            # ranks = []
            # calculated_rank = geometric(0.5)
            # print(f"\nCalculated rank for {k}: {ranks[i]}")
            calculated_rank = ranks[i]
            ranks.append(calculated_rank)
            self.assertTrue(self.tree.insert(Item(k, ord(k)), rank=calculated_rank))
        # stats = gtree_stats_(self.tree, {})
        # print("Tree stats:")
        # pprint(asdict(stats))
        self.assertEqual(self.tree.node.rank, max(ranks), f"Root rank should be {max(ranks)}.\n{str(self.tree.print_structure())}")
        # # Assert leaf nodes are linked
        # prev_node = None
        # for entry in self.tree.node.set:
        #     if entry[1] is None:
        #         continue
        #     if prev_node is None:
        #         prev_node = entry[1].node
        #         continue
        #     cur_tree = entry[1]
        #     self.assertEqual(cur_tree, prev_node.next, f"Leaves should be linked correctly.\n{str(self.tree.print_structure())}")
        #     prev_node = cur_tree.node

        # self.assertEqual(prev_node.next, self.tree.node.right_subtree, f"Last leaf should point to right subtree.\n{str(self.tree.print_structure())}")

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
            first_entry = next(iter(leaf.node.set), None)
            leaf = first_entry[1] if first_entry and first_entry[1] else leaf.node.right_subtree

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

if __name__ == "__main__":
    unittest.main()