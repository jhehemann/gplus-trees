
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

    def test_dummy_element_in_layers(self):
        """(6) A G⁺-Tree includes a dummy element ⊥ as the first item in each layer."""
        # Insert a few items with different ranks so that multiple layers form.
        for key, rank in [("a", 2), ("b", 2), ("c", 2)]:
            self.tree.insert(Item(key, ord(key)), rank=rank)
        # Check root's k-list: first entry must be the dummy.
        root_kl = self.tree.node.set
        first_entry = next(iter(root_kl), None)
        self.assertIsNotNone(first_entry, "Root k-list is empty")
        self.assertEqual(first_entry[0].key, DUMMY_KEY,
                         "Root k-list first entry is not the dummy element")
        # Check each leaf’s k-list.
        for leaf in self.tree.iter_leaf_nodes():
            first_entry = next(iter(leaf.set), None)
            self.assertIsNotNone(first_entry, "Leaf k-list is empty")
            self.assertEqual(first_entry[0].key, DUMMY_KEY,
                             "Leaf k-list first entry is not the dummy element")

    def test_leaf_stores_full_representation(self):
        """(2) Leaf layer stores the full representation of each item."""
        # Insert several items at rank 1; these should end up in the leaf layer.
        keys = ["d", "a", "c", "b", "e"]
        for key in keys:
            self.tree.insert(Item(key, ord(key)), rank=1)
        # In the leaves, check that each non-dummy item has a non-None value.
        for leaf in self.tree.iter_leaf_nodes():
            for entry in leaf.set:
                # Skip dummy items.
                if entry[0].key == DUMMY_KEY:
                    continue
                self.assertIsNotNone(entry[0].value,
                                     f"Leaf entry for key {entry[0].key} does not have full representation")
                self.assertEqual(entry[0].value, ord(entry[0].key),
                                 f"Leaf entry for key {entry[0].key} has an incorrect value")

    def test_internal_nodes_store_replicas(self):
        """(1) A G⁺-Tree stores replicas of selected item’s keys in internal nodes.
           (7) The item rank defines the maximum layer at which an item is stored.
        """
        # Insert items with rank > 1 so that internal nodes must store replicas.
        items = [("x", 2), ("y", 2), ("z", 2)]
        for key, rank in items:
            self.tree.insert(Item(key, ord(key)), rank=rank)
        # For an internal node (if tree rank > 1) check that non-dummy entries are replicas (i.e. value is None).
        if self.tree.node.rank > 1:
            for entry in self.tree.node.set:
                if entry[0].key == DUMMY_KEY:
                    continue
                self.assertIsNone(entry[0].value,
                                  f"Internal node entry for key {entry[0].key} should be a replica (value None)")
        # Additionally, in the leaf layer, the item with key "x" (for example) should appear fully.
        full_found = False
        for leaf in self.tree.iter_leaf_nodes():
            for entry in leaf.set:
                if entry[0].key == "x" and entry[0].value is not None:
                    full_found = True
        self.assertTrue(full_found, "Item 'x' full representation not found in a leaf layer")

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
                if entry[0].key == DUMMY_KEY:
                    continue
                collected.append(entry[0].key)
        expected = sorted(collected)
        self.assertEqual(collected, expected,
                         f"Leaf keys are not in sorted order. Collected: {collected}, Expected: {expected}")

    def test_lower_layers_sorted_right_of_replicas(self):
        """(3) Items in lower layers are sorted to the right of their replicas in higher layers."""
        # Insert items so that internal replicas are created.
        for key, rank in [("a", 2), ("b", 2), ("c", 2), ("d", 2), ("e", 2)]:
            self.tree.insert(Item(key, ord(key)), rank=2)
        # For each entry in the root's k-list (except the dummy), if it has an associated left subtree,
        # verify that the minimal key in that subtree is greater than or equal to the parent's key.
        for entry in self.tree.node.set:
            if entry[0].key == DUMMY_KEY:
                continue
            subtree = entry[1]
            if subtree is None or subtree.is_empty():
                continue
            min_result = subtree.node.set.get_min()
            # Guard against empty subtree.
            self.assertIsNotNone(min_result, f"Subtree for key {entry[0].key} is empty")
            min_item, _ = min_result
            self.assertLessEqual(entry[0].key, min_item.key,
                                 f"Replica key {entry[0].key} not less than or equal to minimum key {min_item.key} in its subtree")

    def test_node_boundaries(self):
        """(5) Node boundaries are defined by the items in the layer above."""
        # Insert a sufficient number of items to produce multiple node splits.
        for key in "abcdefghi":
            self.tree.insert(Item(key, ord(key)), rank=2)
        # For each entry in the parent's k-list, check that the left subtree's maximal key is less than the parent's boundary.
        if self.tree.node.rank > 1:
            for entry in self.tree.node.set:
                subtree = entry[1]
                if subtree is None or subtree.is_empty():
                    continue
                # Collect all keys from the subtree.
                subtree_keys = []
                for leaf in subtree.iter_leaf_nodes():
                    for sub_entry in leaf.set:
                        if sub_entry[0].key == DUMMY_KEY:
                            continue
                        subtree_keys.append(sub_entry[0].key)
                if subtree_keys:
                    max_key = max(subtree_keys)
                    # The parent's key should be greater than max_key from the subtree.
                    self.assertLess(max_key, entry[0].key,
                                    f"Boundary violation: max key {max_key} in subtree is not less than parent's key {entry[0].key}")

    def test_item_rank_limits_layer(self):
        """(7) The item rank defines the maximum layer an item or replica is stored in.
           An item inserted with rank r should appear as a full item in a leaf (rank 1)
           and as a replica (value is None) in nodes of higher rank (up to r).
        """
        # Insert an item with a specific rank.
        inserted = Item("z", ord("z"))
        rank = 3
        self.tree.insert(inserted, rank=rank)
        # Traverse the tree, count appearances in leaf vs. higher layers.
        full_count = 0
        replica_count = 0
        # Traverse all nodes by level. (For simplicity, assume iter_leaf_nodes() gives only leaves.)
        for leaf in self.tree.iter_leaf_nodes():
            for entry in leaf.set:
                if entry[0].key == "z":
                    # In a leaf, we expect a full item.
                    if leaf.node.rank == 1 and entry[0].value is not None:
                        full_count += 1
                    # In an internal node, the replica should have a None value.
                    elif leaf.node.rank > 1 and entry[0].value is None:
                        replica_count += 1
        # We expect at least one full occurrence and one replica (if the tree has layers above).
        self.assertGreaterEqual(full_count, 1, "Full representation of inserted item not found in leaf layer")
        if self.tree.node.rank > 1:
            self.assertGreaterEqual(replica_count, 1, "Replica of inserted item not found in an internal node")

if __name__ == "__main__":
    unittest.main()