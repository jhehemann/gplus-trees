
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

from pprint import pprint
from dataclasses import asdict

from packages.jhehemann.customs.gtree.gplus_tree import (
    GPlusTree,
    GPlusNode,
    gtree_stats_,
)
from packages.jhehemann.customs.gtree.base import Item

BASE_TIMESTAMP = datetime.datetime(2021, 1, 1, tzinfo=datetime.timezone.utc)

class TestGPlusTreeInsert(unittest.TestCase):
    def setUp(self):
        self.tree = GPlusTree()

    def test_insert_into_empty_tree_rank_1(self):
        item = Item("a", 1, BASE_TIMESTAMP)
        success = self.tree.insert(item, rank=1)
        stats = gtree_stats_(self.tree, {})
        self.assertTrue(success)
        self.assertIsNotNone(self.tree.node)
        self.assertEqual(self.tree.node.rank, 1)
        self.assertEqual(stats.item_count, 2)
        found, _ = self.tree.node.set.retrieve("a")
        self.assertEqual(found, item)
        dummy, _ = self.tree.node.set.retrieve("0" * 64)
        self.assertTrue(dummy is not None)

    def test_insert_into_empty_tree_rank_gt_1(self):
        item = Item("a", 10, BASE_TIMESTAMP)
        success = self.tree.insert(item, rank=3)
        stats = gtree_stats_(self.tree, {})
        self.assertTrue(success)
        self.assertIsNotNone(self.tree.node)
        self.assertEqual(self.tree.node.rank, 3)
        self.assertEqual(stats.item_count, 4)
        self.assertEqual(self.tree.node.set.item_count(), 2)

    # def test_insert_multiple_increasing_keys(self):
    #     for key in ["a", "b", "c", "d"]:
    #         self.assertTrue(self.tree.insert(Item(key, ord(key)), rank=1))
    #     for key in ["a", "b", "c", "d"]:
    #         found, _ = self.tree.node.set.retrieve(key)
    #         self.assertEqual(found.key, key)

    # def test_insert_multiple_decreasing_keys(self):
    #     for key in reversed(["a", "b", "c", "d"]):
    #         self.assertTrue(self.tree.insert(Item(key, ord(key)), rank=1))
    #     for key in ["a", "b", "c", "d"]:
    #         found, _ = self.tree.node.set.retrieve(key)
    #         self.assertEqual(found.key, key)

    # def test_insert_duplicate_key_updates_value(self):
    #     item1 = Item("x", 5)
    #     item2 = Item("x", 99)
    #     self.tree.insert(item1, rank=1)
    #     self.tree.insert(item2, rank=1)
    #     found, _ = self.tree.node.set.retrieve("x")
    #     self.assertEqual(found.value, 99)

    # def test_insert_higher_rank_creates_root(self):
    #     item_low = Item("a", 1)
    #     item_high = Item("z", 100)
    #     self.tree.insert(item_low, rank=1)
    #     self.tree.insert(item_high, rank=3)
    #     self.assertEqual(self.tree.node.rank, 3)

    # def test_leaf_and_internal_insertions(self):
    #     self.assertTrue(self.tree.insert(Item("m", 50), rank=2))
    #     self.assertTrue(self.tree.insert(Item("a", 1), rank=1))
    #     self.assertTrue(self.tree.insert(Item("z", 100), rank=1))
    #     # Only leaf nodes should contain full values
    #     # Internal nodes only contain keys with None values
    #     for entry in self.tree.node.set.entries:
    #         item = entry[0]
    #         self.assertIsNone(item.value)

    # def test_insert_structure_consistency(self):
    #     keys = ["c", "a", "e", "b", "d"]
    #     for k in keys:
    #         self.assertTrue(self.tree.insert(Item(k, ord(k)), rank=2))
    #     self.assertEqual(self.tree.node.rank, 2)

    # def test_invalid_search_tree(self):
    #     # Right subtree starts with 'a', but last key in set is 'b'
    #     bad_tree = GPlusTree(GPlusNode(
    #         rank=1,
    #         set=[('b', GPlusTree())],
    #         right_subtree=GPlusTree(GPlusNode(
    #             rank=1,
    #             set=[('a', GPlusTree())],
    #             right_subtree=GPlusTree(),
    #             next=GPlusTree()
    #         )),
    #         next=GPlusTree()
    #     ))
    #     stats = gtree_stats_(bad_tree, {})
    #     print("Bad tree stats:")
    #     pprint(asdict(stats))
    #     self.assertFalse(stats.is_search_tree)
        


if __name__ == "__main__":
    unittest.main()