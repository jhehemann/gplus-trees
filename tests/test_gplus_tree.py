"""Tests for k-lists"""
# pylint: skip-file

import unittest

from typing import Tuple, Optional, List

from gplus_trees.gplus_tree import (
    GPlusTree,
    GPlusNode,
    DUMMY_ITEM,
    gtree_stats_,
    collect_leaf_keys,
)
from gplus_trees.base import (
    Item,
    Entry,
    _create_replica
)
from stats_gplus_tree import (
    check_leaf_keys_and_values,
)

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
                f"Item count {stats.item_count} does not match"
                f"expected {expected_item_count}\n"
                f"Tree structure:\n{self.tree.print_structure()}"
            )

        expected_root_rank = getattr(self, 'expected_root_rank', None)
        if expected_root_rank is not None:
            self.assertEqual(
                self.tree.node.rank, expected_root_rank,
                f"Root rank {self.tree.node.rank} does not match expected"
                f"{expected_root_rank}"
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
                f"GNode count {stats.gnode_count} does not match expected"
                f"{expected_gnode_count}"
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
    
    def _assert_internal_node_properties(
            self, node: GPlusNode, items: List[Item], rank: int
        )-> Tuple[Optional[Entry], Optional[Entry]]:
        """
        Verify that `node` is an internal node with the expected rank
        containing exactly `items`in order, with the first item’s left subtree
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
                f"Entry #{i} key mismatch: "
                f"expected {expected_item.key}, got {entry.item.key}"
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
        key, rank = 1, 1
        item = Item(key, f"val_{key}")
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
        key, rank = 1, 3
        item = Item(key, f"val_{key}")
        self.tree.insert(item, rank)
        root = self.tree.node

        with self.subTest("root"):
            _, replica = self._assert_internal_node_properties(
                root,
                [DUMMY_ITEM, _create_replica(key)],
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
        keys = [2, 4]
        ranks = [1, 1]
        self.item_map = { k: (Item(k, f"val_{k}")) for k in keys}
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
        key, rank = 1, 1
        item = Item(key, f"val_{key}")
        self.tree.insert(item, rank)

        self._assert_leaf_node_properties(
            self.tree.node,
            [DUMMY_ITEM, item, self.item_map[2], self.item_map[4]]
        )
        self.expected_leaf_keys = [1, 2, 4]

    def test_insert_highest_key(self):
        key, rank = 5, 1
        item = Item(key, f"val_{key}")
        self.tree.insert(item, rank)

        self._assert_leaf_node_properties(
            self.tree.node,
            [DUMMY_ITEM, self.item_map[2], self.item_map[4], item]
        )
        self.expected_leaf_keys = [2, 4, 5]

    def test_insert_middle_key(self):
        key, rank = 3, 1
        item = Item(key, f"val_{key}")
        self.tree.insert(item, rank)

        self._assert_leaf_node_properties(
            self.tree.node,
            [DUMMY_ITEM, self.item_map[2], item, self.item_map[4]]
        )
        self.expected_leaf_keys = [2, 3, 4]
        

class TestInsertInNonEmptyTreeGTMaxRankCreatesRoot(TestInsertInTree):
    def setUp(self):
        super().setUp()
        # Create a tree with 1 node and two full items
        keys = [2, 4]
        ranks = [1, 1]
        self.item_map = { k: (Item(k, f"val_{k}")) for k in keys}
        self.rank_map = { key: rank for key, rank in zip(keys, ranks) }
        for k in keys:
            item, rank = self.item_map[k], self.rank_map[k]
            self.tree.insert(item, rank)
        
        self.insert_rank = 3
        self.expected_root_rank = self.insert_rank
        self.expected_gnode_count = 3
        self.expected_item_count = 6    # currently incl. replicas & dummys

    def test_insert_lowest_key(self):
        key, rank = 1, self.insert_rank
        item = Item(key, f"val_{key}")
        self.tree.insert(item, rank)

        with self.subTest("root"):
            _, replica = self._assert_internal_node_properties(
                self.tree.node,
                [DUMMY_ITEM, _create_replica(key)],
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
                [item, self.item_map[2], self.item_map[4]]
            )
            self.assertIsNone(t_right_leaf.node.next,
                              "Right leaf should have no next pointer")

        self.expected_leaf_keys = [1, 2, 4]

    def test_insert_highest_key(self):
        key, rank = 5, self.insert_rank
        item = Item(key, f"val_{key}")
        self.tree.insert(item, rank)

        with self.subTest("root"):
            _, replica = self._assert_internal_node_properties(
                self.tree.node,
                [DUMMY_ITEM, _create_replica(key)],
                rank
            )
        with self.subTest("left leaf"):
            t_left_leaf = replica.left_subtree
            self._assert_leaf_node_properties(
                t_left_leaf.node,
                [DUMMY_ITEM, self.item_map[2], self.item_map[4]]
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

        self.expected_leaf_keys = [2, 4, 5]
    
    def test_insert_middle_key(self):
        key, rank = 3, self.insert_rank
        item = Item(key, f"val_{key}")
        self.tree.insert(item, rank)

        with self.subTest("root"):
            _, replica = self._assert_internal_node_properties(
                self.tree.node,
                [DUMMY_ITEM, _create_replica(key)],
                rank
            )
        with self.subTest("left leaf"):
            t_left_leaf = replica.left_subtree
            self._assert_leaf_node_properties(
                t_left_leaf.node,
                [DUMMY_ITEM, self.item_map[2]]
            )
            self.assertIs(t_left_leaf.node.next, self.tree.node.right_subtree,
                          "Left leaf should point to right leaf")
        with self.subTest("right leaf"):
            t_right_leaf = self.tree.node.right_subtree
            # self.assertIsNotNone(t_right_leaf, "Use empty trees; never None.")
            self._assert_leaf_node_properties(
                t_right_leaf.node,
                [item, self.item_map[4]]
            )
            self.assertIsNone(t_right_leaf.node.next,
                              "Right leaf should have no next pointer")

        self.expected_leaf_keys = [2, 3, 4]
        

class TestInsertInNonEmptyTreeRankGT1(TestInsertInTree):
    def setUp(self):
        super().setUp()
        # Create a tree with two items
        keys = [2, 4, 6]
        ranks = [1, 3, 1]
        self.item_map = { k: (Item(k, f"val_{k}")) for k in keys}
        self.rank_map = { key: rank for key, rank in zip(keys, ranks) }
        for k in keys:
            item, rank = self.item_map[k], self.rank_map[k]
            self.tree.insert(item, rank)
        
        self.insert_rank = 3
        self.expected_root_rank = self.insert_rank
        self.expected_gnode_count = 4
        self.expected_item_count = 8    # currently incl. replicas & dummys

    def test_insert_lowest_key_splits_leaf(self):
        key, rank = 1, self.insert_rank
        item = Item(key, f"val_{key}")
        self.tree.insert(item, rank)
        root = self.tree.node
        root_entries = list(root.set)

        with self.subTest("root"):
            self._assert_internal_node_properties(
                root,
                [DUMMY_ITEM, _create_replica(key), _create_replica(4)],
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
                [item, self.item_map[2]]
            )
            self.assertIs(leaf_2.next, root.right_subtree, 
                          "Leaf 2 should point to leaf 3 tree")
        
        with self.subTest("leaf 3"):
            leaf_3 = root.right_subtree.node
            self._assert_leaf_node_properties(
                leaf_3,
                [self.item_map[4], self.item_map[6]]
            )
            self.assertIsNone(leaf_3.next,
                              "Leaf 3 should have no next pointer")
            
        self.expected_leaf_keys = [1, 2, 4, 6]

    def test_insert_lowest_key_no_leaf_split(self):
        key, rank = 3, self.insert_rank
        item = Item(key, f"val_{key}")
        self.tree.insert(item, rank)
        root = self.tree.node
        root_entries = list(root.set)

        with self.subTest("root"):
            self._assert_internal_node_properties(
                root,
                [DUMMY_ITEM, _create_replica(3), _create_replica(4)],
                rank
            )

        with self.subTest("leaf 1"):
            leaf_1 = root_entries[1].left_subtree.node
            self._assert_leaf_node_properties(
                leaf_1,
                [DUMMY_ITEM, self.item_map[2]]
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
                [self.item_map[4], self.item_map[6]]
            )
            self.assertIsNone(leaf_3.next,
                              "Leaf 3 should have no next pointer")
            
        self.expected_leaf_keys = [2, 3, 4, 6]
            
    def test_insert_highest_key_splits_leaf(self):
        key, rank = 5, self.insert_rank
        item = Item(key, f"val_{key}")
        self.tree.insert(item, rank)
        root = self.tree.node
        root_entries = list(root.set)

        with self.subTest("root"):
            self._assert_internal_node_properties(
                root,
                [DUMMY_ITEM, _create_replica(4), _create_replica(key)],
                rank
            )
        
        with self.subTest("leaf 1"):
            leaf_1 = root_entries[1].left_subtree.node
            self._assert_leaf_node_properties(
                leaf_1,
                [DUMMY_ITEM, self.item_map[2]]
            )
            self.assertIs(leaf_1.next, root_entries[2].left_subtree,
                          "Leaf 1 should point to leaf 2 tree")
            
        with self.subTest("leaf 2"):
            leaf_2 = root_entries[2].left_subtree.node
            self._assert_leaf_node_properties(
                leaf_2,
                [self.item_map[4]]
            )
            self.assertIs(leaf_2.next, root.right_subtree,
                          "Leaf 2 should point to leaf 3 tree")
            
        with self.subTest("leaf 3"):
            leaf_3 = root.right_subtree.node
            self._assert_leaf_node_properties(
                leaf_3,
                [item, self.item_map[6]]
            )
            self.assertIsNone(leaf_3.next,
                              "Leaf 3 should have no next pointer")
            
        self.expected_leaf_keys = [2, 4, 5, 6]

    def test_insert_highest_key_no_leaf_split(self):
        key, rank = 7, self.insert_rank
        item = Item(key, f"val_{key}")
        self.tree.insert(item, rank)
        root = self.tree.node
        root_entries = list(root.set)
        
        with self.subTest("root"):
            self._assert_internal_node_properties(
                root,
                [DUMMY_ITEM, _create_replica(4), _create_replica(key)],
                rank
            )
        
        with self.subTest("leaf 1"):
            leaf_1 = root_entries[1].left_subtree.node
            self._assert_leaf_node_properties(
                leaf_1,
                [DUMMY_ITEM, self.item_map[2]]
            )
            self.assertIs(leaf_1.next, root_entries[2].left_subtree,
                          "Leaf 1 should point to leaf 2 tree")
        
        with self.subTest("leaf 2"):
            leaf_2 = root_entries[2].left_subtree.node
            self._assert_leaf_node_properties(
                leaf_2,
                [self.item_map[4], self.item_map[6]]
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
        
        self.expected_leaf_keys = [2, 4, 6, 7]


class TestInsertInNonEmptyTreeCollapsedLayerCreatesInternal(TestInsertInTree):
    def setUp(self):
        super().setUp()
        # Create a tree with two items
        keys = [2, 4]
        ranks = [1, 3, 1]
        self.item_map = { k: (Item(k, f"val_{k}")) for k in keys}
        self.rank_map = { key: rank for key, rank in zip(keys, ranks) }
        for k in keys:
            item, rank = self.item_map[k], self.rank_map[k]
            self.tree.insert(item, rank)
        
        self.insert_rank = 2
        self.expected_root_rank = max(ranks)
        self.expected_gnode_count = 5
        self.expected_item_count = 8    # currently incl. replicas & dummys

    def test_insert_lowest_key(self):
        key, rank = 1, self.insert_rank
        item = Item(key, f"val_{key}")
        self.tree.insert(item, rank)
        root = self.tree.node

        with self.subTest("root"):
            _, r_replica = self._assert_internal_node_properties(
                root,
                [DUMMY_ITEM, _create_replica(4)],
                3
            )
        
        with self.subTest("created internal"):
            new_internal = r_replica.left_subtree.node
            _, i_replica = self._assert_internal_node_properties(
                new_internal,
                [DUMMY_ITEM, _create_replica(key)],
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
                [item, self.item_map[2]]
            )
            self.assertIs(leaf_2.next, root.right_subtree,
                          "Leaf 2 should point to leaf 3 tree")
        
        with self.subTest("leaf 3"):
            leaf_3 = root.right_subtree.node
            self._assert_leaf_node_properties(
                leaf_3,
                [self.item_map[4]]
            )
            self.assertIsNone(leaf_3.next,
                              "Leaf 3 should have no next pointer")

        self.expected_leaf_keys = [1, 2, 4]

    def test_insert_highest_key(self):
        key, rank = 5, self.insert_rank
        item = Item(key, f"val_{key}")
        self.tree.insert(item, rank)
        root = self.tree.node
        with self.subTest("root"):
            _, r_replica = self._assert_internal_node_properties(
                root,
                [DUMMY_ITEM, _create_replica(4)],
                3
            )
        with self.subTest("created internal"):
            new_internal = root.right_subtree.node
            _, i_replica = self._assert_internal_node_properties(
                new_internal,
                [_create_replica(4), _create_replica(key)],
                self.insert_rank
            )
        
        with self.subTest("leaf 1"):
            leaf_1 = r_replica.left_subtree.node
            self._assert_leaf_node_properties(
                leaf_1,
                [DUMMY_ITEM, self.item_map[2]]
            )
            self.assertIs(leaf_1.next, i_replica.left_subtree,
                          "Leaf 1 should point to leaf 2 tree")
            
        with self.subTest("leaf 2"):
            leaf_2 = i_replica.left_subtree.node
            self._assert_leaf_node_properties(
                leaf_2,
                [self.item_map[4]]
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
        
        self.expected_leaf_keys = [2, 4, 5]


class TestInsertInNonEmptyTreeRankGT2LowestKey(TestInsertInTree):
    def setUp(self):
        super().setUp()
        # Create a tree with two items
        keys = [2, 4, 6, 8]
        ranks = [1, 2, 2, 3]
        self.item_map = { k: (Item(k, f"val_{k}")) for k in keys}
        self.rank_map = { key: rank for key, rank in zip(keys, ranks) }
        for k in keys:
            item, rank = self.item_map[k], self.rank_map[k]
            self.tree.insert(item, rank)
        
        self.insert_rank = 3
        self.expected_root_rank = max(ranks)

    def test_insert_lowest_key_splits_child_lowest_collapses_left_split(self):
        key, rank = 1, self.insert_rank
        item = Item(key, f"val_{key}")
        self.tree.insert(item, rank)
        root = self.tree.node
        root_entries = list(root.set)

        self.expected_gnode_count = 7
        self.expected_item_count = 12    # currently incl. replicas & dummys
        
        with self.subTest("root"):
            self._assert_internal_node_properties(
                root,
                [DUMMY_ITEM, _create_replica(key), _create_replica(8)],
                3
            )
        
        with self.subTest("child right split (internal)"):
            c_right = root_entries[2].left_subtree.node
            self._assert_internal_node_properties(
                c_right,
                [
                    _create_replica(key),
                    _create_replica(4),
                    _create_replica(6)
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
                [item, self.item_map[2]]
            )
            self.assertIs(leaf_2.next, c_right_entries[2].left_subtree,
                          "Leaf 2 should point to leaf 3 tree")
            
        with self.subTest("leaf 3"):
            leaf_3 = c_right_entries[2].left_subtree.node
            self._assert_leaf_node_properties(
                leaf_3,
                [self.item_map[4]]
            )
            self.assertIs(leaf_3.next, c_right.right_subtree,
                          "Leaf 3 should point to leaf 4 tree")
            
        with self.subTest("leaf 4"):
            leaf_4 = c_right.right_subtree.node
            self._assert_leaf_node_properties(
                leaf_4,
                [self.item_map[6]]
            )
            self.assertIs(leaf_4.next, root.right_subtree,
                          "Leaf 4 should point to leaf 5 tree")
        
        with self.subTest("leaf 5"):
            leaf_4 = root.right_subtree.node
            self._assert_leaf_node_properties(
                leaf_4,
                [self.item_map[8]]
            )
            self.assertIsNone(leaf_4.next,
                              "Leaf 4 should have no next pointer")
            
        self.expected_leaf_keys = [1, 2, 4, 6, 8]

    def test_insert_lowest_key_splits_child_middle(self):
        key, rank = 5, self.insert_rank
        item = Item(key, f"val_{key}")
        self.tree.insert(item, rank)
        root = self.tree.node
        root_entries = list(root.set)

        self.expected_gnode_count = 8
        self.expected_item_count = 13    # currently incl. replicas & dummys

        with self.subTest("root"):
            self._assert_internal_node_properties(
                root,
                [DUMMY_ITEM, _create_replica(key), _create_replica(8)],
                3
            )

        with self.subTest("child left split (internal)"):
            c_left = root_entries[1].left_subtree.node
            self._assert_internal_node_properties(
                c_left,
                [DUMMY_ITEM, _create_replica(4)],
                2
            )
            c_left_entries = list(c_left.set)

        with self.subTest("child right split (internal)"):
            c_right = root_entries[2].left_subtree.node
            self._assert_internal_node_properties(
                c_right,
                [_create_replica(key), _create_replica(6)],
                2
            )
            c_right_entries = list(c_right.set)

        with self.subTest("leaf 1"):
            leaf_1 = c_left_entries[1].left_subtree.node
            self._assert_leaf_node_properties(
                leaf_1,
                [DUMMY_ITEM, self.item_map[2]]
            )
            self.assertIs(leaf_1.next, c_left.right_subtree,
                          "Leaf 1 should point to leaf 2 tree")
            
        with self.subTest("leaf 2"):
            leaf_2 = c_left.right_subtree.node
            self._assert_leaf_node_properties(
                leaf_2,
                [self.item_map[4]]
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
                [self.item_map[6]]
            )
            self.assertIs(leaf_4.next, root.right_subtree,
                          "Leaf 4 should point to leaf 5 tree")
        
        with self.subTest("leaf 5"):
            leaf_5 = root.right_subtree.node
            self._assert_leaf_node_properties(
                leaf_5,
                [self.item_map[8]]
            )
            self.assertIsNone(leaf_5.next,
                              "Leaf 5 should have no next pointer")
            
        self.expected_leaf_keys = [2, 4, 5, 6, 8]

    def test_insert_lowest_key_splits_child_highest(self):
        key, rank = 7, self.insert_rank
        item = Item(key, f"val_{key}")
        self.tree.insert(item, rank)
        root = self.tree.node
        root_entries = list(root.set)

        self.expected_gnode_count = 7
        self.expected_item_count = 12    # currently incl. replicas & dummys

        with self.subTest("root"):
            self._assert_internal_node_properties(
                root,
                [DUMMY_ITEM,  _create_replica(key), _create_replica(8)],
                3
            )

        with self.subTest("child left split (internal)"):
            c_left = root_entries[1].left_subtree.node
            self._assert_internal_node_properties(
                c_left,
                [DUMMY_ITEM, _create_replica(4), _create_replica(6)],
                2
            )
            c_left_entries = list(c_left.set)

        with self.subTest("leaf 1"):
            leaf_1 = c_left_entries[1].left_subtree.node
            self._assert_leaf_node_properties(
                leaf_1,
                [DUMMY_ITEM, self.item_map[2]]
            )
            self.assertIs(leaf_1.next, c_left_entries[2].left_subtree,
                          "Leaf 1 should point to leaf 2 tree")
            
        with self.subTest("leaf 2"):
            leaf_2 = c_left_entries[2].left_subtree.node
            self._assert_leaf_node_properties(
                leaf_2,
                [self.item_map[4]]
            )
            self.assertIs(leaf_2.next, c_left.right_subtree,
                          "Leaf 2 should point to leaf 3 tree")
        
        with self.subTest("leaf 3"):
            leaf_3 = c_left.right_subtree.node
            self._assert_leaf_node_properties(
                leaf_3,
                [self.item_map[6]]
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
                [self.item_map[8]]
            )
            self.assertIsNone(leaf_5.next,
                              "Leaf 5 should have no next pointer")
        
        self.expected_leaf_keys = [2, 4, 6, 7, 8]

    
class TestInsertInNonEmptyTreeRankGT2HighestKey(TestInsertInTree):
    def setUp(self):
        super().setUp()
        keys = [1, 3, 5]
        ranks = [3, 2, 2]
        self.item_map = { k: (Item(k, f"val_{k}")) for k in keys}
        self.rank_map = { key: rank for key, rank in zip(keys, ranks) }
        for k in keys:
            item, rank = self.item_map[k], self.rank_map[k]
            self.tree.insert(item, rank)
        
        self.insert_rank = 3
        self.expected_root_rank = max(ranks)

    def test_insert_highest_key_splits_child_low_collapses_left_split(self):
        # print("\nSelf.tree before insert:\n", self.tree.print_structure())
        key, rank = 2, self.insert_rank
        item = Item(key, f"val_{key}")
        self.tree.insert(item, rank)
        # print("\n\nSelf.tree after insert:\n", self.tree.print_structure())
        root = self.tree.node
        root_entries = list(root.set)

        self.expected_gnode_count = 7
        self.expected_item_count = 11    # currently incl. replicas & dummys
        
        with self.subTest("root"):
            self._assert_internal_node_properties(
                root,
                [DUMMY_ITEM, _create_replica(1), _create_replica(key)],
                3
            )
        
        with self.subTest("child right split (internal)"):
            c_right = root.right_subtree.node
            self._assert_internal_node_properties(
                c_right,
                [
                    _create_replica(key),
                    _create_replica(3),
                    _create_replica(5)
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
                [self.item_map[1]]
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
                [self.item_map[3]]
            )
            self.assertIs(leaf_4.next, c_right.right_subtree,
                          "Leaf 4 should point to leaf 5 tree")
        
        with self.subTest("leaf 5"):
            leaf_5 = c_right.right_subtree.node
            self._assert_leaf_node_properties(
                leaf_5,
                [self.item_map[5]]
            )
            self.assertIsNone(leaf_5.next,
                              "Leaf 5 should have no next pointer")
        
        self.expected_leaf_keys = [1, 2, 3, 5]

    def test_insert_highest_key_splits_child_mid(self):
        key, rank = 4, self.insert_rank
        item = Item(key, f"val_{key}")
        self.tree.insert(item, rank)
        root = self.tree.node
        root_entries = list(root.set)

        self.expected_gnode_count = 8
        self.expected_item_count = 12    # currently incl. replicas & dummys

        with self.subTest("root"):
            self._assert_internal_node_properties(
                root,
                [DUMMY_ITEM, _create_replica(1), _create_replica(key)],
                3
            )

        with self.subTest("child left split (internal)"):
            c_left = root_entries[2].left_subtree.node
            self._assert_internal_node_properties(
                c_left,
                [_create_replica(1), _create_replica(3)],
                2
            )
            c_left_entries = list(c_left.set)

        with self.subTest("child right split (internal)"):
            c_right = root.right_subtree.node
            self._assert_internal_node_properties(
                c_right,
                [_create_replica(key), _create_replica(5)],
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
                [self.item_map[1]]
            )
            self.assertIs(leaf_2.next, c_left.right_subtree,
                          "Leaf 2 should point to leaf 3 tree")
        
        with self.subTest("leaf 3"):
            leaf_3 = c_left.right_subtree.node
            self._assert_leaf_node_properties(
                leaf_3,
                [self.item_map[3]]
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
                [self.item_map[5]]
            )
            self.assertIsNone(leaf_5.next,
                              "Leaf 5 should have no next pointer")
            
        self.expected_leaf_keys = [1, 3, 4, 5]

    def test_insert_highest_key_splits_child_high_collapses_right_split(self):
        key, rank = 6, self.insert_rank
        item = Item(key, f"val_{key}")
        self.tree.insert(item, rank)
        root = self.tree.node
        root_entries = list(root.set)
        self.expected_gnode_count = 7
        self.expected_item_count = 11    # currently incl. replicas & dummys
        
        with self.subTest("root"):
            self._assert_internal_node_properties(
                root,
                [DUMMY_ITEM, _create_replica(1), _create_replica(key)],
                3
            )
        with self.subTest("child left split (internal)"):
            c_left = root_entries[2].left_subtree.node
            self._assert_internal_node_properties(
                c_left,
                [
                    _create_replica(1),
                    _create_replica(3),
                    _create_replica(5)
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
                [self.item_map[1]]
            )
            self.assertIs(leaf_2.next, c_left_entries[2].left_subtree,
                          "Leaf 2 should point to leaf 3 tree")
        
        with self.subTest("leaf 3"):
            leaf_3 = c_left_entries[2].left_subtree.node
            self._assert_leaf_node_properties(
                leaf_3,
                [self.item_map[3]]
            )
            self.assertIs(leaf_3.next, c_left.right_subtree,
                          "Leaf 3 should point to leaf 4 tree")
        
        with self.subTest("leaf 4"):
            leaf_4 = c_left.right_subtree.node
            self._assert_leaf_node_properties(
                leaf_4,
                [self.item_map[5]]
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
        
        self.expected_leaf_keys = [1, 3, 5, 6]
        
    
class TestInsertNonemptyTreeHighCollapsingNodesRightLeft(TestInsertInTree):
    def setUp(self):
        super().setUp()
        keys = [1, 2, 4, 5]
        ranks = [4, 3, 1, 2]
        self.item_map = { k: (Item(k, f"val_{k}")) for k in keys}
        self.rank_map = { key: rank for key, rank in zip(keys, ranks) }
        for k in keys:
            item, rank = self.item_map[k], self.rank_map[k]
            self.tree.insert(item, rank)
        
        self.insert_rank = 4
        self.expected_root_rank = max(ranks)

    def test_insert_high_collapse_rank_3_right_rank_2_left_split_leaf(self):
        # print(f"\n\n\nSelf before insert: {self.tree.print_structure()}")
        key, rank = 3, self.insert_rank
        item = Item(key, f"val_{key}")
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
                    _create_replica(1), 
                    _create_replica(key)
                ],
                4
            )
        
        with self.subTest("child rank 3 (left split of right collapsed)"):
            c_left = root_entries[2].left_subtree.node
            self._assert_internal_node_properties(
                c_left,
                [_create_replica(1), _create_replica(2)],
                3
            )
            c_left_entries = list(c_left.set)
        
        with self.subTest("child rank 2 (right split of left collapsed)"):
            c_right = root.right_subtree.node
            self._assert_internal_node_properties(
                c_right,
                [_create_replica(key), _create_replica(5)],
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
                [self.item_map[1]]
            )
            self.assertIs(leaf_2.next, c_left.right_subtree,
                          "Leaf 2 should point to leaf 3 tree")
            
        with self.subTest("leaf 3"):
            leaf_3 = c_left.right_subtree.node
            self._assert_leaf_node_properties(
                leaf_3,
                [self.item_map[2]]
            )
            self.assertIs(leaf_3.next, c_right_entries[1].left_subtree,
                          "Leaf 3 should point to leaf 4 tree")
            
        with self.subTest("leaf 4"):
            leaf_4 = c_right_entries[1].left_subtree.node
            self._assert_leaf_node_properties(
                leaf_4,
                [item, self.item_map[4]]
            )
            self.assertIs(leaf_4.next, c_right.right_subtree,
                          "Leaf 4 should point to leaf 5 tree")
            
        with self.subTest("leaf 5"):
            leaf_5 = c_right.right_subtree.node
            self._assert_leaf_node_properties(
                leaf_5,
                [self.item_map[5]]
            )
            self.assertIsNone(leaf_5.next,
                              "Leaf 5 should have no next pointer")
            
        self.expected_leaf_keys = [1, 2, 3, 4, 5]

    
class TestInsertNonemptyTreeMidCollapsingNodesRightLeft(TestInsertInTree):
    def setUp(self):
        super().setUp()
        keys = [1, 2, 4, 5, 6]
        ranks = [4, 3, 1, 2, 4]
        self.item_map = { k: (Item(k, f"val_{k}")) for k in keys}
        self.rank_map = { key: rank for key, rank in zip(keys, ranks) }
        for k in keys:
            item, rank = self.item_map[k], self.rank_map[k]
            self.tree.insert(item, rank)
        
        self.insert_rank = 4
        self.expected_root_rank = max(ranks)

    def test_insert_mid_collapse_rank_3_right_rank_2_left_split_leaf(self):
        
        # print(f"\n\n\nSelf before insert: {self.tree.print_structure()}")
        key, rank = 3, self.insert_rank
        item = Item(key, f"val_{key}")
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
                    _create_replica(1), 
                    _create_replica(key),
                    _create_replica(6)
                ],
                4
            )
        
        with self.subTest("child rank 3 (left split of right collapsed)"):
            c_left = root_entries[2].left_subtree.node
            self._assert_internal_node_properties(
                c_left,
                [_create_replica(1), _create_replica(2)],
                3
            )
            c_left_entries = list(c_left.set)
        
        with self.subTest("child rank 2 (right split of left collapsed)"):
            c_right = root_entries[3].left_subtree.node
            self._assert_internal_node_properties(
                c_right,
                [_create_replica(key), _create_replica(5)],
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
                [self.item_map[1]]
            )
            self.assertIs(leaf_2.next, c_left.right_subtree,
                          "Leaf 2 should point to leaf 3 tree")
            
        with self.subTest("leaf 3"):
            leaf_3 = c_left.right_subtree.node
            self._assert_leaf_node_properties(
                leaf_3,
                [self.item_map[2]]
            )
            self.assertIs(leaf_3.next, c_right_entries[1].left_subtree,
                          "Leaf 3 should point to leaf 4 tree")
            
        with self.subTest("leaf 4"):
            leaf_4 = c_right_entries[1].left_subtree.node
            self._assert_leaf_node_properties(
                leaf_4,
                [item, self.item_map[4]]
            )
            self.assertIs(leaf_4.next, c_right.right_subtree,
                          "Leaf 4 should point to leaf 5 tree")
            
        with self.subTest("leaf 5"):
            leaf_5 = c_right.right_subtree.node
            self._assert_leaf_node_properties(
                leaf_5,
                [self.item_map[5]]
            )
            self.assertIs(leaf_5.next, root.right_subtree,
                          "Leaf 5 should point to leaf 6 tree")
            
        with self.subTest("leaf 6"):
            leaf_6 = root.right_subtree.node
            self._assert_leaf_node_properties(
                leaf_6,
                [self.item_map[6]]
            )
            self.assertIsNone(leaf_6.next,
                              "Leaf 6 should have no next pointer")
            
        self.expected_leaf_keys = [1, 2, 3, 4, 5, 6]


class TestInsertNonemptyTreeHighCollapsingNodesLeftRight(TestInsertInTree):
    def setUp(self):
        super().setUp()
        keys = [1, 2, 4]
        ranks = [4, 2, 3]
        self.item_map = { k: (Item(k, f"val_{k}")) for k in keys}
        self.rank_map = { key: rank for key, rank in zip(keys, ranks) }
        for k in keys:
            item, rank = self.item_map[k], self.rank_map[k]
            self.tree.insert(item, rank)
        
        self.insert_rank = 4
        self.expected_root_rank = max(ranks)

    def test_insert_high_collapse_rank_3_left_rank_2_right_split_leaf(self):
        # print(f"\n\n\nSelf before insert: {self.tree.print_structure()}")
        key, rank = 3, self.insert_rank
        item = Item(key, f"val_{key}")
        self.tree.insert(item, rank)
        # print(f"\n\n\nSelf after insert: {self.tree.print_structure()}")
        root = self.tree.node
        root_entries = list(root.set)

        self.expected_gnode_count = 8
        self.expected_item_count = 12    # currently incl. replicas & dummys
        
        with self.subTest("root"):
            self._assert_internal_node_properties(
                root,
                [
                    DUMMY_ITEM,
                    _create_replica(1), 
                    _create_replica(key)
                ],
                4
            )
        
        with self.subTest("child rank 3 (right split of left collapsed)"):
            c_right = root.right_subtree.node
            self._assert_internal_node_properties(
                c_right,
                [_create_replica(key), _create_replica(4)],
                3
            )
            c_right_entries = list(c_right.set)
        
        with self.subTest("child rank 2 (left split of right collapsed)"):
            c_left = root_entries[2].left_subtree.node
            self._assert_internal_node_properties(
                c_left,
                [_create_replica(1), _create_replica(2)],
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
                [self.item_map[1]]
            )
            self.assertIs(leaf_2.next, c_left.right_subtree,
                          "Leaf 2 should point to leaf 3 tree")
            
        with self.subTest("leaf 3"):
            leaf_3 = c_left.right_subtree.node
            self._assert_leaf_node_properties(
                leaf_3,
                [self.item_map[2]]
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
                [self.item_map[4]]
            )
            self.assertIsNone(leaf_5.next,
                              "Leaf 5 should have no next pointer")
            
        self.expected_leaf_keys = [1, 2, 3, 4]


class TestInsertNonemptyTreeMidCollapsingNodesLeftRight(TestInsertInTree):
    def setUp(self):
        super().setUp()
        keys = [1, 2, 4, 5]
        ranks = [4, 2, 3, 4]
        self.item_map = { k: (Item(k, f"val_{k}")) for k in keys}
        self.rank_map = { key: rank for key, rank in zip(keys, ranks) }
        for k in keys:
            item, rank = self.item_map[k], self.rank_map[k]
            self.tree.insert(item, rank)
        
        self.insert_rank = 4
        self.expected_root_rank = max(ranks)

    def test_insert_mid_collapse_rank_3_left_rank_2_right_split_leaf(self):
        # print(f"\n\n\nSelf before insert: {self.tree.print_structure()}")
        key, rank = 3, self.insert_rank
        item = Item(key, f"val_{key}")
        self.tree.insert(item, rank)
        # print(f"\n\n\nSelf after insert: {self.tree.print_structure()}")
        root = self.tree.node
        root_entries = list(root.set)

        self.expected_gnode_count = 9
        self.expected_item_count = 14    # currently incl. replicas & dummys
        
        with self.subTest("root"):
            self._assert_internal_node_properties(
                root,
                [
                    DUMMY_ITEM,
                    _create_replica(1), 
                    _create_replica(key),
                    _create_replica(5)
                ],
                4
            )
        
        with self.subTest("child rank 3 (right split of left collapsed)"):
            c_right = root_entries[3].left_subtree.node
            self._assert_internal_node_properties(
                c_right,
                [_create_replica(key), _create_replica(4)],
                3
            )
            c_right_entries = list(c_right.set)
        
        with self.subTest("child rank 2 (left split of right collapsed)"):
            c_left = root_entries[2].left_subtree.node
            self._assert_internal_node_properties(
                c_left,
                [_create_replica(1), _create_replica(2)],
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
                [self.item_map[1]]
            )
            self.assertIs(leaf_2.next, c_left.right_subtree,
                          "Leaf 2 should point to leaf 3 tree")
            
        with self.subTest("leaf 3"):
            leaf_3 = c_left.right_subtree.node
            self._assert_leaf_node_properties(
                leaf_3,
                [self.item_map[2]]
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
                [self.item_map[4]]
            )
            self.assertIs(leaf_5.next, root.right_subtree,
                          "Leaf 5 should point to leaf 6 tree")
            
        with self.subTest("leaf 6"):
            leaf_6 = root.right_subtree.node
            self._assert_leaf_node_properties(
                leaf_6,
                [self.item_map[5]]
            )
            self.assertIsNone(leaf_6.next,
                              "Leaf 6 should have no next pointer")
            
        self.expected_leaf_keys = [1, 2, 3, 4, 5]


class TestInsertNonemptyRank4TreeUnfoldRank2(TestInsertInTree):
    def setUp(self):
        super().setUp()
        keys = [1, 2, 4, 5]
        ranks = [2, 3, 1, 4]
        self.item_map = { k: (Item(k, f"val_{k}")) for k in keys}
        self.rank_map = { key: rank for key, rank in zip(keys, ranks) }
        for k in keys:
            item, rank = self.item_map[k], self.rank_map[k]
            self.tree.insert(item, rank)
        
        self.insert_rank = 2
        self.expected_root_rank = max(ranks)
        self.expected_gnode_count = 9
        self.expected_item_count = 14    # currently incl. replicas & dummys
        self.expected_leaf_keys = [1, 2, 3, 4, 5]

    def test_insert_split_leaf(self):
        # print(f"\n\n\nSelf before insert: {self.tree.print_structure()}")
        key, rank = 3, self.insert_rank
        item = Item(key, f"val_{key}")
        self.tree.insert(item, rank)
        # print(f"\n\n\nSelf after insert: {self.tree.print_structure()}")
        root = self.tree.node
        root_entries = list(root.set)

        with self.subTest("root"):
            self._assert_internal_node_properties(
                root,
                [
                    DUMMY_ITEM,
                    _create_replica(5)
                ],
                4
            )
        
        with self.subTest("rank 3 internal node"):
            internal_3 = root_entries[1].left_subtree.node
            self._assert_internal_node_properties(
                internal_3,
                [DUMMY_ITEM, _create_replica(2)],
                3
            )
            internal_3_entries = list(internal_3.set)
        
        with self.subTest("rank 2 internal left"):
            internal_2_left = internal_3_entries[1].left_subtree.node
            self._assert_internal_node_properties(
                internal_2_left,
                [DUMMY_ITEM, _create_replica(1)],
                2
            )
            internal_2_left_entries = list(internal_2_left.set)

        with self.subTest("rank 2 internal right"):
            internal_2_right = internal_3.right_subtree.node
            self._assert_internal_node_properties(
                internal_2_right,
                [_create_replica(2), _create_replica(key)],
                2
            )
            internal_2_right_entries = list(internal_2_right.set)

        with self.subTest("leaf 1"):
            leaf_1 = internal_2_left_entries[1].left_subtree.node
            self._assert_leaf_node_properties(
                leaf_1,
                [DUMMY_ITEM]
            )
            self.assertIs(leaf_1.next, internal_2_left.right_subtree,
                          "Leaf 1 should point to leaf 2 tree")
            
        with self.subTest("leaf 2"):
            leaf_2 = internal_2_left.right_subtree.node
            self._assert_leaf_node_properties(
                leaf_2,
                [self.item_map[1]]
            )
            self.assertIs(leaf_2.next, internal_2_right_entries[1].left_subtree,
                          "Leaf 2 should point to leaf 3 tree")

        with self.subTest("leaf 3"):
            leaf_3 = internal_2_right_entries[1].left_subtree.node
            self._assert_leaf_node_properties(
                leaf_3,
                [self.item_map[2]]
            )
            self.assertIs(leaf_3.next, internal_2_right.right_subtree,
                          "Leaf 3 should point to leaf 4 tree")

        with self.subTest("leaf 4"):
            leaf_4 = internal_2_right.right_subtree.node
            self._assert_leaf_node_properties(
                leaf_4,
                [item, self.item_map[4]]
            )
            self.assertIs(leaf_4.next, root.right_subtree,
                          "Leaf 4 should point to leaf 5 tree")

        with self.subTest("leaf 5"):
            leaf_5 = root.right_subtree.node
            self._assert_leaf_node_properties(
                leaf_5,
                [self.item_map[5]]
            )
            self.assertIsNone(leaf_5.next,
                              "Leaf 5 should have no next pointer")

        self.expected_leaf_keys = [1, 2, 3, 4, 5]


class TestInsertInNonEmptyTree(TestInsertInTree):
    def test_insert_multiple_increasing_keys_rank_1(self):
        keys = [1, 2, 3, 4]
        ranks = [1, 1, 1, 1]
        for i, k in enumerate(keys):
            calculated_rank = ranks[i]
            self.tree.insert(Item(k, f"val_{k}"), rank=calculated_rank)
        self.expected_root_rank = max(ranks)
        self.expected_gnode_count = 1   # 1 root/leaf
        self.expected_item_count = 5    # currently incl. replicas & dummys
        self.expected_leaf_keys = [1, 2, 3, 4]

    def test_insert_multiple_decreasing_keys_rank_1(self):
        keys = [4, 3, 2, 1]
        ranks = [1, 1, 1, 1]
        for i, k in enumerate(keys):
            calculated_rank = ranks[i]
            self.tree.insert(Item(k, f"val_{k}"), rank=calculated_rank)
        self.expected_root_rank = max(ranks)
        self.expected_gnode_count = 1   # 1 root/leaf
        self.expected_item_count = 5    # currently incl. replicas & dummys
        self.expected_leaf_keys = [1, 2, 3, 4]
        
    def test_insert_multiple_middle_keys_rank_1(self):
        keys = [1, 5, 2, 4, 3]
        ranks = [1, 1, 1, 1, 1]
        for i, k in enumerate(keys):
            calculated_rank = ranks[i]
            self.tree.insert(Item(k, f"val_{k}"), rank=calculated_rank)
        self.expected_root_rank = max(ranks)
        self.expected_gnode_count = 1   # 1 root/leaf
        self.expected_item_count = 6   # currently incl. replicas & dummys
        self.expected_leaf_keys = [1, 2, 3, 4, 5]
        
    def test_insert_higher_rank_creates_root(self):
        keys = [1, 2]
        ranks = [1, 2]
        for i, k in enumerate(keys):
            calculated_rank = ranks[i]
            self.tree.insert(Item(k, f"val_{k}"), rank=calculated_rank)
        self.expected_root_rank = max(ranks)
        self.expected_gnode_count = 3   # 1 root, 2 leaves
        self.expected_item_count = 5    # currently incl. replicas & dummys
        self.expected_leaf_keys = [1, 2]

    def test_insert_increasing_keys_and_ranks(self):
        keys = [1, 2, 3]
        ranks = [1, 2, 3]
        for i, k in enumerate(keys):
            calculated_rank = ranks[i]
            self.tree.insert(Item(k, f"val_{k}"), rank=calculated_rank)
        self.expected_root_rank = max(ranks)
        self.expected_gnode_count = 5   # 1 root, 1 internal, 3 leaves
        self.expected_item_count = 8    # currently incl. replicas & dummys
        self.expected_leaf_keys = [1, 2, 3]

    def test_insert_decreasing_keys_increasing_ranks(self):
        keys = [3, 2, 1]
        ranks = [1, 2, 3]
        for i, k in enumerate(keys):
            calculated_rank = ranks[i]
            self.tree.insert(Item(k, f"val_{k}"), rank=calculated_rank)
        self.expected_root_rank = max(ranks)
        self.expected_gnode_count = 5   # 1 root, 1 internal, 3 leaves
        self.expected_item_count = 8    # currently incl. replicas & dummys
        self.expected_leaf_keys = [1, 2, 3]

    def test_insert_internal_rank_2_left_creates_node(self):
        keys = [1, 3, 2]
        ranks = [1, 3, 2]
        item_map = {k: (Item(k, f"val_{k}"), r) for k, r in zip(keys, ranks)}
        replica_map = {k: Item(k, None) for k in keys}
        for k in keys:
            item, rank = item_map[k]
            self.tree.insert(item, rank=rank)

        self.expected_root_rank = max(ranks)
        self.expected_gnode_count = 5   # 1 root, 1 internal, 3 leaves
        self.expected_item_count = 8    # currently incl. replicas & dummys
        self.expected_leaf_keys = [1, 2, 3]

        # print(self.tree.print_structure())

        # Check root
        exp_items = [DUMMY_ITEM, replica_map[3]]
        self._assert_internal_node_properties(
            self.tree.node, exp_items, rank=item_map[3][1]
        )

        # Check root's right subtree
        leaf_3 = self.tree.node.right_subtree
        items = [item_map[3][0]]
        self._assert_leaf_node_properties(leaf_3.node, items)
        
        # Check root item's left subtree
        _, next_root = self.tree.node.set.get_min()
        internal_l = next_root.left_subtree
        items = [DUMMY_ITEM, replica_map[2]]
        self._assert_internal_node_properties(
            internal_l.node, items, rank=item_map[2][1]
        )
        
        # Check internal node replica's left subtree
        _, next_internal_l = internal_l.node.set.get_min()
        leaf_1 = next_internal_l.left_subtree
        items = [DUMMY_ITEM, item_map[1][0]]
        self._assert_leaf_node_properties(leaf_1.node, items)
        
        # Check internal node's right subtree
        leaf_2 = internal_l.node.right_subtree
        items = [item_map[2][0]]
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
        keys = [3, 1, 2]
        ranks = [1, 3, 2]
        item_map = {k: (Item(k, f"val_{k}"), r) for k, r in zip(keys, ranks)}
        replica_map = {k: Item(k, None) for k in keys}
        for k in keys:
            item, rank = item_map[k]
            self.tree.insert(item, rank=rank)

        self.expected_root_rank = max(ranks)
        self.expected_gnode_count = 5   # 1 root, 1 internal, 3 leaves
        self.expected_item_count = 8    # currently incl. replicas & dummys
        self.expected_leaf_keys = [1, 2, 3]

        # Check root
        exp_items = [DUMMY_ITEM, replica_map[1]]
        self._assert_internal_node_properties(
            self.tree.node, exp_items, rank=item_map[1][1]
        )
        
        # Check root item's left subtree
        _, next_root = self.tree.node.set.get_min()
        leaf_1 = next_root.left_subtree
        items = [DUMMY_ITEM]
        self._assert_leaf_node_properties(leaf_1.node, items)
        
        # Check root's right subtree
        internal_r = self.tree.node.right_subtree
        items = [replica_map[1], replica_map[2]]
        self._assert_internal_node_properties(
            internal_r.node, items, rank=item_map[2][1]
        )
        
        # Check internal node replica's left subtree
        _, next_internal_r = internal_r.node.set.get_min()
        leaf_2 = next_internal_r.left_subtree
        items = [item_map[1][0]]
        self._assert_leaf_node_properties(leaf_2.node, items)
        
        # Check internal node's right subtree
        leaf_3 = internal_r.node.right_subtree
        items = [item_map[2][0], item_map[3][0]]
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
        keys = [3, 2, 1]
        ranks = [3, 2, 1]
        for i, k in enumerate(keys):
            calculated_rank = ranks[i]
            self.tree.insert(Item(k, f"val_{k}"), rank=calculated_rank)
        self.expected_root_rank = max(ranks)
        self.expected_gnode_count = 5   # 1 root, 1 internal, 3 leaves
        self.expected_item_count = 8    # currently incl. replicas & dummys
        self.expected_leaf_keys = [1, 2, 3]
        
    def test_insert_increasing_keys_decreasing_ranks(self):
        keys = [1, 2, 3]
        ranks = [3, 2, 1]
        for i, k in enumerate(keys):
            calculated_rank = ranks[i]
            self.tree.insert(Item(k, f"val_{k}"), rank=calculated_rank)
        self.expected_root_rank = max(ranks)
        self.expected_gnode_count = 5   # 1 root, 1 internal, 3 leaves
        self.expected_item_count = 8    # currently incl. replicas & dummys
        self.expected_leaf_keys = [1, 2, 3]
        
    def test_insert_multiple_increasing_keys_rank_gt_1(self):
        keys = [1, 2, 3, 4]
        ranks = [3, 3, 3, 3]
        for i, k in enumerate(keys):
            calculated_rank = ranks[i]
            self.tree.insert(Item(k, f"val_{k}"), rank=calculated_rank)
        self.expected_root_rank = max(ranks)
        self.expected_gnode_count = 6   # 1 root, 5 leaves 
        self.expected_item_count = 10   # currently incl. replicas & dummys
        self.expected_leaf_keys = [1, 2, 3, 4]

    def test_insert_multiple_decreasing_keys_rank_gt_1(self):
        keys = [4, 3, 2, 1]
        ranks = [3, 3, 3, 3]
        for i, k in enumerate(keys):
            calculated_rank = ranks[i]
            self.tree.insert(Item(k, f"val_{k}"), rank=calculated_rank)
        self.expected_root_rank = max(ranks)
        self.expected_gnode_count = 6   # 1 root, 5 leaves 
        self.expected_item_count = 10   # currently incl. replicas & dummys
        self.expected_leaf_keys = [1, 2, 3, 4]  
    
    def test_insert_middle_key_rank_2_splits_leaf(self):
        keys = [1, 3, 2]
        ranks = [1, 1, 2]
        for i, k in enumerate(keys):
            calculated_rank = ranks[i]
            self.tree.insert(Item(k, f"val_{k}"), rank=calculated_rank)
        self.expected_root_rank = max(ranks)
        self.expected_gnode_count = 3   # 1 root, 2 leaves
        self.expected_item_count = 6    # currently incl. replicas & dummys
        self.expected_leaf_keys = [1, 2, 3]

        result = self.tree.node.set.retrieve(2)
        found_entry = result.found_entry
        self.assertTrue(found_entry is not None, "Item 'b' should be present in root")
        self.assertEqual(found_entry.left_subtree.node.rank, 1, "New node rank should be 2")
        result = found_entry.left_subtree.node.set.retrieve(1)
        found_entry = result.found_entry
        self.assertTrue(found_entry is not None, "Item 'a' should be present in new node")
        
    def test_insert_duplicate_key_updates_value(self):
        keys = [1, 1]
        values = [99, 100]
        ranks = [1, 1]
        for i, k in enumerate(keys):
            calculated_rank = ranks[i]
            self.tree.insert(Item(k, values[i]), ranks[i])
        self.expected_root_rank = max(ranks)
        self.expected_gnode_count = 1
        self.expected_item_count = 2    # currently incl. replicas & dummys
        self.expected_leaf_keys = [1]
        # Check items
        result = self.tree.node.set.retrieve(1)
        found_entry = result.found_entry
        self.assertTrue(found_entry is not None, "'a' should be in root")
        self.assertEqual(found_entry.item.value, 100,
                         "'a' value should be updated to 100")

    def test_insert_packed_tree(self):
        keys = [1, 2, 3]
        ranks = [2, 2, 2]
        for i, k in enumerate(keys):
            calculated_rank = ranks[i]
            self.tree.insert(Item(k, f"val_{k}"), rank=calculated_rank)
        self.expected_root_rank = max(ranks)
        self.expected_gnode_count = 5   # 1 root, 2 leaves
        self.expected_item_count = 8    # currently incl. replicas & dummys

if __name__ == "__main__":
    unittest.main()