import unittest
import collections
from typing import Dict, List, Any, Optional
from unittest.mock import patch

from pprint import pprint
from dataclasses import asdict

# Import the function we're testing using relative import

from gplus_trees.gplus_tree import (
    GPlusTree,
    GPlusNode,
    DUMMY_KEY,
    DUMMY_ITEM,
    gtree_stats_,
    collect_leaf_keys,
)
from gplus_trees.base import (
    Item,
    Entry,
    _create_replica
)
from gplus_trees.klist import KList, KListNode

class TestGTreeStatsInvalidProperties(unittest.TestCase):
    def setUp(self):
        """Set up test trees of various sizes and structures"""
        # Test items
        self.item1 = Item(1, "val_1")
        self.item2 = Item(2, "val_2")
        self.item3 = Item(3, "val_3")
        self.item4 = Item(4, "val_4")

        self.replica1 = Item(1, None)
        self.replica2 = Item(2, None)
        self.replica3 = Item(3, None)
        self.replica4 = Item(4, None)
        
        # Single-node tree
        single_set1 = (
            KList()
            .insert(DUMMY_ITEM, GPlusTree())
            .insert(self.item1, GPlusTree())
        )
        self.single_node_tree = GPlusTree(
            GPlusNode(1, single_set1, GPlusTree())
        )

        # 3-node tree (1 root, 2 leaves)
        leaf_set2 = (
            KList()
            .insert(self.item3, GPlusTree())
            .insert(self.item4, GPlusTree())
        )
        self.leaf_tree2 = GPlusTree(GPlusNode(1, leaf_set2, GPlusTree()))

        leaf_set1 = (
            KList()
            .insert(DUMMY_ITEM, GPlusTree())
            .insert(self.item1, GPlusTree())
            .insert(self.item2, GPlusTree())
        )
        self.leaf_tree1 = GPlusTree(GPlusNode(1, leaf_set1, GPlusTree()))
        self.leaf_tree1.node.next = self.leaf_tree2

        root_set = (
            KList()
            .insert(DUMMY_ITEM, GPlusTree())
            .insert(self.replica3, self.leaf_tree1)
        )
        self.tree_3_nodes = GPlusTree(GPlusNode(3, root_set, self.leaf_tree2))


    def test_single_node_tree_item_order(self):
        """Test stats computed for an empty tree"""
        # Swap first item (DUMMY_ITEM) with item 3
        self.single_node_tree.node.set.head.entries[0].item = self.item3
        
        stats = gtree_stats_(self.single_node_tree)
        
        self.assertEqual(stats.least_item.key, 3,
                            "Expected least item key to be 3")
        self.assertFalse(stats.is_search_tree, "Expected NO search tree")
        self.assertFalse(stats.leaf_keys_in_order,
                            "Expected unordered leaf keys")
        self.assertTrue(stats.linked_leaf_nodes)
        self.assertEqual(stats.true_item_count, 2)
        
    def test_single_node_tree_linked_leafs(self):
        """Test stats computed for an empty tree"""
        # Add an unexpected leaf node by manipulating next pointer
        self.single_node_tree.node.next = self.leaf_tree2
        
        stats = gtree_stats_(self.single_node_tree)
        
        self.assertEqual(stats.least_item.key, DUMMY_KEY,
                            "Expected least item key to be DUMMY_KEY")
        self.assertEqual(stats.greatest_item.key, 1,
                            "Expected greatest item key to be 1")
        
        self.assertFalse(stats.linked_leaf_nodes)
        self.assertEqual(stats.true_item_count, 3)


class TestGTreeStatsInvalidLargeRandomTree(unittest.TestCase):
    def setUp(self):
        """Test stats computed for a large random tree"""
        # Create a large random tree
        self.tree = GPlusTree()
        keys = [1, 2, 3, 4, 5]
        ranks = [2, 3, 2, 1, 4]
        self.item_map = { k: (Item(k, f"val_{k}")) for k in keys}
        self.rank_map = { key: rank for key, rank in zip(keys, ranks) }
        for k in keys:
            item, rank = self.item_map[k], self.rank_map[k]
            self.tree.insert(item, rank)
    
    def test_internal_eq_child_rank(self):
        # Manipulate ranks to create a non-heap property
        root = self.tree.node
        root_entries = list(root.set)
        internal_rank_3 = root_entries[1].left_subtree
        internal_rank_2_r = internal_rank_3.node.right_subtree
        internal_rank_2_r.node.rank = 1 # Equal to child rank

        stats = gtree_stats_(self.tree)
        self.assertFalse(stats.is_heap, "Expected NO heap property")

    def test_root_lt_child_rank(self):
        # Manipulate ranks to create a non-heap property
        root = self.tree.node
        root.rank = 2

        stats = gtree_stats_(self.tree)
        self.assertFalse(stats.is_heap, "Expected NO heap property")

    def test_internal_packed(self):
        # Manipulate ranks to create a non-heap property
        root = self.tree.node
        root_entries = list(root.set)
        internal_rank_3 = root_entries[1].left_subtree
        internal_rank_2_r = internal_rank_3.node.right_subtree
        internal_rank_2_r.node.set.delete(3)

        stats = gtree_stats_(self.tree)
        self.assertFalse(stats.internal_packed, "Node has less than 2 entries")

    def test_internal_has_replicas(self):
        # Manipulate ranks to create a non-heap property
        root = self.tree.node
        root_entries = list(root.set)
        internal_rank_3 = root_entries[1].left_subtree
        internal_rank_2_r = internal_rank_3.node.right_subtree
        node_set = list(internal_rank_2_r.node.set)
        node_set[1].item.value = "val_3"    # Set value to a non-replica

        stats = gtree_stats_(self.tree)
        self.assertFalse(stats.internal_has_replicas, "Expected item that is not a replica")

        








        

    

        
        
    


# class TestGTreeStatsInvalidProperties(unittest.TestCase):
#     """Test cases for detecting invalid tree properties"""
    
#     def test_invalid_heap_property(self):
#         """Test that is_heap flag is False when heap property is violated"""
#         # Create a custom GPlusNode structure that violates heap property
#         # Child has higher rank/priority than parent
#         parent = GPlusNode(rank=2)
#         child = GPlusNode(rank=5)  # Higher rank than parent
        
#         # Connect nodes (parent should have higher rank in a valid heap)
#         parent.children = [child]
#         child.parent = parent
        
#         # Add some items
#         parent.insert_entry(Entry(Item(1, "value1"), _create_replica(Item(1, "value1"))))
#         child.insert_entry(Entry(Item(2, "value2"), _create_replica(Item(2, "value2"))))
        
#         # Create a tree and manually set its root
#         invalid_tree = GPlusTree()
#         invalid_tree.root = parent
        
#         # Check stats
#         stats = gtree_stats_(invalid_tree)
#         self.assertFalse(stats.is_heap, "Tree with invalid heap property should have is_heap=False")
    
#     def test_invalid_search_tree_property(self):
#         """Test that is_search_tree flag is False when BST property is violated"""
#         # Create a tree with keys in wrong order
#         # In a valid BST, left keys < parent key < right keys
#         parent = GPlusNode(rank=3)
#         left_child = GPlusNode(rank=1)
#         right_child = GPlusNode(rank=1)
        
#         # Connect nodes
#         parent.children = [left_child, right_child]
#         left_child.parent = parent
#         right_child.parent = parent
        
#         # Add items with INVALID ordering (left key > parent key)
#         parent.insert_entry(Entry(Item(5, "parent"), _create_replica(Item(5, "parent"))))
#         left_child.insert_entry(Entry(Item(10, "left"), _create_replica(Item(10, "left"))))  # Should be < 5
#         right_child.insert_entry(Entry(Item(15, "right"), _create_replica(Item(15, "right"))))
        
#         # Create tree and set root
#         invalid_tree = GPlusTree()
#         invalid_tree.root = parent
        
#         # Check stats
#         stats = gtree_stats_(invalid_tree)
#         self.assertFalse(stats.is_search_tree, "Tree violating BST property should have is_search_tree=False")
    
#     def test_missing_internal_replicas(self):
#         """Test that internal_has_replicas flag is False when replicas are missing"""
#         # Create a tree with an internal node missing replicas
#         root = GPlusNode(rank=3)
#         child = GPlusNode(rank=1)
        
#         # Connect nodes
#         root.children = [child]
#         child.parent = root
        
#         # Add items but WITHOUT creating replica in parent
#         child.insert_entry(Entry(Item(1, "leaf"), Item(1, "leaf")))
#         # Internal node should have replica, but we'll add a different item
#         root.insert_entry(Entry(Item(5, "internal"), Item(5, "internal")))
        
#         # Create tree and set root
#         invalid_tree = GPlusTree()
#         invalid_tree.root = root
        
#         # Check stats
#         stats = gtree_stats_(invalid_tree)
#         self.assertFalse(stats.internal_has_replicas, 
#                          "Tree without proper internal replicas should have internal_has_replicas=False")
    
#     def test_unlinked_leaf_nodes(self):
#         """Test that linked_leaf_nodes flag is False when leaf links are broken"""
#         # Create tree with unlinked leaf nodes
#         root = GPlusNode(rank=3)
#         leaf1 = GPlusNode(rank=1)
#         leaf2 = GPlusNode(rank=1)
        
#         # Connect parent-child relationships
#         root.children = [leaf1, leaf2]
#         leaf1.parent = root
#         leaf2.parent = root
        
#         # Add items
#         leaf1.insert_entry(Entry(Item(1, "leaf1"), Item(1, "leaf1")))
#         leaf2.insert_entry(Entry(Item(2, "leaf2"), Item(2, "leaf2")))
#         root.insert_entry(Entry(Item(1, "replica1"), _create_replica(Item(1, "replica1"))))
#         root.insert_entry(Entry(Item(2, "replica2"), _create_replica(Item(2, "replica2"))))
        
#         # Leaf nodes should be linked, but we won't link them
#         # In a valid tree, leaf1.next_leaf = leaf2 and leaf2.prev_leaf = leaf1
        
#         # Create tree and set root
#         invalid_tree = GPlusTree()
#         invalid_tree.root = root
        
#         # Check stats
#         stats = gtree_stats_(invalid_tree)
#         self.assertFalse(stats.linked_leaf_nodes, 
#                          "Tree with unlinked leaf nodes should have linked_leaf_nodes=False")
    
#     def test_missing_leaf_values(self):
#         """Test that all_leaf_values_present flag is False when values are missing"""
#         # Create a tree with a leaf missing values
#         leaf = GPlusNode(rank=1)
        
#         # Add an entry with a None value
#         item_with_none_value = Item(1, None)
#         leaf.insert_entry(Entry(item_with_none_value, item_with_none_value))
        
#         # Create tree and set root
#         invalid_tree = GPlusTree()
#         invalid_tree.root = leaf
        
#         # Check stats
#         stats = gtree_stats_(invalid_tree)
#         self.assertFalse(stats.all_leaf_values_present,
#                          "Tree with missing leaf values should have all_leaf_values_present=False")
    
    # def test_unordered_leaf_keys(self):
    #     """Test that leaf_keys_in_order flag is False when leaf keys aren't ordered"""
    #     # Create a custom function to verify key ordering
    #     # By overriding check_leaf_keys_and_values to always return False for keys_in_order
    #     original_check_function = check_leaf_keys_and_values
        
    #     try:
    #         # Replace with mock function that reports unordered keys
    #         def mock_check_function(keys_list, values_dict):
    #             result = original_check_function(keys_list, values_dict)
    #             # Force the keys_in_order flag to be False
    #             return (False, result[1])
            
    #         # Monkey patch the function
    #         import sys
    #         sys.modules['stats_gplus_tree'].check_leaf_keys_and_values = mock_check_function
            
    #         # Create a simple tree and get stats
    #         tree = GPlusTree()
    #         tree.insert(Item(1, "value1"), 1)
    #         tree.insert(Item(2, "value2"), 1)
            
    #         # Check stats
    #         stats = gtree_stats_(tree)
    #         self.assertFalse(stats.leaf_keys_in_order,
    #                         "Tree with unordered leaf keys should have leaf_keys_in_order=False")
            
    #     finally:
    #         # Restore original function
    #         sys.modules['stats_gplus_tree'].check_leaf_keys_and_values = original_check_function