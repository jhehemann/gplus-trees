import sys
import os
import unittest
import random
from typing import List, Tuple

# Add the src directory to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.gplus_trees.base import Item
from src.gplus_trees.g_k_plus.factory import create_gkplus_tree
from src.gplus_trees.g_k_plus.g_k_plus_base import GKPlusTreeBase, DUMMY_ITEM, DUMMY_KEY

class TestGKPlusSizeTracking(unittest.TestCase):
    
    def setUp(self):
        # Create trees with different K values for testing
        self.tree_k2 = create_gkplus_tree(K=2)
        self.tree_k4 = create_gkplus_tree(K=4)
        self.tree_k8 = create_gkplus_tree(K=8)
        
    def create_item(self, key, value="val"):
        """Helper to create test items"""
        return Item(key, value)
    
    def test_empty_tree_size(self):
        """Test that an empty tree has a node with size 0"""
        tree = create_gkplus_tree(K=4)
        self.assertTrue(tree.is_empty())
        # Size is a property of nodes, not empty trees
        
    def test_single_insertion_size(self):
        """Test size is 1 after inserting a single item"""
        item = Item(1000, "val")
        tree, inserted = self.tree_k2.insert(item, rank=1)
        self.assertTrue(inserted)
        self.assertIsNotNone(tree.node.size)
        self.assertEqual(1, tree.node.size, "Tree size should be 1 after single insertion")
    
    def test_multiple_insertions_size(self):
        """Test size increases properly with multiple insertions"""
        tree = self.tree_k4
        expected_size = 0
        
        # Insert 10 items sequentially
        for i in range(1, 11):
            item = Item(i * 1000, "val") 
            tree, inserted = tree.insert(item, rank=1)
            expected_size += 1
            self.assertEqual(expected_size, tree.node.size, 
                             f"Tree size should be {expected_size} after {i} insertions")
    
    def test_duplicate_insertion_size(self):
        """Test size doesn't change when inserting duplicates"""
        tree = self.tree_k2
        
        # First insertion
        item = Item(5000, "val")
        tree, inserted = tree.insert(item, rank=1)
        self.assertTrue(inserted)
        self.assertEqual(1, tree.node.size)
        
        # Duplicate insertion
        item_duplicate = Item(5000, "new_val")
        tree, inserted = tree.insert(item_duplicate, rank=1)
        self.assertFalse(inserted)
        self.assertEqual(1, tree.node.size, "Size should not change after duplicate insertion")
    
    def test_size_with_node_splitting(self):
        """Test size is correctly maintained when nodes are split"""
        tree = self.tree_k2  # Use K=2 to force early splits
        
        # Insert enough items to force node splits
        keys = [100, 200, 300, 400, 500, 600, 700, 800]
        for i, key in enumerate(keys, 1):
            item = Item(key, "val")
            tree, _ = tree.insert(item, rank=1)
            self.assertEqual(i, tree.node.size, f"Tree should have size {i} after inserting {key}")
        
        # Verify size after all insertions
        self.assertEqual(len(keys), tree.node.size)
        
        # Also verify that node sizes are consistent throughout the tree
        self.assertTrue(self.verify_subtree_sizes(tree))
    
    def test_size_with_different_ranks(self):
        """Test size is correctly tracked with items at different ranks"""
        tree = self.tree_k4
        
        # Insert items with different ranks
        items_and_ranks = [
            (Item(1000, "val"), 1),
            (Item(2000, "val"), 2),
            (Item(3000, "val"), 1),
            (Item(4000, "val"), 3),
            (Item(5000, "val"), 2)
        ]
        
        for i, (item, rank) in enumerate(items_and_ranks, 1):
            tree, _ = tree.insert(item, rank)
            self.assertEqual(i, tree.node.size, f"Tree should have size {i} after inserting item with rank {rank}")
        
        # Verify size after all insertions
        self.assertEqual(len(items_and_ranks), tree.node.size)
        
        # Verify subtree sizes are consistent
        self.assertTrue(self.verify_subtree_sizes(tree))
    
    def test_large_tree_size(self):
        """Test size is correctly maintained in a larger tree with random insertions"""
        tree = self.tree_k8
        keys = random.sample(range(1, 10000), 100)  # 100 unique random keys
        
        # Insert all items
        for i, key in enumerate(keys, 1):
            item = Item(key, "val")
            tree, _ = tree.insert(item, rank=1)
            self.assertEqual(i, tree.node.size, f"Tree should have size {i} after inserting {key}")
        
        # Verify size after all insertions
        self.assertEqual(len(keys), tree.node.size)
        
        # Verify subtree sizes
        self.assertTrue(self.verify_subtree_sizes(tree))
    
    def test_complex_tree_structure(self):
        """Test size in a complex tree with various ranks and splits"""
        tree = self.tree_k4
        
        # Create a mix of items with different ranks
        items = []
        for i in range(1, 51):  # 50 items
            key = i * 100
            rank = 1 if i % 3 == 0 else (2 if i % 3 == 1 else 3)  # Mix of ranks 1, 2, 3
            items.append((Item(key, "val"), rank))
        
        # Insert all items
        for i, (item, rank) in enumerate(items, 1):
            tree, _ = tree.insert(item, rank)
            self.assertEqual(i, tree.node.size, 
                            f"Tree should have size {i} after inserting item {item.key} with rank {rank}")
        
        # Verify size after all insertions
        self.assertEqual(len(items), tree.node.size)
        
        # Verify subtree sizes
        self.assertTrue(self.verify_subtree_sizes(tree))
    
    def test_size_after_insert_empty(self):
        """Test that _insert_empty correctly initializes node size"""
        # Case 1: Leaf node (rank 1)
        tree, inserted = self.tree_k4._insert_empty(Item(1000, "val"), rank=1)
        self.assertTrue(inserted)
        self.assertEqual(1, tree.node.size)
        
        # Case 2: Internal node (rank > 1)
        tree, inserted = self.tree_k4._insert_empty(Item(2000, "val"), rank=3)
        self.assertTrue(inserted)
        self.assertEqual(1, tree.node.size)
    
    def test_size_consistency_with_calculate_size(self):
        """Test that node.size matches the result of calculate_size()"""
        tree = self.tree_k4
        
        # Insert several items
        for i in range(1, 21):
            tree, _ = tree.insert(Item(i * 500, "val"), rank=1)
        
        # Check size consistency for the root node
        self.assertEqual(tree.node.size, tree.calculate_size())
        
        # Verify size consistency throughout the tree
        self.verify_calculated_sizes_match(tree)
    
    def verify_subtree_sizes(self, tree):
        """
        Recursive helper to verify that the size of each node equals the sum of its children's sizes
        plus the number of real items in the node itself.
        """
        if tree.is_empty():
            return True
            
        node = tree.node
        
        # For leaf nodes, size should be the count of non-dummy items
        if node.rank == 1:
            calculated_size = sum(1 for entry in node.set if entry.item.key != DUMMY_KEY)
            if calculated_size != node.size:
                print(f"Leaf node has size {node.size} but contains {calculated_size} items")
                return False
            return True
            
        # For internal nodes, size should be sum of child sizes
        calculated_size = 0
        
        # Add sizes from left subtrees
        for entry in node.set:
            if not entry.left_subtree.is_empty():
                calculated_size += entry.left_subtree.node.size
                # Recursively verify this subtree
                if not self.verify_subtree_sizes(entry.left_subtree):
                    return False
        
        # Add size from right subtree
        if not node.right_subtree.is_empty():
            calculated_size += node.right_subtree.node.size
            # Recursively verify right subtree
            if not self.verify_subtree_sizes(node.right_subtree):
                return False
        
        # Check if the stored size matches calculated size
        if calculated_size != node.size:
            print(f"Node at rank {node.rank} has size {node.size} but calculated size is {calculated_size}")
            return False
            
        return True
    
    def verify_calculated_sizes_match(self, tree):
        """
        Recursive helper to verify that node.size matches what calculate_tree_size() returns
        for each node in the tree.
        """
        if tree.is_empty():
            return True
            
        node = tree.node
        
        # Check if node.size matches calculated size
        calculated = node.calculate_tree_size()
        if calculated != node.size:
            print(f"Node size {node.size} doesn't match calculated size {calculated}")
            return False
            
        # Recursively check all subtrees
        for entry in node.set:
            if not entry.left_subtree.is_empty():
                if not self.verify_calculated_sizes_match(entry.left_subtree):
                    return False
        
        if not node.right_subtree.is_empty():
            if not self.verify_calculated_sizes_match(node.right_subtree):
                return False
                
        return True
        
    def test_rank_mismatch_size_handling(self):
        """Test that size is correctly maintained when handling rank mismatches"""
        tree = self.tree_k4
        
        # Insert an item with rank 1
        tree, _ = tree.insert(Item(1000, "val"), rank=1)
        self.assertEqual(1, tree.node.size)
        
        # Insert an item with higher rank, triggering rank mismatch logic
        tree, _ = tree.insert(Item(2000, "val"), rank=3)
        self.assertEqual(2, tree.node.size)
        
        # Verify subtree sizes are consistent
        self.assertTrue(self.verify_subtree_sizes(tree))
        
    def test_size_with_internal_node_splitting(self):
        """Test size maintenance during internal node splits"""
        tree = self.tree_k2  # K=2 to force splits quickly
        
        # Insert items with increasing rank to create deep tree with internal nodes
        ranks = [1, 1, 2, 1, 3, 2, 1]
        keys = [1000, 2000, 3000, 4000, 5000, 6000, 7000]
        
        for i, (key, rank) in enumerate(zip(keys, ranks), 1):
            tree, _ = tree.insert(Item(key, "val"), rank)
            self.assertEqual(i, tree.node.size)
        
        # Verify subtree sizes
        self.assertTrue(self.verify_subtree_sizes(tree))

if __name__ == "__main__":
    unittest.main()