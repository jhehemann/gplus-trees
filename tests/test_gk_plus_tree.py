import sys
import os
import unittest
import random
from typing import List, Tuple

# Add the src directory to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from gplus_trees.base import Item
from gplus_trees.g_k_plus.factory import create_gkplus_tree
from gplus_trees.g_k_plus.g_k_plus_base import get_dummy

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
        actual_size = tree.node.get_size()
        self.assertIsNotNone(actual_size)
        self.assertEqual(1, actual_size, "Tree size should be 1 after single insertion")
    
    def test_multiple_insertions_size(self):
        """Test size increases properly with multiple insertions"""
        tree = self.tree_k4
        expected_size = 0
        
        # Insert 10 items sequentially
        for i in range(1, 11):
            item = Item(i * 1000, "val") 
            tree, inserted = tree.insert(item, rank=1)
            expected_size += 1
            actual_size = tree.node.get_size()
            self.assertEqual(expected_size, actual_size, 
                             f"Tree size should be {expected_size} after {i} insertions, got {actual_size}")
    
    def test_duplicate_insertion_size(self):
        """Test size doesn't change when inserting duplicates"""
        tree = self.tree_k2
        
        # First insertion
        item = Item(5000, "val")
        tree, inserted = tree.insert(item, rank=1)
        self.assertTrue(inserted)
        self.assertEqual(1, tree.node.get_size())
        
        # Duplicate insertion
        item_duplicate = Item(5000, "new_val")
        tree, inserted = tree.insert(item_duplicate, rank=1)
        self.assertFalse(inserted)
        self.assertEqual(1, tree.node.get_size(), "Size should not change after duplicate insertion")
    
    def test_size_with_node_splitting(self):
        """Test size is correctly maintained when nodes are split"""
        tree = self.tree_k2  # Use K=2 to force early splits
        
        # Insert enough items to force node splits
        keys = [100, 200, 300, 400, 500, 600, 700, 800]
        for i, key in enumerate(keys, 1):
            item = Item(key, "val")
            tree, _ = tree.insert(item, rank=1)
            self.assertEqual(i, tree.node.get_size(), f"Tree should have size {i} after inserting {key}")
        
        tree, _ = tree.insert(Item(450, "val"), rank=2)
        
        # Verify size after all insertions
        self.assertEqual(len(keys) + 1, tree.node.get_size())
        
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
            self.assertEqual(i, tree.node.get_size(), f"Tree should have size {i} after inserting item with rank {rank}")
        
        # Verify size after all insertions
        self.assertEqual(len(items_and_ranks), tree.node.size)
        
        # Verify subtree sizes are consistent
        self.assertTrue(self.verify_subtree_sizes(tree))
    
    def test_large_tree_size(self):
        """Test size is correctly maintained in a larger tree with random insertions"""
        tree = self.tree_k8
        keys = random.sample(range(1, 10000), 100)  # 100 unique random keys
        ranks = random.choices(range(1, 6), k=100)  # Random ranks between 1 and 5
        
        # Insert all items
        for i, (key, rank) in enumerate(zip(keys, ranks), 1):
            item = Item(key, "val")
            tree, _ = tree.insert(item, rank=rank)
            self.assertEqual(i, tree.node.get_size(), f"Tree should have size {i} after inserting {key}")
        
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
            actual_size = tree.node.get_size()
            self.assertEqual(i, actual_size, 
                            f"Tree should have size {i} after inserting item {item.key} with rank {rank}")
        
        # Verify size after all insertions
        self.assertEqual(len(items), actual_size)
        
        # Verify subtree sizes
        self.assertTrue(self.verify_subtree_sizes(tree))
    
    def test_size_after_insert_empty(self):
        """Test that _insert_empty correctly initializes node size"""
        # Case 1: Leaf node (rank 1)
        tree, inserted = self.tree_k4._insert_empty(Item(1000, "val"), rank=1)
        self.assertTrue(inserted)
        self.assertEqual(1, tree.node.get_size())
        
        # Case 2: Internal node (rank > 1)
        tree, inserted = self.tree_k4._insert_empty(Item(2000, "val"), rank=3)
        self.assertTrue(inserted)
        self.assertEqual(1, tree.node.get_size())
    
    def test_size_consistency_with_calculate_size(self):
        """Test that node.size matches the result of calculate_size()"""
        tree = self.tree_k4
        
        # Insert several items
        for i in range(1, 21):
            tree, _ = tree.insert(Item(i * 500, "val"), rank=1)
        
        node_size = tree.node.size
        self.assertIsNone(node_size, "Expected node size to be invalidated (none) after insert calculation")
        calculated_size = tree.node.calculate_tree_size()
        
        # Check if the calculated size has been set
        node_size = tree.node.size
        self.assertEqual(node_size, calculated_size, f"Expected tree size to be set to {calculated_size} after calculate_tree_size(); got {node_size}")
        
        # Check size consistency for the root node
        self.assertEqual(node_size, calculated_size, f"Expected size {node_size} to match calculated size {calculated_size}")
        
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
        dummy_key = get_dummy(tree.__class__.DIM).key
        if node.rank == 1:
            calculated_size = sum(1 for entry in node.set if entry.item.key != dummy_key)
            if calculated_size != node.size:
                print(f"Leaf node has size {node.size} but contains {calculated_size} items")
                return False
            return True
            
        # For internal nodes, size should be sum of child sizes
        calculated_size = 0
        
        # Add sizes from left subtrees
        for entry in node.set:
            if not entry.left_subtree.is_empty():
                if entry.left_subtree.node.size is None:
                    size = entry.left_subtree.node.get_size()
                else:
                    size = entry.left_subtree.node.size
                calculated_size += size
                # Recursively verify this subtree
                if not self.verify_subtree_sizes(entry.left_subtree):
                    return False
        
        # Add size from right subtree
        if not node.right_subtree.is_empty():
            if node.right_subtree.node.size is None:
                size = node.right_subtree.node.get_size()
            else:
                size = node.right_subtree.node.size
            calculated_size += size
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
        node_size = node.size
        
        # Check if node.size matches calculated size
        calculated = node.calculate_tree_size()
        if calculated != node_size:
            print(f"Node size {node_size} doesn't match calculated size {calculated}")
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
        self.assertEqual(1, tree.node.get_size())
        
        # Insert an item with higher rank, triggering rank mismatch logic
        tree, _ = tree.insert(Item(2000, "val"), rank=3)
        self.assertEqual(2, tree.node.get_size())
        
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
            self.assertEqual(i, tree.node.get_size())
        
        # Verify subtree sizes
        self.assertTrue(self.verify_subtree_sizes(tree))

class TestGKPlusItemSlotCount(unittest.TestCase):
    """Tests for the item_slot_count method in GKPlusTreeBase."""
    
    def setUp(self):
        # Create trees with different K values for testing
        self.tree_k2 = create_gkplus_tree(K=2)
        self.tree_k4 = create_gkplus_tree(K=4)
        self.tree_k8 = create_gkplus_tree(K=8)
        
    def test_empty_tree_slot_count(self):
        """Test that an empty tree has 0 item slots."""
        tree = create_gkplus_tree(K=4)
        self.assertTrue(tree.is_empty())
        self.assertEqual(0, tree.item_slot_count())
        
    def test_single_item_tree_slot_count(self):
        """Test slot count for a tree with a single item."""
        # Insert one item
        tree, _ = self.tree_k4.insert(Item(1000, "val"), rank=1)
        
        # A single leaf node will be created with K+1 slots (K=4)
        expected_slots = self.tree_k4.SetClass.KListNodeClass.CAPACITY
        self.assertEqual(expected_slots, tree.item_slot_count(),
                         f"Expected {expected_slots} slots for a single item tree with K=4")
        
    def test_cap_node_slot_count(self):
        """Test slot count for a tree with a node at capacity."""
        tree = self.tree_k4
        cap = 4
        
        # Insert item to fill the node to capacity (beware of dummy item)
        for i in range(1, cap-1):
            tree, _ = tree.insert(Item(i * 1000, "val"), rank=1)
        
        exp_item_count = cap
        expected_slots =  exp_item_count + (exp_item_count % tree.SetClass.KListNodeClass.CAPACITY)
        self.assertEqual(expected_slots, tree.item_slot_count(),
                         f"Expected {expected_slots} slots for a tree with {exp_item_count} items in a single node")

    def test_lt_cap_node_slot_count(self):
        """Test slot count for a tree with multiple items in a single leaf node."""
        tree = self.tree_k8
        
        # Insert items that fit in a single leaf node (K=8)
        for i in range(1, 6):  # Insert 5 items
            tree, _ = tree.insert(Item(i * 1000, "val"), rank=1)
        
        # A single leaf node with K+1 slots
        expected_slots = self.tree_k8.SetClass.KListNodeClass.CAPACITY
        self.assertEqual(expected_slots, tree.item_slot_count(),
                         f"Expected {expected_slots} slots for a tree with 5 items in a single node")
        
    def test_multi_leaf_node_slot_count(self):
        """Test slot count for a tree with multiple leaf nodes."""
        tree = self.tree_k2  # K=2 to force splits quickly
        self.cap = 2
        
        # Insert enough items to cause underlying KlistNode splits
        indices = [i for i in range(1, 2 * self.cap + 1)]
        for i in indices:
            tree, _ = tree.insert(Item(i * 1000, "val"), rank=1)
        
        exp_item_count = len(indices) + 1 # +1 for the dummy item
        expected_slots =  exp_item_count + (exp_item_count % self.tree_k2.SetClass.KListNodeClass.CAPACITY)
        self.assertEqual(expected_slots, tree.item_slot_count(),
                        f"Expected {expected_slots} slots for a tree with {exp_item_count} items and K = {self.cap}")
        
    def test_internal_node_slot_count(self):
        """Test slot count for a tree with internal nodes."""
        tree = self.tree_k2
        
        # Insert items with rank 1 to create leaf nodes
        for i in range(1, 8):
            tree, _ = tree.insert(Item(i * 1000, "val"), rank=1)
            
        # Insert items with rank 2 to create internal nodes
        for i in range(1, 4):
            tree, _ = tree.insert(Item(i * 500, "val"), rank=2)
        
        # Count slots manually by traversing the tree
        total_expected_slots = self._count_slots_manually(tree)
        
        self.assertEqual(total_expected_slots, tree.item_slot_count(),
                        f"Expected {total_expected_slots} slots for a tree with internal nodes")
    
    def test_complex_tree_structure_slot_count(self):
        """Test slot count in a complex tree with various ranks and splits."""
        tree = self.tree_k4
        
        # Create a mix of items with different ranks
        items = []
        for i in range(1, 31):  # 30 items
            key = i * 100
            rank = 1 if i % 3 == 0 else (2 if i % 3 == 1 else 3)  # Mix of ranks 1, 2, 3
            items.append((Item(key, "val"), rank))
        
        # Insert all items
        for item, rank in items:
            tree, _ = tree.insert(item, rank)
        
        # Count slots manually by traversing the tree
        total_expected_slots = self._count_slots_manually(tree)
        
        self.assertEqual(total_expected_slots, tree.item_slot_count(),
                        f"Expected {total_expected_slots} slots for a complex tree structure")
    
    def test_large_tree_slot_count(self):
        """Test slot count in a larger tree with random insertions."""
        tree = self.tree_k8
        keys = random.sample(range(1, 10000), 50)  # 50 unique random keys
        ranks = [1] * 30 + [2] * 15 + [3] * 5  # Mix of ranks
        random.shuffle(ranks)
        
        # Insert all items
        for key, rank in zip(keys, ranks):
            item = Item(key, "val")
            tree, _ = tree.insert(item, rank=rank)
        
        # Count slots manually by traversing the tree
        total_expected_slots = self._count_slots_manually(tree)
        
        self.assertEqual(total_expected_slots, tree.item_slot_count(),
                        f"Expected {total_expected_slots} slots for a large tree")
    
    def test_slot_count_after_changing_tree_structure(self):
        """Test that slot count updates correctly when tree structure changes."""
        # Start with a tree with several items
        tree = self.tree_k4
        for i in range(1, 11):
            tree, _ = tree.insert(Item(i * 1000, "val"), rank=1)
        
        # Get initial slot count
        initial_slots = tree.item_slot_count()
        
        # Insert an item that causes a node split
        tree, _ = tree.insert(Item(500, "val"), rank=2)
        
        # Get new slot count
        new_slots = tree.item_slot_count()
        
        # The slot count should increase after a split
        self.assertGreaterEqual(new_slots, initial_slots,
                              "Slot count should increase or stay the same after structure changes")
        
        # Count slots manually to verify
        total_expected_slots = self._count_slots_manually(tree)
        self.assertEqual(total_expected_slots, new_slots,
                        f"Expected {total_expected_slots} slots after structure change")
        
    def _count_slots_manually(self, tree):
        """Helper method to count slots by traversing the tree structure."""
        if tree.is_empty():
            return 0
            
        total_slots = 0
        node_queue = [tree.node]
        
        while node_queue:
            current_node = node_queue.pop(0)
            
            # Count slots in this node's set
            total_slots += current_node.set.item_slot_count()
            
            # Add child nodes to the queue
            if current_node.rank > 1:
                # Add left subtrees
                for entry in current_node.set:
                    if not entry.left_subtree.is_empty():
                        node_queue.append(entry.left_subtree.node)
                
                # Add right subtree
                if not current_node.right_subtree.is_empty():
                    node_queue.append(current_node.right_subtree.node)
                    
        return total_slots

if __name__ == "__main__":
    unittest.main()