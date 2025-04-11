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

"""G+-tree implementation"""

from typing import Optional, Tuple

from packages.jhehemann.customs.gtree.base import AbstractSetDataStructure
from packages.jhehemann.customs.gtree.base import Item
from packages.jhehemann.customs.gtree.klist import KList

DUMMY_ITEM_KEY = "0" * 64
DUMMY_ITEM_VALUE = None
DUMMY_ITEM_TIMESTAMP = None

class GPlusNode:
    """
    A G+-node is the core component of a G+-tree.
    
    Attributes:
        rank (int): A natural number greater than 0.
        set (AbstractSetDataStructure): A k-list that stores elements which are item subtree pairs (item, left_subtree).
        right_subtree (GPlusTree): The right subtree (a GPlusTree) of this G+-node.
    """
    def __init__(
        self,
        rank: int,
        set: AbstractSetDataStructure,
        right_subtree: 'GPlusTree'
    ):
        if rank <= 0:
            raise ValueError("Rank must be a natural number greater than 0.")
        self.rank = rank
        self.set = set
        self.right_subtree = right_subtree

    def __str__(self):
        return f"GPlusNode(rank={self.rank}, klist=[\n{str(self.set)}\n], right_subtree={self.right_subtree})"

    def __repr__(self):
        return self.__str__()


class GPlusTree(AbstractSetDataStructure):
    """
    A G+-tree is a recursively defined structure that is either empty or contains a single G+-node.
    
    If the attribute 'node' is None, the G+-tree is considered empty.
    """
    def __init__(
        self,
        node: Optional[GPlusNode],
        dim: int = 1,
    ):
        """
        Initialize a G+-tree.
        
        Parameters:
            node (G+Node or None): The G+-node that the tree contains. If None, the tree is empty.
            dim (int): The dimension of the G+-tree.
        """
        self.node = node
        self.dim = dim

    def is_empty(self) -> bool:
        return self.node is None
    
    def instantiate_dummy_item(self):
        """
        Instantiate a dummy item with a key of 64 zero bits.
        
        This is used to represent the first entry in each layer of the G+-tree.
        """
        return Item(DUMMY_ITEM_KEY, DUMMY_ITEM_VALUE, DUMMY_ITEM_TIMESTAMP)

    def __str__(self):
        if self.node is None:
            return "Empty GPlusTree"
        return str(self.node)

    def __repr__(self):
        return self.__str__()
        
    def insert(self, item: Item, rank: int) -> bool:
        cur_tree = self
        cur_node = cur_tree.node
        if self.is_empty():
            # If the tree is empty, create initial nodes
            if rank > 1:
                # The rank of the item is greater than 1, so we need to build a root node along with its left and right subtrees, containing leaf nodes.
                # Create root set and insert dummy item.
                root_set = KList()
                root_set.insert(self.instantiate_dummy_item())

                # Create an item replica for the root node only containing the key.
                replica = Item(item.key, None, None)

                # Create left subtree for item (only containing a dummy item) and insert pair into root node.
                left_subtree_set = KList()
                left_subtree_set.insert(self.instantiate_dummy_item())
                left_subtree = GPlusTree(GPlusNode(1, left_subtree_set, GPlusTree()))
                root_set.insert(replica, left_subtree)

                # Create the root node's right subtree (only containing the item)
                right_subtree_set = KList()
                right_subtree_set.insert(item)
                right_subtree = GPlusTree(GPlusNode(1, right_subtree_set, GPlusTree()))

                # Create the root node
                cur_node = GPlusNode(rank, root_set, right_subtree)
                
                return True
            elif rank == 1:
                # The item's rank is 1, so we can create a single leaf node for the item and a dummy item.
                # This node will be the only node in the tree and will also be the root.
                leaf_set = KList()
                leaf_set.insert(self.instantiate_dummy_item())
                leaf_set.insert(item)
                cur_node = GPlusNode(rank, leaf_set, GPlusTree())
                return True

        else:
            # The tree is not empty, so we can traverse it to find the appropriate place for the item.
            prev_tree = None # Track the previously visited tree.
            prev_node = None # Track the previously visited node.

            # While the rank of the item is less than the current node's rank, we need to traverse down until we find a node with a rank equal to or smaller than rank.
            while cur_node.rank > rank:
                item, next_entry = cur_node.set.retrieve(item.key)
                # Check if an item larger than the insert item exists.
                if next_entry is not None:
                    # Next entry exists - descend into its left subtree.
                    prev_tree = cur_tree
                    cur_tree = next_entry[1]
                else:
                    # No next entry exists - descend into the node's right subtree.
                    prev_tree = cur_tree
                    cur_tree = cur_node.right_subtree
                prev_node = prev_tree.node
                cur_node = cur_tree.node

            # Check if the rank of the item is greater than the node's rank
            if cur_node.rank < rank:
                # The rank of the item is greater than the current node's rank.
                # Check if there was a previous tree node.
                if prev_tree is None:
                    # We are at the current root node of the tree 
                    # Create a new root node with a dummy item and an item replica and its subtrees pointing to the current tree.
                    root_set = KList()
                    root_set.insert(self.instantiate_dummy_item())
                    root_set.insert(Item(item.key, None, None), cur_tree)
                    root_tree = GPlusTree(GPlusNode(rank, root_set, cur_tree))
                    cur_tree = root_tree
                    cur_node = cur_tree.node
                else:
                    # A previous tree node exists
                    # Create a new node in between the current node and the previous node and assign it to the previous node's right subtree.
                    new_set = KList()
                    # TODO: Check if we need to insert a dummy item here. Use prev_tree.retrieve(item.key) somehow to check if a dummy item exists in the previous node and if the current tree is the next right subtree of the dummy item
            
            # Now we are at the tree node with the same rank as the item.
            while not cur_tree.is_empty():
                cur_node = cur_tree.node

                # Insert the item into the current node's set.
                if cur_node.set.insert(item, rank):
                    return True
                else:
                    # If insertion fails, we need to split the node.
                    # Create a new G+-node with a higher rank and insert the item into it.
                    new_set = KList()
                    new_set.insert(item)
                    new_node = GPlusNode(rank, new_set, GPlusTree())
                    cur.node = new_node
                    return True
            else:
                # Descend into the right subtree of the current node.
                cur = cur_node.right_subtree

    
    
    def retrieve(
        self, key: str
    ) -> Tuple[Optional[Item], Tuple[Optional[Item], Optional['GPlusTree']]]:
        """
        Searches for an item with a matching key in the G+-tree.

        Iteratively traverses the tree with O(k) additional memory, where k is the expected G+-node size.
        
        The search algorithm:
          - At each non-empty GPlusTree, let current_node = self.node.
          - For each entry in current_node.set (in sorted order):
                If entry.item.key == key, return the item.
                Else if key < entry.item.key, then descend into that entry's left subtree.
          - If no entry in the KList is suitable, then descend into the node's right subtree.
        
        Parameters:
            key (str): The key to search for.
        
        Returns:
            A tuple of two elements:
            - item (Optional[Item]): The value associated with the key, or None if not found.
            - next_entry (Tuple[Optional[Item], Optional[GPlusTree]]): 
                    A tuple containing:
                    * The next item in the sorted order (if any),
                    * The left subtree associated with the next item (if any).
                    If no subsequent entry exists, returns (None, None).
        """
        cur_tree = self
        item = None
        next_entry = None
        while not cur_tree.is_empty():
            cur_node = cur_tree.node

            # Retrieve possibly non existent item and next entry from current node.
            item, next_entry = cur_node.set.retrieve(key)
            #(next_item, left_subtree) = next_entry
            
            # Determine where to descend.
            if next_entry is not None:
                # An item greater than key exists, so we descend into its left subtree.
                cur_tree = next_entry[1]
                continue
            else:
                # No next item exists in the current node, so we must descend into the node's right subtree.
                # If the node has a next pointer, it is a leaf node and we can use it to find the next entry within the tree, before descending
                if cur_node.next is not None:
                    # The next entry is the first entry in the next node.
                    next_entry = cur_node.next.set.get_min()
                cur_tree = cur_node.right_subtree
                continue
        
        # By the traversal logic and the key ordering in a G+tree, if we reach here, it means that we have reached a leaf node's empty subtree and we can return the last seen item and next entry.
        return (item, next_entry)    
