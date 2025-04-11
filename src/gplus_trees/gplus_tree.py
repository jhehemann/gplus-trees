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
    def __init__(self, node: Optional[GPlusNode]):
        """
        Initialize a G+-tree.
        
        Parameters:
            node (G+Node or None): The G+-node that the tree contains. If None, the tree is empty.
        """
        self.node = node

    def is_empty(self) -> bool:
        return self.node is None

    def __str__(self):
        if self.node is None:
            return "Empty GPlusTree"
        return str(self.node)

    def __repr__(self):
        return self.__str__()
    
    # def insert(self, item: Item, rank: int) -> None:
    #     if self.is_empty():
    #         # Create an internal node storing the version of the item that only stores the key.
    #         cur = self
    #         if rank > 1:
    #             internal_klist = KList()
    #             key_only_item = Item(item.key, None, None)
    #             internal_klist.insert(key_only_item)
    #             right_subtree = GPlusTree()
    #             self.node = GPlusNode(rank, internal_klist, right_subtree)
            
    #         # Create a leaf node (G+-node) for the right subtree.
    #         leaf_klist = KList()
    #         leaf_klist.insert(item)  # Insert the full item into the leaf's KList.
    #         leaf_gplus_node = GPlusNode(rank, leaf_klist, GPlusTree())  # Leaf G+-node; its right subtree is empty.
            
           
    #     else:
    #         if rank == self.node.rank:
    #             self.node.set.insert(item)
    #         else:
    #             self.node.right.insert(item, rank)

    
    
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
