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

from __future__ import annotations
from typing import Dict, Optional, Tuple, Any
from dataclasses import dataclass, asdict
from pprint import pprint
import collections

from packages.jhehemann.customs.gtree.base import (
    AbstractSetDataStructure,
    Item,
    Entry,
    _create_replica,
)
from packages.jhehemann.customs.gtree.klist import KList

DUMMY_KEY_STR = "0" * 64
DUMMY_KEY = int(DUMMY_KEY_STR, 16)
DUMMY_VALUE = None
DUMMY_ITEM = Item(DUMMY_KEY, DUMMY_VALUE)

DEBUG = False

@dataclass
class GPlusNode:
    """
    A G+-node is the core component of a G+-tree.
    
    Attributes:
        rank (int): A natural number greater than 0.
        set (AbstractSetDataStructure): A k-list that stores elements which are item subtree pairs (item, left_subtree).
        right_subtree (GPlusTree): The right subtree (a GPlusTree) of this G+-node.
        next (Optional[GPlusTree]): The next G+-node in the linked list of leaf nodes.
    """
    rank: int
    set: 'AbstractSetDataStructure'
    right_subtree: 'GPlusTree'
    next: Optional['GPlusTree'] = None

    def __post_init__(self):
        if self.rank <= 0:
            raise ValueError("Rank must be a natural number greater than 0.")

    def __str__(self):
        return (f"GPlusNode(rank={self.rank}, klist=[\n{str(self.set)}\n], "
                f"right_subtree={self.right_subtree}, next={self.next})")
    
class GPlusTree(AbstractSetDataStructure):
    """
    A G+-tree is a recursively defined structure that is either empty or contains a single G+-node.
    Attributes:
        node (Optional[GPlusNode]): The G+-node that the tree contains. If None, the tree is empty.
        dim (int): The dimension of the G+-tree.
    """
    def __init__(
        self,
        node: Optional[GPlusNode] = None,
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
    
    def delete(self, item):
        raise NotImplementedError("delete not implemented yet")

    def get_min(self):
        raise NotImplementedError("get_min not implemented yet")

    def split_inplace(self):
        raise NotImplementedError("split_inplace not implemented yet")
    
    def instantiate_dummy_item(self):
        """
        Instantiate a dummy item with a key of 64 zero bits.
        This is used to represent the first entry in each layer of the G+-tree.
        """
        return Item(DUMMY_KEY, DUMMY_VALUE)

    def __str__(self):
        if self.is_empty():
            return "Empty GPlusTree"
        return str(self.node)

    def __repr__(self):
        return self.__str__()
        
    def insert(self, x_item: Item, rank: int) -> 'GPlusTree':
        """
        Insert an item into the G+-tree.
        Attributes:
            x_item (Item): The item to be inserted.
            rank (int): The rank of the item.
        """
        if DEBUG:
            print(f"\n\n\n>>>>>>>>>>>>>> Inserting item {x_item} with rank {rank}")
        # 1) If the tree is empty, handle the initial creation logic.
        if self.is_empty():
            return self._handle_empty_insertion(x_item, rank)

        # 2) Otherwise, insert into a non-empty tree.
        return self._insert_non_empty(x_item, rank)
    
    def _handle_empty_insertion(self, x_item: Item, rank: int) -> 'GPlusTree':
        """
        If the tree is empty, build the initial structure depending on rank.
        Attributes:
            x_item (Item): The item to be inserted.
            rank (int): The rank of the item.
        Returns:
            GPlusTree: The updated G+-tree.
        """
        if DEBUG:
            print("> Handling empty insertion")
        # Rank 1: single leaf node with dummy item.
        if rank == 1:
            if DEBUG:
                print(">> Rank 1: creating single leaf node")
            leaf_set = KList()
            leaf_set = leaf_set.insert(self.instantiate_dummy_item(), GPlusTree())
            leaf_set = leaf_set.insert(x_item, GPlusTree())
            self.node = GPlusNode(rank, leaf_set, GPlusTree())
            return self

        # Rank > 1: create a root node with a dummy, an item replica, 
        # plus left and right subtrees.
        elif rank > 1:
            if DEBUG:
                print(f">> Rank > 1, i.e {rank}: creating root node with dummy item")
            root_set = KList()
            root_set = root_set.insert(self.instantiate_dummy_item(), GPlusTree())

            # Create replica (key only) for internal node
            replica = _create_replica(x_item.key)

            # Left subtree with just a dummy
            left_subtree_set = KList()
            left_subtree_set = left_subtree_set.insert(self.instantiate_dummy_item(), GPlusTree())
            left_subtree = GPlusTree(
                GPlusNode(1, left_subtree_set, GPlusTree())
            )
            root_set.insert(replica, left_subtree)

            # Right subtree with the actual item
            right_subtree_set = KList()
            right_subtree_set = right_subtree_set.insert(x_item, GPlusTree())
            right_subtree = GPlusTree(
                GPlusNode(1, right_subtree_set, GPlusTree())
            )

            # Link leaf nodes
            left_subtree.node.next = right_subtree

            self.node = GPlusNode(rank, root_set, right_subtree)
            return self
        
        else:
            raise ValueError("Rank must be a natural number greater than 0.")

    
    def _insert_non_empty(self, x_item: Item, rank: int) -> 'GPlusTree':
        """
        Handle insertion for a non-empty tree. This includes:
         - Descending until we find a node with rank >= the item rank.
         - Adjusting the tree if we overshoot (rank < item rank).
         - Updating existing item or inserting new leaf/internal node.
        Attributes:
            x_item (Item): The item to be inserted.
            rank (int): The rank of the item.
        Returns:
            GPlusTree: The updated G+-tree.
        """
        if DEBUG:
            print("> Inserting into non-empty tree")
        current_tree = self
        parent_tree = None  # Track the previously visited tree when descending.
        parent_entry = None  # The parent's "next larger item" entry we used.
        # print(f"\nTree structure before insertion:\n{self.print_structure()}")

        # Descend until we find a matching rank or run out of subtrees.

        if DEBUG:
            print(f"> Descending to find gnode with rank <= {rank}")

        while current_tree.node.rank > rank:
            if DEBUG:
                print(f"> \n\nCurrent tree rank: {current_tree.print_structure()}")
            result = current_tree.node.set.retrieve(x_item.key)
            if result.next_entry is not None:
                parent_tree = current_tree
                parent_entry = result.next_entry
                if DEBUG:
                    if result.next_entry.left_subtree.is_empty():
                        print(f"> Next entrie's left subtree is empty")
                        print(f"> Current tree: {current_tree.print_structure()}")
                current_tree = result.next_entry.left_subtree
            else:
                if DEBUG:
                    if current_tree.node.right_subtree.is_empty():
                        print(f"> Current tree's right subtree is empty")
                        print(f"> Current tree: {current_tree.print_structure()}")
                # Descend into right subtree
                parent_tree = current_tree
                parent_entry = None

                current_tree = current_tree.node.right_subtree
            if DEBUG:
                print(f"> Is next current empty? {current_tree.is_empty()}")
        
        if DEBUG:
            print(f"> inserting gnode found")

        # If the current node's rank is less than the required rank,
        # we must "unfold" an in-between node or create a new root.
        if current_tree.node.rank < rank:
            current_tree = self._handle_rank_mismatch(
                current_tree, parent_tree, parent_entry, rank
            )
            #self.node = current_tree.node

        # At this point, current_tree.node.rank == rank.
        # print(f"\nInsert tree node found:\n", current_tree.print_structure())
        # Check if the item exists at this node. 
        result = current_tree.node.set.retrieve(x_item.key)
        # print("\nCurrent tree structure:\n", current_tree.print_structure())
        # print("\nFound item:", existing_item, "next larger entry:", next_larger_entry)

        if result.found_entry is not None:
            # Item key already exists: update its value in the leaf node.
            # print("\nUpdating existing item in G+-tree")
            return self._update_existing_item(current_tree, x_item)
        else:
            # Item does not exist: insert it. Possibly replicate along the path.
            # print("\nInserting new item into G+-tree")
            return self._insert_new_item(
                current_tree, x_item, result.next_entry
            )
        
    def _handle_rank_mismatch(
        self,
        current_tree: 'GPlusTree',
        parent_tree: 'GPlusTree',
        parent_entry: Entry,
        rank: int
    ) -> 'GPlusTree':
        """
        If the current node's rank < rank, we need to create or unfold a 
        node to match the new rank.
        This is done by creating a new G+-node and linking it to the parent.
        Attributes:
            current_tree (GPlusTree): The current G+-tree.
            parent_tree (GPlusTree): The parent G+-tree.
            parent_entry (tuple): The entry in the parent tree.
            rank (int): The rank to match.
        Returns:
            GPlusTree: The updated G+-tree.
        """
        if DEBUG:
            print(">> Handling rank mismatch")
        if parent_tree is None:
            if DEBUG:
                print(">>> No parent tree, creating new root")
            # No parent => we are at the root. Create a new root with dummy.
            old_node = self.node
            root_set = KList()
            root_set = root_set.insert(self.instantiate_dummy_item(), GPlusTree())
            # print("\nCurrent tree:", current_tree.print_structure())
            self.node = GPlusNode(rank, root_set, GPlusTree(old_node))

            # print("\nNew root (current) created:\n", self.print_structure())
            return self
        else:
            if DEBUG:
                print(">>> Unfolding a node in between parent and current tree")
            # Unfold a layer in between parent and current node.
            new_set = KList()
            # Insert a replica of the current node's min to new node.
            result = current_tree.node.set.get_min()
            min_entry = result.found_entry
            if min_entry is None:
                raise RuntimeError(f"Expected nonempty set during rank mismatch handling, but get_min() returned None.\n\nParent Tree:\n {parent_tree.print_structure()}\n\nCurrent tree:\n {current_tree.print_structure()}\n\nSelf:\n {self.print_structure()}")
            new_min_replica = _create_replica(min_entry.item.key)

            new_set = new_set.insert(new_min_replica, GPlusTree())
            new_tree = GPlusTree(
                GPlusNode(rank, new_set, current_tree)
            )
            if parent_entry:
                if DEBUG:
                    print(">>>> Updating parent key's left subtree for", parent_entry.item.key)
                parent_tree.node.set = parent_tree.node.set.update_left_subtree(
                    parent_entry.item.key, new_tree
                )
            else:
                if DEBUG:
                    print(">>>> Updating parent tree's right subtree")
                parent_tree.node.right_subtree = new_tree
            
            # print("\n\nNew tree node unfolded (current):\n", new_tree.print_structure())

        return new_tree
        
    def _update_existing_item(
        self,
        current_tree: 'GPlusTree',
        new_item: Item,
    ) -> 'GPlusTree':
        """
        Traverse down to the leaf layer (rank=1) where the real item is stored 
        and update its value.
        Attributes:
            current_tree (GPlusTree): The current G+-tree.
            new_item (Item): The item to be updated.
            next_entry (tuple): The next entry in the tree.
        Returns:
            GPlusTree: The updated G+-tree.
        """
        if DEBUG:
            print(">> Updating existing item")
        while True:
            result = current_tree.node.set.retrieve(new_item.key)
            if current_tree.node.rank == 1:
                old_entry = result.found_entry
                old_item = old_entry.item if old_entry is not None else None
                if old_item is not None:
                    old_item.value = new_item.value
                return self

            # Descend further:
            if result.next_entry is not None:
                current_tree = result.next_entry.left_subtree
            else:
                current_tree = current_tree.node.right_subtree
    
    def _insert_new_item(
        self,
        current_tree: 'GPlusTree',
        x_item: Item,
        next_entry: Entry
    ) -> 'GPlusTree':
        """
        Insert a new item key. For internal nodes, we only store the key. 
        For leaf nodes, we store the full item.
        Attributes:
            current_tree (GPlusTree): The current G+-tree.
            x_item (Item): The item to be inserted.
            next_entry (tuple): The next entry in the tree.
        Returns:
            GPlusTree: The updated G+-tree.
        """
        if DEBUG:
            print(">> Inserting new item")
        # We may need to propagate splits while descending.
        parent_right_tree = None
        parent_right_next = None
        parent_left_tree = None
        parent_left_x_entry = None

        visited = set()

        while True:
            # --- loop guard: prevent infinite descent into the same tree ---
            id_current = id(current_tree)
            if id_current in visited:
                raise RuntimeError("Infinite loop detected in _insert_new_item — revisiting same tree")
            visited.add(id_current)
            # print(f"\nDescending into GPlusTree with rank {current_tree.node.rank} with insert item key {x_item.key}")

            is_leaf = (current_tree.node.rank == 1)

            # Use an "internal replica" if not a leaf
            insert_instance = x_item if is_leaf else _create_replica(x_item.key)

            if parent_right_tree is None:
                
                # First insertion step at this level
                if DEBUG:
                    print(">>> First Iteration")
                # print(f"\nSelf: {self.print_structure()}")
                # print(f"\nCurrent tree: {current_tree.print_structure()}")
                # print(f"\nNext entry: {next_entry}")
                subtree = next_entry.left_subtree if next_entry else current_tree.node.right_subtree
                # print(f"\nDescent subtree: {subtree.print_structure()}")
                current_tree.node.set = current_tree.node.set.insert(insert_instance, subtree)
                   
                # Prepare for possible next iteration
                parent_right_tree = current_tree
                parent_right_next = next_entry
                parent_left_tree = current_tree
                # result = current_tree.node.set.retrieve(x_item.key)
                # parent_left_x_entry = result.found_entry
                
                current_tree = subtree
                

            else:
                if DEBUG:
                    print(">>>> Subsequent Iteration")
                # print(f"\nSelf: {self.print_structure()}")
                
                # print(f"\nCurrent tree pre split: {current_tree.print_structure()}")
                result = parent_right_tree.node.set.retrieve(x_item.key)
                parent_right_next = result.next_entry
                parent_right_x_entry = result.found_entry

                result = parent_left_tree.node.set.retrieve(x_item.key)
                parent_left_x_entry = result.found_entry

                print_cur_tree = current_tree.print_structure()
                
                # We need to split the current node at x_item.key
                left_split, x_item_left_subtree, right_split = current_tree.node.set.split_inplace(x_item.key)

                msg = None
                print_var = False
                if left_split is None:
                    msg = "Expected non-none left split after split operation."
                    print_var = True
                elif left_split.item_count() == 0:
                    msg = "Expected left split with > 0 items after split operation."
                    print_var = True
                elif left_split.is_empty():
                    msg = "Expected non-empty left split after split operation"
                    print_var = True
                

                if print_var:
                    # print("\n\n\nCurrent tree before split:\n", print_cur_tree)
                    # print("\n\n\nCurrent tree after split:\n", current_tree.print_structure())
                    # print(f"\n\n\nLeft split: {left_split.print_structure()}")
                    # print(f"\n\n\nRight split: {right_split.print_structure()}")
                    # print(f"\n\n\nParent right tree: {parent_right_tree.print_structure()}")
                    # print(f"\n\n\nParent right next: {parent_right_next.item.key if parent_right_next else None}")
                    # print(f"\n\n\nParent left tree: {parent_left_tree.print_structure()}")
                    # print(f"\n\n\nParent left x entry: {parent_left_x_entry.item.key if parent_left_x_entry else None}")
                    raise RuntimeError(msg)

                if DEBUG:
                    print(f"\n\n\nLeft split: {left_split.print_structure()}")
                    print(f"\n\n\nRight split: {right_split.print_structure()}")
                    print(f"\n\n\nCurrent after split: {current_tree.print_structure()}")
                    print(f"\n\n\nParent right tree: {parent_right_tree.print_structure()}")
                    print(f"\n\n\nParent right next: {parent_right_next.item.key if parent_right_next else None}")
                    print(f"\n\n\nParent left tree: {parent_left_tree.print_structure()}")
                    print(f"\n\n\nParent left x entry: {parent_left_x_entry.item.key if parent_left_x_entry else None}")

                if x_item_left_subtree is not None:
                    print(f"\n\n\nLeft split: {left_split.print_structure()}")
                    print(f"\n\n\nRight split: {right_split.print_structure()}")
                    print(f"\n\n\nX item left subtree: {x_item_left_subtree.print_structure()}")
                    print(f"\n\n\nX item: {x_item}")
                    print(f"\n\n\nParent right tree: {parent_right_tree.print_structure()}")
                    print(f"\n\n\nParent right next: {parent_right_next}")
                    raise RuntimeError("Expected non-empty left subtree during split operation. This indicates that the item to insert is already present in the tree with a lower rank.")

                # Right side: if it has data or is a leaf, form a new tree node
                
                if right_split.item_count() == 0 and not is_leaf:
                    if DEBUG:
                        print(">>>>> Empty right split and not leaf. Skipping tree node creation (right).")
                    next_parent_right_tree = parent_right_tree
                else:
                # if not (right_split.item_count() == 0 and not is_leaf):
                #if not right_split.is_empty() or is_leaf:
                    if DEBUG:
                        print(f">>>>> Inserting insert instance item {insert_instance} into right split and creating new tree")
                    right_split = right_split.insert(insert_instance, GPlusTree())
                    if DEBUG:
                        if not right_split.is_empty():
                            print("\n\n\n Non-empty right split after insert:\n", right_split.print_structure())
                    

                    if right_split.item_count() == 0:
                        raise RuntimeError("Expected non-empty right split after insert operation.")
                    
                    new_tree = GPlusTree(
                        GPlusNode(
                            current_tree.node.rank,
                            right_split,
                            current_tree.node.right_subtree
                        )
                    )
                    if new_tree.node.set.item_count() == 0:
                        raise RuntimeError("Expected non-empty new tree after insert operation.")
                    if DEBUG:
                        if is_leaf:
                            print("\n\n\n Leaf node new tree for right split after insert:\n", new_tree.print_structure())
                    # print("\nNew tree created (right):\n", new_tree.print_structure())
                    # Update the parent's reference
                    if parent_right_next:
                        if DEBUG:
                            print(">>>>>>Updating parent right tree at key", parent_right_next.item.key)
                        parent_right_tree.node.set = parent_right_tree.node.set.update_left_subtree(
                            parent_right_next.item.key, new_tree)
                    else:
                        if DEBUG:
                            print(">>>>>> Updating parent right tree's right subtree")
                        # print("\nNew right subtree:", new_tree.print_structure())
                        parent_right_tree.node.right_subtree = new_tree
                        # print("\nParent right tree after update:", parent_right_tree.print_structure())

                    next_parent_right_tree = new_tree                   

                parent_right_tree = next_parent_right_tree

                # Reuse the left split in the current node
                if parent_right_x_entry is None:
                    print("\n\nWARNING: In subsequent iteration the insert item schould be present in the parent right tree node.")
                
                if left_split.item_count() == 0:
                    raise RuntimeError("Always expect non-empty left split.")
                
                # if False:
                if left_split.item_count() == 1 and not is_leaf:
                    # Collapse this node and reassign subtrees
                    if DEBUG:
                        print(">>>>> Left split has only one item, collapsing node")
                    
                    if next_entry:
                        if DEBUG:
                            print(">>>>>> Updating prev left tree outdated current's next entry's left subtree")
                        new_subtree = next_entry.left_subtree
                    else:
                        if DEBUG:
                            print(">>>>>> No next entry in outdated current's tree node, while the left split is collapsed. This means that the node contained a single item before insertion.")
                            raise RuntimeError("This should not happen here.")
                        new_subtree = current_tree.node.right_subtree
                    
                    if parent_left_x_entry is not None:
                        if DEBUG:
                            print(">>>>>> Updating parent left tree at key", parent_right_x_entry.item.key)
                        parent_left_tree.node.set = parent_left_tree.node.set.update_left_subtree(
                            parent_right_x_entry.item.key, new_subtree
                        )
                        # next_parent_left_x_entry = None
                    else:
                        if DEBUG:
                            print(">>>>>> Updating parent left tree's right subtree")
                        parent_left_tree.node.right_subtree = new_subtree
                        # next_parent_left_x_entry = parent_left_x_entry
                        # print("\nParent left tree after update:", parent_left_tree.print_structure())

                    # if next_entry:
                    #     if DEBUG:
                    #         print(">>>>>> Updating parent left tree at key", next_entry.item.key)
                    #     new_subtree = next_entry.left_subtree
                        
                    # parent_left_tree.node.set = parent_left_tree.node.set.update_left_subtree(
                    #     parent_right_x_entry.item.key, new_subtree
                    # )
                    next_parent_left_tree = parent_left_tree
                    new_current_tree = new_subtree
                    
                else:
                    # next_parent_left_x_entry = parent_left_x_entry
                    # Continue with the left split at curent level
                    if DEBUG:
                        print(">>>>> Left split has more than one item, or we are at a leaf. Setting current tree's node set to left split")
                    current_tree.node.set = left_split
                    if next_entry:
                        if DEBUG:
                            print(">>>>>> Updating current tree's right subtree with outdated current's next entry's left subtree")
                        # print(f"\n\n\nNext entry Item: {next_entry.item}")
                        # print(f"\n\n\nNext entry left subtree:\n{next_entry.left_subtree.print_structure()}")
                        # print(f"\n\nSetting current tree nodes right subtree to next entry's left subtree")
                        current_tree.node.right_subtree = next_entry.left_subtree
                    else:
                        if DEBUG:
                            print(">>>>>> No next entry in outdated current's tree node. Keeping current tree's right subtree")
                    if parent_left_x_entry is not None:
                        if DEBUG:
                            print(">>>>>> Updating parent left tree at key", parent_left_x_entry.item.key)
                        parent_left_tree.node.set = parent_left_tree.node.set.update_left_subtree(
                            parent_left_x_entry.item.key, current_tree
                        )
                    next_parent_left_tree = current_tree
                    new_current_tree = current_tree.node.right_subtree
                
                parent_left_tree = next_parent_left_tree
                # parent_left_x_entry = next_parent_left_x_entry

                if is_leaf:
                    # If leaf, link 'next' references if needed
                    new_tree.node.next = current_tree.node.next
                    current_tree.node.next = new_tree
                    if DEBUG:
                        print(">>>>> Leaf node reached, linking next references")

                # print(f"\nNew current tree (left): {current_tree.print_structure()}")
                current_tree = new_current_tree
                if DEBUG:
                    print(f"\nNew Current: {current_tree.print_structure()}")

                # print("\nSelf:", self.print_structure())
                # print("\nParent right tree:", parent_right_tree.print_structure())

            # print("\n\nCurrent tree structure after loop run:\n", current_tree.print_structure())
            # Descend further if it’s not a leaf, otherwise we’re done
            if is_leaf:
                if DEBUG:
                    print(">>> Leaf node reached, stopping descent.")
                    print("\n\nFinal tree structure after insertion:\n", self.print_structure())
                # print("\n\nRight subtree structure after insertion:\n", current_tree.node.right_subtree.print_structure())
                return self

            if current_tree.is_empty():
                # print(f"\n\n\nX item: {x_item}")
                print(f"\n\n\nParent right tree: {parent_right_tree.print_structure()}")
                print(f"\n\n\nParent right next: {parent_right_next}")
                print(f"\n\n\nParent left tree: {parent_left_tree.print_structure()}")
                # print(f"\n\n\nParent left x entry: {parent_left_x_entry}")
                raise RuntimeError("Expected non-empty tree after insertion loop iteration where next tree node is not a leaf.")

            result = current_tree.node.set.retrieve(x_item.key)
            next_entry = result.next_entry
            # print(f"\n\nSelf after loop run:\n", self.print_structure())
            # print(f"\n\nCurrent after loop run:\n", current_tree.print_structure())
            # print(f"\n\n######################################## ITERATION FINISHED ########################################\n")
            # found_item, next_entry = current_tree.node.set.retrieve(x_item.key)

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

    def iter_leaf_nodes(self):
        """
        Iterates over all leaf-level GPlusNodes in the tree,
        starting from the leftmost leaf node and following `next` pointers.

        Yields:
            GPlusNode: Each leaf-level node in left-to-right order.
        """
        if self.is_empty():
            return 

        # Descend to the leftmost leaf
        current = self
        while not current.is_empty() and current.node.rank > 1:
            if current.node.set.is_empty():
                raise RuntimeError("Expected non-empty set for a non-empty GPlusTree during iteration. \nCurrent tree:\n", current.print_structure())
                # current = current.node.right_subtree
            else:
                result = current.node.set.get_min()
                if result.next_entry is not None:
                    current = result.next_entry.left_subtree
                else:
                    current = current.node.right_subtree

        # At this point, current is the leftmost leaf-level GPlusTree
        while current is not None and not current.is_empty():
            yield current.node
            current = current.node.next
    
    def physical_height(self) -> int:
        """
        The “real” pointer‑follow height of the G⁺‑tree:
        –  the number of KListNode segments in this node’s k‑list, plus
        –  the maximum physical_height() of any of its subtrees.
        """
        if self.is_empty():
            return 0

        node = self.node
        # 1) base = how many KListNodes this node’s set chains through
        base = node.set.physical_height()

        # 2) find the tallest child among all left_subtrees and the right_subtree
        max_child = 0
        for entry in node.set:
            left = entry.left_subtree
            if left is not None and not left.is_empty():
                max_child = max(max_child, left.physical_height())
        if node.right_subtree is not None and not node.right_subtree.is_empty():
            max_child = max(max_child, node.right_subtree.physical_height())

        # total physical height = this node’s chain length + deepest child
        return base + max_child

    def print_structure(self, indent: int = 0, depth: int = 0, max_depth: int = 2):
        prefix = ' ' * indent
        if self.is_empty() or self is None:
            return f"{prefix}Empty GPlusTree"
        
        if depth > max_depth:
            return f"{prefix}... (max depth reached)"
            
        result = []
        node = self.node
        result.append(f"{prefix}GPlusNode(rank={node.rank}, set={type(node.set).__name__})")
        
        result.append(node.set.print_structure(indent + 4))

        # Print right subtree
        if node.right_subtree and not node.right_subtree.is_empty():
            right_node = node.right_subtree.node
            result.append(f"{prefix}    Right: GPlusNode(rank={right_node.rank}, set={type(right_node.set).__name__})")
            # result.append(f"{prefix}    GPlusNode(rank={right_node.rank}, set={type(right_node.set).__name__})")
            # result.append(f"{prefix}      Entries:")
            # print_klist_entries(right_node.set, indent + 8)
            result.append(right_node.set.print_structure(indent + 8))
        else:
            result.append(f"{prefix}    Right: Empty")

        # Print next node if rank == 1
        if node.rank == 1 and hasattr(node, 'next') and node.next:
            if not node.next.is_empty():
                
                next_node = node.next.node
                result.append(f"{prefix}    Next: GPlusNode(rank={next_node.rank}, set={(type(next_node.set).__name__)})")
                # result.append(f"{prefix}    GPlusNode(rank={next_node.rank}, set={(type(next_node.set).__name__)})")
                #result.append(f"{prefix}      Entries:")
                # print_klist_entries(next_node.set, indent + 8)
                result.append(next_node.set.print_structure(indent + 8))
            else:
                result.append(f"{prefix}    Next: Empty")
        elif node.rank == 1 and hasattr(node, 'next') and node.next is None:
                result.append(f"{prefix}    Next: Empty")
        return "\n".join(result)

@dataclass
class Stats:
    gnode_height: int
    gnode_count: int
    item_count: int
    item_slot_count: int
    rank: int
    is_heap: bool
    least_item: Optional[Any]
    greatest_item: Optional[Any]
    is_search_tree: bool
    internal_has_replicas: bool
    internal_packed: bool
    linked_leaf_nodes: bool
    all_leaf_values_present: bool
    leaf_keys_in_order: bool

    # def __post_init__(self):
    #     # 1) all of these must be True
    #     for flag in (
    #         "is_heap",
    #         "is_search_tree",
    #         "internal_has_replicas",
    #         "internal_packed",
    #         "linked_leaf_nodes",
    #         "all_leaf_values_present",
    #         "leaf_keys_in_order",
    #     ):
    #         if not getattr(self, flag):
    #             raise AssertionError(f"{flag!r} must be True, got {getattr(self, flag)}")
    

# def gtree_stats_(t: GPlusTree, rank_distribution: Dict[int, int]) -> Stats:
#     if t is None or t.is_empty():
#         return Stats(
#             gnode_height=0,
#             gnode_count=0,
#             item_count=0,
#             item_slot_count=0,
#             rank=-1,
#             is_heap=True,
#             least_item=None,
#             greatest_item=None,
#             is_search_tree=True,
#             internal_has_replicas=True,
#             internal_packed=True,
#             linked_leaf_nodes=True,
#             all_leaf_values_present=True,
#             leaf_keys_in_order=True,
            
#         )

#     node = t.node
#     node_set = node.set
#     if DEBUG:
#         print("Node_set items:", node_set.item_count())
#     if node_set.item_count() == 0:
#         print("Node_set is empty: ", t.print_structure())
#     rank_distribution[node.rank] = (
#         rank_distribution.get(node.rank, 0) + node_set.item_count()
#     )

#     set_stats = [(entry, gtree_stats_(entry.left_subtree, rank_distribution))
#                   for entry in node_set]
#     right_stats = gtree_stats_(node.right_subtree, rank_distribution)

#     stats = Stats(**vars(right_stats))
#     max_child_height = max((s.gnode_height for _, s in set_stats), default=0)
#     stats.gnode_height = 1 + max(max_child_height, right_stats.gnode_height)
#     stats.gnode_count += 1
#     for _, s in set_stats:
#         stats.gnode_count += s.gnode_count

#     # print(f"Adding node's item count {node_set.item_count()} to current stats item count {stats.item_count}")
#     stats.item_count += node_set.item_count()
#     # print(f"Resulting item count {stats.item_count}")
    
#     stats.item_slot_count += node_set.item_slot_count()
#     for _, s in set_stats:
#         # print("Adding subtree item count", s.item_count, "to current stats item count", stats.item_count)
#         stats.item_count += s.item_count
#         # print(f"Resulting item count {stats.item_count}")
#         stats.item_slot_count += s.item_slot_count
#     # print("Adding right subtree item count", right_stats.item_count, "to current stats item count", stats.item_count)
#     # stats.item_count += right_stats.item_count
#     # print(f"Resulting item count {stats.item_count}")

#     # A: Local check: parent.rank <= child.rank
#     heap_local = True
#     for entry in node_set:
#         if entry.left_subtree is not None and not entry.left_subtree.is_empty():
#             if node.rank <= entry.left_subtree.node.rank:
#                 heap_local = False
#                 break

#     if node.right_subtree is not None and not node.right_subtree.is_empty():
#         if node.rank <= node.right_subtree.node.rank:
#             heap_local = False

#     # B: All left subtrees are heaps
#     heap_left_subtrees = all(s.is_heap for _, s in set_stats)

#     # C: Right subtree is a heap
#     heap_right_subtree = right_stats.is_heap

#     # Combine
#     stats.is_heap = heap_local and heap_left_subtrees and heap_right_subtree

#     stats.is_search_tree = all(s.is_search_tree for _, s in set_stats) and right_stats.is_search_tree

#     # Search tree property: right subtree >= last key
#     if right_stats.least_item is not None:
#         last_key = set_stats[-1][0].item.key
#         # print(f"\n\nlast key: {last_key} right least item: {right_stats.least_item.key}")
#         if right_stats.least_item.key < last_key:
#             stats.is_search_tree = False
#             # print("\n\nsearch tree property: right too small\n", t.print_structure())

#     for i, (entry, left_stats) in enumerate(set_stats):
#         if left_stats.least_item is not None and i > 0:
#             prev_key = set_stats[i - 1][0].item.key
#             if left_stats.least_item.key < prev_key:
#                 stats.is_search_tree = False
#                 # print(f"\n\nsearch tree property: left {i} too small\n", t.print_structure())
#                 # print("\set_stats", set_stats)

#         if left_stats.greatest_item is not None:
#             if left_stats.greatest_item.key >= entry.item.key:
#                 stats.is_search_tree = False
#                 # print(f"\n\nitem key {entry.item.key} at position {i} is not greater than left subtree greatest item {left_stats.greatest_item.key}")
#                 # print(f"\n\nsearch tree property: left {i} too great\n", t.print_structure())

#     # Set least and greatest
#     least_pair = set_stats[0]
#     stats.least_item = (
#         least_pair[1].least_item if least_pair[1].least_item is not None
#         else least_pair[0].item
#     )

#     stats.greatest_item = (
#         right_stats.greatest_item if right_stats.greatest_item is not None
#         else set_stats[-1][0].item
#     )

#     stats.linked_leaf_nodes = all(s.linked_leaf_nodes for _, s in set_stats) and right_stats.linked_leaf_nodes


    
#     # Linked leaf nodes
#     # if node.rank == 1:
#     #     # Return always true
#     #     stats.linked_leaf_nodes = True
#     if node.rank == 2:
#         # Check if all entrie's left subtrees (if not empty) are linked to the left subtree of the next entry. If the next entry is None, we are at the end of the list and the left subtree of the last entry is linked to the nodes right subtree.

#         for i, (entry, left_stats) in enumerate(set_stats):
#             if entry.left_subtree is not None and not entry.left_subtree.is_empty():
#                 if i < len(set_stats) - 1:
#                     next_entry = set_stats[i + 1][0]
#                     if entry.left_subtree.node.next != next_entry.left_subtree:
#                         stats.linked_leaf_nodes = False
#                         print(f"\n\nLinked leaf nodes property: left {i} not linked to next entry\n", t.print_structure())
#                 else:
#                     if entry.left_subtree.node.next != node.right_subtree:
#                         stats.linked_leaf_nodes = False
#                         print(f"\n\nLinked leaf nodes property: left {i} not linked to right subtree\n", t.print_structure())
#             else:
#                 # If the left subtree is empty, we can skip this entry
#                 continue
    
#     stats.internal_has_replicas = all(s.is_search_tree for _, s in set_stats) and right_stats.internal_has_replicas

#     node_set_packed = True
#     if node.rank >= 2:
#         node_set_packed = False if node_set.item_count() < 2 else True
#         for entry in node_set:
#             if entry.item.value is not None:
#                 stats.internal_has_replicas = False
#         if node.next is not None:
#             stats.linked_leaf_nodes = False
#             print(f"\n\nLinked leaf nodes property: node with rank {node.rank} should not have a next node")
        
#     stats.internal_packed = all(s.internal_packed for _, s in set_stats) and right_stats.internal_packed and node_set_packed

#     # Walk the leaves exactly once
#     dummy_key = DUMMY_KEY
#     leaf_keys, leaf_values = [], []
#     for leaf in t.iter_leaf_nodes():
#         for entry in leaf.set:
#             k, v = entry.item.key, entry.item.value
#             if k == dummy_key:
#                 continue
#             leaf_keys.append(k)
#             leaf_values.append(v)

#     stats.all_leaf_values_present = all(v is not None for v in leaf_values)
#     stats.leaf_keys_in_order     = (leaf_keys == sorted(leaf_keys))

#     stats.rank = node.rank

#     return stats


def gtree_stats_(t: GPlusTree,
                 rank_hist: Optional[Dict[int, int]] = None,
                 _is_root: bool = True,
                 ) -> Stats:
    """
    Returns aggregated statistics for a G⁺-tree in **O(n)** time.

    The caller can supply an existing Counter / dict for `rank_hist`;
    otherwise a fresh Counter is used.
    """
    if rank_hist is None:
        rank_hist = collections.Counter()

    # ---------- empty tree fast-return ---------------------------------
    if t is None or t.is_empty():
        return Stats(gnode_height        = 0,
                     gnode_count         = 0,
                     item_count          = 0,
                     item_slot_count     = 0,
                     rank                = -1,
                     is_heap             = True,
                     least_item          = None,
                     greatest_item       = None,
                     is_search_tree      = True,
                     internal_has_replicas = True,
                     internal_packed     = True,
                     linked_leaf_nodes   = True,
                     all_leaf_values_present = True,
                     leaf_keys_in_order  = True)

    node       = t.node
    node_set   = node.set
    rank_hist[node.rank] = rank_hist.get(node.rank, 0) + node_set.item_count()


    # ---------- recurse on children ------------------------------------
    child_stats = [gtree_stats_(e.left_subtree, rank_hist, False) for e in node_set]
    right_stats = gtree_stats_(node.right_subtree,   rank_hist, False)

    # ---------- aggregate in one pass ----------------------------------
    stats = Stats(
        gnode_height=0,
        gnode_count=0,
        item_count=0,
        item_slot_count=0,
        rank=-1,
        is_heap=True,
        least_item=None,
        greatest_item=None,
        is_search_tree=True,
        internal_has_replicas=True,
        internal_packed=True,
        linked_leaf_nodes=True,
        all_leaf_values_present=True,
        leaf_keys_in_order=True,
    )
    stats.rank = node.rank
    stats.gnode_count     = 1 + right_stats.gnode_count
    stats.item_count      = node_set.item_count() + right_stats.item_count
    stats.item_slot_count = node_set.item_slot_count() + right_stats.item_slot_count
    stats.gnode_height    = 1 + max(right_stats.gnode_height,
                                    max((cs.gnode_height for cs in child_stats), default=0))

    # Single pass over children to fold in all counts and booleans
    prev_key = None
    for entry, cs in zip(node_set, child_stats):
        stats.gnode_count     += cs.gnode_count
        stats.item_count      += cs.item_count
        stats.item_slot_count += cs.item_slot_count

        stats.is_heap            &= (node.rank > cs.rank) and cs.is_heap
        stats.is_search_tree     &= cs.is_search_tree
        if prev_key is not None and cs.least_item and cs.least_item.key < prev_key:
            stats.is_search_tree = False
        if cs.greatest_item and cs.greatest_item.key >= entry.item.key:
            stats.is_search_tree = False

        stats.internal_has_replicas &= cs.internal_has_replicas
        stats.internal_packed       &= cs.internal_packed
        stats.linked_leaf_nodes     &= cs.linked_leaf_nodes

        prev_key = entry.item.key

    # Fold in right subtree flags and ordering
    stats.is_heap              &= right_stats.is_heap
    stats.is_search_tree       &= right_stats.is_search_tree
    if right_stats.least_item and right_stats.least_item.key < prev_key:
        stats.is_search_tree = False

    stats.internal_has_replicas &= right_stats.internal_has_replicas
    stats.internal_packed       &= right_stats.internal_packed
    stats.linked_leaf_nodes     &= right_stats.linked_leaf_nodes

    # ----- LEAST / GREATEST PATCHED -----
    # Instead of list(node_set)[0] / [-1], do:
    if child_stats and child_stats[0].least_item is not None:
        stats.least_item = child_stats[0].least_item
    else:
        res0 = node_set.get_entry(0)
        stats.least_item = res0.found_entry.item

    if right_stats.greatest_item is not None:
        stats.greatest_item = right_stats.greatest_item
    else:
        last_idx = node_set.item_count() - 1
        res_last = node_set.get_entry(last_idx)
        stats.greatest_item = res_last.found_entry.item

    # ---------- leaf walk ONCE at the root -----------------------------
    if node.rank == 1:          # leaf node: base values
        keys   = [e.item.key   for e in node_set if e.item.key != DUMMY_KEY]
        values = [e.item.value for e in node_set if e.item.key != DUMMY_KEY]
        stats.all_leaf_values_present = all(v is not None for v in values)
        stats.leaf_keys_in_order      = True               # a single node is sorted
    elif _is_root:         # root call (only true once)
        leaf_keys, leaf_values = [], []
        for leaf in t.iter_leaf_nodes():
            for e in leaf.set:
                if e.item.key == DUMMY_KEY:
                    continue
                leaf_keys.append(e.item.key)
                leaf_values.append(e.item.value)
        stats.all_leaf_values_present = all(v is not None for v in leaf_values)
        stats.leaf_keys_in_order      = (leaf_keys == sorted(leaf_keys))


    return stats


def collect_leaf_keys(tree: 'GPlusTree') -> list[str]:
        out = []
        for leaf in tree.iter_leaf_nodes():
            for e in leaf.set:
                if e.item.key != DUMMY_KEY:
                    out.append(e.item.key)
        return out


