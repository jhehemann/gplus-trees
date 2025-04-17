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

import time
from typing import Dict, Optional, Tuple, Any
from dataclasses import dataclass, asdict
from pprint import pprint

from packages.jhehemann.customs.gtree.base import (
    AbstractSetDataStructure,
    Item,
    Entry,
)
from packages.jhehemann.customs.gtree.klist import KList

DUMMY_ITEM_KEY = "0" * 64
DUMMY_ITEM_VALUE = None
DUMMY_ITEM_TIMESTAMP = None

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
        return Item(DUMMY_ITEM_KEY, DUMMY_ITEM_VALUE, DUMMY_ITEM_TIMESTAMP)

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
        print(f"\nInserting item {x_item} with rank {rank} into G+-tree structure:\n", self.print_structure())
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
        # Rank 1: single leaf node with dummy item.
        if rank == 1:
            leaf_set = KList()
            leaf_set = leaf_set.insert(self.instantiate_dummy_item(), GPlusTree())
            leaf_set = leaf_set.insert(x_item, GPlusTree())
            self.node = GPlusNode(rank, leaf_set, GPlusTree())
            return self

        # Rank > 1: create a root node with a dummy, an item replica, 
        # plus left and right subtrees.
        if rank > 1:
            root_set = KList()
            root_set = root_set.insert(self.instantiate_dummy_item(), GPlusTree())

            # Create replica (key only) for internal node
            replica = Item(x_item.key, None, None)

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

        # If rank <= 0, or an unexpected condition occurs, return self.
        return self
    
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
        current_tree = self
        parent_tree = None  # Track the previously visited tree when descending.
        parent_entry = None  # The parent's "next larger item" entry we used.
        # print(f"\nTree structure before insertion:\n{self.print_structure()}")

        # Descend until we find a matching rank or run out of subtrees.
        while current_tree.node.rank > rank:
            result = current_tree.node.set.retrieve(x_item.key)
            if result.next_entry is not None:
                parent_tree = current_tree
                parent_entry = result.next_entry
                current_tree = result.next_entry.left_subtree
            else:
                # Descend into right subtree
                parent_tree = current_tree
                current_tree = current_tree.node.right_subtree

        # If the current node's rank is still less than the required rank,
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
        if parent_tree is None:
            # No parent => we are at the root. Create a new root with dummy.
            old_node = self.node
            root_set = KList()
            root_set = root_set.insert(self.instantiate_dummy_item(), GPlusTree())
            # print("\nCurrent tree:", current_tree.print_structure())
            self.node = GPlusNode(rank, root_set, GPlusTree(old_node))

            # print("\nNew root created:\n", self.print_structure())
            return self
        else:
            # Unfold a layer in between parent and current node.
            new_set = KList()
            # Insert the current node's min (a "replica") to new node.
            result = current_tree.node.set.get_min()
            min_entry = result.found_entry
            if min_entry is None:
                raise RuntimeError(f"Expected nonempty set during rank mismatch handling, but get_min() returned None.\n\nParent Tree:\n {parent_tree.print_structure()}\n\nCurrent tree:\n {current_tree.print_structure()}\n\nSelf:\n {self.print_structure()}")
            
            new_set = new_set.insert(min_entry.item, min_entry.left_subtree)
            new_tree = GPlusTree(
                GPlusNode(rank, new_set, current_tree)
            )
            if parent_entry:
                parent_tree.node.set = parent_tree.node.set.update_left_subtree(
                    parent_entry.item.key, new_tree
                )
            else:
                parent_tree.node.right_subtree = new_tree

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
        while True:
            result = current_tree.node.set.retrieve(new_item.key)
            if current_tree.node.rank == 1:
                old_entry = result.found_entry
                old_item = old_entry.item if old_entry is not None else None
                if old_item is not None:
                    old_item.value = new_item.value
                    old_item.timestamp = new_item.timestamp
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
        # We may need to propagate splits while descending.
        parent_right_tree = None
        parent_right_entry = None

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
            insert_instance = x_item if is_leaf else Item(x_item.key, None, None)

            if parent_right_tree is None:
                # First insertion step at this level
                # print("\n\n\nFirst Iteration")
                # print(f"\nSelf: {self.print_structure()}")
                # print(f"\nCurrent tree: {current_tree.print_structure()}")
                # print(f"\nNext entry: {next_entry}")
                subtree = next_entry.left_subtree if next_entry else current_tree.node.right_subtree
                # print(f"\nDescent subtree: {subtree.print_structure()}")
                current_tree.node.set = current_tree.node.set.insert(insert_instance, subtree)

                # Prepare for possible next iteration
                parent_right_tree = current_tree
                current_tree = subtree
                parent_right_entry = next_entry

            else:
                # print("\n\n\nSubsequent Iteration")
                # print(f"\nSelf: {self.print_structure()}")
                # print(f"\nCurrent tree: {current_tree.print_structure()}")
                # print(f"\nNext entry: {next_entry}")
                # print(f"Parent right tree: {parent_right_tree.print_structure()}")
                # print(f"Parent right entry: {parent_right_entry}")
                
                # We need to split the current node at x_item.key
                left_split, x_item_left_subtree, right_split = current_tree.node.set.split_inplace(x_item.key)

                if x_item_left_subtree is not None:
                    print(f"\n\n\nLeft split: {left_split.print_structure()}")
                    print(f"\n\n\nRight split: {right_split.print_structure()}")
                    print(f"\n\n\nX item left subtree: {x_item_left_subtree.print_structure()}")
                    print(f"\n\n\nX item: {x_item}")
                    print(f"\n\n\nParent right tree: {parent_right_tree.print_structure()}")
                    print(f"\n\n\nParent right entry: {parent_right_entry}")
                    raise RuntimeError("Expected non-empty left subtree during split operation. This indicates that the item to insert is already present in the tree with a lower rank.")

                # Right side: if it has data or is a leaf, form a new node
                if not right_split.is_empty() or is_leaf:
                    right_split = right_split.insert(insert_instance, GPlusTree())
                    # if not right_split.is_empty():
                    #     print("\n\n\n Non-empty right split after insert:\n", right_split.print_structure())
                    # if is_leaf:
                    #     print("\n\n\n Leaf node right split after insert:\n", right_split.print_structure())
                    
                    new_tree = GPlusTree(
                        GPlusNode(
                            current_tree.node.rank,
                            right_split,
                            current_tree.node.right_subtree
                        )
                    )
                    # print("\nNew tree created (right):\n", new_tree.print_structure())
                    # Update the parent's reference
                    if parent_right_entry:
                        # print("\nUpdating parent right tree at key", parent_right_entry.item.key)
                        parent_right_tree.node.set = parent_right_tree.node.set.update_left_subtree(
                            parent_right_entry.item.key, new_tree)
                    else:
                        # print("\nUpdating parent right tree's right subtree")
                        # print("\nNew right subtree:", new_tree.print_structure())
                        parent_right_tree.node.right_subtree = new_tree
                        # print("\nParent right tree after update:", parent_right_tree.print_structure())

                    next_parent_right_tree = new_tree
                else:
                    next_parent_right_tree = parent_right_tree

                parent_right_tree = next_parent_right_tree

                # Reuse the left split in the current node
                current_tree.node.set = left_split
                if next_entry:
                    current_tree.node.right_subtree = next_entry.left_subtree

                if is_leaf:
                    # If leaf, link 'next' references if needed
                    new_tree.node.next = current_tree.node.next
                    current_tree.node.next = new_tree
                # print(f"\nNew current tree (left): {current_tree.print_structure()}")
                current_tree = current_tree.node.right_subtree
                # print(f"\nDescend Subtree: {current_tree.print_structure()}")

                # print("\nSelf:", self.print_structure())
                # print("\nParent right tree:", parent_right_tree.print_structure())

            # print("\n\nCurrent tree structure after loop run:\n", current_tree.print_structure())
            # Descend further if it’s not a leaf, otherwise we’re done
            if is_leaf:
                # print("\nLeaf node reached, stopping descent.")
                # print("Final tree structure:\n", self.print_structure())
                return self

            if current_tree.is_empty():
                print(f"\n\n\nX item: {x_item}")
                print(f"\n\n\nParent right tree: {parent_right_tree.print_structure()}")
                print(f"\n\n\nParent right entry: {parent_right_entry}")
                raise RuntimeError("Expected non-empty tree after insertion loop iteration where next tree node is not a leaf.")

            result = current_tree.node.set.retrieve(x_item.key)
            next_entry = result.next_entry
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
            #result.append(f"{prefix}    GPlusNode(rank={right_node.rank}, set={type(right_node.set).__name__})")
            #result.append(f"{prefix}      Entries:")
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
    linked_leaf_nodes: bool
    internal_has_replicas: bool

def gtree_stats_(t: GPlusTree, rank_distribution: Dict[int, int]) -> Stats:
    if t is None or t.is_empty():
        return Stats(
            gnode_height=0,
            gnode_count=0,
            item_count=0,
            item_slot_count=0,
            rank=-1,
            is_heap=True,
            least_item=None,
            greatest_item=None,
            is_search_tree=True,
            linked_leaf_nodes=True,
            internal_has_replicas=True,
        )

    node = t.node
    node_set = node.set
    rank_distribution[node.rank] = (
        rank_distribution.get(node.rank, 0) + node_set.item_count()
    )

    set_stats = [(entry, gtree_stats_(entry.left_subtree, rank_distribution))
                  for entry in node_set]
    right_stats = gtree_stats_(node.right_subtree, rank_distribution)

    stats = Stats(**vars(right_stats))
    max_child_height = max((s.gnode_height for _, s in set_stats), default=0)
    stats.gnode_height = 1 + max(max_child_height, right_stats.gnode_height)
    stats.gnode_count += 1
    for _, s in set_stats:
        stats.gnode_count += s.gnode_count

    # print(f"Adding node's item count {node_set.item_count()} to current stats item count {stats.item_count}")
    stats.item_count += node_set.item_count()
    # print(f"Resulting item count {stats.item_count}")
    
    stats.item_slot_count += node_set.item_slot_count()
    for _, s in set_stats:
        # print("Adding subtree item count", s.item_count, "to current stats item count", stats.item_count)
        stats.item_count += s.item_count
        # print(f"Resulting item count {stats.item_count}")
        stats.item_slot_count += s.item_slot_count
    # print("Adding right subtree item count", right_stats.item_count, "to current stats item count", stats.item_count)
    # stats.item_count += right_stats.item_count
    # print(f"Resulting item count {stats.item_count}")

    # A: Local check: parent.rank <= child.rank
    heap_local = True
    for entry in node_set:
        if entry.left_subtree is not None and not entry.left_subtree.is_empty():
            if node.rank <= entry.left_subtree.node.rank:
                heap_local = False
                break

    if node.right_subtree is not None and not node.right_subtree.is_empty():
        if node.rank <= node.right_subtree.node.rank:
            heap_local = False

    # B: All left subtrees are heaps
    heap_left_subtrees = all(s.is_heap for _, s in set_stats)

    # C: Right subtree is a heap
    heap_right_subtree = right_stats.is_heap

    # Combine
    stats.is_heap = heap_local and heap_left_subtrees and heap_right_subtree

    stats.is_search_tree = all(s.is_search_tree for _, s in set_stats) and right_stats.is_search_tree

    # Search tree property: right subtree >= last key
    if right_stats.least_item is not None:
        last_key = set_stats[-1][0].item.key
        print(f"\n\nlast key: {last_key} right least item: {right_stats.least_item.key}")
        if right_stats.least_item.key < last_key:
            stats.is_search_tree = False
            print("\n\nsearch tree property: right too small\n", t.print_structure())

    for i, (entry, left_stats) in enumerate(set_stats):
        if left_stats.least_item is not None and i > 0:
            prev_key = set_stats[i - 1][0].item.key
            if left_stats.least_item.key < prev_key:
                stats.is_search_tree = False
                print(f"\n\nsearch tree property: left {i} too small\n", t.print_structure())
                print("\set_stats", set_stats)

        if left_stats.greatest_item is not None:
            if left_stats.greatest_item.key >= entry.item.key:
                stats.is_search_tree = False
                print(f"\n\nitem key {entry.item.key} at position {i} is not greater than left subtree greatest item {left_stats.greatest_item.key}")
                print(f"\n\nsearch tree property: left {i} too great\n", t.print_structure())

    # Set least and greatest
    least_pair = set_stats[0]
    stats.least_item = (
        least_pair[1].least_item if least_pair[1].least_item is not None
        else least_pair[0].item
    )

    stats.greatest_item = (
        right_stats.greatest_item if right_stats.greatest_item is not None
        else set_stats[-1][0].item
    )

    # Linked leaf nodes
    # if node.rank == 1:
    #     # Return always true
    #     stats.linked_leaf_nodes = True
    if node.rank == 2:
        # Check if all entrie's left subtrees (if not empty) are linked to the left subtree of the next entry. If the next entry is None, we are at the end of the list and the left subtree of the last entry is linked to the nodes right subtree.

        for i, (entry, left_stats) in enumerate(set_stats):
            if entry.left_subtree is not None and not entry.left_subtree.is_empty():
                if i < len(set_stats) - 1:
                    next_entry = set_stats[i + 1][0]
                    if entry.left_subtree.node.next != next_entry.left_subtree:
                        stats.linked_leaf_nodes = False
                        print(f"\n\nLinked leaf nodes property: left {i} not linked to next entry\n", t.print_structure())
                else:
                    if entry.left_subtree.node.next != node.right_subtree:
                        stats.linked_leaf_nodes = False
                        print(f"\n\nLinked leaf nodes property: left {i} not linked to right subtree\n", t.print_structure())
            else:
                # If the left subtree is empty, we can skip this entry
                continue
    if node.rank >= 2:
        for entry in node_set:
            if entry.item.value is not None:
                stats.internal_has_replicas = False
        if node.next is not None:
            stats.linked_leaf_nodes = False
            print(f"\n\nLinked leaf nodes property: node with rank {node.rank} should not have a next node")
        

    for _, left_stats in set_stats:
        if not left_stats.linked_leaf_nodes:
            stats.linked_leaf_nodes = False
            print(f"\n\nLinked leaf nodes property: left subtree not linked\nLeft subtree stats:", left_stats)

    stats.rank = node.rank

    return stats

    

# print(t.print_structure())
# print("Stats:")
# pprint(asdict(stats))
# print("\n")


