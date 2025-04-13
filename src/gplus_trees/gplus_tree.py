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

from typing import Dict, Optional, Tuple, Any
from dataclasses import dataclass

from packages.jhehemann.customs.gtree.base import AbstractSetDataStructure
from packages.jhehemann.customs.gtree.base import Item
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
    
    If the attribute 'node' is None, the G+-tree is considered empty.
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
        if self.node is None:
            return "Empty GPlusTree"
        return str(self.node)

    def __repr__(self):
        return self.__str__()
        
    def insert(self, x_item: Item, rank: int) -> bool:
        """
        Insert an item into the G+-tree.
        """
        # 1) If the tree is empty, handle the initial creation logic.
        if self.is_empty():
            return self._handle_empty_insertion(x_item, rank)

        # 2) Otherwise, insert into a non-empty tree.
        return self._insert_non_empty(x_item, rank)
    
    def _handle_empty_insertion(self, x_item: Item, rank: int) -> bool:
        """
        If the tree is empty, build the initial structure depending on rank.
        """
        # Rank 1: single leaf node with dummy item.
        if rank == 1:
            leaf_set = KList()
            leaf_set.insert(self.instantiate_dummy_item())
            leaf_set.insert(x_item)
            self.node = GPlusNode(rank, leaf_set, GPlusTree())
            return True

        # Rank > 1: create a root node with a dummy, an item replica, 
        # plus left and right subtrees.
        if rank > 1:
            root_set = KList()
            root_set.insert(self.instantiate_dummy_item())

            # Create replica (key only) for internal node
            replica = Item(x_item.key, None, None)

            # Left subtree with just a dummy
            left_subtree_set = KList()
            left_subtree_set.insert(self.instantiate_dummy_item())
            left_subtree = GPlusTree(
                GPlusNode(1, left_subtree_set, GPlusTree())
            )
            root_set.insert(replica, left_subtree)

            # Right subtree with the actual item
            right_subtree_set = KList()
            right_subtree_set.insert(x_item)
            right_subtree = GPlusTree(
                GPlusNode(1, right_subtree_set, GPlusTree())
            )

            # Link leaf nodes
            left_subtree.node.next = right_subtree

            self.node = GPlusNode(rank, root_set, right_subtree)
            return True

        # If rank <= 0, or an unexpected condition occurs, return False.
        return False
    
    def _insert_non_empty(self, x_item: Item, rank: int) -> bool:
        """
        Handle insertion for a non-empty tree. This includes:
         - Descending until we find a node with rank >= the item rank.
         - Adjusting the tree if we overshoot (rank < item rank).
         - Updating existing item or inserting new leaf/internal node.
        """
        current_tree = self
        parent_tree = None  # Track the previously visited tree when descending.
        parent_entry = None  # The parent's "next larger item" entry we used.

        # Descend until we find a matching rank or run out of subtrees.
        while current_tree.node.rank > rank:
            found_item, next_larger_entry = current_tree.node.set.retrieve(x_item.key)
            if next_larger_entry is not None:
                parent_tree = current_tree
                parent_entry = next_larger_entry
                current_tree = next_larger_entry[1]
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

        # At this point, current_tree.node.rank == rank.
        # Check if the item exists at this node. 
        existing_item, next_larger_entry = (
            current_tree.node.set.retrieve(x_item.key)
        )
        if existing_item is not None:
            # Item key already exists: update its value in the leaf node.
            self._update_existing_item(current_tree, x_item, next_larger_entry)
            return True
        else:
            # Item does not exist: insert it. Possibly replicate along the path.
            return self._insert_new_item(
                current_tree, x_item, next_larger_entry
            )
        
    def _handle_rank_mismatch(
        self,
        current_tree: 'GPlusTree',
        parent_tree: 'GPlusTree',
        parent_entry: tuple,
        rank: int
    ) -> 'GPlusTree':
        """
        If the current node's rank < rank, we need to create or unfold a 
        node to match the new rank.
        """
        if parent_tree is None:
            # No parent => we are at the root. Create a new root with dummy.
            root_set = KList()
            root_set.insert(self.instantiate_dummy_item())
            # Point new root's right subtree to the current node.
            new_root = GPlusTree(
                GPlusNode(rank, root_set, current_tree)
            )
            self.node = new_root.node
            return new_root
        else:
            # Unfold a layer in between parent and current node.
            new_set = KList()
            # Insert the current node's min (a "replica") to new node.
            new_set.insert(current_tree.node.set.get_min())

            new_tree = GPlusTree(
                GPlusNode(rank, new_set, current_tree)
            )

            # Update the parent's reference to this subtree:
            if parent_entry is not None:
                parent_entry[1] = new_tree
            else:
                parent_tree.node.right_subtree = new_tree

            return new_tree
        
    def _update_existing_item(
        self,
        current_tree: 'GPlusTree',
        new_item: Item,
        next_entry: tuple
    ) -> None:
        """
        Traverse down to the leaf layer (rank=1) where the real item is stored 
        and update its value.
        """
        while True:
            if current_tree.node.rank == 1:
                # Update the item in place.
                # The retrieve() call gave us the same object, so:
                #   existing_item.value = new_item.value
                # but to be explicit, we can do it again:
                old_item, _ = current_tree.node.set.retrieve(new_item.key)
                if old_item:
                    old_item.value = new_item.value
                    old_item.timestamp = new_item.timestamp
                return

            # Descend further:
            if next_entry is not None:
                current_tree = next_entry[1]
            else:
                current_tree = current_tree.node.right_subtree

            # Retrieve again for the next level
            old_item, next_entry = current_tree.node.set.retrieve(new_item.key)

    def _insert_new_item(
        self,
        current_tree: 'GPlusTree',
        x_item: Item,
        next_entry: tuple
    ) -> bool:
        """
        Insert a new item key. For internal nodes, we only store the key. 
        For leaf nodes, we store the full item.
        """
        # We may need to propagate splits while descending.
        parent_right_tree = None
        parent_right_entry = None

        while True:
            is_leaf = (current_tree.node.rank == 1)

            # Use an "internal replica" if not a leaf
            insert_instance = x_item if is_leaf else Item(x_item.key, None, None)

            if parent_right_tree is None:
                # First insertion step at this level
                subtree = next_entry[1] if next_entry else current_tree.node.right_subtree
                current_tree.node.set.insert(insert_instance, subtree)

                # Prepare for possible next iteration
                parent_right_tree = current_tree
                current_tree = subtree
                parent_right_entry = next_entry

            else:
                # We need to split the current node at x_item.key
                left_split, _, right_split = current_tree.node.set.split_inplace(x_item.key)

                # Right side: if it has data or is a leaf, form a new node
                if not right_split.is_empty() or is_leaf:
                    right_split.insert(insert_instance)
                    new_tree = GPlusTree(
                        GPlusNode(
                            current_tree.node.rank,
                            right_split,
                            current_tree.node.right_subtree
                        )
                    )
                    # Update the parent's reference
                    if parent_right_entry:
                        parent_right_entry[1] = new_tree
                    else:
                        parent_right_tree.node.right_subtree = new_tree

                    parent_right_tree = new_tree

                # Reuse the left split in the current node
                current_tree.node.set = left_split
                if next_entry:
                    current_tree.node.right_subtree = next_entry[1]

                if is_leaf:
                    # If leaf, link 'next' references if needed
                    # (This may vary with your G+ tree's design.)
                    new_tree.node.next = current_tree.node.next
                    current_tree.node.next = new_tree.node.next
                current_tree = current_tree.node.right_subtree

            # Descend further if it’s not a leaf, otherwise we’re done
            if is_leaf:
                return True

            found_item, next_entry = current_tree.node.set.retrieve(x_item.key)

            
    #def old_insert(self, x_item: Item, rank: int) -> bool:
        # """
        # Insert an item into the G+-tree.
        # The insertion algorithm:
        #   - If the tree is empty, create a new node with the item and a dummy item.
        #   - If the tree is not empty, traverse the tree to find the appropriate place for the item.
        #   - If the item already exists, update its value in the leaf node.
        #   - If the item does not exist, insert replicas in internal nodes if necessary and the item into the appropriate leaf node.
        # Parameters:
        #     x_item (Item): The item to insert.
        #     rank (int): The rank of the item to insert.
        # Returns:
        #     bool: True if the item was inserted successfully, False otherwise.
        # """
        
        # cur_tree = self
        # if self.is_empty():
        #     # If the tree is empty, create initial nodes
        #     if rank > 1:
        #         # The rank of the item is greater than 1, so we need to build a root node along with its left and right subtrees, containing leaf nodes.
        #         # Create root set and insert dummy item.
        #         root_set = KList()
        #         root_set.insert(self.instantiate_dummy_item())

        #         # Create an item replica for the root node only containing the key.
        #         replica = Item(x_item.key, None, None)

        #         # Create left subtree for item (only containing a dummy item) and insert pair into root node.
        #         left_subtree_set = KList()
        #         left_subtree_set.insert(self.instantiate_dummy_item())
        #         left_subtree = GPlusTree(GPlusNode(1, left_subtree_set, GPlusTree()))
        #         root_set.insert(replica, left_subtree)

        #         # Create the root node's right subtree (only containing the insert item)
        #         right_subtree_set = KList()
        #         right_subtree_set.insert(x_item)
        #         right_subtree = GPlusTree(GPlusNode(1, right_subtree_set, GPlusTree()))

        #         # Create the root node and return
        #         cur_tree.node = GPlusNode(rank, root_set, right_subtree)
        #         return True
            
        #     elif rank == 1:
        #         # The item's rank is 1, so we can create a single leaf node for the item and a dummy item.
        #         # This node will be the only node in the tree and will also be the root.
        #         leaf_set = KList()
        #         leaf_set.insert(self.instantiate_dummy_item())
        #         leaf_set.insert(item)
        #         cur_tree.node = GPlusNode(rank, leaf_set, GPlusTree(), None)
        #         return True

        # else:
        #     # The tree is not empty, so we can traverse it to find the appropriate place for the item.
        #     prev_tree = None # Track the previously visited tree.

        #     # While the rank of the item is less than the current node's rank, we need to traverse down until we find a node with a rank equal to or smaller than rank.
        #     while cur_tree.node.rank > rank:
        #         # Check if an item larger than the insert item exists in the current node.
        #         _ , next_entry = cur_tree.node.set.retrieve(x_item.key)
        #         if next_entry is not None:
        #             # Next entry exists - descend into its left subtree.
        #             prev_tree = cur_tree
        #             cur_tree = next_entry[1]
        #         else:
        #             # No next entry exists - descend into the node's right subtree.
        #             prev_tree = cur_tree
        #             cur_tree = cur_tree.node.right_subtree

            # # Check if the rank of the item is greater than the node's rank
            # if cur_tree.node.rank < rank:
            #     # The rank of the item is greater than the current node's rank.
            #     # Check if there was a previous tree node.
            #     if prev_tree is None:
            #         # We are at the current root node of the tree 
            #         # Create a new root node with a single dummy item and its right subtree pointing to the current node.
            #         root_set = KList()
            #         root_set.insert(self.instantiate_dummy_item())
            #         root_tree = GPlusTree(GPlusNode(rank, root_set, cur_tree))
            #         cur_tree = root_tree
            #         cur_tree.node = cur_tree.node
            #     else:
            #         # A previous tree node exists and a collapsed layer at the current subtree needs to be unfolded.
            #         # Add the the current node's first entry to a new node's set. This is the required replica which has been collapsed at during a past tree operation. 
            #         # Create a new tree node in between the current node and the previous node.
            #         new_set = KList()
            #         new_set.insert(cur_tree.node.get_min())
            #         new_tree = GPlusTree(GPlusNode(rank, new_set, cur_tree))
            #         # Assign new tree as the corresponding subtree of the previous node.
            #         if next_entry is not None:
            #             next_entry[1] = new_tree
            #         else:
            #             prev_tree.node.right_subtree = new_tree
            #         cur_tree = new_tree
                        
            # # Now we are at a non-empty tree (node) with the same rank as the item.
            # # Retrieve the item based key and the next larger item from the current node.
            # item, next_entry = cur_tree.node.set.retrieve(x_item.key)
            # if item is not None:
            #     # Update branch: item exists in the tree
            #     # Traverse the tree and update the item in the leaf layer.
            #     while True:
            #         if cur_tree.node.rank == 1:
            #             item.value = x_item.value
            #             return True

            #         # Descend into the left subtree of the next larger item if it exists.
            #         if next_entry is not None:
            #             cur_tree = next_entry[1]
            #             continue

            #         # Descend into the node's right subtree.
            #         cur_tree = cur_tree.node.right_subtree
            # else:
            #     # Insert branch: item does not exist in the tree.
            #     prev_r_tree = None # Last visited right tree from a node split.
            #     prev_r_next_entry = None # Entry with next larger item than the last inserted item instance.
            #     while True:
            #         is_leaf = cur_tree.node.rank == 1
            #         # Prepare the item variant to insert in the current node
            #         insert_instance = (
            #             x_item if is_leaf 
            #             else Item(x_item.key, None, None)
            #         )
            #         if prev_r_tree is None:
            #             # First iteration: insert item paired with the subtree of the next larger entry or the node's right subtree.
            #             left_subtree = (
            #                 next_entry[1] if next_entry is not None 
            #                 else cur_tree.node.right_subtree
            #             )
            #             cur_tree.node.set.insert(
            #                 insert_instance, left_subtree
            #             )
            #             # Update variables to prepare for the next iteration.
            #             prev_r_tree = cur_tree
            #             cur_tree = left_subtree
            #             prev_r_next_entry = next_entry
                        
            #         else:                      
            #             # Subsequent iterations: split the current node's set at the insert item.
            #             l_split, _, r_split = (
            #                 cur_tree.node.set.split_inplace(x_item.key)
            #             )
                        
            #             # Handle right split
            #             if not r_split.is_empty() or is_leaf:
            #                 # Create new tree node for the right split and insert the item.
            #                 r_split.insert(insert_instance)
            #                 r_tree = GPlusTree(
            #                     GPlusNode(
            #                         cur_tree.node.rank,
            #                         r_split,
            #                         cur_tree.node.right_subtree
            #                     )
            #                 )
            #                 if prev_r_next_entry is not None:
            #                     prev_r_next_entry[1] = r_tree
            #                 else:
            #                     prev_r_tree.node.right_subtree = r_tree
            #                 prev_r_tree = r_tree
            #                 prev_r_next_entry = next_entry

            #             # Handle left split
            #             cur_tree.node.set = l_split # Reuse current node
            #             if next_entry is not None:
            #                 cur_tree.node.right_subtree = next_entry[1]
                        
            #             # Update node's next pointers if they are leaf nodes.
            #             if is_leaf:
            #                 r_tree.node.next = cur_tree.node.next
            #                 cur_tree.node.next = r_tree.node.next
            #             cur_tree = cur_tree.node.right_subtree

            #         if is_leaf:
            #             return True
                    
            #         item, next_entry = cur_tree.node.set.retrieve(item.key)


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
    
    def print_tree(self, indent: int = 0):
        def short_key(key: str) -> str:
            return key if len(key) <= 6 else f"{key[:3]}...{key[-3:]}"

        def print_klist_entries(klist, indent: int):
            node = klist.head
            while node:
                for item, _ in node.entries:
                    print(f"{' ' * indent}• key: {short_key(item.key)}, value: {item.value}")
                node = node.next

        prefix = ' ' * indent
        if self.is_empty():
            print(f"{prefix}Empty GPlusTree")
            return

        node = self.node
        print(f"{prefix}GPlusNode(rank={node.rank})")

        # Print own entries
        print(f"{prefix}  Entries:")
        current_klist_node = node.set.head
        while current_klist_node:
            for item, left_subtree in current_klist_node.entries:
                print(f"{prefix}    - key: {short_key(item.key)}, value: {item.value}")

                # Print left subtree’s root and its entries
                if left_subtree and not left_subtree.is_empty():
                    child_node = left_subtree.node
                    print(f"{prefix}      Left subtree: GPlusNode(rank={child_node.rank})")
                    print(f"{prefix}        Entries:")
                    print_klist_entries(child_node.set, indent + 10)
                else:
                    print(f"{prefix}      Left subtree: Empty")
            current_klist_node = current_klist_node.next

        # Print right subtree
        print(f"{prefix}  Right subtree:")
        if node.right_subtree and not node.right_subtree.is_empty():
            right_node = node.right_subtree.node
            print(f"{prefix}    GPlusNode(rank={right_node.rank})")
            print(f"{prefix}      Entries:")
            print_klist_entries(right_node.set, indent + 8)
        else:
            print(f"{prefix}    Empty")

        # Print next node if rank == 1
        if node.rank == 1 and hasattr(node, 'next') and node.next:
            print(f"{prefix}  Next:")
            if not node.next.is_empty():
                next_node = node.next.node
                print(f"{prefix}    GPlusNode(rank={next_node.rank})")
                print(f"{prefix}      Entries:")
                print_klist_entries(next_node.set, indent + 8)
            else:
                print(f"{prefix}    Empty")


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
            is_search_tree=True
        )

    node = t.node
    pairs = node.set
    rank_distribution[node.rank] = (
        rank_distribution.get(node.rank, 0) + pairs.item_count()
    )

    pair_stats = [(item, gtree_stats_(subtree, rank_distribution)) for item, subtree in pairs]
    right_stats = gtree_stats_(node.right_subtree, rank_distribution)

    stats = Stats(**vars(right_stats))
    max_child_height = max((s.gnode_height for _, s in pair_stats), default=0)
    stats.gnode_height = 1 + max(max_child_height, right_stats.gnode_height)
    stats.gnode_count += 1
    for _, s in pair_stats:
        stats.gnode_count += s.gnode_count

    stats.item_count += pairs.item_count()
    stats.item_slot_count += pairs.item_count()
    for _, s in pair_stats:
        stats.item_count += s.item_count
        stats.item_slot_count += s.item_slot_count
    stats.item_count += right_stats.item_count
    stats.item_slot_count += right_stats.item_slot_count

    # A: Local check: parent.rank <= child.rank
    heap_local = True
    for _, subtree in pairs:
        if subtree is not None and not subtree.is_empty():
            if node.rank > subtree.node.rank:
                heap_local = False
                break

    if node.right_subtree is not None and not node.right_subtree.is_empty():
        if node.rank > node.right_subtree.node.rank:
            heap_local = False

    # B: All left subtrees are heaps
    heap_left_subtrees = all(s.is_heap for _, s in pair_stats)

    # C: Right subtree is a heap
    heap_right_subtree = right_stats.is_heap

    # Combine
    stats.is_heap = heap_local and heap_left_subtrees and heap_right_subtree

    stats.is_search_tree = all(s.is_search_tree for _, s in pair_stats) and right_stats.is_search_tree

    # Search tree property: right subtree >= last key
    if right_stats.least_item is not None:
        last_key = pair_stats[-1][0]
        if right_stats.least_item == last_key:
            stats.is_search_tree = False
            print("\n\nsearch tree property: right too small\n", t.print_tree())

    for i, (item, left_stats) in enumerate(pair_stats):
        if left_stats.least_item is not None and i > 0:
            prev_key = pair_stats[i - 1][0]
            exit
            if left_stats.least_item.key < prev_key.key:
                stats.is_search_tree = False
                print(f"\n\nsearch tree property: left {i} too small\n", t.print_tree())
                print("\npair_stats", pair_stats)

        if left_stats.greatest_item is not None:
            if left_stats.greatest_item.key >= item.key:
                stats.is_search_tree = False
                print(f"\n\nsearch tree property: left {i} too great\n", t.print_tree())

    # Set least and greatest
    least_pair = pair_stats[0]
    stats.least_item = (
        least_pair[1].least_item if least_pair[1].least_item is not None
        else least_pair[0]
    )

    stats.greatest_item = (
        right_stats.greatest_item if right_stats.greatest_item is not None
        else pair_stats[-1][0]
    )

    stats.rank = node.rank
    return stats


