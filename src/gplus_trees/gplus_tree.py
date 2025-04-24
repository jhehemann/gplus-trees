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
    RetrievalResult,
)
from packages.jhehemann.customs.gtree.klist import KList

# Constants
DUMMY_KEY = int("0" * 64, 16)
DUMMY_VALUE = None
DUMMY_ITEM = Item(DUMMY_KEY, DUMMY_VALUE)

DEBUG = False

@dataclass(slots=True)
class GPlusNode:
    """
    A G+-node is the core component of a G+-tree.
    
    Attributes:
        rank (int): A natural number greater than 0.
        set (AbstractSetDataStructure): Set data structure storing node entries
        right_subtree (GPlusTree): The right subtree of this G+-node.
        next [GPlusTree]: The next G+-node in the linked list of leaf nodes or None
    """
    rank: int
    set: AbstractSetDataStructure
    right_subtree: GPlusTree
    next: Optional[GPlusTree] = None

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
    __slots__ = ("node", "dim")
    
    def __init__(self, node: Optional[GPlusNode] = None, dim: int = 1):
        self.node = node
        self.dim = dim

    def is_empty(self) -> bool:
        return self.node is None
    
    def __str__(self):
        return "Empty GPlusTree" if self.is_empty() else f"GPlusTree(dim={self.dim}, node={self.node})"

    __repr__ = __str__
    
    # Public API
    def insert(self, x: Item, rank: int) -> GPlusTree:
        """
        Public method (average-case O(log n)): Insert an item into the G+-tree. 
        If the item already exists, updates its value at the leaf node.
        
        Args:
            x_item (Item): The item (key, value) to be inserted.
            rank (int): The rank of the item. Must be a natural number > 0.
        Returns:
            GPlusTree: The updated G+-tree.

        Raises:
            TypeError: If x_item is not an Item or rank is not a positive int.
        """
        if not isinstance(x, Item):
            raise TypeError(f"insert(): expected Item, got {type(x).__name__}")
        if not isinstance(rank, int) or rank <= 0:
            raise TypeError(f"insert(): rank must be a positive int, got {rank!r}")
        if self.is_empty():
            return self._insert_empty(x, rank)
        return self._insert_non_empty(x, rank)
    
    def retrieve(
        self, key: int
    ) -> Tuple[Optional[Item], Tuple[Optional[Item], Optional['GPlusTree']]]:
        """
        Searches for an item with a matching key in the G+-tree.

        Iteratively traverses the tree (O(log n) with high probability)
        with O(k) additional memory by descending into left or right subtrees
        based on key comparisons.

        Args:
            key (int): The key to search for.

        Returns:
            RetrievalResult: Contains:
                found_item (Optional[Item]): The value associated with the key, or None if not found.
                next_pair (Tuple[Optional[Item], Optional[GPlusTree]]):
                    The next item in sorted order and its associated subtree, or (None, None).
        """
        current_tree = self
        found_item: Optional[Item] = None
        next_pair: Tuple[Optional[Item], Optional[GPlusTree]] = (None, None)

        while not current_tree.is_empty():
            node = current_tree.node
            # Attempt to retrieve in this node's k-list
            found, next_pair = node.set.retrieve(key)
            found_item = found or found_item

            # Descend based on presence of next_pair
            if next_pair is not None:
                _, subtree = next_pair
                current_tree = subtree
            else:
                # If leaf has a linked next node, update next_pair
                if node.next is not None:
                    next_pair = node.next.set.get_min()
                current_tree = node.right_subtree

        return RetrievalResult(found_item, next_pair)
    
    def delete(self, item):
        raise NotImplementedError("delete not implemented yet")

    def get_min(self):
        raise NotImplementedError("get_min not implemented yet")

    def split_inplace(self):
        raise NotImplementedError("split_inplace not implemented yet")
    
    # Private Methods
    def _make_leaf_klist(self, x_item: Item) -> KList:
        """Builds a KList for a single leaf node containing the dummy and x_item."""
        return (
            KList()
            .insert(DUMMY_ITEM, GPlusTree())
            .insert(x_item, GPlusTree())
        )

    def _make_leaf_trees(self, x_item: Item) -> Tuple[GPlusTree, GPlusTree]:
        """Builds two linked leaf-level GPlusTree nodes for x_item insertion.
        and returns the corresponding G+-trees."""
        r_leaf_set = KList().insert(x_item, GPlusTree())       # right leaf tree
        r_leaf_t = GPlusTree(GPlusNode(1, r_leaf_set, GPlusTree()))
        l_leaf_set = KList().insert(DUMMY_ITEM, GPlusTree())   # left leaf tree
        l_leaf_t = GPlusTree(GPlusNode(1, l_leaf_set, GPlusTree()))
        l_leaf_t.node.next = r_leaf_t                          # Link leaves
        return l_leaf_t, r_leaf_t
    
    def _insert_empty(self, x_item: Item, rank: int) -> 'GPlusTree':
        """Build the initial tree structure depending on rank."""
        # Single-level leaf
        if rank == 1:
            leaf_set = self._make_leaf_klist(x_item)
            self.node = GPlusNode(rank, leaf_set, GPlusTree())
            return self

        # Higher-level root with two linked leaf children
        l_leaf_t, r_leaf_t = self._make_leaf_trees(x_item)
        root_set = KList().insert(DUMMY_ITEM, GPlusTree())
        root_set = root_set.insert(_create_replica(x_item.key), l_leaf_t)
        self.node = GPlusNode(rank, root_set, r_leaf_t)
        return self
    
    def _insert_non_empty(self, x_item: Item, rank: int) -> GPlusTree:
        """Descend, preserve parent context for rank mismatches, then update or
        insert."""
        cur = self
        parent = None  # previously visited node (tree).
        p_next_entry = None  # Next larger item than x_item in parent node.

        # find the correct position for insertion.
        while cur.node.rank > rank:
            res = cur.node.set.retrieve(x_item.key)
            parent = cur
            if res.next_entry:
                p_next_entry = res.next_entry
                cur = res.next_entry.left_subtree
            else:
                p_next_entry = None
                cur = cur.node.right_subtree

        # Check for unnfolding an in-between node or creating a new root.
        if cur.node.rank < rank:
            cur = self._handle_rank_mismatch(cur, parent, p_next_entry, rank)

        res = cur.node.set.retrieve(x_item.key)
        if res.found_entry:
            return self._update_existing_item(cur, x_item)
        return self._insert_new_item(cur, x_item, res.next_entry)
    
    def _insert_non_empty(self, x_item: Item, rank: int) -> GPlusTree:
        cur = self
        parent: Optional[GPlusTree] = None
        p_next_entry: Optional[Entry] = None

        # Descend to node with rank <= target
        while cur.node.rank > rank:
            node = cur.node
            res = node.set.retrieve(x_item.key)
            parent = cur
            next_entry = res.next_entry
            if next_entry:
                p_next_entry = next_entry
                cur = next_entry.left_subtree
            else:
                p_next_entry = None
                cur = node.right_subtree

        if cur.node.rank < rank:
            cur = self._handle_rank_mismatch(cur, parent, p_next_entry, rank)

        # Check if existing item must be updated or a new item inserted
        node = cur.node
        res = node.set.retrieve(x_item.key)
        return (
            self._update_existing_item(cur, x_item)
            if res.found_entry
            else self._insert_new_item(cur, x_item, res.next_entry)
        )

    def _handle_rank_mismatch(
        self,
        cur: GPlusTree,
        parent: GPlusTree,
        p_next: Entry,
        rank: int
    ) -> GPlusTree:
        """
        If the current node's rank < rank, we need to create or unfold a 
        node to match the new rank.
        This is done by creating a new G+-node and linking it to the parent.
        Attributes:
            cur (GPlusTree): The current G+-tree.
            parent (GPlusTree): The parent G+-tree.
            p_next (tuple): The next entry in the parent tree.
            rank (int): The rank to match.
        Returns:
            GPlusTree: The updated G+-tree.
        """
        if parent is None:
            # We are at the root. Create a new root with dummy.
            old_node = self.node
            root_set = KList().insert(DUMMY_ITEM, GPlusTree())
            self.node = GPlusNode(rank, root_set, GPlusTree(old_node))
            return self
        else:
            # Unfold intermediate node between parent and current
            # Set replica of the current node's min as first entry.
            min_entry = cur.node.set.get_min().found_entry
            min_replica = _create_replica(min_entry.item.key)
            new_set = KList().insert(min_replica, GPlusTree())
            new_tree = GPlusTree(
                GPlusNode(rank, new_set, cur)
            )
            if p_next:
                parent.node.set = parent.node.set.update_left_subtree(
                    p_next.item.key, new_tree
                )
            else:
                parent.node.right_subtree = new_tree

        return new_tree

    def _update_existing_item(self, cur: GPlusTree, new_item: Item) -> GPlusTree:
        """
        Traverse to leaf (rank==1) and update the entry in-place.
        """
        key = new_item.key
        # Descend to leaf level
        while True:
            node = cur.node
            if node.rank == 1:
                entry = node.set.retrieve(key).found_entry
                if entry:
                    entry.item.value = new_item.value
                return self
            _, next_entry = node.set.retrieve(key)
            cur = next_entry.left_subtree if next_entry else node.right_subtree
        
    
    def _insert_new_item(
        self,
        cur: 'GPlusTree',
        x_item: Item,
        next_entry: Entry
    ) -> 'GPlusTree':
        """
        Insert a new item key. For internal nodes, we only store the key. 
        For leaf nodes, we store the full item.
        Attributes:
            cur (GPlusTree): The current G+-tree.
            x_item (Item): The item to be inserted.
            next_entry (tuple): The next entry in the tree.
        Returns:
            GPlusTree: The updated G+-tree.
        """
        x_key = x_item.key
        replica = _create_replica(x_key)

        # Parent tracking
        p_right: GPlusTree | None = None          # Temporary right parent
        p_r_next_key = None   # Next larger entry than x_item in parent node
        p_left = None           # Temporary left parent
        p_l_x_key = False      # Inserted item in left parent
        p_l_next_entry = None
        

        while True:
            node = cur.node
            is_leaf = node.rank == 1
            # Prepare insert object (replica if not a leaf)
            insert_obj = x_item if is_leaf else replica

            if p_right is None:
                subtree = (
                    next_entry.left_subtree 
                    if next_entry else node.right_subtree
                )
                node.set = node.set.insert(insert_obj, subtree)
                   
                # Prepare for possible next iteration
                p_right = cur
                p_r_next_key = p_l_next_key = (
                    next_entry.item.key if next_entry else None
                )
                p_left = cur
                p_l_x_key = x_key
                cur = subtree
            else:
                
                res = node.set.retrieve(x_key)
                next_entry = res.next_entry

                # We need to split the current node at x_item.key
                left_split, _, right_split = node.set.split_inplace(x_key)

                # res = p_right.node.set.retrieve(x_item.key)
                # p_r_next_entry = res.next_entry
                # p_r_x_entry = res.found_entry

                # res = p_left.node.set.retrieve(x_item.key)
                # p_l_x_entry = res.found_entry

                # Right side: if it has data or is a leaf, form a new tree node
                if right_split.item_count() == 0 and not is_leaf:
                    next_p_right = p_right
                    next_p_r_next_key = p_r_next_key
                else:
                    right_split = right_split.insert(
                        insert_obj, GPlusTree()
                    )
                    
                    new_tree = GPlusTree(
                        GPlusNode(
                            node.rank,
                            right_split,
                            node.right_subtree
                        )
                    )

                    # Update the parent's reference
                    if p_r_next_key is not None:
                        p_right.node.set = p_right.node.set.update_left_subtree(
                            p_r_next_key, new_tree)
                    else:
                        p_right.node.right_subtree = new_tree

                    next_p_right = new_tree
                    next_p_r_next_key = next_entry.item.key if next_entry else None

                p_right = next_p_right
                p_r_next_key = next_p_r_next_key
                
                if left_split.item_count() == 1 and not is_leaf:
                    # Collapse this node and reassign subtrees
                    if next_entry is not None:
                        new_subtree = next_entry.left_subtree

                    if p_l_x_key:
                        p_left.node.set = p_left.node.set.update_left_subtree(
                            x_key, new_subtree
                        )
                        # next_parent_left_x_entry = None
                    else:
                        p_left.node.right_subtree = new_subtree

                    next_p_left = p_left
                    next_p_l_x_key = p_l_x_key
                    new_cur = new_subtree
                    
                else:
                    # next_parent_left_x_entry = parent_left_x_entry
                    # Make the left split the current node's set
                    cur.node.set = left_split
                    if next_entry:
                        cur.node.right_subtree = next_entry.left_subtree

                    if p_l_x_key:
                        p_left.node.set = p_left.node.set.update_left_subtree(
                            x_key, cur
                        )
                    
                    # make the current node the left parent
                    next_p_left = cur

                    # left split never contains x_item
                    next_p_l_x_key = False   
                    new_cur = cur.node.right_subtree
                    
                
                p_left = next_p_left
                p_l_x_key = next_p_l_x_key

                if is_leaf:
                    # If leaf, link 'next' references if needed
                    new_tree.node.next = cur.node.next
                    cur.node.next = new_tree
                cur = new_cur

            # Descend further if it’s not a leaf, otherwise we’re done
            if is_leaf:
                return self

            if cur.is_empty():
                raise RuntimeError("Expected non-empty tree after insertion loop iteration where next tree node is not a leaf.")

            res = cur.node.set.retrieve(x_item.key)
            next_entry = res.next_entry



    def _insert_new_item_try(
        self,
        cur: GPlusTree,
        x_item: Item,
        next_entry: Optional[Entry],
    ) -> GPlusTree:
        """Insert *x_item* beginning at *cur* without altering core semantics.

        Key optimisation points (relative to the original implementation):
        • **One-time replica** – we call ``_create_replica`` exactly once.
        • **Single retrieve per level** – we never call ``retrieve`` on the same
          key twice in one iteration.
        • **Minimal allocations** – left-split re-uses the current node; a
          right-sibling tree is created only if the right split is non‑empty.
        • **No runtime ``visited`` set** – the descent is guaranteed finite by
          rank-decrease, so we drop the defensive set and its hash overhead.
        """
        replica = _create_replica(x_item.key)

        # Parent context (right‑side & left‑side) needed for back‑links.
        p_right: GPlusTree | None = None
        p_right_next: Optional[Entry] = next_entry
        p_left: GPlusTree | None = cur  # initial left parent is *cur*
        first_iter = True


        while True:
            node = cur.node
            is_leaf = node.rank == 1
            insert_obj = x_item if is_leaf else replica

            if p_right is None:
                # first iteration
                next_entry = node.set.retrieve(x_item.key).next_entry
                subtree = (
                    next_entry.left_subtree
                    if next_entry else node.right_subtree
                )
                node.set = node.set.insert(insert_obj, subtree)
            else:
                # split current node around x_item
                left_split, _, right_split = node.set.split_inplace(x_item.key)

                # Build a right sibling only if split produced real content.
                sibling: GPlusTree | None = None
                if right_split.item_count():
                    right_split = right_split.insert(replica, GPlusTree())
                    sibling = GPlusTree(GPlusNode(node.rank, right_split, node.right_subtree))

                    if p_right_next is not None:
                        p_right.node.set = p_right.node.set.update_left_subtree(p_right_next.item.key, sibling)
                    else:
                        p_right.node.right_subtree = sibling

                # Current node keeps the left split only.
                node.set = left_split

            if is_leaf:
                # Maintain leaf "next" chain once, then finish.
                if p_right_next is not None:
                    subtree.node.next = node.next      # type: ignore[attr-defined]
                    node.next = subtree                # type: ignore[attr-defined]
                return self

            

            # ---------- 3. ADVANCE DOWN ONE LEVEL ------------------------
            p_right = sibling if sibling is not None else p_right or cur
            cur = subtree  # move to the child we just inserted under
            p_right_next = cur.node.set.retrieve(x_item.key).next_entry
            p_left = cur  # left parent becomes the subtree for next iter
    


    def _insert_new_item_funktional(
        self,
        cur: 'GPlusTree',
        x_item: Item,
        next_entry: Entry
    ) -> 'GPlusTree':
        """
        Insert a new item key. For internal nodes, we only store the key. 
        For leaf nodes, we store the full item.
        Attributes:
            cur (GPlusTree): The current G+-tree.
            x_item (Item): The item to be inserted.
            next_entry (tuple): The next entry in the tree.
        Returns:
            GPlusTree: The updated G+-tree.
        """
        replica = _create_replica(x_item.key)

        # Parent tracking
        p_right: GPlusTree | None = None          # Temporary right parent
        p_r_next_entry = None   # Next larger entry than x_item in parent node
        p_left = None           # Temporary left parent
        p_l_x_entry = None      # Inserted item in left parent
        

        while True:
            node = cur.node
            is_leaf = node.rank == 1
            # Prepare insert object (replica if not a leaf)
            insert_obj = x_item if is_leaf else replica

            if p_right is None:
                subtree = (
                    next_entry.left_subtree 
                    if next_entry else node.right_subtree
                )
                node.set = node.set.insert(insert_obj, subtree)
                   
                # Prepare for possible next iteration
                p_right = cur
                p_r_next_entry = next_entry
                p_left = cur                
                cur = subtree
            else:
                res = p_right.node.set.retrieve(x_item.key)
                p_r_next_entry = res.next_entry
                p_r_x_entry = res.found_entry

                res = p_left.node.set.retrieve(x_item.key)
                p_l_x_entry = res.found_entry
                
                # We need to split the current node at x_item.key
                (
                    left_split,
                    x_item_left_subtree,
                    right_split
                ) = node.set.split_inplace(x_item.key)

                if x_item_left_subtree is not None:
                    raise RuntimeError("Expected non-empty left subtree during split operation. This indicates that the item to insert is already present in the tree with a lower rank.")

                # Right side: if it has data or is a leaf, form a new tree node
                if right_split.item_count() == 0 and not is_leaf:
                    next_p_right = p_right
                else:
                    right_split = right_split.insert(
                        insert_obj, GPlusTree()
                    )
                    
                    if right_split.item_count() == 0:
                        raise RuntimeError("Expected non-empty right split after insert operation.")
                    
                    new_tree = GPlusTree(
                        GPlusNode(
                            node.rank,
                            right_split,
                            node.right_subtree
                        )
                    )

                    if new_tree.node.set.item_count() == 0:
                        raise RuntimeError("Expected non-empty new tree after insert operation.")

                    # Update the parent's reference
                    if p_r_next_entry:
                        p_right.node.set = p_right.node.set.update_left_subtree(
                            p_r_next_entry.item.key, new_tree)
                    else:
                        p_right.node.right_subtree = new_tree

                    next_p_right = new_tree

                p_right = next_p_right

                # Reuse the left split in the current node
                if p_r_x_entry is None:
                    print("\n\nWARNING: In subsequent iteration the insert item schould be present in the parent right tree node.")
                
                if left_split.item_count() == 0:
                    raise RuntimeError("Always expect non-empty left split.")
                
                if left_split.item_count() == 1 and not is_leaf:
                    # Collapse this node and reassign subtrees
                    if next_entry:
                        new_subtree = next_entry.left_subtree
                    else:
                        raise RuntimeError("The node contained a single item before insertion. This should not happen here.")

                    if p_l_x_entry is not None:
                        p_left.node.set = p_left.node.set.update_left_subtree(
                            p_r_x_entry.item.key, new_subtree
                        )
                        # next_parent_left_x_entry = None
                    else:
                        p_left.node.right_subtree = new_subtree

                    next_p_left = p_left
                    new_cur = new_subtree
                    
                else:
                    # next_parent_left_x_entry = parent_left_x_entry
                    # Continue with the left split at curent level
                    cur.node.set = left_split
                    if next_entry:
                        cur.node.right_subtree = next_entry.left_subtree

                    if p_l_x_entry is not None:
                        p_left.node.set = p_left.node.set.update_left_subtree(
                            p_l_x_entry.item.key, cur
                        )
                    next_p_left = cur
                    new_cur = cur.node.right_subtree
                
                p_left = next_p_left

                if is_leaf:
                    # If leaf, link 'next' references if needed
                    new_tree.node.next = cur.node.next
                    cur.node.next = new_tree
                cur = new_cur

            # Descend further if it’s not a leaf, otherwise we’re done
            if is_leaf:
                return self

            if cur.is_empty():
                raise RuntimeError("Expected non-empty tree after insertion loop iteration where next tree node is not a leaf.")

            res = cur.node.set.retrieve(x_item.key)
            next_entry = res.next_entry


    

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


