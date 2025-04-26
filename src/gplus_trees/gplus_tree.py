"""G+-tree implementation"""

from __future__ import annotations
from typing import Dict, Optional, Tuple, Any
from dataclasses import dataclass, asdict
from pprint import pprint
import collections

from gplus_trees.base import (
    AbstractSetDataStructure,
    Item,
    Entry,
    _create_replica,
    RetrievalResult,
)
from gplus_trees.klist import KList

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
        cur = self
        parent: Optional[GPlusTree] = None
        p_next_entry: Optional[Entry] = None

        while True:
            node = cur.node
            if node.rank == rank:
                res = node.set.retrieve(x_item.key)
                if res.found_entry:
                    return self._update_existing_item(cur, x_item)
                return self._insert_new_item(cur, x_item, res.next_entry)

            if node.rank < rank:
                cur = self._handle_rank_mismatch(
                    cur, parent, p_next_entry, rank
                )
                continue

            # Descend to the next level
            res = node.set.retrieve(x_item.key)
            parent = cur
            if res.next_entry:
                p_next_entry = res.next_entry
                cur = res.next_entry.left_subtree
            else:
                p_next_entry = None
                cur = node.right_subtree

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
            # create a new root node
            old_node = self.node
            root_set = KList().insert(DUMMY_ITEM, GPlusTree())
            self.node = GPlusNode(rank, root_set, GPlusTree(old_node))
            return self

        # Unfold intermediate node between parent and current
        # Set replica of the current node's min as first entry.
        min_entry = cur.node.set.get_min().found_entry
        min_replica = _create_replica(min_entry.item.key)
        new_set = KList().insert(min_replica, GPlusTree())
        new_tree = GPlusTree(GPlusNode(rank, new_set, cur))
        if p_next:
            parent.node.set = parent.node.set.update_left_subtree(
                p_next.item.key, new_tree
            )
        else:
            parent.node.right_subtree = new_tree

        return new_tree

    def _update_existing_item(
        self, cur: GPlusTree, new_item: Item
    ) -> GPlusTree:
        """Traverse to leaf (rank==1) and update the entry in-place."""
        key = new_item.key
        while True:
            node = cur.node
            if node.rank == 1:
                entry = node.set.retrieve(key).found_entry
                if entry:
                    entry.item.value = new_item.value
                return self
            next = node.set.retrieve(key).next_entry
            cur = next.left_subtree if next else node.right_subtree
        

    def _insert_new_item(
        self,
        cur: 'GPlusTree',
        x_item: Item,
        next_entry: Entry
    ) -> 'GPlusTree':
        """
        Insert a new item key. For internal nodes, we only store the key. 
        For leaf nodes, we store the full item.
        
        Args:
            cur: The current G+-tree node where insertion starts
            x_item: The item to be inserted
            next_entry: The next entry in the tree relative to x_item
            
        Returns:
            The updated G+-tree
        """
        x_key = x_item.key
        replica = _create_replica(x_key)

        # Parent tracking variables
        right_parent = None    # Parent node for right-side updates
        right_key = None       # Key in right parent pointing to current subtree
        left_parent = None     # Parent node for left-side updates
        left_has_x = False     # Whether x_key is stored in left parent
        
        while True:
            node = cur.node
            is_leaf = node.rank == 1
            # Use correct item type based on node rank
            insert_obj = x_item if is_leaf else replica

            # First iteration - simple insert without splitting
            if right_parent is None:
                # Determine subtree for potential next iteration
                subtree = (
                    next_entry.left_subtree
                    if next_entry else node.right_subtree
                )   
                
                # Insert the item
                node.set = node.set.insert(insert_obj, subtree)

                # Early return if we're already at a leaf node
                if is_leaf:
                    return self
                
                # Update parent tracking for next iteration
                right_parent = left_parent = cur
                right_key = next_entry.item.key if next_entry else None
                left_has_x = True
                cur = subtree
            else:
                # Node splitting required - get updated next_entry
                res = node.set.retrieve(x_key)
                next_entry = res.next_entry

                # Split node at x_key
                left_split, _, right_split = node.set.split_inplace(x_key)

                # --- Handle right side of the split ---
                # Determine if we need a new tree for the right split
                if right_split.item_count() > 0 or is_leaf:
                    # Insert item into right split and create new tree
                    right_split = right_split.insert(insert_obj, GPlusTree())
                    new_tree = GPlusTree(
                        GPlusNode(node.rank, right_split, node.right_subtree)
                    )

                    # Update parent reference to the new tree
                    if right_key is not None:
                        right_parent.node.set = right_parent.node.set.update_left_subtree(
                            right_key, new_tree
                        )
                    else:
                        right_parent.node.right_subtree = new_tree

                    # Update right parent tracking
                    next_right_parent = new_tree
                    next_right_key = next_entry.item.key if next_entry else None
                else:
                    # Keep existing parent references
                    next_right_parent = right_parent
                    next_right_key = right_key

                # Update right parent variables for next iteration
                right_parent = next_right_parent
                right_key = next_right_key
                
                # --- Handle left side of the split ---
                # Determine if we need to create/update using left split
                if left_split.item_count() > 1 or is_leaf:
                    # Update current node to use left split
                    cur.node.set = left_split
                    if next_entry:
                        cur.node.right_subtree = next_entry.left_subtree

                    # Update parent reference if needed
                    if left_has_x:
                        left_parent.node.set = (
                            left_parent.node.set
                            .update_left_subtree(x_key, cur)
                        )
                    
                    # Make current node the new left parent
                    next_left_parent = cur
                    next_left_has_x = False  # Left split never contains x_item
                    next_cur = cur.node.right_subtree
                else:
                    # Collapse single-item nodes for non-leaves
                    new_subtree = (
                        next_entry.left_subtree if next_entry else GPlusTree()
                    )
                    
                    # Update parent reference
                    if left_has_x:
                        left_parent.node.set = (
                            left_parent.node.set
                            .update_left_subtree(x_key, new_subtree)
                        )
                    else:
                        left_parent.node.right_subtree = new_subtree

                    # Prepare for next iteration
                    next_left_parent = left_parent
                    next_left_has_x = left_has_x
                    next_cur = new_subtree
                
                # Update left parent variables for next iteration
                left_parent = next_left_parent
                left_has_x = next_left_has_x

                # Update leaf node 'next' pointers if at leaf level
                if is_leaf:
                    new_tree.node.next = cur.node.next
                    cur.node.next = new_tree
                    return self  # Early return when leaf is processed
                    
                # Continue to next iteration with updated current node
                cur = next_cur

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
        # while not current.is_empty() and current.node.rank > 1:
        while current.node.rank > 1:
            result = current.node.set.get_min()
            if result.next_entry is not None:
                current = result.next_entry.left_subtree
            else:
                current = current.node.right_subtree

        # At this point, current is the leftmost leaf-level GPlusTree
        while current is not None:
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


