"""G+-tree base implementation"""

from __future__ import annotations
import logging
from typing import Dict, Optional, Tuple, Any, Type
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
from gplus_trees.klist_base import KListBase
from gplus_trees.profiling import (
    track_performance,
    PerformanceTracker
)

# Configure logging
logger = logging.getLogger(__name__)
if not logger.handlers:
    # Avoid duplicated handlers when module is imported multiple times
    handler = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)

t = Type["GPlusTreeBase"]

# Constants
DUMMY_KEY = int("0" * 64, 16)
DUMMY_VALUE = None
DUMMY_ITEM = Item(DUMMY_KEY, DUMMY_VALUE)

DEBUG = False

class GPlusNodeBase:
    """
    Base class for G+-tree nodes. Factory will set:
      - SetClass  : which AbstractSetDataStructure to use for entries
      - TreeClass : which GPlusTree to build for child/subtree pointers
    """
    __slots__ = ("rank", "set", "right_subtree", "next")

    # these two get injected by your factory.py
    SetClass: Type[AbstractSetDataStructure]
    TreeClass: Type[GPlusTreeBase]

    def __init__(
        self,
        rank: int,
        set: AbstractSetDataStructure,
        right: Optional[GPlusTreeBase] = None
    ) -> None:
        if rank <= 0:
            raise ValueError("rank must be > 0")
        self.rank = rank
        self.set = set
        self.right_subtree = right if right is not None else self.TreeClass()
        self.next = None    # leaf‐chain pointer
    
class GPlusTreeBase(AbstractSetDataStructure):
    """
    A G+-tree is a recursively defined structure that is either empty or contains a single G+-node.
    Attributes:
        node (Optional[GPlusNode]): The G+-node that the tree contains. If None, the tree is empty.
    """
    __slots__ = ("node",)
    
    # Will be set by the factory
    NodeClass: Type[GPlusNodeBase]
    SetClass: Type[AbstractSetDataStructure]
    
    def __init__(self, node: Optional[GPlusNodeBase] = None):
        self.node: Optional[GPlusNodeBase] = node

    @classmethod
    def from_root(cls: Type[t], root_node: GPlusNodeBase) -> t:
        """
        Create a new tree instance wrapping an existing node.
        """
        tree = cls.__new__(cls)
        tree.node = root_node
        return tree

    # @track_performance
    def is_empty(self) -> bool:
        return self.node is None
    
    def __str__(self):
        return "Empty GPlusTree" if self.is_empty() else f"GPlusTree(node={self.node})"

    __repr__ = __str__
    
    # Public API
    # @track_performance
    def insert(self, x: Item, rank: int) -> GPlusTreeBase:
        """
        Public method (average-case O(log n)): Insert an item into the G+-tree. 
        If the item already exists, updates its value at the leaf node.
        
        Args:
            x_item (Item): The item (key, value) to be inserted.
            rank (int): The rank of the item. Must be a natural number > 0.
        Returns:
            GPlusTreeBase: The updated G+-tree.

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
    
    # @track_performance
    def retrieve(
        self, key: int
    ) -> RetrievalResult:
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
                next_pair (Tuple[Optional[Item], Optional[GPlusTreeBase]]):
                    The next item in sorted order and its associated subtree, or (None, None).
        """
        current_tree = self
        found_entry: Optional[Entry] = None
        next_entry: Optional[Entry] = None

        while not current_tree.is_empty():
            node = current_tree.node
            # Attempt to retrieve in this node's k-list
            res = node.set.retrieve(key)
            found_entry = res.found_entry
            next_entry = res.next_entry

            # Descend based on presence of next_entry
            if next_entry is not None:
                subtree = next_entry.left_subtree
                current_tree = subtree
            else:
                # If leaf has a linked next node, update next_entry
                if node.next is not None:
                    next_entry = node.next.node.set.get_min().found_entry
                current_tree = node.right_subtree

        return RetrievalResult(found_entry, next_entry)
    
    def delete(self, item):
        raise NotImplementedError("delete not implemented yet")

    # Private Methods
    def _make_leaf_klist(self, x_item: Item) -> AbstractSetDataStructure:
        """Builds a KList for a single leaf node containing the dummy and x_item."""
        TreeClass = type(self)
        SetClass = self.SetClass
        
        # start with a fresh empty set of entries
        leaf_set = SetClass()
        
        # insert the dummy entry, pointing at an empty subtree
        leaf_set = leaf_set.insert(DUMMY_ITEM, TreeClass())
        
        # now insert the real item, also pointing at an empty subtree
        leaf_set = leaf_set.insert(x_item, TreeClass())
        
        return leaf_set
    
    def _make_leaf_trees(self, x_item) -> Tuple[GPlusTreeBase, GPlusTreeBase]:
        """
        Builds two linked leaf-level GPlusTreeBase nodes for x_item insertion.
        and returns the corresponding G+-trees.
        """
        TreeK = type(self)
        NodeK = self.NodeClass
        SetK = self.SetClass

        # Build right leaf
        right_set = SetK()
        right_set = right_set.insert(x_item, TreeK())
        right_node = NodeK(1, right_set, TreeK())
        right_leaf = TreeK.from_root(right_node)

        # Build left leaf with dummy entry
        left_set = SetK()
        left_set = left_set.insert(DUMMY_ITEM, TreeK())
        left_node = NodeK(1, left_set, TreeK())
        left_leaf = TreeK.from_root(left_node)

        # Link leaves
        left_leaf.node.next = right_leaf
        return left_leaf, right_leaf
    
    def _insert_empty(self, x_item: Item, rank: int) -> GPlusTreeBase:
        """Build the initial tree structure depending on rank."""
        logger.debug(f"_insert_empty called with item: {x_item}, rank: {rank}")
        logger.debug(f"Using tree class: {type(self).__name__}")
        logger.debug(f"NodeClass: {self.NodeClass.__name__}")
        logger.debug(f"SetClass: {self.SetClass.__name__}")
        
        # Single-level leaf
        if rank == 1:
            leaf_set = self._make_leaf_klist(x_item)
            logger.debug(f"Created leaf set of type: {type(leaf_set).__name__}")
            self.node = self.NodeClass(rank, leaf_set, type(self)())
            logger.debug(f"Created node: rank={rank}, set type={type(leaf_set).__name__}, node type={type(self.node).__name__}")
            # logger.debug(f"Tree after empty insert rank 1:\n{self.print_structure()}")
            return self

        # Higher-level root with two linked leaf children
        l_leaf_t, r_leaf_t = self._make_leaf_trees(x_item)
        logger.debug(f"Created leaf trees: left={type(l_leaf_t).__name__}, right={type(r_leaf_t).__name__}")
        root_set = self.SetClass().insert(DUMMY_ITEM, type(self)())
        root_set = root_set.insert(_create_replica(x_item.key), l_leaf_t)
        self.node = self.NodeClass(rank, root_set, r_leaf_t)
        logger.debug(f"Created root node: rank={rank}, set type={type(root_set).__name__}, node type={type(self.node).__name__}")
        logger.debug(f"Tree after empty insert rank > 1:\n{self.print_structure()}")
        return self

    # @track_performance
    def _insert_non_empty(self, x_item: Item, rank: int) -> GPlusTreeBase:
        """Optimized version for inserting into a non-empty tree."""
        cur = self
        parent = None
        p_next_entry = None

        # Loop until we find where to insert
        while True:
            node = cur.node
            node_rank = node.rank  # Cache attribute access
            
            # Case 1: Found node with matching rank - ready to insert
            if node_rank == rank:
                # Only retrieve once
                res = node.set.retrieve(x_item.key)
                
                # Fast path: update existing item
                if res.found_entry:
                    # Direct update for leaf nodes (common case)
                    if rank == 1:
                        res.found_entry.item.value = x_item.value
                        return self
                    return self._update_existing_item(cur, x_item)
                
                # Insert new item
                return self._insert_new_item(cur, x_item, res.next_entry)

            # Case 2: Current rank too small - handle rank mismatch
            if node_rank < rank:
                cur = self._handle_rank_mismatch(cur, parent, p_next_entry, rank)
                continue

            # Case 3: Descend to next level (current rank > rank)
            res = node.set.retrieve(x_item.key)
            parent = cur
            
            # Cache the next_entry to avoid repeated access
            next_entry = res.next_entry
            if next_entry:
                p_next_entry = next_entry
                cur = next_entry.left_subtree
            else:
                p_next_entry = None
                cur = node.right_subtree

    def _handle_rank_mismatch(
        self,
        cur: GPlusTreeBase,
        parent: GPlusTreeBase,
        p_next: Entry,
        rank: int
    ) -> GPlusTreeBase:
        """
        If the current node's rank < rank, we need to create or unfold a 
        node to match the new rank.
        This is done by creating a new G+-node and linking it to the parent.
        Attributes:
            cur (GPlusTreeBase): The current G+-tree.
            parent (GPlusTreeBase): The parent G+-tree.
            p_next (tuple): The next entry in the parent tree.
            rank (int): The rank to match.
        Returns:
            GPlusTreeBase: The updated G+-tree.
        """
        TreeClass = type(self)
        
        if parent is None:
            # create a new root node
            old_node = self.node
            root_set = self.SetClass().insert(DUMMY_ITEM, TreeClass())
            self.node = self.NodeClass(rank, root_set, TreeClass(old_node))
            return self

        # Unfold intermediate node between parent and current
        # Set replica of the current node's min as first entry.
        min_entry = cur.node.set.get_min().found_entry
        min_replica = _create_replica(min_entry.item.key)
        new_set = self.SetClass().insert(min_replica, TreeClass())
        new_tree = TreeClass()
        new_tree.node = self.NodeClass(rank, new_set, cur)
        
        if p_next:
            p_next.left_subtree = new_tree
        else:
            parent.node.right_subtree = new_tree

        return new_tree

    # @track_performance
    def _update_existing_item(
        self, cur: GPlusTreeBase, new_item: Item
    ) -> GPlusTreeBase:
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
        
    # @track_performance
    def _insert_new_item(
        self,
        cur: 'GPlusTreeBase',
        x_item: Item,
        next_entry: Entry
    ) -> 'GPlusTreeBase':
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
        TreeClass = type(self)

        # Parent tracking variables
        right_parent = None    # Parent node for right-side updates
        right_entry = None     # Entry in right parent points to current subtree
        left_parent = None     # Parent node for left-side updates
        left_x_entry = None    # x_item stored in left parent
        
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
                
                # Assign parent tracking for next iteration
                right_parent = left_parent = cur
                right_entry = next_entry if next_entry else None
                left_x_entry = node.set.retrieve(x_key).found_entry
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
                    right_split = right_split.insert(insert_obj, TreeClass())
                    new_tree = TreeClass()
                    new_tree.node = self.NodeClass(node.rank, right_split, node.right_subtree)

                    # Update parent reference to the new tree
                    if right_entry is not None:
                        right_entry.left_subtree = new_tree
                    else:
                        right_parent.node.right_subtree = new_tree

                    # Update right parent tracking
                    next_right_parent = new_tree
                    next_right_entry = next_entry if next_entry else None
                else:
                    # Keep existing parent references
                    next_right_parent = right_parent
                    next_right_entry = right_entry

                # Update right parent variables for next iteration
                right_parent = next_right_parent
                right_entry = next_right_entry
                
                # --- Handle left side of the split ---
                # Determine if we need to create/update using left split
                if left_split.item_count() > 1 or is_leaf:
                    # Update current node to use left split
                    cur.node.set = left_split
                    if next_entry:
                        cur.node.right_subtree = next_entry.left_subtree

                    # Update parent reference if needed
                    if left_x_entry is not None:
                        left_x_entry.left_subtree = cur
                    
                    # Make current node the new left parent
                    next_left_parent = cur
                    next_left_x_entry = None  # Left split never contains x_item
                    next_cur = cur.node.right_subtree
                else:
                    # Collapse single-item nodes for non-leaves
                    new_subtree = (
                        next_entry.left_subtree if next_entry else TreeClass()
                    )
                    
                    # Update parent reference
                    if left_x_entry is not None:
                        left_x_entry.left_subtree = new_subtree
                    else:
                        left_parent.node.right_subtree = new_subtree

                    # Prepare for next iteration
                    next_left_parent = left_parent
                    next_left_x_entry = left_x_entry
                    next_cur = new_subtree
                
                # Update left parent variables for next iteration
                left_parent = next_left_parent
                left_x_entry = next_left_x_entry

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
        while current.node.rank > 1:
            result = current.node.set.get_min()
            if result.next_entry is not None:
                current = result.next_entry.left_subtree
            else:
                current = current.node.right_subtree

        # At this point, current is the leftmost leaf-level GPlusTreeBase
        while current is not None and not current.is_empty():
            yield current.node
            current = current.node.next
    
    # @track_performance
    def physical_height(self) -> int:
        """
        The “real” pointer-follow height of the G⁺-tree:
        –  the number of KListNode segments in this node’s k-list, plus
        –  the maximum physical_height() of any of its subtrees.
        """
        node = self.node
        base = node.set.physical_height()

        # If this is a leaf node, return the base height
        if node.rank == 1:
            return base

        # Find the tallest child among all left_subtrees and the right_subtree
        max_child = 0
        for entry in node.set:
            left = entry.left_subtree
            if left is not None and not left.is_empty():
                max_child = max(max_child, left.physical_height())
        if node.right_subtree is not None and not node.right_subtree.is_empty():
            max_child = max(max_child, node.right_subtree.physical_height())

        # total physical height = this node’s chain length + deepest child
        return base + max_child

    # @track_performance
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
    
    # Add a method to get performance metrics
    @classmethod
    def get_performance_report(cls, sort_by: str = 'total_time') -> str:
        """
        Get a formatted report of performance metrics.
        
        Args:
            sort_by: Field to sort by ('total_time', 'avg_time', 'call_count', etc.)
            
        Returns:
            str: Formatted performance report
        """
        return PerformanceTracker.get_instance().report(sort_by=sort_by)
    
    @classmethod
    def reset_performance_metrics(cls) -> None:
        """Reset all performance metrics."""
        PerformanceTracker.get_instance().reset()
    
    @classmethod
    def enable_performance_tracking(cls) -> None:
        """Enable performance tracking."""
        PerformanceTracker.get_instance().enable()
    
    @classmethod
    def disable_performance_tracking(cls) -> None:
        """Disable performance tracking."""
        PerformanceTracker.get_instance().disable()

@dataclass
class Stats:
    gnode_height: int
    gnode_count: int
    item_count: int
    real_item_count: int
    item_slot_count: int
    leaf_count: int
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

def gtree_stats_(t: GPlusTreeBase,
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

    # ---------- empty tree return ---------------------------------
    if t is None or t.is_empty():
        return Stats(gnode_height        = 0,
                     gnode_count         = 0,
                     item_count          = 0,
                     real_item_count     = 0,
                     item_slot_count     = 0,
                     leaf_count          = 0,
                     rank                = -1,
                     is_heap             = True,
                     least_item          = None,
                     greatest_item       = None,
                     is_search_tree      = True,
                     internal_has_replicas = True,
                     internal_packed     = True,
                     linked_leaf_nodes   = True,
                     all_leaf_values_present = True,
                     leaf_keys_in_order  = True,)

    node       = t.node
    node_set   = node.set
    node_rank = node.rank
    node_item_count = node_set.item_count()
    rank_hist[node_rank] = rank_hist.get(node_rank, 0) + node_set.item_count()

    # ---------- recurse on children ------------------------------------
    child_stats = [gtree_stats_(e.left_subtree, rank_hist, False) for e in node_set]
    right_stats = gtree_stats_(node.right_subtree,   rank_hist, False)

    # ---------- aggregate ----------------------------------
    # Initialize with default values for the current node
    stats = Stats(
        gnode_height=0,
        gnode_count=0,
        item_count=0,
        real_item_count=0,
        item_slot_count=0,
        leaf_count=0,
        rank=node_rank,
        is_heap=True,
        least_item=None,
        greatest_item=None,
        is_search_tree=True,
        internal_has_replicas=True,
        internal_packed=(node_rank <= 1 or node_item_count > 1),
        linked_leaf_nodes=True,
        all_leaf_values_present=True,
        leaf_keys_in_order=True,
    )
    
    # Precompute common values using right subtree stats
    stats.gnode_count     = 1 + right_stats.gnode_count
    stats.item_count      = node_item_count + right_stats.item_count
    stats.real_item_count += right_stats.real_item_count
    stats.item_slot_count = node_set.item_slot_count() + right_stats.item_slot_count
    stats.leaf_count += right_stats.leaf_count
    # stats.gnode_height    = 1 + max(right_stats.gnode_height,
    #                                 max((cs.gnode_height for cs in child_stats), default=0))

    max_child_height = 0

    # Single pass over children to fold in all counts and booleans
    prev_key = None
    for entry, cs in zip(node_set, child_stats):
        current_key = entry.item.key
        
        max_child_height = max(max_child_height, cs.gnode_height)

        if node_rank >= 2 and entry.item.value is not None:
            stats.internal_has_replicas = False
            # print(f"Internal node item is not a replica (rank {node.rank}): {current_key} -> {entry.item.value}")
            # print(f"Node: {node.set!r}")
        
        # Accumulate counts for common values
        stats.gnode_count += cs.gnode_count
        stats.item_count += cs.item_count
        stats.item_slot_count += cs.item_slot_count
        stats.leaf_count += cs.leaf_count
        stats.real_item_count += cs.real_item_count

        # Update boolean flags
        if stats.is_heap and not ((node_rank > cs.rank) and cs.is_heap):
            stats.is_heap = False
            
        stats.internal_has_replicas &= cs.internal_has_replicas
        stats.internal_packed &= cs.internal_packed
        stats.linked_leaf_nodes &= cs.linked_leaf_nodes
        
        # Check search tree property
        if stats.is_search_tree:
            if not cs.is_search_tree:
                stats.is_search_tree = False
            elif prev_key is not None:
                if prev_key >= current_key or (cs.least_item and cs.least_item.key < prev_key):
                    stats.is_search_tree = False
                elif cs.greatest_item and cs.greatest_item.key >= current_key:
                    stats.is_search_tree = False
        
        prev_key = current_key
    
    # Calculate final height
    stats.gnode_height = 1 + max(right_stats.gnode_height, max_child_height)

    # Fold in right subtree flags
    if stats.is_heap and not right_stats.is_heap:
        stats.is_heap = False
    
    if stats.is_search_tree:
        if not right_stats.is_search_tree:
            stats.is_search_tree = False
        elif right_stats.least_item and prev_key is not None and right_stats.least_item.key < prev_key:
            stats.is_search_tree = False


    stats.is_search_tree       &= right_stats.is_search_tree
    if right_stats.least_item and right_stats.least_item.key < prev_key:
        stats.is_search_tree = False

    stats.internal_has_replicas &= right_stats.internal_has_replicas
    stats.internal_packed       &= right_stats.internal_packed
    stats.linked_leaf_nodes     &= right_stats.linked_leaf_nodes

    # ----- LEAST / GREATEST -----
    if child_stats and child_stats[0].least_item is not None:
        stats.least_item = child_stats[0].least_item
    else:
        stats.least_item = node_set.get_entry(0).found_entry.item

    if right_stats.greatest_item is not None:
        stats.greatest_item = right_stats.greatest_item
    else:
        stats.greatest_item = node_set.get_entry(node_item_count - 1).found_entry.item

    # ---------- leaf walk ONCE at the root -----------------------------
    if node_rank == 1:          # leaf node: base values
        # Count non-dummy items
        true_count = 0
        all_values_present = True
        
        for entry in node_set:
            item = entry.item
            if item.key != DUMMY_KEY:
                true_count += 1
                if item.value is None:
                    all_values_present = False

        stats.all_leaf_values_present = all_values_present
        stats.real_item_count = true_count
        stats.leaf_count = 1

    # Root-level validation (only occurs once)
    if _is_root:         # root call (only true once)
        leaf_keys, leaf_values = [], []
        leaf_count, item_count = 0, 0
        last_leaf, prev_key = None, None
        keys_in_order = True
        for leaf in t.iter_leaf_nodes():
            last_leaf = leaf
            leaf_count += 1
            
            for entry in leaf.set:
                item = entry.item
                if item == DUMMY_ITEM:
                    continue

                item_count += 1
                key = item.key

                # Check orderin
                if prev_key is not None and key < prev_key:
                    keys_in_order = False
                prev_key = key

                leaf_keys.append(key)
                leaf_values.append(item.value)

        # Set values from leaf traversal
        stats.leaf_keys_in_order = keys_in_order

        # Check leaf_count and real_item_count consistency
        if leaf_count != stats.leaf_count or item_count != stats.real_item_count:
            stats.linked_leaf_nodes = False
            stats.leaf_count = max(leaf_count, stats.leaf_count)
            stats.real_item_count = max(item_count, stats.real_item_count)
        elif last_leaf is not None:
            # Check if greatest item matches last leaf's greatest item
            last_count = last_leaf.set.item_count()
            last_item = last_leaf.set.get_entry(last_count - 1).found_entry.item
            if stats.greatest_item is not last_item:
                stats.linked_leaf_nodes = False

    return stats

# @track_performance
def collect_leaf_keys(tree: 'GPlusTreeBase') -> list[str]:
    out = []
    for leaf in tree.iter_leaf_nodes():
        for e in leaf.set:
            if e.item.key != DUMMY_KEY:
                out.append(e.item.key)
    return out


