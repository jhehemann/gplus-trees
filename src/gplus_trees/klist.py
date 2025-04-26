"""K-list implementation"""

from typing import TYPE_CHECKING, List, Optional, Tuple
import bisect

from gplus_trees.base import (
    Item,
    AbstractSetDataStructure,
    RetrievalResult,
    Entry
)

if TYPE_CHECKING:
    from gplus_trees.gplus_tree import GPlusTree

class KListNode:
    """
    A node in the k-list.

    Each node stores up to CAPACITY entries.
    Each entry is a tuple of the form:
        (item, left_subtree)
    where `left_subtree` is a G-tree associated with this entry.
    """
    CAPACITY = 4

    def __init__(self):
        self.entries: list[Entry] = []
        self.next: Optional['KListNode'] = None

    def insert_entry(
            self, 
            entry: Entry
    ) -> Optional[Entry]:
        """
        Inserts an entry into a sorted KListNode by key.
        If capacity exceeds, last entry is returned for further processing.
        
        Attributes:
            entry (Entry): The entry to insert into the KListNode.
        Returns:
            Optional[Entry]: The last entry if the node overflows; otherwise, None.
        """
        bisect.insort(self.entries, entry)  # nutzt nun __lt__-Logik
        if len(self.entries) > KListNode.CAPACITY:
            return self.entries.pop()
        return None
    
    def retrieve_entry(
        self, key: int
    ) -> Tuple[Optional[Entry], Optional[Entry], bool]:
        """
        Returns (found, in_node_successor, go_next):
         - found: the Entry with .item.key == key, or None
         - in_node_successor: the next Entry *within this node* if key < max_key
                              else None
         - go_next: True if key > max_key OR (found at max_key) → caller should
                    continue into next node to find the true successor.
        """
        entries = self.entries
        if not entries:
            return None, None, True

        keys = [e.item.key for e in entries]
        i = bisect.bisect_left(keys, key)

        # Case A: found exact
        if i < len(entries) and keys[i] == key:
            found = entries[i]
            # if not last in this node, return node-local successor
            if i + 1 < len(entries):
                return found, entries[i+1], False
            # otherwise we must scan next node
            return found, None, True

        # Case B: i < len → this entries[i] is the in-node successor
        if i < len(entries):
            return None, entries[i], False

        # Case C: key > max_key in this node → skip to next
        return None, None, True
    
    def get_by_offset(self, offset: int) -> Tuple[Entry, Optional[Entry], bool]:
        """
        offset: 0 <= offset < len(self.entries)
        Returns (entry, in_node_successor, needs_next_node)
        """
        entry = self.entries[offset]
        if offset + 1 < len(self.entries):
            return entry, self.entries[offset+1], False
        else:
            return entry, None, True


class KList(AbstractSetDataStructure):
    """
    A k-list implemented as a linked list of nodes.
    Each node holds up to CAPACITY (4) sorted entries.
    An entry is of the form (item, left_subtree), where left_subtree is a G+-tree (or None).
    The overall order is maintained lexicographically by key.
    """

    def __init__(self):
        self.head = self.tail = None
        # auxiliary index
        self._nodes = []           # List[KListNode]
        self._prefix_counts = []   # List[int]
        self._bounds = []          # List[int], max key per node (optional)

    def _rebuild_index(self):
        """Rebuild the node list and prefix-sum of entry counts."""
        self._nodes.clear()
        self._prefix_counts.clear()
        self._bounds.clear()
        
        total = 0
        node = self.head
        while node:
            self._nodes.append(node)
            total += len(node.entries)
            self._prefix_counts.append(total)
            
            # Add the maximum key in this node to bounds
            if node.entries:
                self._bounds.append(node.entries[-1].item.key)
            
            node = node.next

    def is_empty(self) -> bool:
        return self.head is None
    
    def item_count(self) -> int:
        count = 0
        current = self.head
        while current is not None:
            count += len(current.entries)
            current = current.next
        return count
    
    def item_slot_count(self) -> int:
        """
        Returns the total number of slots available
        in the k-list, which is the sum of the capacities of all nodes.
        """
        count = 0
        current = self.head
        while current is not None:
            count += KListNode.CAPACITY
            current = current.next
        return count
    
    def physical_height(self) -> int:
        """
        Returns the number of KListNode segments in this k-list.
        (i.e. how many times you must follow `next` before you reach None).
        """
        height = 0
        node = self.head
        while node is not None:
            height += 1
            node = node.next
        return height
    

    def insert(
            self, 
            item: Item,
            left_subtree: Optional['GPlusTree'] = None
    ) -> 'KList':
        """
        Inserts an item with an optional left subtree into the k-list.
        It is stored as an Entry(item, left_subtree).

        The insertion ensures that the keys are kept in lexicographic order.
        If a node overflows (more than k entries), the extra entry is recursively inserted into the next node.

        Parameters:
            item (Item): The item to insert.
            left_subtree (GPlusTree or None): Optional G+-tree to attach as the left subtree.
        """
        entry = Entry(item, left_subtree)
        
        # If the k-list is empty, create a new node.
        if self.head is None:
            node = KListNode()
            self.head = self.tail = node
        else:
            # Fast-Path: If the new key > the last key in the tail, insert there.
            if self.tail.entries and item.key > self.tail.entries[-1].item.key:
                node = self.tail
            else:
                # linear search from the head
                node = self.head
                while node.next is not None and node.entries and item.key > node.entries[-1].item.key:
                    node = node.next
        
        overflow = node.insert_entry(entry)

        if node is self.tail and overflow is None:
            self._rebuild_index()
            return self

        MAX_OVERFLOW_DEPTH = 10000
        depth = 0

        # Propagate overflow if needed.
        while overflow is not None:
            if node.next is None:
                node.next = KListNode()
                self.tail = node.next
            node = node.next
            overflow = node.insert_entry(overflow)
            depth += 1
            if depth > MAX_OVERFLOW_DEPTH:
                raise RuntimeError("KList insert overflowed too deeply – likely infinite loop.")
            
        self._rebuild_index()
        return self

    def delete(self, key: int) -> "KList":
        node = self.head
        prev = None
        found = False

        # 1) Find and remove the entry.
        while node:
            for i, entry in enumerate(node.entries):
                if entry.item.key == key:
                    del node.entries[i]
                    found = True
                    break
            if found:
                break
            prev, node = node, node.next

        if not found:
            self._rebuild_index()
            return self

        # 2) If head is now empty, advance head.
        if node is self.head and not node.entries:
            self.head = node.next
            if self.head is None:
                # list became empty
                self.tail = None
                self._rebuild_index()
                return self
            # reset for possible rebalance, but prev stays None
            node = self.head

        # 3) If *any other* node is now empty, splice it out immediately.
        elif not node.entries:
            # remove node from chain
            prev.next = node.next
            # if we removed the tail, update it
            if prev.next is None:
                self.tail = prev
            self._rebuild_index()
            return self

        # 4) Otherwise, rebalance by borrowing one entry from next node.
        while node.next and len(node.entries) < KListNode.CAPACITY:
            next_node = node.next
            shifted = next_node.entries.pop(0)
            node.entries.append(shifted)
            # if next_node now empty, splice it out and update tail
            if not next_node.entries:
                node.next = next_node.next
                if node.next is None:
                    self.tail = node
            break

        self._rebuild_index()
        return self
    
    def retrieve(self, key: int) -> RetrievalResult:
        """
        Search for `key` in O(l·log k) time:
        – for each node, do a bisect on its sorted entries (size k).
        – skip whole nodes when key < first or key > last.
        – return (found_entry, successor) or (None, successor) correctly.
        """
        if not isinstance(key, int):
            raise TypeError(f"key must be int, got {type(key).__name__!r}")

        node = self.head
        while node:
            entries = node.entries
            if not entries:
                node = node.next
                continue

            first = entries[0].item.key
            last  = entries[-1].item.key

            # if key < first, successor is entries[0]
            if key < first:
                return RetrievalResult(found_entry=None,
                                    next_entry=entries[0])

            # if key > last, skip to next node
            if key > last:
                node = node.next
                continue

            # now key is in [first, last], find it by bisect
            # build a local list of keys (cheap when k is small)
            keys = [e.item.key for e in entries]
            i    = bisect.bisect_left(keys, key)

            # exact match?
            if i < len(entries) and entries[i].item.key == key:
                found = entries[i]
                # in‐node successor?
                if i + 1 < len(entries):
                    succ = entries[i+1]
                else:
                    # boundary: pull from next node if exists
                    succ = (node.next.entries[0]
                            if node.next and node.next.entries
                            else None)
                return RetrievalResult(found_entry=found, next_entry=succ)

            # not found, but entries[i] is the next larger key
            return RetrievalResult(found_entry=None,
                                next_entry=entries[i])

        # fell off the end
        return RetrievalResult(found_entry=None, next_entry=None)    
    
    def get_entry(self, index: int) -> RetrievalResult:
        """
        Returns the entry at the given overall index in the sorted KList along with the next entry. O(log l) node-lookup plus O(1) in-node offset.

        Parameters:
            index (int): Zero-based index to retrieve.

        Returns:
            RetrievalResult: A structured result containing:
                - found_entry: The requested Entry if present, otherwise None.
                - next_entry: The subsequent Entry, or None if no next entry exists.
        """
        # 0) validate
        if not isinstance(index, int):
            raise TypeError(f"index must be int, got {type(index).__name__!r}")

        # 1) empty list?
        if not self._prefix_counts:
            return RetrievalResult(found_entry=None, next_entry=None)

        total_items = self._prefix_counts[-1]
        # 2) out‐of‐bounds?
        if index < 0 or index >= total_items:
            return RetrievalResult(found_entry=None, next_entry=None)

        # 3) find the node in O(log l)
        node_idx = bisect.bisect_right(self._prefix_counts, index)
        node = self._nodes[node_idx]

        # 4) compute offset within that node
        prev_count = self._prefix_counts[node_idx - 1] if node_idx else 0
        offset = index - prev_count

        # 5) delegate to node
        entry, in_node_succ, needs_next = node.get_by_offset(offset)

        # 6) if we hit the end of this node, pull the true successor
        if needs_next:
            if node.next and node.next.entries:
                next_entry = node.next.entries[0]
            else:
                next_entry = None
        else:
            next_entry = in_node_succ

        return RetrievalResult(found_entry=entry, next_entry=next_entry)
    
    def get_min(self) -> RetrievalResult:
        """Retrieve the minimum entry from the sorted KList."""
        return self.get_entry(index=0)
    
    def update_left_subtree(self, key: int, new_tree: 'GPlusTree') -> 'KList':
        """
        Updates the left subtree of the item in the k-list.

        Parameters:
            key (int): The key of the item to update.
            new_tree (GPlusTree or None): The new left subtree to associate with the item.

        Returns:
            KList: The updated k-list.
        """
        if not isinstance(key, int):
            raise TypeError(f"key must be int, got {type(key).__name__!r}")

        result = self.retrieve(key)
        if result.found_entry is not None:
            result.found_entry.left_subtree = new_tree
        return self

    
    def split_inplace(
        self, key: int
    ) -> Tuple["KList", Optional["GPlusTree"], "KList"]:

        if not isinstance(key, int):
            raise TypeError(f"key must be int, got {type(key).__name__!r}")

        if self.head is None:                        # ··· (1) empty
            return KList(), None, KList()

        # --- locate split node ------------------------------------------------
        node_idx = bisect.bisect_right(self._bounds, key)
        if node_idx >= len(self._nodes):             # ··· (2) key > max
            return self, None, KList()

        split_node   = self._nodes[node_idx]
        prev_node    = self._nodes[node_idx - 1] if node_idx else None
        original_next = split_node.next

        # --- bisect inside that node -----------------------------------------
        keys = [e.item.key for e in split_node.entries]
        i    = bisect.bisect_left(keys, key)
        exact = i < len(keys) and keys[i] == key

        left_entries   = split_node.entries[:i]
        right_entries  = split_node.entries[i + 1 if exact else i :]
        left_subtree   = split_node.entries[i].left_subtree if exact else None

        # ------------- build LEFT --------------------------------------------
        left = KList()
        if left_entries:                             # reuse split_node
            split_node.entries = left_entries
            split_node.next    = None
            left.head = self.head
            left.tail = split_node
        else:                                        # nothing in split node
            if prev_node:                            # skip it
                prev_node.next = None
                left.head = self.head
                left.tail = prev_node
            else:                                    # key at very first entry
                left.head = left.tail = None

        # ------------- build RIGHT -------------------------------------------
        right = KList()
        if right_entries:
            if left_entries:                         # both halves non-empty
                new_node = KListNode()
                new_node.entries = right_entries
                new_node.next    = original_next
                right.head       = new_node
            else:                                    # left empty → reuse split_node
                split_node.entries = right_entries
                split_node.next    = original_next
                right.head         = split_node
        else:                                        # no right_entries
            right.head = original_next

        # find right.tail
        tail = right.head
        while tail and tail.next:
            tail = tail.next
        right.tail = tail

        # ------------- rebuild indexes ---------------------------------------
        left._rebuild_index()
        right._rebuild_index()

        return left, left_subtree, right

    
    def print_structure(self, indent: int = 0, depth: int = 0, max_depth: int = 2):
        """
        Returns a string representation of the k-list for debugging.
        
        Parameters:
            indent (int): Number of spaces for indentation.
            depth (int): Current recursion depth.
            max_depth (int): Maximum allowed recursion depth.
        """
        if self.is_empty():
            return f"{' ' * indent}Empty"

        if depth > max_depth:
            return f"{' ' * indent}... (max depth reached)"

        result = []
        node = self.head
        index = 0
        while node:
            result.append(f"{' ' * indent}KListNode(idx={index}, K={KListNode.CAPACITY})")
            for entry in node.entries:
                result.append(f"{' ' * indent}• {str(entry.item)}")
                if entry.left_subtree is None:
                    result.append(f"{' ' * indent}  Left: None")
                else:
                    result.append(entry.left_subtree.print_structure(indent + 2, depth + 1, max_depth))
            node = node.next
            index += 1
        return "\n".join(result)



    def __iter__(self):
        """
        Yields each entry of the k-list in lexicographic order.
        Each entry is of the form (item, left_subtree).
        """
        node = self.head
        while node:
            for entry in node.entries:
                yield entry
            node = node.next

    def __str__(self):
        """
        Returns a string representation of the k-list for debugging.
        """
        result = []
        node = self.head
        index = 0
        while node:
            result.append(f"Node {index}: {node.entries}")
            node = node.next
            index += 1
        return "\n".join(result)
    
    def check_invariant(self) -> None:
        """
        Verifies that:
          1) Each KListNode.entries is internally sorted by item.key.
          2) For each consecutive pair of nodes, 
             last_key(node_i) <= first_key(node_{i+1}).
          3) self.tail.next is always None (tail really is the last node).

        Raises:
            AssertionError: if any of these conditions fails.
        """
        # 1) Tail pointer must point to the true last node
        assert (self.head is None and self.tail is None) or (
            self.tail is not None and self.tail.next is None
        ), "Invariant violated: tail must reference the final node"

        node = self.head
        previous_last_key = None

        # 2) Walk through all nodes
        while node is not None:
            # 2a) Entries within this node are sorted
            for i in range(1, len(node.entries)):
                k0 = node.entries[i-1].item.key
                k1 = node.entries[i].item.key
                assert k0 <= k1, (
                    f"Intra-node sort order violated in node {node}: "
                    f"{k0} > {k1}"
                )

            # 2b) Boundary with the previous node
            if previous_last_key is not None and node.entries:
                first_key = node.entries[0].item.key
                assert previous_last_key <= first_key, (
                    f"Inter-node invariant violated between nodes: "
                    f"{previous_last_key} > {first_key}"
                )

            # Update for the next iteration
            if node.entries:
                previous_last_key = node.entries[-1].item.key

            node = node.next
