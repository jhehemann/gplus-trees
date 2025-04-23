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

"""K-list implementation"""

from typing import TYPE_CHECKING, Optional, Tuple
import bisect

from packages.jhehemann.customs.gtree.base import (
    Item,
    AbstractSetDataStructure,
    RetrievalResult,
    Entry
)

if TYPE_CHECKING:
    from packages.jhehemann.customs.gtree.gplus_tree import GPlusTree

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


class KList(AbstractSetDataStructure):
    """
    A k-list implemented as a linked list of nodes.
    Each node holds up to CAPACITY (4) sorted entries.
    An entry is of the form (item, left_subtree), where left_subtree is a G+-tree (or None).
    The overall order is maintained lexicographically by key.
    """

    def __init__(self):
        self.head = self.tail = None

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
        
        return self

    def delete(self, key):
        """
        Deletes an entry by key from the k-list.
        After deletion, rebalances nodes by shifting entries from subsequent nodes to fill gaps.

        Parameters:
            key (str): The key to delete.

        Returns:
            KList: The updated k-list after deletion.
        """
        node = self.head
        found = False

        # Find and remove the entry with the given key.
        while node:
            for i, entry in enumerate(node.entries):
                if entry.item.key == key:
                    del node.entries[i]
                    found = True
                    break
            if found:
                break
            node = node.next

        if not found:
            return self
        
        # If we emptied the head node, advance head
        if node is self.head and not node.entries:
            self.head = node.next
            if self.head is None:
                self.tail = None
                return self
            node = self.head

        # Rebalance: fill this node back up by borrowing one
        # smallest entry from node.next, if it exists.
        while node.next and len(node.entries) < KListNode.CAPACITY:
            next_node = node.next

            # pull the smallest entry from next_node
            shifted = next_node.entries.pop(0)

            # since invariants guarantee shifted ≥ node.entries[-1],
            # we can append without re-sorting
            node.entries.append(shifted)

            # if next_node is now empty, splice it out
            if not next_node.entries:
                node.next = next_node.next
                # update tail if we removed the last node
                if node.next is None:
                    self.tail = node
                break
            else:
                # we only needed one shift to fill this node
                break

        # If the very last node lost its only entry, update tail
        if self.tail and not self.tail.entries:
            # find the new tail by walking from head
            t = self.head
            while t and t.next:
                t = t.next
            self.tail = t

        return self
    
    def retrieve(self, key: str) -> RetrievalResult:
        """
        Retrieve the entry associated with the given key from the KList.
        
        Parameters:
            key (str): The key of the entry to retrieve.
        
        Returns:
            RetrievalResult: A structured result containing:
            - found_entry: The entry (an Entry instance) corresponding to the searched key if found; otherwise, None.
            - next_entry: The subsequent entry in sorted order (an Entry), or None if no subsequent entry exists.
        """
        current_node = self.head
        while current_node is not None:
            for i, entry in enumerate(current_node.entries):
                if entry.item.key == key:
                    if i + 1 < len(current_node.entries):
                        next_entry = current_node.entries[i + 1]
                    elif current_node.next is not None and current_node.next.entries:
                        next_entry = current_node.next.entries[0]
                    else:
                        next_entry = None
                    return RetrievalResult(found_entry=entry, next_entry=next_entry)
                elif entry.item.key > key:
                    # Item not found; return the next candidate.
                    return RetrievalResult(found_entry=None,
                                            next_entry=entry)
            current_node = current_node.next
        return RetrievalResult(found_entry=None, next_entry=None)
    
    def get_entry(self, index: int) -> RetrievalResult:
        """
        Returns the entry at the given overall index in the KList along with the next entry in sorted order.

        This method traverses the linked list of KListNodes and returns a RetrievalResult.

        Parameters:
            index (int): Zero-based index to retrieve.

        Returns:
            RetrievalResult: A structured result containing:
                - found_entry: The requested Entry if present, otherwise None.
                - next_entry: The subsequent Entry, or None if no next entry exists.
        """
        current = self.head
        count = 0
        while current is not None:
            if count + len(current.entries) > index:
                # Target entry is in the current node.
                entry = current.entries[index - count]
                # Determine the next entry:
                if (index - count + 1) < len(current.entries):
                    next_entry = current.entries[index - count + 1]
                elif current.next is not None and current.next.entries:
                    next_entry = current.next.entries[0]
                else:
                    next_entry = None
                return RetrievalResult(found_entry=entry, next_entry=next_entry)
            count += len(current.entries)
            current = current.next
        return RetrievalResult(found_entry=None, next_entry=None)

    
    def get_min(self) -> RetrievalResult:
        """Retrieve the minimum entry from the sorted KList."""
        return self.get_entry(index=0)
    
    def update_left_subtree(
            self,
            key: str,
            new_tree: 'GPlusTree'
    ) -> 'KList':
        """
        Updates the left subtree of the item in the k-list.

        If the item is not found, it returns the original k-list.
        If found, it updates the left subtree and returns the updated k-list.

        Parameters:
            key (str): The key of the item to update.
            new_tree (GPlusTree or None): The new left subtree to associate with the item.

        Returns:
            KList: The updated k-list.
        """
        current_node = self.head
        while current_node is not None:
            for i, entry in enumerate(current_node.entries):
                if entry.item.key == key:
                    # Update the left subtree of the found entry.
                    current_node.entries[i].left_subtree = new_tree
                    return self
            current_node = current_node.next
        return self
    
    def split_inplace(
            self, key: str
    ) -> Tuple['KList', Optional['GPlusTree'], 'KList']:
        """
        NOTE: Not yet in-place splitting. Currently creates new KLists.
        
        Partitions the current KList in-place based on the provided key.
        
        This method splits the KList into:
        - A left partition containing all entries with keys < key.
        - The left subtree of the entry with key == key (if found), otherwise None.
        - A right partition containing all entries with keys > key.
        
        The original KList is modified (its nodes are re-wired).
        
        Returns:
            A tuple (left_klist, left_subtree, right_klist)
        """
        # Create new KLists for the partitions.
        left_klist = KList()
        right_klist = KList()
        left_subtree = None

        # We will rewire nodes into the new lists.
        left_tail = None
        right_tail = None

        current = self.head
        while current is not None:
            # Temporary lists to hold entries of the current node for each partition.
            left_entries = []
            right_entries = []
            # Process each entry in the current node.
            for entry in current.entries:
                if entry.item.key < key:
                    left_entries.append(entry)
                elif entry.item.key == key:
                    # Mark that we found an exact match and store its left subtree.
                    # (We do not include this entry in either partition.)
                    if left_subtree is None:
                        left_subtree = entry.left_subtree
                else:  # item.key > key
                    right_entries.append(entry)
            # If there are entries for the left partition, create a node and append it.
            if left_entries:
                new_left_node = KListNode()
                new_left_node.entries = left_entries
                if left_klist.head is None:
                    left_klist.head = new_left_node
                    left_tail = new_left_node
                else:
                    left_tail.next = new_left_node
                    left_tail = new_left_node
            # Similarly for the right partition.
            if right_entries:
                new_right_node = KListNode()
                new_right_node.entries = right_entries
                if right_klist.head is None:
                    right_klist.head = new_right_node
                    right_tail = new_right_node
                else:
                    right_tail.next = new_right_node
                    right_tail = new_right_node
            current = current.next

        # At this point the new left_klist and right_klist represent the in-place partitioning.
        return (left_klist, left_subtree, right_klist)
    
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
