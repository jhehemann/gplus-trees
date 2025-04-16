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
    where `left_subtree` is a G-tree (or None) associated with this entry.
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
        Inserts an entry into a sorted KListNode.
        If the node exceeds its capacity, the last entry is returned for further processing.
        The entries are kept sorted based on the key (entry.item.key).
        If the node is not full, it simply appends the entry.
        
        Attributes:
            entry (Entry): The entry to insert into the KListNode.
        Returns:
            Optional[Entry]: The last entry if the node overflows; otherwise, None.
        """
        self.entries.append(entry)
        # Sort entries based on the key (located at entry.item.key)
        self.entries.sort(key=lambda entry: entry.item.key)
        if len(self.entries) > KListNode.CAPACITY:
            # Remove and return the largest entry (the last one)
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
        self.head = None

    def is_empty(self) -> bool:
        return self.head is None
    
    def item_count(self) -> int:
        count = 0
        current = self.head
        while current is not None:
            count += len(current.entries)
            current = current.next
        return count

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
        if self.head is None:
            self.head = KListNode()
        
        node = self.head
        entry = Entry(item, left_subtree)
    
        # Traverse nodes if the key should come later.
        # Compare with the last key in the current node (if any).
        while node.next is not None and node.entries and item.key > node.entries[-1].item.key:
            node = node.next
        
        overflow = node.insert_entry(entry)
        # print(f"Inserted Item: {item}")
        MAX_OVERFLOW_DEPTH = 100
        depth = 0
        # Propagate overflow if needed.
        while overflow is not None:
            if node.next is None:
                node.next = KListNode()
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

        # Rebalance: shift the first entry from the next node into the current node if space exists.
        current = node
        while current and current.next and len(current.entries) < KListNode.CAPACITY:
            if current.next.entries:
                shifted_entry = current.next.entries.pop(0)
                current.entries.append(shifted_entry)
                current.entries.sort(key=lambda entry: entry.item.key)
                if not current.next.entries:
                    current.next = current.next.next
            else:
                break

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
            left_subtree: 'GPlusTree'
    ) -> 'KList':
        """
        Updates the left subtree of the item in the k-list.

        If the item is not found, it returns the original k-list.
        If found, it updates the left subtree and returns the updated k-list.

        Parameters:
            key (str): The key of the item to update.
            left_subtree (GPlusTree or None): The new left subtree to associate with the item.

        Returns:
            KList: The updated k-list.
        """
        current_node = self.head
        while current_node is not None:
            for i, entry in enumerate(current_node.entries):
                if entry.item.key == key:
                    # Update the left subtree of the found entry.
                    current_node.entries[i].left_subtree = left_subtree
                    return self
            current_node = current_node.next
        return self
    
    def split_inplace(
            self, key: str
    ) -> Tuple['KList', Optional['GPlusTree'], 'KList']:
        """
        Partitions the current KList in place based on the provided key.
        
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
                        left_subtree = entry.subtree
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
    
    def print_structure(self, indent: int = 0, depth: int = 0, max_depth: int = 10):
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
