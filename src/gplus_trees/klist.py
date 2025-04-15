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

from packages.jhehemann.customs.gtree.base import Item
from packages.jhehemann.customs.gtree.base import AbstractSetDataStructure

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
        self.entries = []  # List of entries: each is (item, left_subtree)
        self.next = None

    def insert_entry(
            self, 
            item: Item,
            left_subtree: Optional['GPlusTree'] = None
    ):
        """
        Inserts the item pair (with an optional left_subtree) into this node in sorted order.

        Sorting is done lexicographically on the key.
        If the node exceeds its capacity after insertion, the largest entry (by key) is removed and returned.

        Parameters:
            key (str): The key used to order entries.
            value (any): The associated value.
            left_subtree (GPlusTree or None): An optional G+-tree to attach as the left subtree.

        Returns:
            tuple or None: The overflow entry (item, left_subtree) if the node exceeds capacity; otherwise, None.
        """
        new_entry = (item, left_subtree)
        # print(f"Inserting Entry: {new_entry}")
        self.entries.append(new_entry)
        # Sort entries based on the key (located at entry[0].key)
        self.entries.sort(key=lambda entry: entry[0].key)
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
        Inserts a key-value pair (with an optional left subtree) into the k-list.
        The entry is stored as (item, left_subtree).

        The insertion ensures that the keys are kept in lexicographic order.
        If a node overflows (more than 4 entries), the extra entry is recursively inserted into the next node.

        Parameters:
            item (Item): The item to insert.
            left_subtree (GPlusTree or None): Optional G+-tree to attach as the left subtree.
        """
        if self.head is None:
            self.head = KListNode()
        
        node = self.head
    
        # Traverse nodes if the key should come later.
        # Compare with the last key in the current node (if any).
        while node.next is not None and node.entries and item.key > node.entries[-1][0].key:
            node = node.next

        
        overflow = node.insert_entry(item, left_subtree)
        # print(f"Inserted Item: {item}")
        MAX_OVERFLOW_DEPTH = 100
        depth = 0
        # Propagate overflow if needed.
        while overflow is not None:
            if node.next is None:
                node.next = KListNode()
            node = node.next
            # Overflow is of the form (item, left_subtree); insert it into the new node.
            overflow_item, overflow_left_subtree = overflow
            overflow = node.insert_entry(overflow_item, overflow_left_subtree)
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
            bool: True if deletion was successful, False if the key was not found.
        """
        node = self.head
        found = False

        # Find and remove the entry with the given key.
        while node:
            for i, (item, _) in enumerate(node.entries):
                if item.key == key:
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
                current.entries.sort(key=lambda entry: entry[0].key)
                if not current.next.entries:
                    current.next = current.next.next
            else:
                break

        return self
    
    def retrieve(
        self, key: str
    ) -> Tuple[Optional[Item], Tuple[Optional[Item], Optional['GPlusTree']]]:
        """
        Retrieve the item associated with the given key from the KList.
        
        Parameters:
            key (str): The key of the item to retrieve.
        
        Returns:
            A tuple of two elements:
            - item: The value associated with the key, or None if not found.
            - next_entry: 
                    A tuple containing:
                    * The next item in the sorted order (if any),
                    * The left subtree associated with the next item (if any).
                    If no subsequent entry exists, returns None.
        """
        current_node = self.head
        while current_node is not None:
            # Iterate over the entries in the current KListNode.
            for i, (item, left_subtree) in enumerate(current_node.entries):
                if item.key == key:
                    # Item found; determine the next entry.
                    if i + 1 < len(current_node.entries):
                        # There is a subsequent entry in the same node.
                        next_entry = current_node.entries[i + 1]
                    elif current_node.next is not None and current_node.next.entries:
                        # Otherwise, take the first entry from the next node.
                        next_entry = current_node.next.entries[0]
                    else:
                        # No further entry exists.
                        next_entry = None
                    return (item, next_entry)
                elif item.key > key:
                    # Since entries are sorted, if we hit an item with a key greater
                    # than the search key, the key is not present; return the "next entry".
                    return (None, (item, left_subtree))
            current_node = current_node.next
        # If we have traversed all nodes and found nothing, return (None, None).
        return (None, None)
    
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
            for i, (entry_item, _) in enumerate(current_node.entries):
                if entry_item.key == key:
                    # Update the left subtree of the found entry.
                    current_node.entries[i] = (entry_item, left_subtree)
                    return self
            current_node = current_node.next
        return self

    
    def get_min(self) -> Optional[Tuple['Item', 'AbstractSetDataStructure']]:
        """
        Retrieve the minimum entry from the set.

        An entry is defined as a tuple consisting of:
            - An Item, which represents the entry.
            - A left subtree of type AbstractSetDataStructure.

        Returns:
            Optional[Tuple[Item, AbstractSetDataStructure]]:
                The minimum entry if the set is non-empty; otherwise, None.
        """
        current_node = self.head
        # Iterate through nodes until a node with entries is found.
        while current_node is not None:
            if current_node.entries:
                # The first entry is the minimal one due to the lexicographic sorting.
                return current_node.entries[0]
            current_node = current_node.next
        return None
    
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
                item, subtree = entry
                if item.key < key:
                    left_entries.append(entry)
                elif item.key == key:
                    # Mark that we found an exact match and store its left subtree.
                    # (We do not include this entry in either partition.)
                    if left_subtree is None:
                        left_subtree = subtree
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
                result.append(f"{' ' * indent}• {str(entry[0])}")
                if entry[1] is None:
                    result.append(f"{' ' * indent}  Left: None")
                else:
                    result.append(entry[1].print_structure(indent + 2, depth + 1, max_depth))
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
