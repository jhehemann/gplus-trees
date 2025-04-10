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

from typing import Optional

from packages.jhehemann.customs.gtree.item import Item
from packages.jhehemann.customs.gtree.gtree import GTree

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

    def insert_entry(self, item: Item, left_subtree: Optional[GTree] = None):
        """
        Inserts the item pair (with an optional left_subtree) into this node in sorted order.

        Sorting is done lexicographically on the key.
        If the node exceeds its capacity after insertion, the largest entry (by key) is removed and returned.

        Parameters:
            key (str): The key used to order entries.
            value (any): The associated value.
            left_subtree (GTree or None): An optional g-tree to attach as the left subtree.

        Returns:
            tuple or None: The overflow entry (item, left_subtree) if the node exceeds capacity; otherwise, None.
        """
        new_entry = (item, left_subtree)
        self.entries.append(new_entry)
        # Sort entries based on the key (located at entry[0].key)
        self.entries.sort(key=lambda entry: entry[0].key)
        if len(self.entries) > KListNode.CAPACITY:
            # Remove and return the largest entry (the last one)
            return self.entries.pop()
        return None


class KList:
    """
    A k-list implemented as a linked list of nodes.
    Each node holds up to CAPACITY (4) sorted entries.
    An entry is of the form (item, left_subtree), where left_subtree is a g-tree (or None).
    The overall order is maintained lexicographically by key.
    """

    def __init__(self):
        self.head = KListNode()

    def insert(self, item: Item, left_subtree: Optional[GTree] = None):
        """
        Inserts a key-value pair (with an optional left subtree) into the k-list.
        The entry is stored as (item, left_subtree).

        The insertion ensures that the keys are kept in lexicographic order.
        If a node overflows (more than 4 entries), the extra entry is recursively inserted into the next node.

        Parameters:
            item (Item): The item to insert.
            left_subtree (GTree or None): Optional g-tree to attach as the left subtree.
        """
        node = self.head
        # Traverse nodes if the key should come later.
        # Compare with the last key in the current node (if any).
        while node.next is not None and node.entries and item.key > node.entries[-1][0].key:
            node = node.next

        overflow = node.insert_entry(item, left_subtree)
        # Propagate overflow if needed.
        while overflow is not None:
            if node.next is None:
                node.next = KListNode()
            node = node.next
            # Overflow is of the form (item, left_subtree); insert it into the new node.
            overflow_item = overflow[0]
            overflow_left_subtree = overflow[1]
            overflow = node.insert_entry(overflow_item, overflow_left_subtree)

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
            return False

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

        return True

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
