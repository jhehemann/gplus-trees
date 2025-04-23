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

from abc import ABC, abstractmethod
from dataclasses import dataclass

from typing import NamedTuple, Optional, Tuple, TYPE_CHECKING
import math
import hashlib


if TYPE_CHECKING:
    from .gplus_tree import GPlusTree
    from .klist import KList


class Item:
    """
    Represents an item (a key-value pair) for insertion in G-trees and k-lists.
    """

    def __init__(
            self,
            key: str,
            value: str
    ):
        """
        Initialize an Item.

        Parameters:
            key (str): The item's key.
            value (str): The item's value.
        """
        self.key = key
        self.value = value
    

    def short_key(self) -> str:
        return self.key if len(self.key) <= 10 else f"{self.key[:3]}...{self.key[-3:]}"
    
    def get_rank(self, k):
        """
        Calculate and return the rank for this item using its key.
        
        The calculation is performed using `calculate_item_rank` and the
        provided parameter k.
        
        Parameters:
            k (int): The desired g-node size, must be a positive power of 2.
        
        Returns:
            int: The rank calculated for this item.
        """
        return calculate_item_rank(self.key, k)

    def __repr__(self):
        return (f"Item(key={self.key!r}, value={self.value!r}")
        return self.__str__()

    def __str__(self):
        # return (f"Item(key={self.key}, value={self.value}")
        return (f"Item(key={self.short_key()}, value={self.value}")

def _create_replica(key):
    """Create a replica item with given key and no value."""
    return Item(key, None)

class AbstractSetDataStructure(ABC):
    """
    Abstract base class for a set data structure storing tuples of items and their left subtrees.
    """
    
    @abstractmethod
    def insert(self, item: 'Item', rank: int) -> 'AbstractSetDataStructure':
        """
        Insert an item into the set with the provided rank.
        
        Parameters:
            item (Item): The item to be inserted.
            rank (int): The rank for the item.
        
        Returns:
            AbstractSetDataStructure: The set data structure instance where the item was inserted.
        """
        pass

    @abstractmethod
    def delete(self, key: str) -> 'AbstractSetDataStructure':
        """
        Delete the item corresponding to the given key from the corresponding set data structure.
        
        Parameters:
            key (str): The key of the item to be deleted.
        
        Returns:
            AbstractSetDataStructure: The set data structure instance after deletion.
        """
        pass

    @abstractmethod
    def retrieve(
        self, key: str
    ) -> 'RetrievalResult':
        """
        Retrieve the entry associated with the given key from the set data structure.

        Parameters:
            key (str): The key of the entry to retrieve.
        
        Returns:
            RetrievalResult: A named tuple containing:
                - found_entry: A tuple (item, left_subtree) if the key is found; otherwise, None.
                - next_entry: A tuple (next_item, left_subtree) representing the next entry in sorted order,
                            or None if no subsequent entry exists.
        """
        pass

    @abstractmethod
    def get_min(self) -> 'RetrievalResult':
        """
        Retrieve the minimum entry in the set data structure.
        Returns:
            RetrievalResult: A named tuple containing:
                - found_entry: The minimum entry in the set.
                - next_entry: The next entry in sorted order after the minimum entry.
        """
        pass

    def item_count(self) -> int:
        """
        Get the count of items in the set data structure.
        Returns:
            int: The number of items in the set.
        """
        pass

    @abstractmethod
    def split_inplace(
            self, key: str
    ) -> Tuple['AbstractSetDataStructure', Optional['AbstractSetDataStructure'], 'AbstractSetDataStructure']:
        """
        Split the set into two parts based on the provided key.
        
        The first part contains all items less than or equal to the key,
        and the second part contains all items greater than the key.
        
        Parameters:
            key (str): The key to split the set by.
        
        Returns:
            Tuple[AbstractSetDataStructure, Optional[AbstractSetDataStructure], AbstractSetDataStructure]:
                A tuple containing:
                    - The left set (items < key).
                    - The left subtree of the item (item = key).
                    - The right subtree (items > key).
        """
        pass

@dataclass
class Entry:
    """
    Represents an entry in the KList or G⁺-tree.

    Attributes:
        item (Item): The item contained in the entry.
        left_subtree (AbstractSetDataStructure): The left subtree associated with this item.
            This is always provided, even if the subtree is empty.
    """
    __slots__ = ("item", "left_subtree")
    
    item: Item
    left_subtree: AbstractSetDataStructure

    def __lt__(self, other: "Entry") -> bool:
        return self.item.key < other.item.key


class RetrievalResult(NamedTuple):
    """
    A container for the result of a lookup in an AbstractSetDataStructure.

    Attributes:
        found_entry (Optional[Entry]):
            The entry corresponding to the searched key if found;
            otherwise, None.
        next_entry (Optional[Entry]):
            The subsequent entry in the sorted order, which serves as a candidate for
            further operations or in-order traversal; None if no subsequent entry exists.
    """
    found_entry: Optional[Entry]
    next_entry: Optional[Entry]

def calculate_item_rank(key, k):
    """
    Calculate the rank for an item based on its key and a desired g-node size k.
    
      1. Verify that k is a power of 2.
      2. Compute n = log2(k). (n is the group size of zeroes in bits.)
      3. Compute the SHA-256 hash of the key and convert it to a binary string.
      4. Count the number of trailing zero bits in the binary representation.
      5. Group these trailing zeros into groups of n consecutive zeros.
      6. The final rank is the number of such complete groups plus 1.
    
    Parameters:
        key (str): The key for which to calculate the rank.
        k (int): The desired g-node size. Must be a power of 2.
        
    Returns:
        int: The rank for the key.
    
    Raises:
        ValueError: If k is not a power of 2.
    """
    # Check that k is a power of 2.
    if k <= 0 or (k & (k - 1)) != 0:
        raise ValueError("k must be a positive power of 2.")
    
    # Calculate the grouping size: n = log2(k)
    group_size = int(math.log2(k))
    
    # Compute the SHA-256 hash of the key.
    sha_hex = hashlib.sha256(key.encode('utf-8')).hexdigest()
    
    # Convert the hexadecimal hash into an integer, then get its binary representation.
    hash_int = int(sha_hex, 16)
    binary_str = bin(hash_int)[2:]  # Remove the '0b' prefix.
    
    # Count the trailing zeros in the binary string.
    trailing_zero_count = len(binary_str) - len(binary_str.rstrip('0'))
    
    # Count the number of complete groups of trailing zeros, each of size group_size.
    group_count = trailing_zero_count // group_size
    
    # The final rank is group_count + 1.
    return group_count + 1
