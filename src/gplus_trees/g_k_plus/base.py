from abc import ABC, abstractmethod
from dataclasses import dataclass

from typing import NamedTuple, Optional, Tuple, TypeVar, Generic, Iterator
import hashlib

from gplus_trees.base import (
    AbstractSetDataStructure,
    Item,
    Entry,
    RetrievalResult
)

T = TypeVar("T", bound="GKTreeSetDataStructure")

class GKTreeSetDataStructure(AbstractSetDataStructure):
    """
    Abstract base class for G+ trees and k-lists.
    """

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

    @abstractmethod
    def get_entry(self, index: int) -> 'RetrievalResult':
        """
        Returns the entry at the given overall index in the sorted set data structure along with the next entry.

        Parameters:
            index (int): Zero-based index to retrieve.

        Returns:
            RetrievalResult: A structured result containing:
                - found_entry: The requested Entry if present, otherwise None.
                - next_entry: The subsequent Entry, or None if no next entry exists.
        """
        pass

    @abstractmethod
    def item_count(self) -> int:
        """
        Get the count of items in the set data structure.
        Returns:
            int: The number of items in the set.
        """
        pass

    @abstractmethod
    def item_slot_count(self) -> int:
        """
        Get the total number of item slots reserved by the set data structure. This is the sum of the capacity of all nodes.
        Returns:
            int: The number of item slots in the set.
        """
        pass
    
    @abstractmethod
    def physical_height(self) -> int:
        """
        Get the physical height of the set data structure which is the maximum number of traversal steps needed to reach the deepest node.
        Returns:
            int: The physical height of the set.
        """
        pass

    @abstractmethod
    def split_inplace(
            self, key: int
    ) -> Tuple[T, Optional[T], T]:
        """
        Split the set into two parts based on the provided key.
        
        The first part contains all items less than or equal to the key,
        and the second part contains all items greater than the key.
        
        Parameters:
            key (int): The key to split the set by.
        
        Returns:
            Tuple[AbstractSetDataStructure, Optional[AbstractSetDataStructure], AbstractSetDataStructure]:
                A tuple containing:
                    - The left set (items < key).
                    - The left subtree of the item (item = key).
                    - The right subtree (items > key).
        """
        pass

    @abstractmethod
    def __iter__(self) -> Iterator['Entry']:
        """
        Iterate over the entries in the set data structure.
        
        Returns:
            Iterator[Entry]: An iterator over the entries in the set.
        """
        pass