from abc import ABC, abstractmethod
from dataclasses import dataclass

from typing import NamedTuple, Optional, Tuple
import hashlib


# if TYPE_CHECKING:
#     from gplus_trees.gplus_tree import GPlusTree
#     from gplus_trees.klist import KList

class Item:
    """
    Represents an item (a key-value pair) for insertion in G-trees and k-lists.
    """

    __slots__ = ("key", "value")  # Define slots for memory efficiency

    def __init__(
            self,
            key: int,
            value: str
    ):
        """
        Initialize an Item.

        Parameters:
            key (int): The item's key.
            value (str): The item's value.
        """
        self.key = key
        self.value = value
    

    def short_key(self) -> str:
        # 1) Turn your key into a plain string of characters to slice:
        if isinstance(self.key, (bytes, bytearray)):
            s = self.key.hex()
        else:
            # treat everything else—including int—as decimal-string
            s = str(self.key)

        # 2) If it’s already short, just return it; otherwise elide the middle
        return s if len(s) <= 10 else f"{s[:3]}...{s[-3:]}"
    
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

    def __repr__(self) -> str:
        cls = self.__class__.__name__
        return f"{cls}(key={self.key!r}, value={self.value!r})"

    def __str__(self):
        cls = self.__class__.__name__
        return f"{cls}(key={self.short_key()}, value={self.value})"


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
    def delete(self, key: int) -> 'AbstractSetDataStructure':
        """
        Delete the item corresponding to the given key from the corresponding set data structure.
        
        Parameters:
            key (int): The key of the item to be deleted.
        
        Returns:
            AbstractSetDataStructure: The set data structure instance after deletion.
        """
        pass

    @abstractmethod
    def retrieve(
        self, key: int
    ) -> 'RetrievalResult':
        """
        Retrieve the entry associated with the given key from the set data structure.

        Parameters:
            key (int): The key of the entry to retrieve.
        
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
            self, key: int
    ) -> Tuple['AbstractSetDataStructure', Optional['AbstractSetDataStructure'], 'AbstractSetDataStructure']:
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


def calculate_group_size(k: int) -> int:
    """
    Calculate the group size of trailing zero-groupings of an item key's hash to count based on an expected g-node size k (power of 2).
    
    Parameters:
        k (int): The g-node size, must be a positive power of 2.
    
    Returns:
        int: The group size, which is log2(k).
    
    Raises:
        ValueError: If k is not a positive power of 2.
    """
    if k <= 0 or (k & (k - 1)) != 0:
        raise ValueError("k must be a positive power of 2")
    
    return k.bit_length() - 1

def calculate_item_rank(key: int, group_size: int):
    """
    Calculate the rank for an item by counting the number of complete groups of trailing zero-bits in the SHA-256 hash of its key.
    """
    # hash the decimal string of the int key
    key_bytes = str(key).encode("utf-8")
    digest = hashlib.sha256(key_bytes).digest()
    
    # convert to integer.
    hash_int = int.from_bytes(digest, byteorder="big", signed=False)
    
    # count trailing-zero bits
    tz = (hash_int & -hash_int).bit_length() - 1
    
    # final rank = complete groups + 1
    return (tz // group_size) + 1
