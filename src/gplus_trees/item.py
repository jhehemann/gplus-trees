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

"""Item implementation"""

import math
import hashlib

class Item:
    """
    Represents an item (a key-value pair) for insertion in G-trees and k-lists.
    Each item has an associated UTC timestamp, which denotes the last time the item was changed.
    This timestamp must be provided during initialization.
    """

    def __init__(self, key, value, timestamp):
        """
        Initialize an Item.

        Parameters:
            key (str): The item's key.
            value (any): The item's value.
            timestamp (datetime.datetime): The UTC timestamp of the item's last change.
        """
        self.key = key
        self.value = value
        self.timestamp = timestamp

    def update_value(self, new_value, new_timestamp):
        """
        Update the item's value, provided the new timestamp differs from the current one.

        Parameters:
            new_value (any): The new value to update the item with.
            new_timestamp (datetime.datetime): The timestamp associated with the new value.

        Raises:
            ValueError: If the new timestamp is exactly the same as the current timestamp.
        """
        if new_timestamp == self.timestamp:
            raise ValueError("Update failed: the provided timestamp is identical to the current timestamp.")
        self.value = new_value
        self.timestamp = new_timestamp
    
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
        return (f"Item(key={self.key!r}, value={self.value!r}, "
                f"timestamp={self.timestamp.isoformat()})")

    def __str__(self):
        return (f"Item(key={self.key}, value={self.value}, "
                f"last_updated={self.timestamp.isoformat()})")


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
