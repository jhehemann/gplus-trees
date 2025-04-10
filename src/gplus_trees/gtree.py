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

"""G-tree implementation"""

import math
import hashlib

class GNode:
    """
    A G-node is the core component of a G-tree.
    
    Attributes:
        rank (int): A natural number greater than 0.
        klist (KList): A k-list that stores elements; each element is a pair ((key, value), left_subtree).
        right (GTree): The right subtree (a GTree) of this G-node.
    """
    def __init__(self, rank, klist, right):
        if rank <= 0:
            raise ValueError("Rank must be a natural number greater than 0.")
        self.rank = rank
        self.klist = klist  # A non-empty instance of KList.
        self.right = right  # A GTree.

    def __str__(self):
        return f"GNode(rank={self.rank}, klist=[\n{str(self.klist)}\n], right={self.right})"

    def __repr__(self):
        return self.__str__()


class GTree:
    """
    A G-tree is a recursively defined structure that is either empty or contains a single G-node.
    
    If the attribute 'node' is None, the G-tree is considered empty.
    """
    def __init__(self, node=None):
        """
        Initialize a G-tree.
        
        Parameters:
            node (GNode or None): The G-node that the tree contains. If None, the tree is empty.
        """
        self.node = node

    def is_empty(self):
        """Return True if the G-tree is empty."""
        return self.node is None

    def __str__(self):
        if self.node is None:
            return "Empty GTree"
        return str(self.node)

    def __repr__(self):
        return self.__str__()


def calculate_item_rank(key, k):
    """
    Calculate the rank for an item based on its key and a desired g-node size k.
    
    The function works as follows:
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