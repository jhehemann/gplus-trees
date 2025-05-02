"""Merkle extension for GPlus-trees.

This module provides Merkle tree functionality for GPlus-trees,
allowing each node to store and update cryptographic hashes of its subtrees.
"""

import hashlib
from typing import Dict, Optional, Tuple, Type, Any
import logging

from gplus_trees.base import (
    Item, 
    Entry, 
    AbstractSetDataStructure, 
    RetrievalResult, 
    _create_replica
)
from gplus_trees.gplus_tree_base import (
    GPlusTreeBase, 
    GPlusNodeBase, 
    DUMMY_ITEM,
    DUMMY_KEY,
)

# Configure logging
logger = logging.getLogger(__name__)
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)


class MerkleGPlusNodeBase(GPlusNodeBase):
    """
    Extension of GPlusNodeBase that adds Merkle tree functionality.
    Each node maintains a hash of its subtree's contents.
    """
    __slots__ = GPlusNodeBase.__slots__ + ("merkle_hash",)

    def __init__(
        self,
        rank: int,
        set: AbstractSetDataStructure,
        right: Optional['GPlusTreeBase'] = None
    ) -> None:
        super().__init__(rank, set, right)
        self.merkle_hash = None  # Will be computed on-demand

    def compute_hash(self) -> bytes:
        """
        Compute the Merkle hash for this node and its subtrees.
        
        Returns:
            bytes: The hash value for this node and its subtrees.
        """
        # Initialize a hash object
        h = hashlib.sha256()

        # Add the rank to the hash input
        h.update(str(self.rank).encode())

        # Add all entries from the set
        for entry in self.set:
            # Add entry's item key and value to hash
            h.update(str(entry.item.key).encode())
            if entry.item.value is not None:
                h.update(str(entry.item.value).encode())
            
            # Add left subtree hash if it exists and is not empty
            if entry.left_subtree is not None and not entry.left_subtree.is_empty():
                if isinstance(entry.left_subtree.node, MerkleGPlusNodeBase):
                    left_hash = entry.left_subtree.node.get_hash()
                    h.update(left_hash)

        # Add right subtree hash if it exists and is not empty
        if self.right_subtree is not None and not self.right_subtree.is_empty():
            if isinstance(self.right_subtree.node, MerkleGPlusNodeBase):
                right_hash = self.right_subtree.node.get_hash()
                h.update(right_hash)

        # Store and return the hash
        self.merkle_hash = h.digest()
        return self.merkle_hash

    def get_hash(self) -> bytes:
        """
        Get the hash value, computing it if necessary.
        
        Returns:
            bytes: The hash value for this node and its subtrees.
        """
        if self.merkle_hash is None:
            return self.compute_hash()
        return self.merkle_hash

    def invalidate_hash(self):
        """
        Invalidate the stored hash value so it will be recomputed on next access.
        This should be called whenever the node or its subtrees are modified.
        """
        self.merkle_hash = None


class MerkleGPlusTreeBase(GPlusTreeBase):
    """
    Extension of GPlusTreeBase that adds Merkle tree functionality.
    Each tree maintains a hash of its contents and updates it as modifications occur.
    """
    
    # These will be set by the factory
    NodeClass: Type[MerkleGPlusNodeBase]
    
    def get_root_hash(self) -> Optional[bytes]:
        """
        Get the Merkle root hash of the tree.
        
        Returns:
            bytes: The Merkle root hash, or None if the tree is empty.
        """
        if self.is_empty():
            return None
            
        if isinstance(self.node, MerkleGPlusNodeBase):
            return self.node.get_hash()
        return None

    def verify_integrity(self) -> bool:
        """
        Verify the integrity of the entire tree by recomputing hashes.
        
        Returns:
            bool: True if tree integrity is intact, False otherwise
        """
        if self.is_empty():
            return True
            
        # Get the current hash
        old_hash = self.get_root_hash()
        
        # Force a recomputation by invalidating all hashes
        self._invalidate_all_hashes()
        
        # Get the new hash
        new_hash = self.get_root_hash()
        
        # Compare them
        return old_hash == new_hash

    def _invalidate_all_hashes(self):
        """
        Recursively invalidate all hashes in the tree.
        """
        if self.is_empty():
            return
            
        # Invalidate this node's hash
        if isinstance(self.node, MerkleGPlusNodeBase):
            self.node.invalidate_hash()
            
        # Invalidate all left subtree hashes
        for entry in self.node.set:
            if entry.left_subtree and not entry.left_subtree.is_empty():
                if isinstance(entry.left_subtree, MerkleGPlusTreeBase):
                    entry.left_subtree._invalidate_all_hashes()
                    
        # Invalidate right subtree hash
        if self.node.right_subtree and not self.node.right_subtree.is_empty():
            if isinstance(self.node.right_subtree, MerkleGPlusTreeBase):
                self.node.right_subtree._invalidate_all_hashes()

    # Override insertion methods to update hashes
    def insert(self, x: Item, rank: int) -> 'MerkleGPlusTreeBase':
        """
        Insert an item with the given rank, then update Merkle hashes.
        
        Parameters:
            x (Item): The item to insert
            rank (int): The rank to assign
            
        Returns:
            MerkleGPlusTreeBase: The updated tree
        """
        # Perform the standard insert operation
        result = super().insert(x, rank)
        
        # Invalidate the hash so it will be recomputed when needed
        self._invalidate_all_hashes()
            
        return result
    
    # Method to get a proof of inclusion for a specific key
    def get_inclusion_proof(self, key: int) -> list:
        """
        Generate a Merkle inclusion proof for a given key.
        
        Parameters:
            key (int): The key to generate a proof for
            
        Returns:
            list: A list of hash values forming the inclusion proof
        """
        proof = []
        if self.is_empty():
            return proof
            
        # Build the proof by traversing the tree to the key
        self._build_inclusion_proof(key, proof)
        return proof
        
    def _build_inclusion_proof(self, key: int, proof: list) -> bool:
        """
        Helper method to build an inclusion proof for a key.
        
        Parameters:
            key (int): The key to generate a proof for
            proof (list): The proof list to append to
            
        Returns:
            bool: True if the key was found, False otherwise
        """
        if self.is_empty():
            return False
            
        node = self.node
        
        # Get the retrieval result for this key at this node
        res = node.set.retrieve(key)
        
        # If this is a leaf node, check if we found the item
        if node.rank == 1:
            # If we found it, the proof is complete
            return res.found_entry is not None
            
        # For internal nodes, we need to continue down
        # Add sibling hashes to the proof
        for entry in node.set:
            # Skip the entry for our key
            if res.found_entry and entry.item.key == res.found_entry.item.key:
                continue
            
            # Add sibling hashes
            if entry.left_subtree and not entry.left_subtree.is_empty():
                if isinstance(entry.left_subtree.node, MerkleGPlusNodeBase):
                    proof.append(entry.left_subtree.node.get_hash())
                
        # Add the right subtree hash if we're not going that way
        next_subtree = res.next_entry.left_subtree if res.next_entry else node.right_subtree
        if node.right_subtree != next_subtree and not node.right_subtree.is_empty():
            if isinstance(node.right_subtree.node, MerkleGPlusNodeBase):
                proof.append(node.right_subtree.node.get_hash())
            
        # Continue building the proof in the next subtree
        return next_subtree._build_inclusion_proof(key, proof)