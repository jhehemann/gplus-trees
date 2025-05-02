"""GKPlusTree base implementation"""

from __future__ import annotations
from typing import Optional, Type, TypeVar, Dict, Tuple, Any
import logging
from dataclasses import dataclass
import collections

from gplus_trees.base import (
    AbstractSetDataStructure,
    Item,
    Entry,
    _create_replica,
    RetrievalResult,
)
from gplus_trees.klist_base import KListBase
from gplus_trees.gplus_tree_base import GPlusTreeBase, GPlusNodeBase
from gplus_trees.profiling import track_performance

# Configure logging
logger = logging.getLogger(__name__)
if not logger.handlers:
    # Avoid duplicated handlers when module is imported multiple times
    handler = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)

t = TypeVar('t', bound='GKPlusTreeBase')

# Constants
DUMMY_KEY = int("0" * 64, 16)
DUMMY_VALUE = None
DUMMY_ITEM = Item(DUMMY_KEY, DUMMY_VALUE)
DEFAULT_L_FACTOR = 0.75  # Default threshold factor for KList to GKPlusTree conversion

class GKPlusNodeBase(GPlusNodeBase):
    """
    Base class for GK+-tree nodes.
    Extends GPlusNodeBase with dimension support.
    """
    __slots__ = ("rank", "set", "right_subtree", "next")

    # These will be injected by the factory
    SetClass: Type[AbstractSetDataStructure]
    TreeClass: Type[GKPlusTreeBase]

    def __init__(
        self,
        rank: int,
        set: AbstractSetDataStructure,
        right: Optional[GKPlusTreeBase] = None
    ) -> None:
        if rank <= 0:
            raise ValueError("rank must be > 0")
        self.rank = rank
        self.set = set
        self.right_subtree = right if right is not None else self.TreeClass()
        self.next = None  # leaf-chain pointer

class GKPlusTreeBase(GPlusTreeBase):
    """
    A GK+-tree is an extension of G+-tree with dimension support.
    It can automatically transform between KList and GKPlusTree based on item count.
    
    Attributes:
        node (Optional[GKPlusNodeBase]): The GK+-node that the tree contains.
        DIM (int): The dimension of the GK+-tree (class attribute).
        l_factor (float): The threshold factor for conversion between KList and GKPlusTree.
    """
    __slots__ = ("node", "l_factor")
    
    # Default dimension value that will be overridden by factory-created subclasses
    DIM: int = 1  # Default dimension value, will usually be set by the factory
    
    # Will be set by the factory
    NodeClass: Type[GKPlusNodeBase]
    SetClass: Type[AbstractSetDataStructure]
    KListClass: Type[KListBase]
    
    def __init__(self, node: Optional[GKPlusNodeBase] = None, 
                 l_factor: float = DEFAULT_L_FACTOR):
        """
        Initialize a new GKPlusTree.
        
        Args:
            node: The root node of the tree (if not empty)
            l_factor: Threshold factor for KList-to-GKPlusTree conversion (default: 0.75)
        """
        self.node = node
        self.l_factor = l_factor
    
    @classmethod
    def from_root(cls: Type[t], root_node: GKPlusNodeBase) -> t:
        """Create a new tree instance wrapping an existing node."""
        tree = cls.__new__(cls)
        tree.node = root_node
        tree.l_factor = DEFAULT_L_FACTOR
        return tree
    
    def __str__(self):
        return "Empty GKPlusTree" if self.is_empty() else f"GKPlusTree(dim={self.__class__.DIM}, node={self.node})"

    __repr__ = __str__
    
    @classmethod
    def with_dimension(cls: Type[t], dim: int) -> Type[t]:
        """
        Create a new GKPlusTreeBase subclass with a specific dimension.
        
        Args:
            dim: The dimension value for the new class
            
        Returns:
            A new class with the specified dimension
        """
        new_class = type(f"{cls.__name__}Dim{dim}", (cls,), {"DIM": dim})
        return new_class
    
    # Main conversion methods
    @classmethod
    def from_klist(cls: Type[t], klist: KListBase, dim: int = None) -> t:
        """
        Convert a KList to a GKPlusTree.
        
        Args:
            klist: The KList to convert
            dim: The dimension for the new tree (if specified, creates a new class)
            
        Returns:
            A new GKPlusTree containing all items from the KList
        """
        # Use the specified dimension or increment the current one
        target_class = cls
        if dim is not None:
            target_class = cls.with_dimension(dim)
        
        new_tree = target_class()
        
        # If the KList is empty, return an empty tree
        if klist.is_empty():
            return new_tree
        
        # Insert all items from the KList into the tree
        # For each entry in the KList
        for entry in klist:
            # Insert the item with rank 1 (leaf level)
            new_tree = new_tree.insert(entry.item, 1)
            
            # If the entry has a left subtree, integrate it
            if entry.left_subtree is not None and not entry.left_subtree.is_empty():
                # Get the updated entry after insertion
                result = new_tree.retrieve(entry.item.key)
                if result.found_entry is not None:
                    # If the left subtree is a GKPlusTree, use it directly
                    if isinstance(entry.left_subtree, GKPlusTreeBase):
                        left_subtree = entry.left_subtree
                    # If it's another type of tree, recursively convert its content
                    else:
                        # This is a placeholder - in a real implementation, you'd need
                        # a more sophisticated conversion from other tree types
                        left_subtree = cls.from_tree(entry.left_subtree)
                    
                    # Update the left subtree for this entry
                    result.found_entry.left_subtree = left_subtree
        
        return new_tree
    
    def to_klist(self) -> KListBase:
        """
        Convert this GKPlusTree to a KList.
        
        Returns:
            A new KList containing all items from this tree
        """
        if self.is_empty():
            return self.KListClass()
        
        klist = self.KListClass()
        
        # Process leaf nodes to build the KList
        for leaf_node in self.iter_leaf_nodes():
            for entry in leaf_node.set:
                # Skip dummy items
                if entry.item.key == DUMMY_KEY:
                    continue
                
                # Insert each item into the KList
                klist = klist.insert(entry.item, entry.left_subtree)
        
        return klist
    
    @classmethod
    def from_tree(cls: Type[t], tree: GPlusTreeBase, dim: int = None) -> t:
        """
        Convert any GPlusTree-like object to a GKPlusTree.
        
        Args:
            tree: The tree to convert
            dim: The dimension for the new tree (if specified, creates a new class)
            
        Returns:
            A new GKPlusTree with the same content as the original tree
        """
        # Use the specified dimension or the current one
        target_class = cls
        if dim is not None:
            target_class = cls.with_dimension(dim)
        
        # If it's already the right type, just return it
        if isinstance(tree, target_class):
            return tree
            
        # If it's empty, return an empty GKPlusTree
        if tree is None or tree.is_empty():
            return target_class()
            
        # Create a new GKPlusTree with the same structure
        new_tree = target_class()
        
        # Process leaf nodes to build the new tree
        for leaf_node in tree.iter_leaf_nodes():
            for entry in leaf_node.set:
                # Skip dummy items
                if entry.item.key == DUMMY_KEY:
                    continue
                
                # Insert the item
                new_tree = new_tree.insert(entry.item, 1)
                
                # If there's a left subtree, recursively convert it
                if entry.left_subtree is not None and not entry.left_subtree.is_empty():
                    left_subtree = cls.from_tree(entry.left_subtree)
                    # Update the reference in the new entry
                    result = new_tree.retrieve(entry.item.key)
                    if result.found_entry:
                        result.found_entry.left_subtree = left_subtree
                        
        return new_tree
    
    # Extension methods to check threshold and perform conversions
    def check_and_expand_klist(self, klist: KListBase) -> AbstractSetDataStructure:
        """
        Check if a KList exceeds the threshold and should be converted to a GKPlusTree.
        
        Args:
            klist: The KList to check
            
        Returns:
            Either the original KList or a new GKPlusTree based on the threshold
        """
        if klist.is_empty():
            return klist
            
        # Check if the item count exceeds l_factor * CAPACITY
        capacity = klist.KListNodeClass.CAPACITY
        threshold = int(capacity * self.l_factor)
        
        if klist.item_count() > threshold:
            # Convert to GKPlusTree with increased dimension
            new_dim = self.__class__.DIM + 1
            new_tree_class = type(self).with_dimension(new_dim)
            return new_tree_class.from_klist(klist)
        
        return klist
    
    def check_and_collapse_tree(self, tree: 'GKPlusTreeBase') -> AbstractSetDataStructure:
        """
        Check if a GKPlusTree has few enough items to be collapsed into a KList.
        
        Args:
            tree: The GKPlusTree to check
            
        Returns:
            Either the original tree or a new KList based on the threshold
        """
        if tree.is_empty():
            return tree
            
        # Get the threshold based on the KList capacity
        capacity = self.KListClass.KListNodeClass.CAPACITY
        threshold = int(capacity * self.l_factor)
        
        # Count the actual items (excluding dummy items)
        item_count = 0
        for leaf_node in tree.iter_leaf_nodes():
            for entry in leaf_node.set:
                if entry.item.key != DUMMY_KEY:
                    item_count += 1
        
        if item_count <= threshold:
            # Collapse into a KList
            return tree.to_klist()
            
        return tree
        
    # Override methods as needed to integrate dimension handling
    def _insert_empty(self, x_item: Item, rank: int) -> 'GKPlusTreeBase':
        """Build the initial tree structure depending on rank."""
        # Call the parent implementation and ensure it uses the same class
        result = super()._insert_empty(x_item, rank)
        return result