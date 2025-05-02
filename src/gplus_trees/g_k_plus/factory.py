"""GKPlusTree factory module"""

from typing import Type, Dict, Any, Optional, Callable
import logging

from gplus_trees.base import AbstractSetDataStructure
from gplus_trees.klist_base import KListBase, KListNodeBase
from gplus_trees.g_k_plus.g_k_plus_base import (
    GKPlusTreeBase,
    GKPlusNodeBase,
    DEFAULT_L_FACTOR
)

logger = logging.getLogger(__name__)

def create_gkplus_tree(
    klist_class: Type[KListBase],
    knode_class: Type[KListNodeBase],
    dimension: int = 1,
    l_factor: float = DEFAULT_L_FACTOR,
    **kwargs: Dict[str, Any]
) -> Type[GKPlusTreeBase]:
    """
    Creates a concrete GKPlusTree class with specific KList implementation.
    
    Args:
        klist_class: The KList implementation to use
        knode_class: The KListNode implementation to use
        dimension: The initial dimension for the tree (default: 1)
        l_factor: The threshold factor for KList <-> GKPlusTree conversions
        **kwargs: Additional configuration options
        
    Returns:
        A concrete GKPlusTree class configured with the specified dependencies
    """
    logger.debug(f"Creating GKPlusTree with KList: {klist_class.__name__}, node: {knode_class.__name__}, dimension: {dimension}")
    
    # Create concrete node class
    class GKPlusNode(GKPlusNodeBase):
        SetClass = klist_class
        
        # TreeClass will be assigned after GKPlusTree is defined
        
        def __str__(self):
            return f"GKPlusNode(rank={self.rank}, items={self.set.item_count()})"

    # Create concrete tree class
    class GKPlusTree(GKPlusTreeBase):
        NodeClass = GKPlusNode
        SetClass = klist_class
        KListClass = klist_class
        DIM = dimension  # Set the class attribute for dimension
        
        def __init__(self, node=None, l_factor=l_factor):
            super().__init__(node, l_factor)
        
        def insert(self, x_item, rank):
            """
            Override insert to handle automatic conversion between KList and GKPlusTree.
            
            Args:
                x_item: The item to insert
                rank: The rank for the item
                
            Returns:
                The updated tree
            """
            # First perform the regular insertion
            result = super().insert(x_item, rank)
            
            if result.is_empty():
                return result
            
            # Check if any KLists need to be expanded to GKPlusTrees
            def process_node(node):
                if node is None:
                    return
                
                # Check if the set is a KList
                if isinstance(node.set, klist_class):
                    # Check if it should be expanded to a GKPlusTree
                    node.set = self.check_and_expand_klist(node.set)
                
                # Recursively check entries in the node's set
                for entry in node.set:
                    if entry.left_subtree and not entry.left_subtree.is_empty():
                        process_tree(entry.left_subtree)
                
                # Check right subtree
                if node.right_subtree and not node.right_subtree.is_empty():
                    process_tree(node.right_subtree)
            
            def process_tree(tree):
                if tree.is_empty():
                    return
                
                # Process the root node
                process_node(tree.node)
            
            # Start processing from the result tree's root
            process_tree(result)
            
            return result
        
        def delete(self, key):
            """
            Override delete to handle automatic conversion between GKPlusTree and KList.
            
            Args:
                key: The key to delete
                
            Returns:
                The updated tree
            """
            # Implementation required - this is a placeholder
            # Would first call super().delete(key) then check if any GKPlusTrees
            # should be collapsed to KLists
            raise NotImplementedError("delete not yet implemented")

    # Assign the tree class to the node class
    GKPlusNode.TreeClass = GKPlusTree
    
    return GKPlusTree

def create_klist_aware_gkplus_tree(
    base_factory: Callable,
    dimension: int = 1,
    l_factor: float = DEFAULT_L_FACTOR,
    **kwargs: Dict[str, Any]
) -> Dict[str, Type]:
    """
    Creates a GKPlusTree class that is aware of the KList implementation.
    Uses an existing factory function to create the base components.
    
    Args:
        base_factory: Factory function that creates the base KList types
        dimension: The initial dimension for the tree
        l_factor: The threshold factor for KList <-> GKPlusTree conversions
        **kwargs: Additional configuration options
        
    Returns:
        A dict with the created GKPlusTree class and supporting components
    """
    # Get the KList and KListNode classes from the base factory
    klist_result = base_factory(**kwargs)
    klist_class = klist_result.get("klist_class")
    knode_class = klist_result.get("knode_class")
    
    if klist_class is None or knode_class is None:
        raise ValueError("Base factory must provide klist_class and knode_class")
    
    # Create the GKPlusTree class
    gkplus_tree_class = create_gkplus_tree(klist_class, knode_class, dimension, l_factor, **kwargs)
    
    return {
        "gkplus_tree_class": gkplus_tree_class,
        "klist_class": klist_class,
        "knode_class": knode_class
    }