"""Factory for the creation of set data structures"""

from typing import Type, Tuple, Dict, Optional
import logging

from gplus_trees.gplus_tree_base import GPlusTreeBase, GPlusNodeBase
from gplus_trees.klist_base import KListBase, KListNodeBase

# Configure logging
logger = logging.getLogger(__name__)
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)

# Cache for previously created classes to avoid recreating them
_class_cache: Dict[int, Tuple[Type, Type, Type, Type]] = {}


def make_gplustree_classes(K: int) -> Tuple[
    Type[GPlusTreeBase],
    Type[GPlusNodeBase],
    Type[KListBase],
    Type[KListNodeBase]
]:
    """
    Factory function to generate GPlus-tree and KList classes specialized for a given capacity K.

    Returns:
        GPlusTreeK    – subclass of GPlusTreeBase with NodeClass=GPlusNodeK and SetClass=KListK.
        GPlusNodeK    – subclass of GPlusNodeBase with SetClass=KListK.
        KListK        – subclass of KListBase with KListNodeClass=KListNodeK.
        KListNodeK    – subclass of KListNodeBase with CAPACITY=K.
    """
    # Check if we've already created classes for this K value
    if K in _class_cache:
        logger.debug(f"Using cached classes for K={K}")
        return _class_cache[K]

    logger.debug(f"Creating new classes for K={K}")
    
    # 1) Leaf-list node: capacity K
    KListNodeK = type(
        f"KListNode_K{K}",
        (KListNodeBase,),
        {
            "CAPACITY": K,
            "__slots__": KListNodeBase.__slots__ if hasattr(KListNodeBase, "__slots__") else ()
        }
    )
    logger.debug(f"Created KListNode_K{K} with CAPACITY={K}")
    logger.debug(f"KListNode_K{K}.__slots__: {KListNodeK.__slots__ if hasattr(KListNodeK, '__slots__') else 'None'}")

    # 2) KList class points at the node class
    KListK = type(
        f"KList_K{K}",
        (KListBase,),
        {
            "KListNodeClass": KListNodeK,
            "__slots__": KListBase.__slots__ if hasattr(KListBase, "__slots__") else ()
        }
    )
    logger.debug(f"Created KList_K{K} with KListNodeClass={KListNodeK.__name__}")
    logger.debug(f"KList_K{K}.__slots__: {KListK.__slots__ if hasattr(KListK, '__slots__') else 'None'}")

    # 3) Create GPlusTreeK with a forward reference (to be set later)
    GPlusTreeK = type(
        f"GPlusTree_K{K}",
        (GPlusTreeBase,),
        {
            "SetClass": KListK,
            "__slots__": GPlusTreeBase.__slots__ if hasattr(GPlusTreeBase, "__slots__") else ()
        }
    )
    logger.debug(f"Created GPlusTree_K{K} with SetClass={KListK.__name__}")
    logger.debug(f"GPlusTree_K{K}.__slots__: {GPlusTreeK.__slots__ if hasattr(GPlusTreeK, '__slots__') else 'None'}")
    
    # 4) GPlus-node class uses the KListK and references the already created GPlusTreeK
    GPlusNodeK = type(
        f"GPlusNode_K{K}",
        (GPlusNodeBase,),
        {
            "SetClass": KListK,
            "TreeClass": GPlusTreeK,
            "__slots__": GPlusNodeBase.__slots__ if hasattr(GPlusNodeBase, "__slots__") else ()
        }
    )
    logger.debug(f"Created GPlusNode_K{K} with SetClass={KListK.__name__}, TreeClass={GPlusTreeK.__name__}")
    logger.debug(f"GPlusNode_K{K}.__slots__: {GPlusNodeK.__slots__ if hasattr(GPlusNodeK, '__slots__') else 'None'}")

    # 5) Set NodeClass on GPlusTreeK now that GPlusNodeK exists
    setattr(GPlusTreeK, "NodeClass", GPlusNodeK)
    logger.debug(f"Set NodeClass={GPlusNodeK.__name__} on {GPlusTreeK.__name__}")

    # Cache the created classes
    _class_cache[K] = (GPlusTreeK, GPlusNodeK, KListK, KListNodeK)
    logger.debug(f"Cached classes for K={K}")

    return GPlusTreeK, GPlusNodeK, KListK, KListNodeK


def create_gplustree(K: int) -> GPlusTreeBase:
    """
    Create a new GPlusTree with the specified capacity K.
    
    Args:
        K (int): The capacity of the tree's KListNodes
        
    Returns:
        A new empty GPlusTree with the specified capacity
    """
    logger.debug(f"Creating new tree with K={K}")
    GPlusTreeK, _, _, _ = make_gplustree_classes(K)
    tree = GPlusTreeK()
    logger.debug(f"Created tree instance of type {type(tree).__name__}")
    return tree
