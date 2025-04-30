"""Factory for the creation of set data structures"""

from typing import Type, Tuple

from gplus_trees.gplus_tree_base import GPlusTreeBase, GPlusNodeBase
from gplus_trees.klist_base import KListBase, KListNodeBase


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
    # 1) Leaf-list node: capacity K
    KListNodeK = type(
        f"KListNode_K{K}",
        (KListNodeBase,),
        {
            "CAPACITY": K,
            "__slots__": KListNodeBase.__slots__,
        }
    )

    # 2) KList class points at the node class
    KListK = type(
        f"KList_K{K}",
        (KListBase,),
        {
            "KListNodeClass": KListNodeK,
            "__slots__": KListBase.__slots__,
        }
    )

    # 3) GPlus-node class uses the KListK
    GPlusNodeK = type(
        f"GPlusNode_K{K}",
        (GPlusNodeBase,),
        {
            "SetClass": KListK,
            "TreeClass": GPlusTreeK,
            "__slots__": GPlusNodeBase.__slots__
        }
        )

    # 4) GPlus-tree class uses GPlusNodeK and KListK
    GPlusTreeK = type(
        f"GPlusTree_K{K}",
        (GPlusTreeBase,),
        {
            "NodeClass": GPlusNodeK,
            "SetClass": KListK,
            "__slots__": GPlusTreeBase.__slots__,
        }
    )

    return GPlusTreeK, GPlusNodeK, KListK, KListNodeK
