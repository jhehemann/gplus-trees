"""
GKPlusTree module - Extension of G+Trees with dimensional support.

This module provides GKPlusTrees that can automatically transform between
KLists and GKPlusTrees based on item count thresholds.
"""

from gplus_trees.g_k_plus.g_k_plus_base import (
    GKPlusTreeBase,
    GKPlusNodeBase,
    DEFAULT_DIMENSION,
    DEFAULT_L_FACTOR
)
from gplus_trees.g_k_plus.factory import (
    create_gkplus_tree,
    create_klist_aware_gkplus_tree
)

__all__ = [
    'GKPlusTreeBase',
    'GKPlusNodeBase',
    'create_gkplus_tree',
    'create_klist_aware_gkplus_tree',
    'DEFAULT_DIMENSION',
    'DEFAULT_L_FACTOR'
]