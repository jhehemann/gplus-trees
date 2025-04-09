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
        self.klist = klist  # An instance of KList.
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