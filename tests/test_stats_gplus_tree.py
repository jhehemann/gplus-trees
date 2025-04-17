
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
"""Tests for gplus tree abstract data structure."""
# pylint: skip-file

import math
import random
from statistics import mean
from typing import List, Optional, Tuple

from packages.jhehemann.customs.gtree.base import (
    Item,
    calculate_item_rank
)
from packages.jhehemann.customs.gtree.gplus_tree import (
    GPlusTree,
    gtree_stats_
)

def geometric(p: float) -> int:
    """
    Draw a sample from a geometric distribution with success probability p.
    The result is in {1, 2, 3, ...}.
    """
    u = random.random()  # uniform in [0, 1)
    # Use the inverse transform: X = floor(log(u) / log(1-p)) + 1
    return math.floor(math.log(u) / math.log(1 - p)) + 1

# Assume create_gtree(items) builds a GPlusTree from a list of (Item, rank) pairs.
def create_gtree(items):
    """
    Mimics the Rust create_gtree: build a tree by inserting each (item, rank) pair.
    Replace this with your actual tree‐creation logic.
    """
    tree = GPlusTree()
    for (item, rank) in items:
        tree.insert(item, rank)
    return tree

# Create a random GPlusTree with n items and target node size (K) determining the rank distribution.
def random_gtree_of_size(n: int, target_node_size: int) -> GPlusTree:
    # cache globals
    calc_rank = calculate_item_rank
    make_item = Item

    # we need at least n unique values; 2^24 = 16 777 216 > 1 000 000
    space = 1 << 24
    assert space >= n, "key‑space too small!"

    # sample unique indices once (random order baked in)
    indices = random.sample(range(space), k=n)

    items = []
    append = items.append

    for idx in indices:
        # 3 bytes → 6 hex digits, all lowercase, C‐level speed
        key = idx.to_bytes(3, 'big').hex()
        # for your demo value:
        val = ord(key[3])

        item = make_item(key, val)
        rank = calc_rank(item.key, target_node_size)
        append((item, rank))

    return create_gtree(items)

# The function random_klist_tree just wraps random_gtree_of_size with a given K.
def random_klist_tree(n: int, K: int) -> GPlusTree:
    return random_gtree_of_size(n, K)

# # Stub for collecting statistics from a tree.
# # Replace this with your real gtree_stats_ implementation.
# class TreeStats:
#     def __init__(self, gnode_count, gnode_height, item_count, item_slot_count, rank):
#         self.gnode_count = gnode_count
#         self.gnode_height = gnode_height
#         self.item_count = item_count
#         self.item_slot_count = item_slot_count
#         self.rank = rank
        
# def gtree_stats_(tree: GPlusTree, dummy: dict) -> TreeStats:
#     # Return dummy statistics – replace with your actual logic.
#     # For demonstration, we use arbitrary values.
#     return TreeStats(
#         gnode_count=random.randint(5, 15),
#         gnode_height=random.randint(2, 5),
#         item_count=random.randint(10, 20),
#         item_slot_count=random.randint(20, 30),
#         rank=random.randint(1, 5)
#     )

# # Stub for computing the physical height of the tree.
# def compute_physical_height(tree: GPlusTree) -> int:
#     # Replace with your actual physical height computation.
#     return random.randint(2, 6)

def check_leaf_keys_and_values(
    tree: GPlusTree,
    expected_keys: Optional[List[str]] = None
) -> Tuple[bool, bool, bool]:
    """
    Traverse all leaf nodes exactly once, gathering their real items (key, value),
    then compute three invariants:
      1. presence_ok: if `expected_keys` is provided, do we have exactly that set of keys?
                      otherwise always True.
      2. all_have_values: are all those items’ values not None?
      3. order_ok: are the keys in strictly sorted order?
    
    Parameters:
        tree:           The GPlusTree to examine.
        expected_keys:  Optional list of keys that must match exactly. If None, skip presence test.
    
    Returns:
        (presence_ok, all_have_values, order_ok)
    """
    actual = []
    for leaf in tree.iter_leaf_nodes():
        for entry in leaf.set:
            # skip dummy or internal replicas (they have value=None)
            if entry.item.value is None:
                continue
            actual.append((entry.item.key, entry.item.value))

    # unzip into separate lists
    keys   = [k for k, _ in actual]
    values = [v for _, v in actual]

    # 1) presence: only if expected_keys was provided
    if expected_keys is None:
        presence_ok = True
    else:
        presence_ok = set(keys) == set(expected_keys)

    # 2) all real items have non-None values
    all_have_values = all(v is not None for v in values)

    # 3) keys are in sorted order
    order_ok = (keys == sorted(keys))

    return keys, presence_ok, all_have_values, order_ok

# --- The repeated_experiment function ---
def repeated_experiment(size: int, repetitions: int, K: int, p_override: float = None):
    """
    Repeatedly builds random GPlusTrees (with size items) using ranks drawn from a geometric distribution.
    Uses K as target node size to compute the geometric parameter. Aggregates statistics over many trees.
    
    Parameters:
      size : number of items in each tree
      repetitions : number of trees to generate
      K : target node size (used to compute the rank distribution as p = 1 - 1/(K+1))
      p_override : If provided, use this probability instead (optional)
    """
    results = []  # List of tuples: (stats, physical_height)

    # Generate results from repeated experiments.
    for _ in range(repetitions):
        tree = random_klist_tree(size, K)
        stats = gtree_stats_(tree, {})
        phy_height = compute_physical_height(tree)
        results.append((stats, phy_height))
    
    # Perfect height: ceil( log_{K+1}(size) )
    perfect_height = math.ceil(math.log(size, K + 1)) if size > 0 else 0

    # Aggregate averages
    avg_gnode_height    = mean(stats.gnode_height for stats, _ in results)
    avg_gnode_count     = mean(stats.gnode_count for stats, _ in results)
    avg_item_count      = mean(stats.item_count for stats, _ in results)
    avg_item_slot_count = mean(stats.item_slot_count for stats, _ in results)
    avg_space_amp       = mean((stats.item_slot_count / stats.item_count)
                               for stats, _ in results)
    avg_physical_height = mean(phy_height for _, phy_height in results)
    avg_height_amp      = mean((phy_height / perfect_height) for _, phy_height in results) if perfect_height else 0
    avg_avg_gnode_size  = mean((stats.item_count / stats.gnode_count)
                               for stats, _ in results)
    avg_max_rank        = mean(stats.rank for stats, _ in results)
    
    # Compute variances
    var_gnode_height    = mean((stats.gnode_height - avg_gnode_height)**2 for stats, _ in results)
    var_gnode_count     = mean((stats.gnode_count - avg_gnode_count)**2 for stats, _ in results)
    var_item_count      = mean((stats.item_count - avg_item_count)**2 for stats, _ in results)
    var_item_slot_count = mean((stats.item_slot_count - avg_item_slot_count)**2 for stats, _ in results)
    var_space_amp       = mean(((stats.item_slot_count / stats.item_count) - avg_space_amp)**2 for stats, _ in results)
    var_physical_height = mean((phy_height - avg_physical_height)**2 for _, phy_height in results)
    var_height_amp      = mean(((phy_height / perfect_height) - avg_height_amp)**2 for _, phy_height in results) if perfect_height else 0
    var_avg_gnode_size  = mean(((stats.item_count / stats.gnode_count) - avg_avg_gnode_size)**2 for stats, _ in results)
    var_max_rank        = mean((stats.rank - avg_max_rank)**2 for stats, _ in results)

    print(f"n = {size}; K = {K}; {repetitions} repetitions")
    print("Legend: name <average> (<variance>)")
    print("---------------------------------------")
    print(f"Item count: {avg_item_count:.2f} ({var_item_count:.2f})")
    print(f"Item slot count: {avg_item_slot_count:.2f} ({var_item_slot_count:.2f})")
    print(f"Space amplification: {avg_space_amp:.2f} ({var_space_amp:.2f})")
    print(f"G-node count: {avg_gnode_count:.2f} ({var_gnode_count:.2f})")
    print(f"Average G-node size: {avg_avg_gnode_size:.2f} ({var_avg_gnode_size:.2f})")
    print(f"Maximum rank: {avg_max_rank:.2f} ({var_max_rank:.2f})")
    print(f"G-node height: {avg_gnode_height:.2f} ({var_gnode_height:.2f})")
    print(f"Actual height: {avg_physical_height:.2f} ({var_physical_height:.2f})")
    print(f"Perfect height: {perfect_height}")
    print(f"Height amplification: {avg_height_amp:.2f} ({var_height_amp:.2f})")
    # print(f"Tree structure:\n{tree.print_structure()}")
    print("\n\n")

if __name__ == "__main__":
    # List of tree sizes to test.
    sizes = [10, 100, 1000, 10000]
    # List of K values for which we want to run experiments.
    Ks = [2, 4, 16, 64]
    repetitions = 200

    for n in sizes:
        for K in Ks:
            print(f"\n--- Running experiment: n = {n}, K = {K} ---")
            repeated_experiment(size=n, repetitions=repetitions, K=K)