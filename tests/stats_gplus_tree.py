"""Statistics for gplus trees."""
# pylint: skip-file

import os
import logging
import math
import random
import time
from statistics import mean
from typing import List, Optional, Tuple
from pprint import pprint
from dataclasses import asdict
from datetime import datetime
import numpy as np

from gplus_trees.base import (
    Item,
    calculate_item_rank,
    calculate_group_size,
)
from gplus_trees.gplus_tree import (
    GPlusTree,
    gtree_stats_,
    Stats,
    DUMMY_ITEM,
)
from gplus_trees.profiling import (
    track_performance,
)

TREE_FLAGS = (
    "is_heap",
    "is_search_tree",
    "internal_has_replicas",
    "internal_packed",
    "linked_leaf_nodes",
    "all_leaf_values_present",
    "leaf_keys_in_order",
)

# @track_performance
def assert_invariants(t: GPlusTree, stats: Stats) -> None:
    """Check all invariants, but only log ERROR messages on failures."""
    for flag in TREE_FLAGS:
        if not getattr(stats, flag):
            logging.error("Invariant failed: %s is False", flag)

    if not t.is_empty():
        if stats.item_count <= 0:
            logging.error(
                "Invariant failed: item_count=%d ≤ 0 for non-empty tree",
                stats.item_count
            )
        if stats.item_slot_count <= 0:
            logging.error(
                "Invariant failed: item_slot_count=%d ≤ 0 for non-empty tree",
                stats.item_slot_count
            )
        if stats.gnode_count <= 0:
            logging.error(
                "Invariant failed: gnode_count=%d ≤ 0 for non-empty tree",
                stats.gnode_count
            )
        if stats.gnode_height <= 0:
            logging.error(
                "Invariant failed: gnode_height=%d ≤ 0 for non-empty tree",
                stats.gnode_height
            )
        if stats.rank <= 0:
            logging.error(
                "Invariant failed: rank=%d ≤ 0 for non-empty tree",
                stats.rank
            )
        if stats.least_item is None:
            logging.error(
                "Invariant failed: least_item is None for non-empty tree"
            )
        if stats.greatest_item is None:
            logging.error(
                "Invariant failed: greatest_item is None for non-empty tree"
            )

# Assume create_gtree(items) builds a GPlusTree from a list of (Item, rank) pairs.
# @track_performance
def create_gtree(items):
    """
    Mimics the Rust create_gtree: build a tree by inserting each (item, rank) pair.
    Replace this with your actual tree-creation logic.
    """
    tree = GPlusTree()
    tree_insert = tree.insert
    for (item, rank) in items:
        tree_insert(item, rank)
    return tree

# Create a random GPlusTree with n items and target node size (K) determining the rank distribution.
# @track_performance
def random_gtree_of_size(n: int, target_node_size: int) -> GPlusTree:
    # cache globals
    # calc_rank = calculate_item_rank
    # group_size = calculate_group_size(target_node_size)
    make_item = Item
    p = 1.0 - (1.0 / (target_node_size))    # probability for geometric dist

    # we need at least n unique values; 2^24 = 16 777 216 > 1 000 000
    space = 1 << 24
    if space <= n:
        raise ValueError(f"Key-space too small! Required: {n + 1}, Available: {space}")

    indices = random.sample(range(space), k=n)

    # Pre-allocate items list
    items = [(None, None)] * n

    ranks = np.random.geometric(p, size=n)

    # Process all items in a single pass
    for i, idx in enumerate(indices):
        # Use the index directly as the key
        key = idx
        val = "val"
        items[i] = (make_item(key, val), int(ranks[i]))

    return create_gtree(items)

# The function random_klist_tree just wraps random_gtree_of_size with a given K.
def random_klist_tree(n: int, K: int) -> GPlusTree:
    return random_gtree_of_size(n, K)

# @track_performance
def check_leaf_keys_and_values(
    tree: GPlusTree,
    expected_keys: Optional[List[str]] = None
) -> Tuple[List[str], bool, bool, bool]:
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
        (keys, presence_ok, all_have_values, order_ok)
    """
    keys = []
    all_have_values = True
    order_ok = True
    
    # Traverse leaf nodes and collect keys
    prev_key = None
    for leaf in tree.iter_leaf_nodes():
        leaf_set = leaf.set
        for entry in leaf_set:
            item = entry.item
            key = item.key
            if prev_key is None:
                if not item is DUMMY_ITEM:
                    order_ok = False
            else:
                keys.append(key)

                # Check if value is non-None
                if item.value is None:
                    all_have_values = False

                # Check if keys are in sorted order
                if key < prev_key:
                    order_ok = False
            prev_key = key

    # Check presence only if expected_keys is provided
    presence_ok = True
    if expected_keys is not None:
        if len(keys) != len(expected_keys):
            presence_ok = False
        else:
            # Use a single pass to check presence equivalence
            presence_ok = set(keys) == set(expected_keys)

    return keys, presence_ok, all_have_values, order_ok

# @track_performance
def repeated_experiment(
        size: int,
        repetitions: int,
        K: int,
    ) -> None:
    """
    Repeatedly builds random GPlusTrees (with size items) using ranks drawn from a geometric distribution.
    Uses K as target node size to compute the geometric parameter. Aggregates statistics and timings over many trees.
    """
    t_all_0 = time.perf_counter()

    # Storage for stats and timings
    results = []  # List of tuples: (stats, phy_height)
    times_build = []
    times_stats = []
    times_phy = []

    # Generate results from repeated experiments.
    for _ in range(repetitions):
        # Time tree construction
        t0 = time.perf_counter()
        tree = random_klist_tree(size, K)
        times_build.append(time.perf_counter() - t0)

        # Time stats computation
        t0 = time.perf_counter()
        stats = gtree_stats_(tree, {})
        times_stats.append(time.perf_counter() - t0)

        # Time physical height computation
        t0 = time.perf_counter()
        phy_height = tree.physical_height()
        times_phy.append(time.perf_counter() - t0)

        results.append((stats, phy_height))

        assert_invariants(tree, stats)
        print("Tree stats:")
        pprint(asdict(stats))

    # Perfect height: ceil( log_{K+1}(size) )
    perfect_height = math.ceil(math.log(size, K)) if size > 0 else 0

    # Aggregate averages for stats
    avg_gnode_height    = mean(s.gnode_height for s, _ in results)
    avg_gnode_count     = mean(s.gnode_count for s, _ in results)
    avg_item_count      = mean(s.item_count for s, _ in results)
    avg_item_slot_count = mean(s.item_slot_count for s, _ in results)
    avg_space_amp       = mean((s.item_slot_count / s.item_count) for s, _ in results)
    avg_physical_height = mean(h for _, h in results)
    avg_height_amp      = mean((h / perfect_height) for _, h in results) if perfect_height else 0
    avg_avg_gnode_size  = mean((s.item_count / s.gnode_count) for s, _ in results)
    avg_max_rank        = mean(s.rank for s, _ in results)

    # Aggregate averages for timings
    avg_build_time      = mean(times_build)
    avg_stats_time      = mean(times_stats)
    avg_phy_time        = mean(times_phy)

    # Compute variances for stats
    var_gnode_height    = mean((s.gnode_height - avg_gnode_height)**2 for s, _ in results)
    var_gnode_count     = mean((s.gnode_count - avg_gnode_count)**2 for s, _ in results)
    var_item_count      = mean((s.item_count - avg_item_count)**2 for s, _ in results)
    var_item_slot_count = mean((s.item_slot_count - avg_item_slot_count)**2 for s, _ in results)
    var_space_amp       = mean(((s.item_slot_count / s.item_count) - avg_space_amp)**2 for s, _ in results)
    var_physical_height = mean((h - avg_physical_height)**2 for _, h in results)
    var_height_amp      = mean(((h / perfect_height) - avg_height_amp)**2 for _, h in results) if perfect_height else 0
    var_avg_gnode_size  = mean(((s.item_count / s.gnode_count) - avg_avg_gnode_size)**2 for s, _ in results)
    var_max_rank        = mean((s.rank - avg_max_rank)**2 for s, _ in results)

    # Compute variances for timings
    var_build_time      = mean((t - avg_build_time)**2 for t in times_build)
    var_stats_time      = mean((t - avg_stats_time)**2 for t in times_stats)
    var_phy_time        = mean((t - avg_phy_time)**2 for t in times_phy)

    # Prepare rows for stats and timings
    rows = [
        ("Item count",            avg_item_count,       var_item_count),
        ("Item slot count",       avg_item_slot_count,  var_item_slot_count),
        ("Space amplification",    avg_space_amp,        var_space_amp),
        ("G-node count",          avg_gnode_count,      var_gnode_count),
        ("Avg G-node size",       avg_avg_gnode_size,   var_avg_gnode_size),
        ("Maximum rank",          avg_max_rank,         var_max_rank),
        ("G-node height",         avg_gnode_height,     var_gnode_height),
        ("Actual height",         avg_physical_height,  var_physical_height),
        ("Perfect height",        perfect_height,       None),
        ("Height amplification",  avg_height_amp,       var_height_amp),
    ]

    # Log table
    header = f"{'Metric':<20} {'Avg':>15} {'(Var)':>15}"
    sep_line = "-" * len(header)

    # logging.info(f"n = {size}; K = {K}; {repetitions} repetitions")
    logging.info(header)
    logging.info(sep_line)
    for name, avg, var in rows:
        if var is None:
            logging.info(f"{name:<20} {avg:>15}")
        else:
            var_str = f"({var:.2f})"
            avg_fmt = f"{avg:15.2f}"
            logging.info(f"{name:<20} {avg_fmt} {var_str:>15}")
    
    # Performance metrics
    sum_build = sum(times_build)
    sum_stats = sum(times_stats)
    sum_phy   = sum(times_phy)
    total_sum = sum_build + sum_stats + sum_phy

    pct_build = (sum_build / total_sum * 100) if total_sum else 0
    pct_stats = (sum_stats / total_sum * 100) if total_sum else 0
    pct_phy   = (sum_phy   / total_sum * 100) if total_sum else 0

    perf_rows = [
        ("Build time (s)", avg_build_time, var_build_time, sum_build, pct_build),
        ("Stats time (s)", avg_stats_time, var_stats_time, sum_stats, pct_stats),
        ("Phy height time (s)", avg_phy_time, var_phy_time, sum_phy, pct_phy),
    ]
    
    # 2) Log a separate performance table
    header = f"{'Metric':<20}{'Avg(s)':>13}{'Var(s)':>13}{'Total(s)':>13}{'%Total':>10}"
    sep    = "-" * len(header)

    logging.info("")  # blank line for separation
    logging.info("Performance summary:")
    logging.info(header)
    logging.info(sep)
    for name, avg, var, total, pct in perf_rows:
        logging.info(
            f"{name:<20}"
            f"{avg:13.6f}"
            f"{var:13.6f}"
            f"{total:13.6f}"
            f"{pct:10.2f}%"
        )
    

    # # Add method-level performance breakdown
    # logging.info("")
    # logging.info("Method-level performance breakdown:")
    # report = GPlusTree.get_performance_report(sort_by='total_time')
    # for line in report.split('\n'):
    #     logging.info(line)

    logging.info(sep)
    t_all_1 = time.perf_counter() - t_all_0
    logging.info("Execution time: %.3f seconds", t_all_1)

if __name__ == "__main__":
    log_dir = os.path.join(os.getcwd(), "tests/logs")
    os.makedirs(log_dir, exist_ok=True)

    # 2) Create a timestamped logfile name
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = os.path.join(log_dir, f"run_{ts}.log")

    # 3) Configure logging to write to that file (and still print to console, if you like)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler(log_path, mode="w"),
            logging.StreamHandler()         # comment this out if you don't want console output
        ]
    )
    
    # # Enable performance tracking before experiments
    # GPlusTree.enable_performance_tracking()
    # logging.info("Performance tracking enabled")

    # List of tree sizes to test.
    sizes = [1000]
    # sizes = [10, 100, 1000, 10_000, 100_000]
    # List of K values for which we want to run experiments.
    # Ks = [2, 4, 16, 64]
    Ks = [2]
    repetitions = 1

    for n in sizes:
        for K in Ks:
            logging.info("")
            logging.info("")
            logging.info(f"---------------- NOW RUNNING EXPERIMENT: n = {n}, K = {K}, repetitions = {repetitions} ----------------")
            t0 = time.perf_counter()
            repeated_experiment(size=n, repetitions=repetitions, K=K)
            elapsed = time.perf_counter() - t0

            # Reset performance metrics for next experiment
            GPlusTree.reset_performance_metrics()
            logging.info(f"Total experiment time: {elapsed:.3f} seconds")
            # logging.info("Performance metrics reset for next experiment")

    # # Disable tracking when completely done
    # GPlusTree.disable_performance_tracking()
    # logging.info("")
    # logging.info("Performance tracking disabled")
            