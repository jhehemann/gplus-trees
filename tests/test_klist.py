
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
"""Tests for jhehemann/customs/k-list abstract data structure."""
# pylint: skip-file

import unittest
import random
import json
import os
import statistics

from packages.jhehemann.customs.klist.klist import KList
from packages.jhehemann.customs.gtree.gtree import calculate_item_rank

class TestKList(unittest.TestCase):

    def setUp(self):
        self.klist = KList()

    def _count_nodes(self, klist):
        count = 0
        node = klist.head
        while node:
            count += 1
            node = node.next
        return count
    
    def test_random_order_insertion(self):
        """
        Insert entries in random order and verify that the k-list stores them in lexicographic order.
        """
        # Define key-value pairs (keys not in sorted order)
        entries = [
            ("delta", 4),
            ("alpha", 1),
            ("charlie", 3),
            ("bravo", 2),
            ("echo", 5),
            ("foxtrot", 6),
            ("golf", 7),
            ("hotel", 8)
        ]
        # Shuffle entries to simulate unordered input.
        random.shuffle(entries)
        
        
        for key, value in entries:
            self.klist.insert(key, value)
        
        # Print the keys for debugging
        print("Inserted keys order:", [(key, value) for key, value in entries])
        
        # # Print the klist for debugging
        # print("Random order keys:", [key for key, _ in entries])
        # print(self.klist)

        # Retrieve keys from the klist.
        stored_keys = [(k, v) for (k, v), _ in self.klist]
        expected_keys = sorted([(key, value) for key, value in entries])

        # Validate that the keys are in the expected (sorted) order.
        self.assertEqual(stored_keys, expected_keys, "Keys should be stored in lexicographic order after insertion.")


    def test_node_overflow(self):
        """Test that inserting more than 4 entries creates new nodes."""
        # Insert 10 entries to force overflow into multiple nodes.
        for i in range(10):
            self.klist.insert(f"key{i}", i)
        num_nodes = self._count_nodes(self.klist)
        self.assertGreater(num_nodes, 1, "Expected multiple nodes due to overflow")

    def test_delete_existent(self):
        """Test that deleting an existing key works correctly and rebalances nodes."""
        entries = [
            ("a", 1),
            ("b", 2),
            ("c", 3),
            ("d", 4),
            ("e", 5)
        ]
        for key, value in entries:
            self.klist.insert(key, value)
        # Delete an entry and verify deletion
        result = self.klist.delete("c")
        self.assertTrue(result)
        keys_after = [key for key, _ in self.klist]
        self.assertNotIn("c", keys_after)
        # Total count should be one less than before.
        self.assertEqual(len(list(self.klist)), len(entries) - 1)

    def test_delete_nonexistent(self):
        """Test that deleting a nonexistent key returns False."""
        entries = [("x", 10), ("y", 20)]
        for key, value in entries:
            self.klist.insert(key, value)
        result = self.klist.delete("z")
        self.assertFalse(result)

    def test_insertion_from_file(self):
        """Test that entries from the dummy data file are inserted in order."""
        file_path = "tests/dummy_vector_data_A.json"
        self.assertTrue(os.path.exists(file_path), f"File {file_path} not found")
        with open(file_path, "r") as f:
            data = json.load(f)
        initial_count = len(data)
        for key, value in data.items():
            self.klist.insert(key, value)
        inserted_keys = [key for key, _ in self.klist]

        # Check that the number of keys matches
        self.assertEqual(len(inserted_keys), initial_count)
        # And check that keys are in sorted order
        self.assertEqual(inserted_keys, sorted(inserted_keys))

    def test_rebalance_after_deletion(self):
        """Test that deleting an element properly rebalances the nodes."""
        # Insert enough entries to span multiple nodes.
        for i in range(12):
            self.klist.insert(f"key{i}", i)
        nodes_before = self._count_nodes(self.klist)
        # Delete a key and expect rebalancing (nodes could merge)
        self.klist.delete("key1")
        nodes_after = self._count_nodes(self.klist)
        self.assertLessEqual(nodes_after, nodes_before,
                             "Rebalancing should merge nodes if possible")
        # Verify the deleted key is no longer present
        keys = [key for key, _ in self.klist]
        self.assertNotIn("key1", keys)


class TestRankStatistics(unittest.TestCase):
    def test_rank_statistics_from_file(self):
        """
        For each element in dummy_vector_data_A.json, check whether there is a 'rank'
        in the value dictionary. If not, calculate the rank using calculate_gnode_rank with k=8.
        Then, output the maximum rank, mean, and standard deviation of all ranks.
        """
        # Locate the file; adjust the path if needed.
        file_path = "tests/dummy_vector_data_A.json"
        self.assertTrue(os.path.exists(file_path), f"File {file_path} not found")
        
        with open(file_path, "r") as f:
            data = json.load(f)
        
        # Test rounds for different values of k.
        for k in [2, 4, 8, 16]:
            ranks = []
            for key, value in data.items():
                if "rank" in value:
                    rank_value = value["rank"]
                else:
                    rank_value = calculate_item_rank(key, k)
                ranks.append(rank_value)

            max_rank = max(ranks) if ranks else None
            mean_rank = statistics.mean(ranks) if ranks else 0.0
            stdev_rank = statistics.stdev(ranks) if len(ranks) > 1 else 0.0

            print(f"\nRank Statistics for k={k}:")
            print("  Maximum Rank:", max_rank)
            print("  Mean Rank:", mean_rank)
            print("  Standard Deviation:", stdev_rank)

            # Ensure that a rank is at least 1.
            self.assertIsNotNone(max_rank)
            self.assertGreaterEqual(max_rank, 1)


if __name__ == "__main__":
    unittest.main()