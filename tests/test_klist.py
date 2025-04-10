
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
import datetime

from packages.jhehemann.customs.gtree.klist import KList
from packages.jhehemann.customs.gtree.base import Item
from packages.jhehemann.customs.gtree.base import calculate_item_rank

BASE_TIMESTAMP = datetime.datetime(2021, 1, 1, tzinfo=datetime.timezone.utc)

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
            ("delta", 4, BASE_TIMESTAMP),
            ("alpha", 1, BASE_TIMESTAMP),
            ("charlie", 3, BASE_TIMESTAMP),
            ("bravo", 2, BASE_TIMESTAMP),
            ("echo", 5, BASE_TIMESTAMP),
            ("foxtrot", 6, BASE_TIMESTAMP),
            ("golf", 7, BASE_TIMESTAMP),
            ("hotel", 8, BASE_TIMESTAMP)
        ]
        
        # Create an Item instance for each (key, value) pair.
        #items = [Item(k, v, timestamp) for k, v, timestamp in entries]
        
        # Shuffle entries to simulate unordered input.
        random.shuffle(entries)
        
        for k, v, timestamp in entries:
            self.klist.insert(Item(k, v, timestamp))
        
        # Print the keys for debugging
        print("\nInserted keys order:\n", [k for k, _, _ in entries])

        # Retrieve keys from the klist.
        stored_keys = [item.key for item, _ in self.klist]
        expected_keys = sorted([k for k, _, _ in entries])

        # Print the stored and expected keys for debugging
        print("\nExpected keys order:\n", expected_keys)
        print("\nStored keys order:\n", stored_keys)

        # Validate that the keys are in the expected (sorted) order.
        self.assertEqual(stored_keys, expected_keys, "Keys should be stored in lexicographic order after insertion.")


    def test_node_overflow(self):
        """Test that inserting more than 4 entries creates new nodes."""
        # Insert 10 entries to force overflow into multiple nodes.
        for i in range(10):
            self.klist.insert(Item(f"key{i}", i, BASE_TIMESTAMP))
        num_nodes = self._count_nodes(self.klist)
        self.assertGreater(num_nodes, 1, "Expected multiple nodes due to overflow")

    def test_delete_existent(self):
        """Test that deleting an existing key works correctly and rebalances nodes."""
        entries = [
            ("a", 1, BASE_TIMESTAMP),
            ("b", 2, BASE_TIMESTAMP),
            ("c", 3, BASE_TIMESTAMP),
            ("d", 4, BASE_TIMESTAMP),
            ("e", 5, BASE_TIMESTAMP)
        ]
        for k, v, timestamp in entries:
            self.klist.insert(Item(k, v, timestamp))
        # Delete an entry and verify deletion
        result = self.klist.delete("c")
        self.assertTrue(result)
        keys_after = [item.key for item, _ in self.klist]
        self.assertNotIn("c", keys_after)
        # Total count should be one less than before.
        self.assertEqual(len(list(self.klist)), len(entries) - 1)

    def test_delete_nonexistent(self):
        """Test that deleting a nonexistent key returns False."""
        entries = [("x", 10, BASE_TIMESTAMP), ("y", 20, BASE_TIMESTAMP)]
        for k, v, timestamp in entries:
            self.klist.insert(Item(k, v, timestamp))
        result = self.klist.delete("z")
        self.assertFalse(result)

    def test_insertion_from_file(self):
        """Test that entries from the dummy data file are inserted in order."""
        file_path = "tests/dummy_vector_data_A.json"
        self.assertTrue(os.path.exists(file_path), f"File {file_path} not found")
        with open(file_path, "r") as f:
            data = json.load(f)
        initial_count = len(data)
        for k, v in data.items():
            self.klist.insert(Item(k, v, BASE_TIMESTAMP))
        inserted_keys = [item.key for item, _ in self.klist]

        # Check that the number of keys matches
        self.assertEqual(len(inserted_keys), initial_count)
        # And check that keys are in sorted order
        self.assertEqual(inserted_keys, sorted(inserted_keys))

    def test_rebalance_after_deletion(self):
        """Test that deleting an element properly rebalances the nodes."""
        # Insert enough entries to span multiple nodes.
        for i in range(12):
            self.klist.insert(Item(f"key{i}", i, BASE_TIMESTAMP))
        nodes_before = self._count_nodes(self.klist)
        # Delete a key and expect rebalancing (nodes could merge)
        self.klist.delete("key1")
        nodes_after = self._count_nodes(self.klist)
        self.assertLessEqual(nodes_after, nodes_before,
                             "Rebalancing should merge nodes if possible")
        # Verify the deleted key is no longer present
        keys = [item.key for item, _ in self.klist]
        self.assertNotIn("key1", keys)

    def test_retrieve_existing(self):
        """
        Insert several items and test retrieve for a key that exists.
        The test verifies that the returned value is correct and that the "next entry"
        corresponds to the entry immediately following the found item.
        """
        items = [
            Item("alpha", "A", BASE_TIMESTAMP),
            Item("bravo", "B", BASE_TIMESTAMP),
            Item("charlie", "C", BASE_TIMESTAMP),
            Item("delta", "D", BASE_TIMESTAMP)
        ]
        for i in items:
            self.klist.insert(i)
        
        # Retrieve an existing key "bravo"
        item, next_entry = self.klist.retrieve("bravo")
        self.assertEqual(item, items[1], "Retrieve should return the correct item for 'bravo'.")
        # Expect the next entry to be the one with key "charlie"
        next_item, _ = next_entry
        self.assertIsNotNone(next_item, "Next entry should not be None.")
        self.assertEqual(next_item.key, "charlie",
                         "The next entry after 'bravo' should be 'charlie'.")

    def test_retrieve_nonexistent(self):
        """
        Insert several items and test retrieve for keys that do not exist.
        The test verifies that retrieve returns None for the item and
        an appropriate "next entry" (or (None, None) if no such entry exists).
        """
        items = [
            Item("alpha", "A", BASE_TIMESTAMP),
            Item("bravo", "B", BASE_TIMESTAMP),
            Item("charlie", "C", BASE_TIMESTAMP),
            Item("delta", "D", BASE_TIMESTAMP)
        ]
        for i in items:
            self.klist.insert(i)
        
        # Test a key that is less than the smallest key.
        item, next_entry = self.klist.retrieve("aardvark")
        self.assertIsNone(item, "Retrieving 'aardvark' should return None.")
        next_item, _ = next_entry
        self.assertIsNotNone(next_item, "There should be a next entry for 'aardvark'.")
        self.assertEqual(next_item.key, "alpha",
                         "The next entry for 'aardvark' should be 'alpha'.")

        # Test a key that lies between two items (e.g., between 'bravo' and 'charlie').
        item, next_entry = self.klist.retrieve("bri")
        self.assertIsNone(item, "Retrieving a non-existent key 'bri' should return None.")
        next_item, _ = next_entry
        self.assertIsNotNone(next_item, "There should be a next entry for key 'bri'.")
        self.assertEqual(next_item.key, "charlie",
                         "The next entry for 'bri' should be 'charlie'.")

        # Test a key that is greater than the maximum key.
        item, next_entry = self.klist.retrieve("zeta")
        self.assertIsNone(item, "Retrieving 'zeta' should return None.")
        self.assertEqual(next_entry, (None, None),
                         "The next entry for 'zeta' should be (None, None), since it is greater than all keys.")


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