
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

from packages.jhehemann.customs.gtree.klist import KList
from packages.jhehemann.customs.gtree.base import Item
from packages.jhehemann.customs.gtree.base import calculate_item_rank

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
        insert_entries = [
            ("delta", 4),
            ("bravo", 2),
            ("alpha", 1),
            ("charlie", 3),
            ("echo", 5),
            ("foxtrot", 6),
            ("golf", 7),
            ("hotel", 8)
        ]
        
        # Shuffle entries to simulate unordered input.
        random.shuffle(insert_entries)
        
        for k, v in insert_entries:
            self.klist.insert(Item(k, v))
        
        # Print the keys for debugging
        print("\nInserted keys order:\n", [k for k, _, _ in insert_entries])

        # Retrieve keys from the klist.
        stored_keys = [entry.item.key for entry in self.klist]
        expected_keys = sorted([k for k, _, _ in insert_entries])

        # Print the stored and expected keys for debugging
        print("\nExpected keys order:\n", expected_keys)
        print("\nStored keys order:\n", stored_keys)

        # Validate that the keys are in the expected (sorted) order.
        self.assertEqual(stored_keys, expected_keys, "Keys should be stored in lexicographic order after insertion.")


    def test_node_overflow(self):
        """Test that inserting more than 4 entries creates new nodes."""
        # Insert 10 entries to force overflow into multiple nodes.
        for i in range(10):
            self.klist.insert(Item(f"key{i}", i))
        num_nodes = self._count_nodes(self.klist)
        self.assertGreater(num_nodes, 1, "Expected multiple nodes due to overflow")

    def test_delete_existent(self):
        """Test that deleting an existing key works correctly and rebalances nodes."""
        insert_entries = [
            ("a", 1),
            ("b", 2),
            ("c", 3),
            ("d", 4),
            ("e", 5),
        ]
        for k, v in insert_entries:
            self.klist.insert(Item(k, v))
        # Delete an entry and verify deletion
        result = self.klist.delete("c")
        self.assertTrue(result)
        keys_after = [entry.item.key for entry in self.klist]
        self.assertNotIn("c", keys_after)
        # Total count should be one less than before.
        self.assertEqual(len(list(self.klist)), len(insert_entries) - 1)

    def test_delete_nonexistent(self):
        initial_keys = ["a", "b", "c"]
        # Insert some entries
        for k in initial_keys:
            self.klist.insert(Item(k, ord(k)))
            
        initial_count = self.klist.item_count()
        updated_klist = self.klist.delete("d")
        
        # The delete method should return the original KList unmodified.
        self.assertIs(updated_klist, self.klist,
                      "Deleting a non-existent key should return the same KList instance.")
        
        self.assertEqual(self.klist.item_count(), initial_count,
                         "KList item count should remain unchanged after deleting a non-existent key.")
        
        # Check that the order and content of keys remain unchanged.
        keys_after = []
        current = self.klist.head
        while current is not None:
            for entry in current.entries:
                keys_after.append(entry.item.key)
            current = current.next
        
        # Since the entries are sorted, keys should match our initial insertion.
        self.assertEqual(keys_after, initial_keys,
                         "The keys in the KList should remain unchanged after an unsuccessful delete.")

    def test_insertion_from_file(self):
        """Test that entries from the dummy data file are inserted in order."""
        file_path = "tests/dummy_vector_data_A.json"
        self.assertTrue(os.path.exists(file_path), f"File {file_path} not found")
        with open(file_path, "r") as f:
            data = json.load(f)
        initial_count = len(data)
        for k, v in data.items():
            self.klist.insert(Item(k, v))
        inserted_keys = [entry.item.key for entry in self.klist]

        # Check that the number of keys matches
        self.assertEqual(len(inserted_keys), initial_count)
        # And check that keys are in sorted order
        self.assertEqual(inserted_keys, sorted(inserted_keys))

    def test_rebalance_after_deletion(self):
        """Test that deleting an element properly rebalances the nodes."""
        # Insert enough entries to span multiple nodes.
        for i in range(12):
            self.klist.insert(Item(f"key{i}", i))
        nodes_before = self._count_nodes(self.klist)
        # Delete a key and expect rebalancing (nodes could merge)
        self.klist.delete("key1")
        nodes_after = self._count_nodes(self.klist)
        self.assertLessEqual(nodes_after, nodes_before,
                             "Rebalancing should merge nodes if possible")
        # Verify the deleted key is no longer present
        keys = [entry.item.key for entry in self.klist]
        self.assertNotIn("key1", keys)

    def test_retrieve_existing(self):
        """
        Insert several items and test retrieve for a key that exists.
        The test verifies that the returned value is correct and that the "next entry"
        corresponds to the entry immediately following the found item.
        """
        insert_entries = [
            ("alpha", "A"),
            ("bravo", "B"),
            ("charlie", "C"),
            ("delta", "D")
        ]
        items = [Item(k, v) for k, v in insert_entries]
        for item in items:
            self.klist.insert(item)
        
        # Retrieve an existing key "bravo"
        result = self.klist.retrieve("bravo")
        self.assertEqual(result.found_entry.item, items[1], "Retrieve should return the correct item for 'bravo'.")
        # Expect the next entry to be the one with key "charlie"
        self.assertIsNotNone(result.next_entry, "Next entry should not be None.")
        self.assertEqual(result.next_entry.item.key, "charlie",
                         "The next entry after 'bravo' should be 'charlie'.")

    def test_retrieve_nonexistent(self):
        """
        Insert several items and test retrieve for keys that do not exist.
        The test verifies that retrieve returns None for the item and
        an appropriate "next entry" (or (None, None) if no such entry exists).
        """
        items = [
            Item("alpha", "A"),
            Item("bravo", "B"),
            Item("charlie", "C"),
            Item("delta", "D")
        ]
        for i in items:
            self.klist.insert(i)
        
        # Test a key that is less than the smallest key.
        result = self.klist.retrieve("aardvark")
        self.assertIsNone(result.found_entry, "Retrieving 'aardvark' should return None.")
        self.assertIsNotNone(result.next_entry, "There should be a next entry for 'aardvark'.")
        self.assertEqual(result.next_entry.item.key, "alpha",
                         "The next entry for 'aardvark' should be 'alpha'.")

        # Test a key that lies between two items (e.g., between 'bravo' and 'charlie').
        result = self.klist.retrieve("bri")
        self.assertIsNone(result.found_entry, "Retrieving a non-existent key 'bri' should return None.")
        self.assertIsNotNone(result.next_entry, "There should be a next entry for key 'bri'.")
        self.assertEqual(result.next_entry.item.key, "charlie",
                         "The next entry for 'bri' should be 'charlie'.")

        # Test a key that is greater than the maximum key.
        result = self.klist.retrieve("zeta")
        self.assertIsNone(result.found_entry, "Retrieving 'zeta' should return None.")
        self.assertIsNone(result.next_entry, "The next entry for 'zeta' should be None, since it is greater than all keys.")

# class TestSplitInPlace(unittest.TestCase):
#     def test_split_in_place(self):
#         """
#         Test the split_in_place method of KList.
#         This method should split the list into two halves and return the second half.
#         """
#         # Create a KList and insert some items
#         klist = KList()
#         for i in range(10):
#             klist.insert(Item(f"key{i}", i))
        
#         # Split the list
#         second_half = klist.split_in_place()
        
#         # Check that the first half is empty and the second half contains the items
#         self.assertEqual(len(klist), 0, "The first half should be empty after splitting.")
#         self.assertEqual(len(second_half), 5, "The second half should contain 5 items.")


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