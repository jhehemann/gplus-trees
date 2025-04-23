
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

from packages.jhehemann.customs.gtree.klist import KList, KListNode
from packages.jhehemann.customs.gtree.base import Item
from packages.jhehemann.customs.gtree.base import calculate_item_rank

class TestKList(unittest.TestCase):

    def setUp(self):
        self.klist = KList()

    def tearDown(self):
        # automatically verify invariants after each test
        self.klist.check_invariant()

    def _count_nodes(self, klist):
        count = 0
        node = klist.head
        while node:
            count += 1
            node = node.next
        return count

    # def test_insert_in_order(self):
    #     for key in ["a", "b", "c", "d", "e", "f", "g", "h", "i", "j", "j"]:
    #         self.klist.insert(Item(key, ord(key)))
    #     # invariant is checked in tearDown()

    # def test_insert_out_of_order(self):
    #     for key in ["d", "b", "a", "c", "e", "h", "g", "f", "j", "i", "h"]:
    #         self.klist.insert(Item(key, ord(key)))
    #     # invariant is checked in tearDown()
    
    # def test_random_order_insertion(self):
    #     """
    #     Insert entries in random order and verify that the k-list stores them in lexicographic order.
    #     """

    #     # Define key-value pairs (keys not in sorted order)
    #     insert_entries = [
    #         ("delta", 4),
    #         ("bravo", 2),
    #         ("alpha", 1),
    #         ("charlie", 3),
    #         ("echo", 5),
    #         ("foxtrot", 6),
    #         ("golf", 7),
    #         ("hotel", 8)
    #     ]
        
    #     # Shuffle entries to simulate unordered input.
    #     random.shuffle(insert_entries)
        
    #     for k, v in insert_entries:
    #         self.klist.insert(Item(k, v))
        
    #     # Print the keys for debugging
    #     print("\nInserted keys order:\n", [k for k, _ in insert_entries])

    #     # Retrieve keys from the klist.
    #     stored_keys = [entry.item.key for entry in self.klist]
    #     expected_keys = sorted([k for k, _ in insert_entries])

    #     # Print the stored and expected keys for debugging
    #     print("\nExpected keys order:\n", expected_keys)
    #     print("\nStored keys order:\n", stored_keys)

    #     # Validate that the keys are in the expected (sorted) order.
    #     self.assertEqual(stored_keys, expected_keys, "Keys should be stored in lexicographic order after insertion.")


    # def test_node_overflow(self):
    #     """Test that inserting more than 4 entries creates new nodes."""
    #     # Insert 10 entries to force overflow into multiple nodes.
    #     for i in range(10):
    #         self.klist.insert(Item(f"key{i}", i))
    #     num_nodes = self._count_nodes(self.klist)
    #     self.assertGreater(num_nodes, 1, "Expected multiple nodes due to overflow")

    # def test_delete_existent(self):
    #     """Test that deleting an existing key works correctly and rebalances nodes."""
    #     insert_entries = [
    #         ("a", 1),
    #         ("b", 2),
    #         ("c", 3),
    #         ("d", 4),
    #         ("e", 5),
    #     ]
    #     for k, v in insert_entries:
    #         self.klist.insert(Item(k, v))
    #     # Delete an entry and verify deletion
    #     result = self.klist.delete("c")
    #     self.assertTrue(result)
    #     keys_after = [entry.item.key for entry in self.klist]
    #     self.assertNotIn("c", keys_after)
    #     # Total count should be one less than before.
    #     self.assertEqual(len(list(self.klist)), len(insert_entries) - 1)

    # def test_delete_nonexistent(self):
    #     initial_keys = ["a", "b", "c"]
    #     # Insert some entries
    #     for k in initial_keys:
    #         self.klist.insert(Item(k, ord(k)))
            
    #     initial_count = self.klist.item_count()
    #     updated_klist = self.klist.delete("d")
        
    #     # The delete method should return the original KList unmodified.
    #     self.assertIs(updated_klist, self.klist,
    #                   "Deleting a non-existent key should return the same KList instance.")
        
    #     self.assertEqual(self.klist.item_count(), initial_count,
    #                      "KList item count should remain unchanged after deleting a non-existent key.")
        
    #     # Check that the order and content of keys remain unchanged.
    #     keys_after = []
    #     current = self.klist.head
    #     while current is not None:
    #         for entry in current.entries:
    #             keys_after.append(entry.item.key)
    #         current = current.next
        
    #     # Since the entries are sorted, keys should match our initial insertion.
    #     self.assertEqual(keys_after, initial_keys,
    #                      "The keys in the KList should remain unchanged after an unsuccessful delete.")

    # def test_insertion_from_file(self):
    #     """Test that entries from the dummy data file are inserted in order."""
    #     file_path = "tests/dummy_vector_data_A.json"
    #     self.assertTrue(os.path.exists(file_path), f"File {file_path} not found")
    #     with open(file_path, "r") as f:
    #         data = json.load(f)
    #     initial_count = len(data)
    #     for k, v in data.items():
    #         self.klist.insert(Item(k, v))
    #     inserted_keys = [entry.item.key for entry in self.klist]

    #     # Check that the number of keys matches
    #     self.assertEqual(len(inserted_keys), initial_count)
    #     # And check that keys are in sorted order
    #     self.assertEqual(inserted_keys, sorted(inserted_keys))

    # def test_rebalance_after_deletion(self):
    #     """Test that deleting an element properly rebalances the nodes."""
    #     # Insert enough entries to span multiple nodes.
    #     for i in range(12):
    #         self.klist.insert(Item(f"key{i}", i))
    #     nodes_before = self._count_nodes(self.klist)
    #     # Delete a key and expect rebalancing (nodes could merge)
    #     self.klist.delete("key1")
    #     nodes_after = self._count_nodes(self.klist)
    #     self.assertLessEqual(nodes_after, nodes_before,
    #                          "Rebalancing should merge nodes if possible")
    #     # Verify the deleted key is no longer present
    #     keys = [entry.item.key for entry in self.klist]
    #     self.assertNotIn("key1", keys)

    # def test_retrieve_existing(self):
    #     """
    #     Insert several items and test retrieve for a key that exists.
    #     The test verifies that the returned value is correct and that the "next entry"
    #     corresponds to the entry immediately following the found item.
    #     """
    #     insert_entries = [
    #         ("alpha", "A"),
    #         ("bravo", "B"),
    #         ("charlie", "C"),
    #         ("delta", "D")
    #     ]
    #     items = [Item(k, v) for k, v in insert_entries]
    #     for item in items:
    #         self.klist.insert(item)
        
    #     # Retrieve an existing key "bravo"
    #     result = self.klist.retrieve("bravo")
    #     self.assertEqual(result.found_entry.item, items[1], "Retrieve should return the correct item for 'bravo'.")
    #     # Expect the next entry to be the one with key "charlie"
    #     self.assertIsNotNone(result.next_entry, "Next entry should not be None.")
    #     self.assertEqual(result.next_entry.item.key, "charlie",
    #                      "The next entry after 'bravo' should be 'charlie'.")

    # def test_retrieve_nonexistent(self):
    #     """
    #     Insert several items and test retrieve for keys that do not exist.
    #     The test verifies that retrieve returns None for the item and
    #     an appropriate "next entry" (or (None, None) if no such entry exists).
    #     """
    #     items = [
    #         Item("alpha", "A"),
    #         Item("bravo", "B"),
    #         Item("charlie", "C"),
    #         Item("delta", "D")
    #     ]
    #     for i in items:
    #         self.klist.insert(i)
        
    #     # Test a key that is less than the smallest key.
    #     result = self.klist.retrieve("aardvark")
    #     self.assertIsNone(result.found_entry, "Retrieving 'aardvark' should return None.")
    #     self.assertIsNotNone(result.next_entry, "There should be a next entry for 'aardvark'.")
    #     self.assertEqual(result.next_entry.item.key, "alpha",
    #                      "The next entry for 'aardvark' should be 'alpha'.")

    #     # Test a key that lies between two items (e.g., between 'bravo' and 'charlie').
    #     result = self.klist.retrieve("bri")
    #     self.assertIsNone(result.found_entry, "Retrieving a non-existent key 'bri' should return None.")
    #     self.assertIsNotNone(result.next_entry, "There should be a next entry for key 'bri'.")
    #     self.assertEqual(result.next_entry.item.key, "charlie",
    #                      "The next entry for 'bri' should be 'charlie'.")

    #     # Test a key that is greater than the maximum key.
    #     result = self.klist.retrieve("zeta")
    #     self.assertIsNone(result.found_entry, "Retrieving 'zeta' should return None.")
    #     self.assertIsNone(result.next_entry, "The next entry for 'zeta' should be None, since it is greater than all keys.")

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

class TestKListInsert(unittest.TestCase):
    def setUp(self):
        self.klist = KList()
        self.cap = KListNode.CAPACITY

    def tearDown(self):
        # Always verify the core invariants after each test
        self.klist.check_invariant()

    def extract_all_keys(self):
        """Traverse the KList and collect all item keys in order."""
        keys = []
        node = self.klist.head
        while node:
            keys.extend(e.item.key for e in node.entries)
            node = node.next
        return keys

    def test_insert_into_empty(self):
        # Inserting into an empty list should set head and tail
        self.assertIsNone(self.klist.head)
        self.klist.insert(Item("a", 1))
        self.assertIsNotNone(self.klist.head)
        self.assertIs(self.klist.head, self.klist.tail)
        self.assertEqual(self.extract_all_keys(), ["a"])

    def test_insert_in_order(self):
        # Insert keys in sorted order one by one
        for key in ["a", "b", "c", "d", "e"]:
            self.klist.insert(Item(key, ord(key)))
        self.assertEqual(self.extract_all_keys(), ["a", "b", "c", "d", "e"])

    def test_insert_out_of_order(self):
        # Insert keys in random order, final list must be sorted
        for key in ["d", "a", "e", "b", "c"]:
            self.klist.insert(Item(key, ord(key)))
        self.assertEqual(self.extract_all_keys(), ["a", "b", "c", "d", "e"])

    def test_single_node_overflow(self):
        # Fill exactly one node to capacity, then insert one more
        keys = [chr(65 + i) for i in range(self.cap)]
        for k in keys:
            self.klist.insert(Item(k, ord(k)))
        # one more causes a second node
        self.klist.insert(Item("Z", ord("Z")))
        all_keys = self.extract_all_keys()
        self.assertEqual(len(all_keys), self.cap + 1)
        # First node must have cap entries, second node the overflow
        node = self.klist.head
        self.assertEqual([e.item.key for e in node.entries], keys)
        self.assertIsNotNone(node.next)
        self.assertEqual([e.item.key for e in node.next.entries], ["Z"])
        self.assertIs(self.klist.tail, node.next)

    def test_multiple_node_overflows(self):
        # Insert 3*cap + 2 items, ensure we get 4 nodes (since last may be partial)
        total = 3 * self.cap + 2
        keys = [f"{i:03d}" for i in range(total)]
        for k in keys:
            self.klist.insert(Item(k, int(k)))
        # Traverse and count nodes & entries
        node = self.klist.head
        counts = []
        while node:
            counts.append(len(node.entries))
            node = node.next
        # All but the last should be full
        for cnt in counts[:-1]:
            self.assertEqual(cnt, self.cap)
        # Last node has the remainder
        self.assertEqual(sum(counts), total)
        # head/tail correct
        self.assertIsNotNone(self.klist.head)
        self.assertIs(self.klist.tail.entries, node.entries if node else self.klist.tail.entries)

    def test_duplicate_keys(self):
        # Insert duplicate keys – they all should appear, in stable order
        for _ in range(3):
            self.klist.insert(Item("dup", 42))
        self.assertEqual(self.extract_all_keys(), ["dup", "dup", "dup"])

    def test_tail_fast_path(self):
        # Repeatedly append monotonic keys should always hit the tail fast-path
        # and never trigger a linear scan.
        # We can't directly assert “fast-path used,” but we can assert correctness and performance-like behavior.
        for i in range(100):
            self.klist.insert(Item(str(i), i))
        self.assertEqual(self.extract_all_keys(), [str(i) for i in range(100)])
        # Only one node (or more if cap < 100) but always sorted
        self.klist.check_invariant()

    def test_interleaved_inserts_and_checks(self):
        # Interleave inserts with invariant checks to catch transient issues
        sequence = ["m", "a", "z", "b", "y", "c", "x"]
        for key in sequence:
            self.klist.insert(Item(key, ord(key)))
            # After each insert, list so far must be sorted
            so_far = self.extract_all_keys()
            self.assertEqual(so_far, sorted(so_far))

    def test_complex_pattern(self):
        # Insert a complex shuffled pattern repeatedly and verify final sort
        import random
        keys = [chr(65 + i) for i in range(self.cap * 2)]
        for _ in range(5):
            random.shuffle(keys)
            for k in keys:
                self.klist.insert(Item(k, ord(k)))
        # After many insertions, extract and ensure global sort
        all_keys = self.extract_all_keys()
        self.assertEqual(all_keys, sorted(all_keys))
        self.assertEqual(len(all_keys), self.cap * 2 * 5)


# class TestKListDelete(unittest.TestCase):
#     def setUp(self):
#         self.klist = KList()
#         # shorthand capacity
#         self.cap = KListNode.CAPACITY

#     def insert_keys(self, keys):
#         """Helper: insert a sequence of single-character keys with dummy values."""
#         for k in keys:
#             self.klist.insert(Item(k, ord(k)))
#         self.klist.check_invariant()

#     def extract_all_keys(self):
#         """Helper: traverse KList and return all keys in order."""
#         keys = []
#         node = self.klist.head
#         while node:
#             keys.extend([e.item.key for e in node.entries])
#             node = node.next
#         return keys

#     def test_delete_on_empty_list(self):
#         # deleting from an empty KList should do nothing
#         before = self.extract_all_keys()
#         self.klist.delete("x")
#         after = self.extract_all_keys()
#         self.assertEqual(before, after)
#         self.assertIsNone(self.klist.head)
#         self.assertIsNone(self.klist.tail)

#     def test_delete_nonexistent_key(self):
#         # insert some items, then delete a missing key
#         self.insert_keys(["a","b","c"])
#         before = self.extract_all_keys()
#         self.klist.delete("z")
#         after = self.extract_all_keys()
#         self.assertEqual(before, after)

#     def test_delete_only_item(self):
#         # after deleting the sole element, head and tail should be None
#         self.insert_keys(["m"])
#         self.klist.delete("m")
#         self.assertIsNone(self.klist.head)
#         self.assertIsNone(self.klist.tail)

#     def test_delete_head_key(self):
#         # delete the first key in a multi-element, single-node list
#         keys = ["a","b","c"]
#         self.insert_keys(keys)
#         self.klist.delete("a")
#         result = self.extract_all_keys()
#         self.assertEqual(result, ["b","c"])
#         # head should remain the same node
#         self.assertIsNotNone(self.klist.head)
#         self.klist.check_invariant()

#     def test_delete_tail_key(self):
#         # delete the last key in a single-node list
#         keys = ["a","b","c"]
#         self.insert_keys(keys)
#         self.klist.delete("c")
#         result = self.extract_all_keys()
#         self.assertEqual(result, ["a","b"])
#         self.klist.check_invariant()

#     def test_delete_middle_key(self):
#         # delete a middle key and ensure rebalance keeps packing
#         keys = ["a","b","c","d","e"]
#         self.insert_keys(keys)
#         # force at least two nodes by setting CAPACITY small
#         self.assertGreater(len(self.klist.head.entries), 0)
#         self.klist.delete("c")
#         result = self.extract_all_keys()
#         # 'c' is gone, others remain in sorted order
#         self.assertEqual(result, ["a","b","d","e"])
#         self.klist.check_invariant()

#     def test_delete_causes_node_removal(self):
#         # build exactly two nodes, then delete enough to remove the second node
#         # fill first node to capacity, next with 1 entry
#         keys = [chr(i) for i in range(65, 65+self.cap+1)]  # 'A'.. up to capacity+1
#         self.insert_keys(keys)
#         # now head.entries == capacity, second node has 1 element
#         # delete the one in the second node
#         last_key = keys[-1]
#         self.klist.delete(last_key)
#         # the second node should be spliced out
#         self.assertIsNone(self.klist.head.next)
#         # head still has all capacity elements
#         self.assertEqual(len(self.klist.head.entries), self.cap)
#         self.klist.check_invariant()

#     def test_multiple_deletes(self):
#         # delete multiple keys in succession
#         keys = ["a","b","c","d","e","f","g"]
#         self.insert_keys(keys)
#         for k in ["b","e","a","g","d"]:
#             self.klist.delete(k)
#             self.assertNotIn(k, self.extract_all_keys())
#             self.klist.check_invariant()
#         # remaining should be ['c','f']
#         self.assertEqual(self.extract_all_keys(), ["c","f"])

#     def test_repeated_delete_same_key(self):
#         # inserting duplicates—only first matching should be removed
#         # assume KList allows duplicates for this test
#         dup_keys = ["x","x","x"]
#         self.insert_keys(dup_keys)
#         self.klist.delete("x")
#         # exactly two 'x' should remain
#         self.assertEqual(self.extract_all_keys(), ["x","x"])
#         self.klist.delete("x")
#         self.klist.delete("x")
#         # now list is empty
#         self.assertIsNone(self.klist.head)
#         self.assertIsNone(self.klist.tail)

#     def test_delete_all_nodes(self):
#         # insert enough to create 3 nodes, then delete everything one by one
#         keys = [chr(65+i) for i in range(3*self.cap + 2)]  # at least 3 nodes
#         self.insert_keys(keys)
#         for k in keys:
#             self.klist.delete(k)
#         # list should be empty afterwards
#         self.assertIsNone(self.klist.head)
#         self.assertIsNone(self.klist.tail)



# class TestRankStatistics(unittest.TestCase):
#     def test_rank_statistics_from_file(self):
#         """
#         For each element in dummy_vector_data_A.json, check whether there is a 'rank'
#         in the value dictionary. If not, calculate the rank using calculate_gnode_rank with k=8.
#         Then, output the maximum rank, mean, and standard deviation of all ranks.
#         """
#         # Locate the file; adjust the path if needed.
#         file_path = "tests/dummy_vector_data_A.json"
#         self.assertTrue(os.path.exists(file_path), f"File {file_path} not found")
        
#         with open(file_path, "r") as f:
#             data = json.load(f)
        
#         # Test rounds for different values of k.
#         for k in [2, 4, 8, 16]:
#             ranks = []
#             for key, value in data.items():
#                 if "rank" in value:
#                     rank_value = value["rank"]
#                 else:
#                     rank_value = calculate_item_rank(key, k)
#                 ranks.append(rank_value)

#             max_rank = max(ranks) if ranks else None
#             mean_rank = statistics.mean(ranks) if ranks else 0.0
#             stdev_rank = statistics.stdev(ranks) if len(ranks) > 1 else 0.0

#             print(f"\nRank Statistics for k={k}:")
#             print("  Maximum Rank:", max_rank)
#             print("  Mean Rank:", mean_rank)
#             print("  Standard Deviation:", stdev_rank)

#             # Ensure that a rank is at least 1.
#             self.assertIsNotNone(max_rank)
#             self.assertGreaterEqual(max_rank, 1)


if __name__ == "__main__":
    unittest.main()