
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

from src.gplus_trees.klist import KList, KListNode
from src.gplus_trees.gplus_tree import GPlusTree
from src.gplus_trees.base import Item
from src.gplus_trees.base import calculate_item_rank

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

    def test_insert_in_order(self):
        for key in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 10]:
            self.klist.insert(Item(key, f"val_{key}"))
        # invariant is checked in tearDown()

    def test_insert_out_of_order(self):
        for key in [4, 2, 1, 3, 5, 8, 7, 6, 10, 9, 8]:
            self.klist.insert(Item(key, f"val_{key}"))
        # invariant is checked in tearDown()
    
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
    #         (1, 1),
    #         (2, 2),
    #         (3, 3),
    #         (4, 4),
    #         (5, 5),
    #     ]
    #     for k, v in insert_entries:
    #         self.klist.insert(Item(k, v))
    #     # Delete an entry and verify deletion
    #     result = self.klist.delete(3)
    #     self.assertTrue(result)
    #     keys_after = [entry.item.key for entry in self.klist]
    #     self.assertNotIn(3, keys_after)
    #     # Total count should be one less than before.
    #     self.assertEqual(len(list(self.klist)), len(insert_entries) - 1)

    # def test_delete_nonexistent(self):
    #     initial_keys = [1, 2, 3]
    #     # Insert some entries
    #     for k in initial_keys:
    #         self.klist.insert(Item(k, f"val_{k}"))
            
    #     initial_count = self.klist.item_count()
    #     updated_klist = self.klist.delete(4)
        
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
        self.klist.insert(Item(1, "val_1"))
        self.assertIsNotNone(self.klist.head)
        self.assertIs(self.klist.head, self.klist.tail)
        self.assertEqual(self.extract_all_keys(), [1])

    def test_insert_in_order(self):
        # Insert keys in sorted order one by one
        for key in [1, 2, 3, 4, 5]:
            self.klist.insert(Item(key, f"val_{key}"))
        self.assertEqual(self.extract_all_keys(), [1, 2, 3, 4, 5])

    def test_insert_out_of_order(self):
        # Insert keys in random order, final list must be sorted
        for key in [4, 1, 5, 2, 3]:
            self.klist.insert(Item(key, f"val_{key}"))
        self.assertEqual(self.extract_all_keys(), [1, 2, 3, 4, 5])

    def test_single_node_overflow(self):
        # Fill exactly one node to capacity, then insert one more
        keys = list(range(self.cap))
        for k in keys:
            self.klist.insert(Item(k, f"val_{k}"))
        # one more causes a second node
        extra = self.cap
        self.klist.insert(Item(extra, f"val_{extra}"))

        all_keys = self.extract_all_keys()
        self.assertEqual(len(all_keys), self.cap + 1)

        # First node must have cap entries, second node the overflow
        node = self.klist.head
        self.assertEqual([e.item.key for e in node.entries], keys)
        self.assertIsNotNone(node.next)
        self.assertEqual([e.item.key for e in node.next.entries], [extra])
        self.assertIs(self.klist.tail, node.next)

    def test_multiple_node_overflows(self):
        # Insert 3*cap + 2 items, ensure proper node counts
        total = 3 * self.cap + 2
        keys = list(range(total))
        for k in keys:
            self.klist.insert(Item(k, f"val_{k}"))

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

    def test_duplicate_keys(self):
        # Insert duplicate keys – they all should appear in order
        for _ in range(3):
            self.klist.insert(Item(7, "duplicate"))
        self.assertEqual(self.extract_all_keys(), [7, 7, 7])

    def test_tail_fast_path(self):
        # Repeatedly append monotonic integer keys
        for i in range(100):
            self.klist.insert(Item(i, f"val_{i}"))
        self.assertEqual(self.extract_all_keys(), list(range(100)))

    def test_interleaved_inserts_and_checks(self):
        # Interleave inserts with invariant checks to catch transient issues
        sequence = [5, 2, 8, 3, 9, 1, 7]
        for key in sequence:
            self.klist.insert(Item(key, f"val_{key}"))
            so_far = self.extract_all_keys()
            self.assertEqual(so_far, sorted(so_far))

    def test_complex_pattern(self):
        # Insert a complex shuffled pattern repeatedly and verify final sort
        import random
        keys = list(range(self.cap * 2))
        for _ in range(5):
            random.shuffle(keys)
            for k in keys:
                self.klist.insert(Item(k, f"val_{k}"))

        all_keys = self.extract_all_keys()
        self.assertEqual(all_keys, sorted(all_keys))
        self.assertEqual(len(all_keys), self.cap * 2 * 5)


class TestKListDelete(unittest.TestCase):
    def setUp(self):
        self.klist = KList()
        self.cap = KListNode.CAPACITY

    def insert_keys(self, keys):
        """Helper: insert a sequence of integer keys with dummy values."""
        for k in keys:
            self.klist.insert(Item(k, f"val_{k}"))
        self.klist.check_invariant()

    def extract_all_keys(self):
        """Helper: traverse KList and return all keys in order."""
        keys = []
        node = self.klist.head
        while node:
            keys.extend(e.item.key for e in node.entries)
            node = node.next
        return keys

    def test_delete_on_empty_list(self):
        # deleting from an empty KList should do nothing
        before = self.extract_all_keys()
        self.klist.delete(999)           # nonexistent int
        after = self.extract_all_keys()
        self.assertEqual(before, after)
        self.assertIsNone(self.klist.head)
        self.assertIsNone(self.klist.tail)

    def test_delete_nonexistent_key(self):
        # insert some items, then delete a missing key
        self.insert_keys([1, 2, 3])
        before = self.extract_all_keys()
        self.klist.delete(999)
        after = self.extract_all_keys()
        self.assertEqual(before, after)

    def test_delete_only_item(self):
        # after deleting the sole element, head and tail should be None
        self.insert_keys([5])
        self.klist.delete(5)
        self.assertIsNone(self.klist.head)
        self.assertIsNone(self.klist.tail)

    def test_delete_head_key(self):
        # delete the first key in a multi-element, single-node list
        keys = [1, 2, 3]
        self.insert_keys(keys)
        self.klist.delete(1)
        result = self.extract_all_keys()
        self.assertEqual(result, [2, 3])
        # head should remain the same node
        self.assertIsNotNone(self.klist.head)
        self.klist.check_invariant()

    def test_delete_tail_key(self):
        # delete the last key in a single-node list
        keys = [1, 2, 3]
        self.insert_keys(keys)
        self.klist.delete(3)
        result = self.extract_all_keys()
        self.assertEqual(result, [1, 2])
        self.klist.check_invariant()

    def test_delete_middle_key(self):
        # delete a middle key and ensure rebalance keeps packing
        keys = [1, 2, 3, 4, 5]
        self.insert_keys(keys)
        # ensure at least two nodes exist
        self.assertGreater(len(self.klist.head.entries), 0)
        self.klist.delete(3)
        result = self.extract_all_keys()
        self.assertEqual(result, [1, 2, 4, 5])
        self.klist.check_invariant()

    def test_delete_causes_node_removal(self):
        # build exactly two nodes: first full, second with 1 entry
        keys = list(range(self.cap + 1))
        self.insert_keys(keys)
        # delete the lone entry in the second node
        last_key = keys[-1]
        self.klist.delete(last_key)
        # the second node should be spliced out
        self.assertIsNone(self.klist.head.next)
        # head still has all capacity entries
        self.assertEqual(len(self.klist.head.entries), self.cap)
        self.klist.check_invariant()

    def test_multiple_deletes(self):
        # delete multiple keys in succession
        keys = [1, 2, 3, 4, 5, 6, 7]
        self.insert_keys(keys)
        for k in [2, 5, 1, 7, 4]:
            self.klist.delete(k)
            self.assertNotIn(k, self.extract_all_keys())
            self.klist.check_invariant()
        # remaining should be [3,6]
        self.assertEqual(self.extract_all_keys(), [3, 6])

    def test_repeated_delete_same_key(self):
        # inserting duplicates—only first matching should be removed each time
        dup_key = 7
        self.insert_keys([dup_key, dup_key, dup_key])
        self.klist.delete(dup_key)
        # exactly two remain
        self.assertEqual(self.extract_all_keys(), [dup_key, dup_key])
        self.klist.delete(dup_key)
        self.klist.delete(dup_key)
        # now list is empty
        self.assertIsNone(self.klist.head)
        self.assertIsNone(self.klist.tail)

    def test_delete_all_nodes(self):
        # insert enough to create 3+ nodes, then delete everything one by one
        keys = list(range(3 * self.cap + 2))
        self.insert_keys(keys)
        for k in keys:
            self.klist.delete(k)
        # list should be empty afterwards
        self.assertIsNone(self.klist.head)
        self.assertIsNone(self.klist.tail)


class TestKListRetrieve(unittest.TestCase):
    def setUp(self):
        self.klist = KList()
        self.cap = KListNode.CAPACITY

    def insert_sequence(self, keys):
        """Helper to insert integer keys with dummy values."""
        for k in keys:
            self.klist.insert(Item(k, f"val_{k}"))
        # ensure invariants
        self.klist.check_invariant()

    def assertRetrieval(self, key, found_key, next_key):
        """
        Helper: call retrieve(key) and assert that
          result.found_entry.item.key == found_key  (or None)
          result.next_entry.item.key == next_key    (or None)
        """
        res = self.klist.retrieve(key)
        if found_key is None:
            self.assertIsNone(res.found_entry, f"Expected no entry for {key}")
        else:
            self.assertIsNotNone(res.found_entry)
            self.assertEqual(res.found_entry.item.key, found_key)
        if next_key is None:
            self.assertIsNone(res.next_entry, f"Expected no successor for {key}")
        else:
            self.assertIsNotNone(res.next_entry)
            self.assertEqual(res.next_entry.item.key, next_key)

    def test_retrieve_empty(self):
        # empty list returns (None, None)
        res = self.klist.retrieve(123)
        self.assertIsNone(res.found_entry)
        self.assertIsNone(res.next_entry)

    def test_type_error_on_non_int(self):
        with self.assertRaises(TypeError):
            self.klist.retrieve("not-an-int")

    def test_single_node_exact_middle(self):
        # fill one node without overflow
        keys = [10, 20, 30]
        self.insert_sequence(keys)
        # exact match in middle
        self.assertRetrieval(20, 20, 30)

    def test_single_node_exact_first(self):
        keys = [5, 15, 25]
        self.insert_sequence(keys)
        self.assertRetrieval(5, 5, 15)

    def test_single_node_exact_last(self):
        keys = [1, 2, 3]
        self.insert_sequence(keys)
        # found at last position → successor None
        self.assertRetrieval(3, 3, None)

    def test_single_node_between(self):
        keys = [100, 200, 300]
        self.insert_sequence(keys)
        # between 100 and 200
        self.assertRetrieval(150, None, 200)

    def test_single_node_below_min(self):
        keys = [50, 60]
        self.insert_sequence(keys)
        # below first
        self.assertRetrieval(40, None, 50)

    def test_single_node_above_max(self):
        keys = [7, 8, 9]
        self.insert_sequence(keys)
        # above last
        self.assertRetrieval(100, None, None)

    def test_cross_node_exact_and_successor(self):
        # overflow into two nodes
        # capacity = 4, so use 5 items
        keys = [1, 2, 3, 4, 5]
        self.insert_sequence(keys)
        # 4 is last of first node, successor should be first of second node (5)
        self.assertRetrieval(4, 4, 5)
        # 5 is in second node, exact last → successor None
        self.assertRetrieval(5, 5, None)

    def test_cross_node_between(self):
        # retrieve only accepts int keys, so passing a float should raise
        with self.assertRaises(TypeError):
            self.klist.retrieve(4.5)

    def test_cross_node_below_head(self):
        keys = [10, 20, 30, 40, 50]
        self.insert_sequence(keys)
        # below first in head
        self.assertRetrieval(5, None, 10)

    def test_cross_node_above_tail(self):
        keys = [10, 20, 30, 40, 50]
        self.insert_sequence(keys)
        # above max across all nodes
        self.assertRetrieval(1000, None, None)

    def test_bulk_retrieval_all_keys(self):
        # retrieve each real key should find itself and next
        keys = list(range(1, self.cap * 2 + 1))  # generate enough to overflow
        self.insert_sequence(keys)
        for idx, k in enumerate(keys):
            expected_next = keys[idx+1] if idx+1 < len(keys) else None
            self.assertRetrieval(k, k, expected_next)

    def test_random_nonexistent(self):
        keys = list(range(0, self.cap * 3))
        self.insert_sequence(keys)
        low, high = -10, max(keys) + 10

        for _ in range(20):
            x = random.randint(low, high)
            if x in keys:
                continue  # skip existing keys

            # find the smallest key > x
            next_candidates = [k for k in keys if k > x]
            nxt = min(next_candidates) if next_candidates else None

            # now x is an int, so retrieve(x) works
            self.assertRetrieval(x, None, nxt)


class TestKListGetEntry(unittest.TestCase):
    def setUp(self):
        self.klist = KList()
        self.cap = KListNode.CAPACITY

    def insert_sequence(self, keys):
        """Helper: insert integer keys with dummy values."""
        for k in keys:
            self.klist.insert(Item(k, f"val_{k}"))
        self.klist.check_invariant()

    def assertGet(self, index, found_key, next_key):
        """Helper: assert get_entry(index) returns expected keys."""
        res = self.klist.get_entry(index)
        if found_key is None:
            self.assertIsNone(res.found_entry, f"Expected no entry at index {index}")
        else:
            self.assertIsNotNone(res.found_entry, f"Expected entry at index {index}")
            self.assertEqual(res.found_entry.item.key, found_key)
        if next_key is None:
            self.assertIsNone(res.next_entry, f"Expected no successor for index {index}")
        else:
            self.assertIsNotNone(res.next_entry, f"Expected successor for index {index}")
            self.assertEqual(res.next_entry.item.key, next_key)

    def test_empty_list(self):
        # retrieving any index from empty list returns (None, None)
        for idx in [0, 1, -1, 100]:
            self.assertGet(idx, None, None)

    def test_type_error_non_int(self):
        with self.assertRaises(TypeError):
            self.klist.get_entry('0')
        with self.assertRaises(TypeError):
            self.klist.get_entry(1.5)

    def test_single_node_boundaries(self):
        keys = [10, 20, 30]
        self.insert_sequence(keys)
        # valid indices
        self.assertGet(0, 10, 20)
        self.assertGet(1, 20, 30)
        self.assertGet(2, 30, None)
        # out of range
        self.assertGet(-1, None, None)
        self.assertGet(3, None, None)

    def test_single_node_varied(self):
        keys = [5]
        self.insert_sequence(keys)
        self.assertGet(0, 5, None)
        self.assertGet(1, None, None)

    def test_two_nodes_indexing(self):
        # fill first node, overflow one into second
        keys = list(range(self.cap + 1))
        self.insert_sequence(keys)
        # first node: indices 0..cap-1
        for i in range(self.cap):
            next_key = i+1
            if next_key == self.cap:
                # next entry is first of second node
                expected_next = self.cap
            else:
                expected_next = next_key
            self.assertGet(i, i, expected_next)
        # index cap is first of second node
        self.assertGet(self.cap, self.cap, None)

    def test_multi_node_full_scan(self):
        total = 3 * self.cap + 2
        keys = list(range(total))
        self.insert_sequence(keys)
        # test all indices
        for i in range(total):
            expected_next = i+1 if i+1 < total else None
            self.assertGet(i, i, expected_next)
        # out of bounds
        self.assertGet(total, None, None)
        self.assertGet(total+5, None, None)

    def test_rebuild_index_on_modification(self):
        # if index is maintained, ensure it updates
        if not hasattr(self.klist, '_prefix_counts'):
            self.skipTest("Index not implemented")
        # initial insert
        keys = [1, 2, 3, 4]
        self.insert_sequence(keys)
        # delete middle
        self.klist.delete(2)
        self.klist._rebuild_index()
        # now index 1 should be key=3
        self.assertGet(1, 3, 4)


class TestKListIndex(unittest.TestCase):
    def setUp(self):
        self.klist = KList()
        self.cap = KListNode.CAPACITY

    def test_empty_index(self):
        # Before any operations, index lists should exist and be empty
        self.assertTrue(hasattr(self.klist, "_nodes"))
        self.assertEqual(self.klist._nodes, [], "_nodes should be initialized empty")
        self.assertTrue(hasattr(self.klist, "_prefix_counts"))
        self.assertEqual(self.klist._prefix_counts, [], "_prefix_counts should be initialized empty")

    def test_index_after_single_insert(self):
        # Insert one item, rebuild, then index should have 1 node, prefix_counts [1]
        self.klist.insert(Item(10, "val_10"))
        self.assertEqual(len(self.klist._nodes), 1)
        self.assertEqual(self.klist._prefix_counts, [1])
        # Node in list should be the head
        self.assertIs(self.klist._nodes[0], self.klist.head)

    def test_index_after_multiple_inserts_no_overflow(self):
        # Insert fewer than CAPACITY items
        keys = list(range(self.cap - 1))
        for k in keys:
            self.klist.insert(Item(k, f"v{k}"))
        # Still one node, prefix_counts = [len(keys)]
        self.assertEqual(len(self.klist._nodes), 1)
        self.assertEqual(self.klist._prefix_counts, [len(keys)])
    
    def test_index_after_overflow(self):
        # Insert exactly CAPACITY + 2 items → 2 nodes
        total = self.cap + 2
        for k in range(total):
            self.klist.insert(Item(k, f"v{k}"))
        # Should have 2 nodes
        self.assertEqual(len(self.klist._nodes), 2)
        # prefix_counts: first node cap, second cap+2
        expected = [self.cap, total]
        self.assertEqual(self.klist._prefix_counts, expected)
        # Check that _nodes entries match actual chain
        node = self.klist.head
        for idx, n in enumerate(self.klist._nodes):
            self.assertIs(n, node)
            node = node.next

    def test_prefix_counts_monotonic_and_correct(self):
        # Random insertion pattern, then check prefix sums
        keys = [5, 1, 9, 2, 8, 3, 7, 4, 6, 0]
        for k in keys:
            self.klist.insert(Item(k, f"v{k}"))
        # Now delete a few to force structure change
        for k in [9, 0]:
            self.klist.delete(k)
        # Compute expected prefix sums by traversing
        running = 0
        expected = []
        node = self.klist.head
        while node:
            running += len(node.entries)
            expected.append(running)
            node = node.next
        self.assertEqual(self.klist._prefix_counts, expected)
        # Ensure strictly increasing
        for a, b in zip(expected, expected[1:]):
            self.assertLess(a, b)

    def test_index_after_bulk_deletes(self):
        # Fill three nodes exactly, then remove the middle node
        total = 3 * self.cap
        for k in range(total):
            self.klist.insert(Item(k, f"v{k}"))
        # delete all keys in the middle node
        middle_start = self.cap
        middle_end   = 2 * self.cap - 1
        for k in range(middle_start, middle_end + 1):
            self.klist.delete(k)
        # Now exactly two nodes remain (the head and the tail)
        self.assertEqual(len(self.klist._nodes), 2)
        # And the prefix sums should be [CAPACITY, 3*CAPACITY]
        self.assertEqual(
            self.klist._prefix_counts,
            [self.cap, 2 * self.cap]
        )


class TestUpdateLeftSubtree(unittest.TestCase):
    def setUp(self):
        self.klist = KList()
        self.cap = KListNode.CAPACITY
        # Trees to attach
        self.treeA = GPlusTree()
        self.treeB = GPlusTree()

    def extract_left_subtrees(self):
        """Helper to collect left_subtree pointers for all entries."""
        subs = []
        node = self.klist.head
        while node:
            subs.extend(entry.left_subtree for entry in node.entries)
            node = node.next
        return subs

    def test_update_on_empty_list(self):
        # Updating an empty list should do nothing and return self
        returned = self.klist.update_left_subtree(1, self.treeA)
        self.assertIs(returned, self.klist)
        # List still empty
        self.assertIsNone(self.klist.head)
        self.assertIsNone(self.klist.tail)

    def test_update_nonexistent_key(self):
        # Insert some keys, then update a non-existent one
        keys = [1, 2, 3]
        for k in keys:
            self.klist.insert(Item(k, f"val_{k}"))
        before = self.extract_left_subtrees()
        returned = self.klist.update_left_subtree(99, self.treeA)
        self.assertIs(returned, self.klist)
        after = self.extract_left_subtrees()
        self.assertEqual(before, after)

    def test_update_first_entry(self):
        # Insert keys and update the first key
        keys = [10, 20, 30]
        for k in keys:
            self.klist.insert(Item(k, f"val_{k}"))
        returned = self.klist.update_left_subtree(10, self.treeA)
        self.assertIs(returned, self.klist)
        # First entry gets treeA, others remain None
        subs = self.extract_left_subtrees()
        self.assertEqual(subs[0], self.treeA)
        self.assertTrue(all(s is None for s in subs[1:]))

    def test_update_last_entry(self):
        # Insert keys and update the last key
        keys = [5, 6, 7]
        for k in keys:
            self.klist.insert(Item(k, f"val_{k}"))
        returned = self.klist.update_left_subtree(7, self.treeB)
        self.assertIs(returned, self.klist)
        subs = self.extract_left_subtrees()
        # Last subtree updated
        self.assertEqual(subs[-1], self.treeB)
        self.assertTrue(all(s is None for s in subs[:-1]))

    def test_update_middle_entry(self):
        # Insert multiple keys and update a middle one
        keys = [1, 2, 3, 4, 5]
        for k in keys:
            self.klist.insert(Item(k, f"val_{k}"))
        returned = self.klist.update_left_subtree(3, self.treeA)
        self.assertIs(returned, self.klist)
        subs = self.extract_left_subtrees()
        # Only the third entry is updated
        for i, s in enumerate(subs):
            if keys[i] == 3:
                self.assertIs(s, self.treeA)
            else:
                self.assertIsNone(s)

    def test_update_after_overflow(self):
        # Force two nodes and update an entry in second node
        total = self.cap + 2
        for k in range(total):
            self.klist.insert(Item(k, f"val_{k}"))
        # Update key = cap (first entry in second node)
        returned = self.klist.update_left_subtree(self.cap, self.treeB)
        self.assertIs(returned, self.klist)
        # Traverse to the second node:
        node = self.klist.head.next
        # First entry in that node should have left_subtree = treeB
        self.assertIs(node.entries[0].left_subtree, self.treeB)

    def test_chained_updates(self):
        # Multiple updates in sequence
        keys = [1, 2, 3]
        for k in keys:
            self.klist.insert(Item(k, f"val_{k}"))
        self.klist.update_left_subtree(1, self.treeA)
        self.klist.update_left_subtree(2, self.treeB)
        subs = self.extract_left_subtrees()
        self.assertIs(subs[0], self.treeA)
        self.assertIs(subs[1], self.treeB)
        self.assertIsNone(subs[2])

    def test_type_error_on_non_int_key(self):
        with self.assertRaises(TypeError):
            self.klist.update_left_subtree("not-int", self.treeA)


class TestSplitInplace(unittest.TestCase):
    def setUp(self):
        self.klist = KList()
        self.cap = KListNode.CAPACITY

    def extract_keys(self, kl: KList):
        """Return list of keys in order from KList."""
        keys = []
        node = kl.head
        while node:
            keys.extend(entry.item.key for entry in node.entries)
            node = node.next
        return keys

    def test_empty_split(self):
        left, subtree, right = self.klist.split_inplace(5)
        # both sides empty
        self.assertEqual(self.extract_keys(left), [])
        self.assertIsNone(subtree)
        self.assertEqual(self.extract_keys(right), [])
        # invariants hold
        left.check_invariant()
        right.check_invariant()

    def test_split_before_all_keys(self):
        # Insert some keys
        keys = [10, 20, 30]
        for k in keys:
            self.klist.insert(Item(k, f"v{k}"))
        # split before smallest
        left, subtree, right = self.klist.split_inplace(5)
        self.assertEqual(self.extract_keys(left), [])
        self.assertIsNone(subtree)
        self.assertEqual(self.extract_keys(right), keys)
        left.check_invariant()
        right.check_invariant()

    def test_split_after_all_keys(self):
        keys = [1, 2, 3]
        for k in keys:
            self.klist.insert(Item(k, f"v{k}"))
        # split after largest
        left, subtree, right = self.klist.split_inplace(10)
        self.assertEqual(self.extract_keys(left), keys)
        self.assertIsNone(subtree)
        self.assertEqual(self.extract_keys(right), [])
        left.check_invariant()
        right.check_invariant()

    def test_split_exact_middle(self):
        keys = [1, 2, 3, 4, 5]
        for k in keys:
            self.klist.insert(Item(k, f"v{k}"))
        # split on 3
        left, subtree, right = self.klist.split_inplace(3)
        self.assertEqual(self.extract_keys(left), [1, 2])
        self.assertIsNone(subtree)  # default left_subtree None
        self.assertEqual(self.extract_keys(right), [4, 5])
        left.check_invariant()
        right.check_invariant()

    def test_split_nonexistent_between(self):
        keys = [10, 20, 30, 40]
        for k in keys:
            self.klist.insert(Item(k, f"v{k}"))
        # split on a key not present but between 20 and 30
        left, subtree, right = self.klist.split_inplace(25)
        self.assertEqual(self.extract_keys(left), [10, 20])
        self.assertIsNone(subtree)
        self.assertEqual(self.extract_keys(right), [30, 40])
        left.check_invariant()
        right.check_invariant()

    def test_split_at_node_boundary(self):
        # make at least two nodes
        total = self.cap + 2
        for k in range(total):
            self.klist.insert(Item(k, f"v{k}"))
        # first node has keys 0..cap-1, second has [cap+1]
        # split exactly at cap (first of second node)
        left, subtree, right = self.klist.split_inplace(self.cap)
        self.assertEqual(self.extract_keys(left), list(range(self.cap)))
        self.assertIsNone(subtree)
        self.assertEqual(self.extract_keys(right), [self.cap+1])
        left.check_invariant()
        right.check_invariant()

    def test_split_with_subtree_propagation(self):
        # insert and assign a left_subtree for a particular key
        keys = [1, 2, 3]
        for k in keys:
            self.klist.insert(Item(k, f"v{k}"))
        # update left_subtree for key=2
        subtree = GPlusTree()
        self.klist.update_left_subtree(2, subtree)
        left, st, right = self.klist.split_inplace(2)
        # left contains [1], subtree returned
        self.assertEqual(self.extract_keys(left), [1])
        self.assertIs(st, subtree)
        self.assertEqual(self.extract_keys(right), [3])
        left.check_invariant()
        right.check_invariant()

    def test_split_multiple_times(self):
        # perform sequential splits
        keys = list(range(6))
        for k in keys:
            self.klist.insert(Item(k, f"v{k}"))
        # split on 2
        l1, _, r1 = self.klist.split_inplace(2)
        # split r1 on 4
        l2, _, r2 = r1.split_inplace(4)
        self.assertEqual(self.extract_keys(l1), [0, 1])
        self.assertEqual(self.extract_keys(l2), [3])
        self.assertEqual(self.extract_keys(r2), [5])
        l1.check_invariant()
        l2.check_invariant()
        r2.check_invariant()

    def test_type_error_non_int_key(self):
        with self.assertRaises(TypeError):
            self.klist.split_inplace("not-int")


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