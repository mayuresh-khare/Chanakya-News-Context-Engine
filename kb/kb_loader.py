"""
CHANAKYA - Knowledge Base Loader
Loads all JSON knowledge base files into a single unified list of entries,
which kb_matcher.py then uses to match extracted entities.
"""

import json
import os

KB_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "knowledge_base")

KB_FILES = [
    "schemes.json",
    "acts.json",
    "ministries.json",
    "people.json",
    "international_orgs.json",
]


def load_knowledge_base() -> list:
    """
    Loads every JSON file in the knowledge base directory and merges
    them into a single flat list of entries. Each entry keeps its
    original fields plus a 'source_file' tag for traceability.
    """
    all_entries = []

    for filename in KB_FILES:
        filepath = os.path.join(KB_DIR, filename)
        if not os.path.exists(filepath):
            print(f"  [WARNING] KB file not found: {filepath}")
            continue

        with open(filepath, "r", encoding="utf-8") as f:
            entries = json.load(f)
            for entry in entries:
                entry["source_file"] = filename
            all_entries.extend(entries)

    return all_entries


def build_lookup_index(kb_entries: list) -> dict:
    """
    Builds a fast lookup dictionary mapping every name AND alias
    (lowercased) directly to its KB entry. This gives us O(1) exact
    matching before falling back to fuzzy matching in kb_matcher.py.
    """
    index = {}
    for entry in kb_entries:
        index[entry["name"].lower()] = entry
        for alias in entry.get("aliases", []):
            index[alias.lower()] = entry
    return index


# ---------------------------------------------------------
# Quick manual test
# ---------------------------------------------------------
if __name__ == "__main__":
    kb = load_knowledge_base()
    print(f"Loaded {len(kb)} total KB entries.\n")

    # Count entries per category
    categories = {}
    for entry in kb:
        cat = entry.get("category", "Unknown")
        categories[cat] = categories.get(cat, 0) + 1

    print("Entries by category:")
    for cat, count in categories.items():
        print(f"  {cat:<30}: {count}")

    index = build_lookup_index(kb)
    print(f"\nLookup index size (names + aliases): {len(index)}")

    print("\nSample lookup - 'PM-KISAN':")
    print(f"  {index.get('pm-kisan')}")

    print("\nSample lookup - 'RBI':")
    print(f"  {index.get('rbi')}")
