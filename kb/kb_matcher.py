"""
CHANAKYA - Knowledge Base Matcher
Matches entities extracted by NER (Experiment 7) against the Knowledge
Base, using a three-tier strategy:
  1. Exact match (fast, via lookup index)
  2. Substring/partial match (handles "Sitharaman" -> "Nirmala Sitharaman")
  3. Fuzzy match via Edit Distance (handles typos, from Experiment 3)

This is Stage 7 of CHANAKYA's architecture -- the step that actually
connects a news article's entities to real contextual information.
"""

import sys
import os
import re

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from kb.kb_loader import load_knowledge_base, build_lookup_index
from nlp.spelling_correction import edit_distance


def _contains_as_phrase(needle: str, haystack: str) -> bool:
    """
    Checks if `needle` appears in `haystack` as a whole word/phrase,
    using word boundaries -- NOT a raw substring check.

    This avoids false positives like "MoD" (Ministry of Defence alias)
    incorrectly matching inside "Modi" (na-render-MOD-i), which a naive
    `needle in haystack` substring check would wrongly allow.
    """
    if not needle or not haystack:
        return False
    pattern = r'\b' + re.escape(needle) + r'\b'
    return re.search(pattern, haystack) is not None


class KnowledgeBaseMatcher:
    """
    Wraps the loaded Knowledge Base and provides matching methods
    against entity mentions extracted from news articles.
    """

    def __init__(self):
        self.kb_entries = load_knowledge_base()
        self.lookup_index = build_lookup_index(self.kb_entries)

        # All known names/aliases as a flat list, used for fuzzy matching
        self.all_known_terms = list(self.lookup_index.keys())

    def exact_match(self, mention: str) -> dict:
        """Tier 1: Direct lookup by name or alias (case-insensitive)."""
        return self.lookup_index.get(mention.lower())

    def partial_match(self, mention: str) -> dict:
        """
        Tier 2: Checks if the mention is a substring of a known name,
        or a known name is a substring of the mention.
        Handles cases like "Sitharaman" matching "Nirmala Sitharaman".
        """
        mention_lower = mention.lower()
        for entry in self.kb_entries:
            full_name = entry["name"].lower()
            if _contains_as_phrase(mention_lower, full_name) or _contains_as_phrase(full_name, mention_lower):
                return entry

            for alias in entry.get("aliases", []):
                alias_lower = alias.lower()
                if _contains_as_phrase(mention_lower, alias_lower) or _contains_as_phrase(alias_lower, mention_lower):
                    return entry
        return None

    def fuzzy_match(self, mention: str, max_distance: int = 2) -> dict:
        """
        Tier 3: Uses Minimum Edit Distance (Experiment 3) to find the
        closest known term, catching typos and minor spelling variants.
        """
        best_term = None
        best_distance = max_distance + 1

        for term in self.all_known_terms:
            dist = edit_distance(mention, term)
            if dist < best_distance:
                best_distance = dist
                best_term = term

        if best_term is not None:
            return self.lookup_index[best_term]
        return None

    def match_entity(self, mention: str, allow_partial: bool = True, allow_fuzzy: bool = True) -> dict:
        """
        Main entry point: tries tiers in order and returns the first
        successful match, tagged with WHICH tier found it.

        allow_partial / allow_fuzzy let callers restrict matching for
        entity types where those tiers are unreliable -- e.g. generic
        location names like "India" or "Delhi" are common substrings
        inside many unrelated KB aliases (e.g. "India" inside "Clean
        India Mission"), producing misleading matches. The caller
        (pipeline.py) disables partial/fuzzy matching for LOCATION-type
        entities as a result, keeping only exact matches for those.
        """
        result = self.exact_match(mention)
        if result:
            return {"mention": mention, "matched_entry": result, "match_tier": "exact"}

        if allow_partial:
            result = self.partial_match(mention)
            if result:
                return {"mention": mention, "matched_entry": result, "match_tier": "partial"}

        if allow_fuzzy:
            result = self.fuzzy_match(mention)
            if result:
                return {"mention": mention, "matched_entry": result, "match_tier": "fuzzy"}

        return {"mention": mention, "matched_entry": None, "match_tier": "no_match"}

    def match_entities(self, mentions: list) -> list:
        """Batch version -- matches a list of entity mentions (e.g. from NER output)."""
        return [self.match_entity(m) for m in mentions]


# ---------------------------------------------------------
# Quick manual test
# ---------------------------------------------------------
if __name__ == "__main__":
    matcher = KnowledgeBaseMatcher()

    # Simulating entity mentions as they'd come out of Experiment 7 (NER)
    # -- including exact names, partial names, and a typo, to test all 3 tiers.
    test_mentions = [
        "Reserve Bank of India",   # exact match expected
        "Sitharaman",              # partial match expected (last name only)
        "Narender Modi",           # fuzzy match expected (typo)
        "PM-KISAN",                # exact match via alias
        "Ministry of Fiance",      # fuzzy match expected (typo: Fiance -> Finance)
        "Eiffel Tower",            # should NOT match anything (out of scope)
    ]

    print("=" * 80)
    print("KNOWLEDGE BASE ENTITY MATCHING TEST")
    print("=" * 80)

    results = matcher.match_entities(test_mentions)

    for r in results:
        print(f"\nMention: '{r['mention']}'")
        if r["matched_entry"]:
            entry = r["matched_entry"]
            print(f"  Matched To : {entry['name']}")
            print(f"  Category   : {entry['category']}")
            print(f"  Match Tier : {r['match_tier']}")
            print(f"  Description: {entry['description'][:80]}...")
        else:
            print(f"  NO MATCH FOUND (tier: {r['match_tier']})")

    print("\n" + "=" * 80)
    print("KEY OBSERVATION:")
    print("Notice the 3-tier strategy catches different real-world cases:")
    print("exact names, partial mentions (just a surname), and typos --")
    print("this is what makes CHANAKYA's KB matching (Stage 7) robust")
    print("enough to handle real, messy news article text.")
