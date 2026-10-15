"""
CHANAKYA - Syntactic Parsing Module
Experiment 6: Context-Free Grammar (CFG) Parsing

Defines a small CFG covering simplified news-sentence structures and
parses sentences using both a top-down (Recursive Descent) parser and
a bottom-up-style (Chart) parser from NLTK, comparing their behavior.
"""

import nltk
from nltk import CFG
from nltk.parse import RecursiveDescentParser, ChartParser


# ---------------------------------------------------------
# A SMALL CFG FOR SIMPLIFIED NEWS SENTENCES
# ---------------------------------------------------------
# Covers patterns like:
#   "the government announced the policy"
#   "the minister met the leader in delhi"
# NOTE: Avoid left-recursive rules like "NP -> NP PP" -- they cause
# infinite recursion in top-down (Recursive Descent) parsers.
GRAMMAR = CFG.fromstring("""
    S -> NP VP
    NP -> Det N | Det Adj N | 'Nirmala' | 'Modi' | 'delhi'
    VP -> V NP | V NP PP | Adv V NP
    PP -> P NP
    Det -> 'the' | 'The' | 'a'
    N -> 'government' | 'policy' | 'minister' | 'leader' | 'parliament' | 'bill' | 'delhi'
    V -> 'announced' | 'met' | 'passed' | 'presented'
    Adj -> 'new' | 'foreign' | 'senior'
    Adv -> 'quickly' | 'formally'
    P -> 'in' | 'with' | 'after'
""")


def parse_top_down(sentence: str) -> dict:
    """
    Parses a sentence using Recursive Descent Parsing (top-down).
    Starts from the S (Sentence) symbol and expands rules downward,
    trying to match against the actual words. Backtracks on failure.
    """
    tokens = sentence.lower().split()
    parser = RecursiveDescentParser(GRAMMAR)

    trees = []
    try:
        for tree in parser.parse(tokens):
            trees.append(tree)
    except ValueError as e:
        return {"error": str(e), "trees": []}

    return {"error": None, "trees": trees}


def parse_bottom_up(sentence: str) -> dict:
    """
    Parses a sentence using Chart Parsing (bottom-up style).
    Builds up from individual words toward larger constituents,
    using a chart to avoid re-computing the same sub-parses repeatedly
    (more efficient than naive bottom-up shift-reduce parsing).
    """
    tokens = sentence.lower().split()
    parser = ChartParser(GRAMMAR)

    trees = []
    try:
        for tree in parser.parse(tokens):
            trees.append(tree)
    except ValueError as e:
        return {"error": str(e), "trees": []}

    return {"error": None, "trees": trees}


def parse_sentence(sentence: str, method: str = "both") -> dict:
    """
    Main entry point. Parses a sentence with the selected method(s)
    and returns tree structures (as strings, for easy display/storage).
    """
    result = {"sentence": sentence}

    if method in ("top_down", "both"):
        td_result = parse_top_down(sentence)
        result["top_down"] = {
            "success": td_result["error"] is None and len(td_result["trees"]) > 0,
            "error": td_result["error"],
            "trees": [str(t) for t in td_result["trees"]]
        }

    if method in ("bottom_up", "both"):
        bu_result = parse_bottom_up(sentence)
        result["bottom_up"] = {
            "success": bu_result["error"] is None and len(bu_result["trees"]) > 0,
            "error": bu_result["error"],
            "trees": [str(t) for t in bu_result["trees"]]
        }

    return result


# ---------------------------------------------------------
# Quick manual test
# ---------------------------------------------------------
if __name__ == "__main__":
    test_sentences = [
        "the government announced the policy",
        "the minister met the leader in delhi",
        "Purple elephants dance quickly",  # should FAIL -- not in our grammar
    ]

    for sentence in test_sentences:
        print("=" * 75)
        print(f"SENTENCE: \"{sentence}\"")
        print("=" * 75)

        result = parse_sentence(sentence)

        print("\n--- TOP-DOWN (Recursive Descent) ---")
        if result["top_down"]["success"]:
            for tree_str in result["top_down"]["trees"]:
                print(tree_str)
        else:
            print("  FAILED to parse -- sentence not covered by grammar.")

        print("\n--- BOTTOM-UP (Chart Parser) ---")
        if result["bottom_up"]["success"]:
            for tree_str in result["bottom_up"]["trees"]:
                print(tree_str)
        else:
            print("  FAILED to parse -- sentence not covered by grammar.")
        print()

    print("=" * 75)
    print("KEY OBSERVATION:")
    print("Both parsers agree on valid sentences (same grammar, same result).")
    print("The nonsense sentence fails on BOTH -- because it fails, not the")
    print("parser design, but because our vocabulary/grammar is deliberately")
    print("small (a toy CFG can't cover the full richness of real news text).")
    print("This is a known, expected limitation of hand-written CFGs, and")
    print("why CHANAKYA's production pipeline uses spaCy's dependency parser")
    print("for real sentence structure, keeping this CFG as a demonstration")
    print("of top-down vs bottom-up parsing theory.")