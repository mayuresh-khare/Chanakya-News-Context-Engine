"""
CHANAKYA - Spelling Correction Module
Experiment 3: Minimum Edit Distance (Levenshtein Distance)

Implements the classic dynamic programming algorithm for edit distance,
then uses it for two purposes in CHANAKYA:
  1. General spell-correction of article text
  2. Fuzzy entity matching against the Knowledge Base
     (e.g. "Sitaraman" in article -> matches "Sitharaman" in KB)
"""


def edit_distance(word1: str, word2: str) -> int:
    """
    Computes the Minimum Edit Distance between two strings using
    dynamic programming (Levenshtein Distance).

    Three allowed operations, each costing 1:
      - Insertion
      - Deletion
      - Substitution

    Returns the minimum number of operations to transform word1 into word2.
    """
    word1, word2 = word1.lower(), word2.lower()
    m, n = len(word1), len(word2)

    # dp[i][j] = edit distance between word1[:i] and word2[:j]
    dp = [[0] * (n + 1) for _ in range(m + 1)]

    # Base cases: transforming empty string requires all insertions/deletions
    for i in range(m + 1):
        dp[i][0] = i
    for j in range(n + 1):
        dp[0][j] = j

    # Fill the DP table
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if word1[i - 1] == word2[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]  # characters match, no cost
            else:
                dp[i][j] = 1 + min(
                    dp[i - 1][j],      # deletion
                    dp[i][j - 1],      # insertion
                    dp[i - 1][j - 1]   # substitution
                )

    return dp[m][n]


def find_closest_match(word: str, candidates: list, max_distance: int = 2) -> dict:
    """
    Given a (possibly misspelled) word and a list of candidate words
    (e.g. known entity names from the Knowledge Base), finds the closest
    match within max_distance edits.

    Returns None if no candidate is close enough.
    """
    best_match = None
    best_distance = max_distance + 1  # anything found must beat this

    for candidate in candidates:
        dist = edit_distance(word, candidate)
        if dist < best_distance:
            best_distance = dist
            best_match = candidate

    if best_match is not None and best_distance <= max_distance:
        return {
            "original": word,
            "matched": best_match,
            "edit_distance": best_distance,
            "confidence": round(1 - (best_distance / max(len(word), len(best_match))), 2)
        }
    return None


def correct_typos_in_tokens(tokens: list, known_vocabulary: list, max_distance: int = 2) -> list:
    """
    Runs spell correction across a list of tokens against a known vocabulary
    (e.g. dictionary words, or entity names from the KB).

    Only corrects tokens NOT already in the known vocabulary, and only
    if a close-enough match exists.
    """
    known_set = set(w.lower() for w in known_vocabulary)
    corrected = []

    for token in tokens:
        if token.lower() in known_set:
            corrected.append(token)  # already correct, no change needed
            continue

        match = find_closest_match(token, known_vocabulary, max_distance)
        if match:
            corrected.append(match["matched"])
        else:
            corrected.append(token)  # no close match found, leave as-is

    return corrected


# ---------------------------------------------------------
# Quick manual test
# ---------------------------------------------------------
if __name__ == "__main__":
    print("=" * 60)
    print("TEST 1: Basic Edit Distance Calculations")
    print("=" * 60)
    test_pairs = [
        ("Sitaraman", "Sitharaman"),
        ("Loksabha", "Lok Sabha"),
        ("recieve", "receive"),
        ("goverment", "government"),
    ]
    for w1, w2 in test_pairs:
        dist = edit_distance(w1, w2)
        print(f"  '{w1}' -> '{w2}'  |  Edit Distance: {dist}")

    print("\n" + "=" * 60)
    print("TEST 2: Fuzzy Entity Matching (simulating KB lookup)")
    print("=" * 60)

    kb_entities = [
        "Nirmala Sitharaman", "Narendra Modi", "Reserve Bank of India",
        "Lok Sabha", "Rajya Sabha", "Ministry of Finance"
    ]

    misspelled_mentions = ["Sitaraman", "Lok Sabaha", "Narender Modi"]

    for mention in misspelled_mentions:
        match = find_closest_match(mention, kb_entities, max_distance=3)
        if match:
            print(f"  '{mention}' --> matched to '{match['matched']}' "
                  f"(distance: {match['edit_distance']}, confidence: {match['confidence']})")
        else:
            print(f"  '{mention}' --> no close match found")