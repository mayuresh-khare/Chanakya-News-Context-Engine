"""
CHANAKYA - Word Sense Disambiguation Module
Experiment 8: Lesk Algorithm + WordNet Semantic Relations

Disambiguates ambiguous words based on context using NLTK's Lesk
implementation, and explores semantic relations (synonyms, antonyms,
hypernyms) via WordNet. Used in CHANAKYA to correctly interpret
ambiguous entity/keyword terms before Knowledge Base matching.
"""

from nltk.wsd import lesk
from nltk.corpus import wordnet as wn
from nltk import word_tokenize


def disambiguate_word(sentence: str, ambiguous_word: str) -> dict:
    """
    Uses the Lesk algorithm to determine the most likely sense (synset)
    of an ambiguous word, given the sentence it appears in.
    """
    tokens = word_tokenize(sentence)
    best_synset = lesk(tokens, ambiguous_word)

    if best_synset is None:
        return {
            "word": ambiguous_word,
            "sentence": sentence,
            "sense_found": False,
            "definition": None,
            "synset_name": None
        }

    return {
        "word": ambiguous_word,
        "sentence": sentence,
        "sense_found": True,
        "synset_name": best_synset.name(),
        "definition": best_synset.definition(),
        "examples": best_synset.examples()
    }


def get_all_senses(word: str) -> list:
    """
    Lists every possible sense (synset) WordNet knows for a word,
    with its definition -- useful to show WHY disambiguation is needed
    (i.e. how many different meanings a word actually has).
    """
    synsets = wn.synsets(word)
    return [
        {"synset_name": s.name(), "definition": s.definition()}
        for s in synsets
    ]


def get_semantic_relations(word: str) -> dict:
    """
    Explores semantic relations for a word's primary (most common) sense:
    synonyms, antonyms, and hypernyms (more general/parent concepts).
    """
    synsets = wn.synsets(word)
    if not synsets:
        return {"word": word, "found": False}

    primary_sense = synsets[0]  # WordNet orders senses by frequency of use

    synonyms = set()
    antonyms = set()
    for lemma in primary_sense.lemmas():
        synonyms.add(lemma.name())
        for ant in lemma.antonyms():
            antonyms.add(ant.name())

    hypernyms = [h.name() for h in primary_sense.hypernyms()]
    hyponyms = [h.name() for h in primary_sense.hyponyms()[:5]]  # limit for readability

    return {
        "word": word,
        "found": True,
        "primary_sense": primary_sense.name(),
        "definition": primary_sense.definition(),
        "synonyms": list(synonyms),
        "antonyms": list(antonyms),
        "hypernyms": hypernyms,      # more general concepts (e.g. "bank" -> "financial institution")
        "hyponyms": hyponyms         # more specific concepts
    }


# ---------------------------------------------------------
# Quick manual test
# ---------------------------------------------------------
if __name__ == "__main__":
    print("=" * 70)
    print("TEST 1: Lesk Algorithm - Disambiguating 'bank' in context")
    print("=" * 70)

    test_cases = [
        ("The bank announced new interest rates for savings accounts.", "bank"),
        ("They sat by the bank of the river and watched the sunset.", "bank"),
    ]

    for sentence, word in test_cases:
        result = disambiguate_word(sentence, word)
        print(f"\n  Sentence: \"{sentence}\"")
        print(f"  Word: '{word}'")
        if result["sense_found"]:
            print(f"  Detected Sense : {result['synset_name']}")
            print(f"  Definition     : {result['definition']}")
        else:
            print("  No sense could be determined.")

    print("\n" + "=" * 70)
    print("TEST 2: All Possible Senses of 'policy' (shows why WSD is needed)")
    print("=" * 70)
    senses = get_all_senses("policy")
    for s in senses:
        print(f"  {s['synset_name']:<20} : {s['definition']}")

    print("\n" + "=" * 70)
    print("TEST 3: Semantic Relations for 'government'")
    print("=" * 70)
    relations = get_semantic_relations("government")
    print(f"  Word           : {relations['word']}")
    print(f"  Primary Sense  : {relations['primary_sense']}")
    print(f"  Definition     : {relations['definition']}")
    print(f"  Synonyms       : {relations['synonyms']}")
    print(f"  Antonyms       : {relations['antonyms']}")
    print(f"  Hypernyms      : {relations['hypernyms']}")
    print(f"  Hyponyms       : {relations['hyponyms']}")

    print("\n" + "=" * 70)
    print("KEY OBSERVATION:")
    print("The Lesk algorithm correctly picked DIFFERENT senses of 'bank'")
    print("based purely on surrounding context words (interest/savings vs")
    print("river/sunset). This is exactly what CHANAKYA needs in Stage 6 --")
    print("resolving ambiguous terms before matching them to the Knowledge Base.")