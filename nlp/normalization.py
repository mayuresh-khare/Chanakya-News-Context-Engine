"""
CHANAKYA - Normalization Module
Experiment 2: Stemming vs Lemmatization

Compares Porter Stemmer (rule-based, crude) against WordNet Lemmatizer
(dictionary-based, accurate) and explains why CHANAKYA uses lemmatization
for Knowledge Base matching.
"""

from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk import pos_tag
from nltk.corpus import wordnet

stemmer = PorterStemmer()
lemmatizer = WordNetLemmatizer()


def get_wordnet_pos(nltk_pos_tag: str):
    """
    WordNet's lemmatizer needs to know the part of speech to work correctly.
    'running' lemmatized as a VERB -> 'run'
    'running' lemmatized as a NOUN -> 'running' (e.g. "running is healthy")

    This maps NLTK's POS tags (like 'VBG', 'NNS') to WordNet's simpler
    categories (NOUN, VERB, ADJ, ADV).
    """
    tag = nltk_pos_tag[0].upper()
    tag_map = {
        'J': wordnet.ADJ,
        'V': wordnet.VERB,
        'N': wordnet.NOUN,
        'R': wordnet.ADV
    }
    return tag_map.get(tag, wordnet.NOUN)  # default to noun if unknown


def stem_word(word: str) -> str:
    """Reduce a word using Porter Stemmer (rule-based, may not be a real word)."""
    return stemmer.stem(word)


def lemmatize_word(word: str, pos: str = None) -> str:
    """
    Reduce a word to its dictionary root form using WordNet.
    If pos is provided (e.g. from POS tagging), lemmatization is more accurate.
    """
    if pos:
        wn_pos = get_wordnet_pos(pos)
        return lemmatizer.lemmatize(word, pos=wn_pos)
    return lemmatizer.lemmatize(word)


def compare_normalization(tokens: list) -> list:
    """
    Runs both stemming and lemmatization on a list of tokens and
    returns a side-by-side comparison. Uses POS tagging so lemmatization
    is context-aware (not just default noun-form guessing).
    """
    tagged = pos_tag(tokens)  # [('running', 'VBG'), ('policies', 'NNS'), ...]

    results = []
    for word, pos in tagged:
        results.append({
            "word": word,
            "pos_tag": pos,
            "stemmed": stem_word(word),
            "lemmatized": lemmatize_word(word, pos)
        })
    return results


def normalize_tokens_for_kb_matching(tokens: list) -> list:
    """
    Production function used elsewhere in CHANAKYA's pipeline.
    Lemmatizes a list of tokens (with POS context) for accurate
    Knowledge Base entity matching. This is what Stage 7 will call.
    """
    tagged = pos_tag(tokens)
    return [lemmatize_word(word, pos) for word, pos in tagged]


# ---------------------------------------------------------
# Quick manual test
# ---------------------------------------------------------
if __name__ == "__main__":
    test_words = [
        "studies", "running", "policies", "governments",
        "announced", "better", "children", "was", "flying"
    ]

    print("=" * 70)
    print(f"{'WORD':<15}{'POS':<8}{'STEMMED':<15}{'LEMMATIZED':<15}")
    print("=" * 70)

    comparison = compare_normalization(test_words)
    for item in comparison:
        print(f"{item['word']:<15}{item['pos_tag']:<8}{item['stemmed']:<15}{item['lemmatized']:<15}")

    print("\n" + "=" * 70)
    print("KEY OBSERVATION:")
    print("Notice 'studies' -> stem: 'studi' (not a real word)")
    print("                 -> lemma: 'study' (real dictionary word)")
    print("This is why CHANAKYA uses lemmatization for KB entity matching.")