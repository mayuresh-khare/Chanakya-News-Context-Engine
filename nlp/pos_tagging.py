"""
CHANAKYA - POS Tagging Module
Experiment 5: Rule-based vs Probabilistic POS Tagging

Implements a hand-written rule-based tagger (using suffix/pattern rules)
and compares it against NLTK's probabilistic (statistically trained)
tagger. Reports agreement/accuracy between the two.
"""

from nltk import pos_tag, word_tokenize
from nltk.tag import RegexpTagger


# ---------------------------------------------------------
# RULE-BASED TAGGER
# ---------------------------------------------------------
# Each rule is (regex_pattern, tag). Order matters -- first match wins.
# This is a simplified but real rule-based tagger built on suffix patterns.
RULE_PATTERNS = [
    (r'.*ing$', 'VBG'),               # gerunds: running, announcing
    (r'.*ed$', 'VBD'),                # past tense verbs: announced, passed
    (r'.*es$', 'VBZ'),                # 3rd person singular verbs: goes, passes
    (r'.*ly$', 'RB'),                 # adverbs: quickly, formally
    (r'.*ould$', 'MD'),               # modals: would, should, could
    (r'.*\'s$', 'POS'),               # possessives: Modi's
    (r'.*s$', 'NNS'),                 # plural nouns: policies, leaders
    (r'^[A-Z].*$', 'NNP'),            # capitalized -> proper noun (crude heuristic)
    (r'^-?[0-9]+(\.[0-9]+)?$', 'CD'), # numbers
    (r'.*', 'NN'),                    # default fallback: common noun
]

rule_based_tagger = RegexpTagger(RULE_PATTERNS)


def tag_rule_based(tokens: list) -> list:
    """
    Tags tokens using our hand-written rule-based tagger.
    Fast, transparent, but less accurate on irregular words.
    """
    return rule_based_tagger.tag(tokens)


def tag_probabilistic(tokens: list) -> list:
    """
    Tags tokens using NLTK's built-in probabilistic tagger
    (Averaged Perceptron, trained on a large annotated corpus).
    """
    return pos_tag(tokens)


def compare_taggers(sentence: str) -> dict:
    """
    Runs both taggers on a sentence and computes their agreement rate.
    This is our "accuracy" proxy: how often do the two approaches agree?
    (In a full academic evaluation you'd compare both against a human-
    annotated gold standard; here we compare rule-based against the
    probabilistic tagger, treating the probabilistic one as the stronger
    baseline since it's trained on real annotated data.)
    """
    tokens = word_tokenize(sentence)

    rule_tags = tag_rule_based(tokens)
    prob_tags = tag_probabilistic(tokens)

    matches = 0
    comparison = []
    for (word_r, tag_r), (word_p, tag_p) in zip(rule_tags, prob_tags):
        agree = (tag_r == tag_p)
        if agree:
            matches += 1
        comparison.append({
            "word": word_r,
            "rule_based_tag": tag_r,
            "probabilistic_tag": tag_p,
            "agree": agree
        })

    accuracy = matches / len(tokens) if tokens else 0.0

    return {
        "tokens": tokens,
        "comparison": comparison,
        "agreement_rate": round(accuracy, 3)
    }


# ---------------------------------------------------------
# Quick manual test
# ---------------------------------------------------------
if __name__ == "__main__":
    test_sentences = [
        "The government announced new policies today.",
        "Nirmala Sitharaman quickly presented the budget.",
        "Parliament passed the bill after long debate.",
    ]

    for sentence in test_sentences:
        print("=" * 75)
        print(f"SENTENCE: \"{sentence}\"")
        print("=" * 75)

        result = compare_taggers(sentence)

        print(f"{'WORD':<15}{'RULE-BASED':<15}{'PROBABILISTIC':<18}{'AGREE?':<8}")
        print("-" * 56)
        for item in result["comparison"]:
            agree_mark = "YES" if item["agree"] else "NO"
            print(f"{item['word']:<15}{item['rule_based_tag']:<15}"
                  f"{item['probabilistic_tag']:<18}{agree_mark:<8}")

        print(f"\nAgreement Rate: {result['agreement_rate'] * 100:.1f}%\n")

    print("=" * 75)
    print("KEY OBSERVATION:")
    print("Rule-based tagging fails on irregular patterns (e.g. capitalized")
    print("common words at sentence start get mistagged as NNP).")
    print("The probabilistic tagger, trained on real annotated text, handles")
    print("these cases correctly -- this is why CHANAKYA's production")
    print("pipeline (Stage 3) uses spaCy's statistical tagger, not rules.")