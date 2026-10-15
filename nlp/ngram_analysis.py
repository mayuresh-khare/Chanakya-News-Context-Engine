"""
CHANAKYA - N-gram Language Model Module
Experiment 4: Unigram, Bigram, Trigram models; sequence probability; perplexity

Builds N-gram frequency models from a training corpus, computes the
probability of a sentence under that model, and evaluates perplexity
(a measure of how "natural" a sentence looks to the model).

Used in CHANAKYA as a lightweight text-quality signal and for surfacing
frequent multi-word phrases before NER runs.
"""

import math
from collections import defaultdict, Counter
from nltk import ngrams, word_tokenize


class NGramModel:
    """
    A simple N-gram language model with add-one (Laplace) smoothing.
    Smoothing is necessary because without it, any unseen word sequence
    gets probability 0, which makes perplexity calculations undefined
    (division by zero / log(0)).
    """

    def __init__(self, n: int):
        self.n = n
        self.ngram_counts = defaultdict(int)        # counts of full n-grams
        self.context_counts = defaultdict(int)      # counts of (n-1)-gram contexts
        self.vocabulary = set()

    def train(self, tokenized_sentences: list):
        """
        Trains the model on a list of tokenized sentences.
        Each sentence is a list of tokens, e.g. [['the', 'government', ...], ...]
        """
        for tokens in tokenized_sentences:
            self.vocabulary.update(tokens)

            # Generate n-grams for this sentence
            padded = tokens  # (keeping simple: no explicit <s>/</s> padding for V1)
            n_grams = list(ngrams(padded, self.n)) if len(padded) >= self.n else []

            for gram in n_grams:
                self.ngram_counts[gram] += 1
                context = gram[:-1]  # everything except the last word
                self.context_counts[context] += 1

    def probability(self, ngram: tuple) -> float:
        """
        Computes P(last word | preceding n-1 words) using add-one smoothing.

        P(w_n | context) = (Count(context, w_n) + 1) / (Count(context) + V)
        where V = vocabulary size
        """
        context = ngram[:-1]
        V = len(self.vocabulary)

        ngram_count = self.ngram_counts.get(ngram, 0)
        context_count = self.context_counts.get(context, 0)

        return (ngram_count + 1) / (context_count + V)

    def sentence_probability(self, tokens: list) -> float:
        """
        Computes the probability of an entire sentence under this model
        by multiplying the probability of each n-gram in it.
        """
        if len(tokens) < self.n:
            return 0.0

        n_grams = list(ngrams(tokens, self.n))
        prob = 1.0
        for gram in n_grams:
            prob *= self.probability(gram)
        return prob

    def perplexity(self, tokens: list) -> float:
        """
        Computes perplexity of a sentence under this model.

        Perplexity = 2^(-average log2 probability per n-gram)

        Lower = model finds this sentence "natural" (matches training patterns)
        Higher = model finds this sentence "surprising" (unusual/garbled)
        """
        if len(tokens) < self.n:
            return float('inf')

        n_grams = list(ngrams(tokens, self.n))
        log_prob_sum = 0.0

        for gram in n_grams:
            p = self.probability(gram)
            log_prob_sum += math.log2(p)

        avg_log_prob = log_prob_sum / len(n_grams)
        return 2 ** (-avg_log_prob)

    def top_ngrams(self, top_k: int = 10) -> list:
        """Returns the most frequent n-grams seen during training."""
        counter = Counter(self.ngram_counts)
        return counter.most_common(top_k)


def build_ngram_models(training_texts: list) -> dict:
    """
    Trains unigram, bigram, and trigram models on a list of raw text strings.
    Returns a dict of {1: model, 2: model, 3: model}.
    """
    tokenized_sentences = [word_tokenize(text.lower()) for text in training_texts]

    models = {}
    for n in [1, 2, 3]:
        model = NGramModel(n)
        model.train(tokenized_sentences)
        models[n] = model

    return models


# ---------------------------------------------------------
# Quick manual test
# ---------------------------------------------------------
if __name__ == "__main__":
    # Small training corpus (in a real pipeline, this would be your
    # collected news articles dataset)
    training_corpus = [
        "The government announced a new policy today.",
        "The Prime Minister met with foreign leaders yesterday.",
        "The Finance Minister presented the budget in parliament.",
        "The government announced new reforms for the economy.",
        "Parliament passed the new bill after long debate.",
        "The Prime Minister announced new economic reforms.",
    ]

    print("=" * 70)
    print("TRAINING N-GRAM MODELS on sample news corpus")
    print("=" * 70)
    models = build_ngram_models(training_corpus)

    for n in [1, 2, 3]:
        print(f"\n--- Top {n}-grams (n={n}) ---")
        for gram, count in models[n].top_ngrams(5):
            print(f"  {gram} : {count}")

    print("\n" + "=" * 70)
    print("SEQUENCE PROBABILITY & PERPLEXITY TEST")
    print("=" * 70)

    test_sentences = [
        "The government announced new reforms",      # similar to training data
        "Parliament passed the new bill",             # very close to training data
        "Purple elephants dance quickly under water", # nonsense, unrelated to training
    ]

    bigram_model = models[2]

    for sentence in test_sentences:
        tokens = word_tokenize(sentence.lower())
        prob = bigram_model.sentence_probability(tokens)
        perp = bigram_model.perplexity(tokens)
        print(f"\n  Sentence: \"{sentence}\"")
        print(f"    Probability : {prob:.10f}")
        print(f"    Perplexity  : {perp:.2f}")

    print("\n" + "=" * 70)
    print("KEY OBSERVATION:")
    print("Notice the nonsense sentence has MUCH higher perplexity --")
    print("the model is 'surprised' by it because it doesn't match")
    print("patterns seen in the training corpus. This is our text-quality signal.")