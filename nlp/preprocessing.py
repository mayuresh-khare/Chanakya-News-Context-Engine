"""
CHANAKYA - Text Preprocessing Module
Experiment 1: Tokenization, Script Validation, Stop-word Removal

This module cleans raw news article text and prepares it for the rest
of the NLP pipeline. It produces TWO outputs:
  1. cleaned_text   -> original sentences, just whitespace-normalized
                        (used later for POS tagging, NER, CFG parsing)
  2. tokens          -> tokenized, lowercased, stop-words removed
                        (used later for N-grams, classification)
"""

import re
import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords

# Load English stopwords once (module-level, so we don't reload every call)
STOP_WORDS = set(stopwords.words('english'))


def is_devanagari(token: str) -> bool:
    """
    Checks if a token contains Devanagari script characters.
    Devanagari Unicode block: U+0900 to U+097F
    """
    return bool(re.search(r'[\u0900-\u097F]', token))


def validate_script(tokens: list) -> dict:
    """
    Splits tokens into English and Devanagari groups.
    For V1, we only fully process English tokens downstream;
    Devanagari tokens are flagged and kept aside.
    """
    english_tokens = []
    devanagari_tokens = []

    for token in tokens:
        if is_devanagari(token):
            devanagari_tokens.append(token)
        else:
            english_tokens.append(token)

    return {
        "english_tokens": english_tokens,
        "devanagari_tokens": devanagari_tokens,
        "has_devanagari": len(devanagari_tokens) > 0
    }


def clean_text(raw_text: str) -> str:
    """
    Normalizes whitespace and removes stray control characters.
    Does NOT lowercase or remove punctuation -- this version is kept
    'human-readable' for later stages like NER and POS tagging.
    """
    text = raw_text.strip()
    text = re.sub(r'\s+', ' ', text)          # collapse multiple spaces/newlines
    text = re.sub(r'[\r\t]', ' ', text)       # remove tabs/carriage returns
    return text


def tokenize_text(text: str) -> list:
    """
    Tokenizes text into word-level tokens using NLTK's tokenizer.
    Handles punctuation, contractions, and abbreviations properly
    (unlike a naive .split()).
    """
    return word_tokenize(text)


def tokenize_sentences(text: str) -> list:
    """
    Splits text into sentences. Needed later for NER, POS tagging,
    and classification, since we process news article sentence-by-sentence.
    """
    return sent_tokenize(text)


def remove_stopwords(tokens: list) -> list:
    """
    Removes common English stop-words (the, is, and, etc.)
    Also lowercases tokens and strips pure punctuation tokens.
    """
    cleaned = []
    for token in tokens:
        lower_token = token.lower()
        if lower_token in STOP_WORDS:
            continue
        if not re.search(r'[a-zA-Z0-9]', lower_token):  # skip pure punctuation
            continue
        cleaned.append(lower_token)
    return cleaned


def preprocess_article(raw_text: str) -> dict:
    """
    Main entry point for Experiment 1.
    Runs the full preprocessing pipeline on a raw news article
    and returns a structured dictionary with all intermediate outputs.
    """
    # Step 1: Clean raw text (whitespace normalization)
    cleaned = clean_text(raw_text)

    # Step 2: Sentence tokenization (needed for later stages)
    sentences = tokenize_sentences(cleaned)

    # Step 3: Word tokenization
    tokens = tokenize_text(cleaned)

    # Step 4: Script validation (English vs Devanagari)
    script_info = validate_script(tokens)

    # Step 5: Stop-word removal (only on English tokens)
    filtered_tokens = remove_stopwords(script_info["english_tokens"])

    return {
        "cleaned_text": cleaned,
        "sentences": sentences,
        "raw_tokens": tokens,
        "english_tokens": script_info["english_tokens"],
        "devanagari_tokens": script_info["devanagari_tokens"],
        "has_devanagari": script_info["has_devanagari"],
        "filtered_tokens": filtered_tokens,   # stop-words removed, lowercased
        "token_count_raw": len(tokens),
        "token_count_filtered": len(filtered_tokens),
    }


# ---------------------------------------------------------
# Quick manual test (run this file directly to sanity-check)
# ---------------------------------------------------------
if __name__ == "__main__":
    sample_article = """
    The Reserve Bank of India (RBI) announced a new monetary policy today.
    The move is expected to affect interest rates across the country.
    Finance Minister Nirmala Sitharaman welcomed the decision, calling it
    a step towards economic stability.
    """

    result = preprocess_article(sample_article)

    print("=" * 60)
    print("CLEANED TEXT:")
    print(result["cleaned_text"])

    print("\n" + "=" * 60)
    print(f"SENTENCES ({len(result['sentences'])}):")
    for i, s in enumerate(result["sentences"], 1):
        print(f"  {i}. {s}")

    print("\n" + "=" * 60)
    print(f"RAW TOKENS ({result['token_count_raw']}):")
    print(result["raw_tokens"])

    print("\n" + "=" * 60)
    print(f"FILTERED TOKENS ({result['token_count_filtered']}) - stopwords removed:")
    print(result["filtered_tokens"])

    print("\n" + "=" * 60)
    print(f"HAS DEVANAGARI: {result['has_devanagari']}")