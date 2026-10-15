"""
CHANAKYA - NER and Chunking Module
Experiment 7: Named Entity Recognition + Noun Phrase Chunking

Uses spaCy for NER (production-grade, statistically trained) and NLTK
for rule-based chunking (to fulfill the "chunking" requirement using
POS-pattern grammar, distinct from spaCy's built-in NP extraction).

This is the CORE experiment for CHANAKYA -- entities extracted here feed
directly into Knowledge Base matching (Stage 7 of the architecture).
"""

import spacy
import nltk
from nltk import pos_tag, word_tokenize, RegexpParser

nlp = spacy.load("en_core_web_sm")

# Entity types we care about for CHANAKYA's Knowledge Base matching.
# spaCy has more types (MONEY, PERCENT, etc.) -- we filter to the ones
# relevant to news context: people, orgs, places, dates, laws/events.
RELEVANT_ENTITY_TYPES = {
    "PERSON": "PERSON",
    "ORG": "ORGANIZATION",
    "GPE": "LOCATION",       # Geo-Political Entity (countries, states, cities)
    "LOC": "LOCATION",
    "DATE": "DATE",
    "EVENT": "EVENT",
    "LAW": "POLICY",
    "NORP": "GROUP",          # Nationalities, religious/political groups
}

# Chunking grammar: defines what counts as a Noun Phrase (NP) chunk
# using POS tag patterns.
#   <DT>?   = optional determiner (the, a)
#   <JJ>*   = zero or more adjectives (new, senior)
#   <NN.*>+ = one or more nouns (NN, NNS, NNP, NNPS)
CHUNK_GRAMMAR = r"""
    NP: {<DT>?<JJ>*<NN.*>+}
"""
chunk_parser = RegexpParser(CHUNK_GRAMMAR)


def extract_entities(text: str) -> list:
    """
    Extracts named entities from text using spaCy's statistical NER model.
    Filters to entity types relevant for CHANAKYA's Knowledge Base matching.
    """
    doc = nlp(text)
    entities = []

    for ent in doc.ents:
        if ent.label_ in RELEVANT_ENTITY_TYPES:
            entities.append({
                "text": ent.text,
                "spacy_label": ent.label_,
                "chanakya_type": RELEVANT_ENTITY_TYPES[ent.label_],
                "start_char": ent.start_char,
                "end_char": ent.end_char
            })

    return entities


def extract_noun_chunks_spacy(text: str) -> list:
    """
    Uses spaCy's built-in dependency-based noun chunk extraction.
    More accurate than rule-based chunking since it uses the full
    dependency parse, not just POS tag patterns.
    """
    doc = nlp(text)
    return [chunk.text for chunk in doc.noun_chunks]


def extract_noun_chunks_nltk(text: str) -> list:
    """
    Rule-based chunking using NLTK's RegexpParser and a hand-written
    grammar over POS tags. This fulfills the "chunking" experiment
    requirement using a classic NLP technique (POS-pattern grammars),
    distinct from spaCy's dependency-based approach above.
    """
    tokens = word_tokenize(text)
    tagged = pos_tag(tokens)
    tree = chunk_parser.parse(tagged)

    chunks = []
    for subtree in tree.subtrees(filter=lambda t: t.label() == "NP"):
        chunk_words = " ".join(word for word, tag in subtree.leaves())
        chunks.append(chunk_words)

    return chunks


def analyze_article(text: str) -> dict:
    """
    Main entry point for Experiment 7. Runs entity extraction and
    both chunking approaches on a piece of text, returning a full
    comparison -- this is what feeds Stage 4 of CHANAKYA's pipeline.
    """
    return {
        "entities": extract_entities(text),
        "noun_chunks_spacy": extract_noun_chunks_spacy(text),
        "noun_chunks_nltk": extract_noun_chunks_nltk(text),
    }


# ---------------------------------------------------------
# Quick manual test
# ---------------------------------------------------------
if __name__ == "__main__":
    sample_article = """
    Prime Minister Narendra Modi announced a new economic policy on Monday.
    Finance Minister Nirmala Sitharaman said the Reserve Bank of India will
    implement the changes starting next month. The announcement came after
    a meeting with senior officials from the Ministry of Finance in New Delhi.
    """

    result = analyze_article(sample_article)

    print("=" * 70)
    print("NAMED ENTITIES (spaCy)")
    print("=" * 70)
    for ent in result["entities"]:
        print(f"  {ent['text']:<35} -> {ent['chanakya_type']} (spaCy: {ent['spacy_label']})")

    print("\n" + "=" * 70)
    print("NOUN CHUNKS (spaCy - dependency based)")
    print("=" * 70)
    for chunk in result["noun_chunks_spacy"]:
        print(f"  - {chunk}")

    print("\n" + "=" * 70)
    print("NOUN CHUNKS (NLTK - POS pattern based)")
    print("=" * 70)
    for chunk in result["noun_chunks_nltk"]:
        print(f"  - {chunk}")

    print("\n" + "=" * 70)
    print("KEY OBSERVATION:")
    print("spaCy's entities directly identify WHO/WHAT/WHERE in the news --")
    print("this is the exact input CHANAKYA needs for Knowledge Base matching")
    print("in Stage 7 of the architecture (Nirmala Sitharaman -> KB entry,")
    print("Ministry of Finance -> KB entry, etc.)")