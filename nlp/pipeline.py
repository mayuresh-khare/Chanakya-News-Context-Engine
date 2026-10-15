"""
CHANAKYA - Main Pipeline
Experiment 10 (Part 1): Integration of all NLP modules into one flow.

Takes a raw news article and produces the 5 required outputs:
  1. Event Summary
  2. Why is this in the News?
  3. Background of the News
  4. Connected Topics
  5. Important Entities

This wires together preprocessing, classification, NER, and Knowledge
Base matching -- Experiments 1 through 9 all feed into this pipeline.
"""

import sys
import os

sys.path.append(os.path.dirname(__file__))
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from nlp.preprocessing import clean_text, tokenize_sentences
from nlp.ner_chunking import extract_entities
from nlp.classification import TextClassifier
from kb.kb_matcher import KnowledgeBaseMatcher


# ---------------------------------------------------------
# TRAINING DATA for the sentence-type classifier
# (same data used in Experiment 9 -- in production this would be a
# much larger labeled corpus of real news sentences)
# ---------------------------------------------------------
TRAINING_TEXTS = [
    "The government announced a new policy on Monday.",
    "Parliament passed the bill after a long debate.",
    "The Reserve Bank raised interest rates by 0.5 percent.",
    "The Prime Minister met with foreign leaders yesterday.",
    "The Finance Minister presented the annual budget today.",
    "Officials signed a new trade agreement this week.",
    "The Supreme Court delivered its verdict on the case.",
    "The cabinet approved the new infrastructure project.",
    "The ministry launched a nationwide awareness campaign.",
    "The president addressed the nation on television.",

    "This decision affects millions of farmers across the country.",
    "The move is expected to impact inflation rates nationwide.",
    "Experts say this could reshape India's trade relationships.",
    "The ruling has major implications for future elections.",
    "This policy could significantly alter the job market.",
    "The verdict sets an important precedent for future cases.",
    "This reform is likely to affect small businesses nationwide.",
    "The decision could strain diplomatic ties with neighboring countries.",
    "This change may influence how future budgets are structured.",
    "The move raises concerns about long-term economic stability.",

    "The policy was first proposed back in 2019 amid debate.",
    "This follows a similar reform attempted a decade ago.",
    "The ministry has been working on this issue since 2020.",
    "Previous governments had also considered similar measures.",
    "The debate over this issue dates back several years.",
    "A similar law was struck down by courts in 2015.",
    "This proposal builds on recommendations made in 2018.",
    "The controversy has its roots in a decades-old dispute.",
    "Talks on this topic began well before the last election.",
    "The scheme replaces an older program launched in 2010.",
]
TRAINING_LABELS = (["event_summary"] * 10) + (["why_news"] * 10) + (["background"] * 10)


class ChanakyaPipeline:
    """
    The main CHANAKYA pipeline. Initializes the sentence classifier
    (trained once) and the Knowledge Base matcher, then processes
    articles through the full flow to produce the 5 required outputs.
    """

    def __init__(self):
        print("Initializing CHANAKYA pipeline...")

        # Train the sentence-type classifier once at startup
        self.classifier = TextClassifier()
        self.classifier.train(TRAINING_TEXTS, TRAINING_LABELS)

        # Load the Knowledge Base matcher once at startup
        self.kb_matcher = KnowledgeBaseMatcher()

        print(f"  - Sentence classifier trained on {len(TRAINING_TEXTS)} examples")
        print(f"  - Knowledge Base loaded with {len(self.kb_matcher.kb_entries)} entries")
        print("Pipeline ready.\n")

    def process_article(self, raw_text: str) -> dict:
        """
        Main entry point. Runs a raw article through the full CHANAKYA
        pipeline and returns the 5 required outputs, plus intermediate
        data for transparency/debugging.
        """
        # STAGE 1-2: Preprocessing
        cleaned = clean_text(raw_text)
        sentences = tokenize_sentences(cleaned)

        # STAGE 3: Classify each sentence into event_summary / why_news / background
        classified_sentences = {"event_summary": [], "why_news": [], "background": []}
        for sentence in sentences:
            result = self.classifier.predict(sentence)
            category = result["predicted_class"]
            classified_sentences[category].append(sentence)

        # STAGE 4: Named Entity Recognition (on the full cleaned article)
        raw_entities = extract_entities(cleaned)

        # DATE entities are NOT run through KB name-matching -- a bare year
        # like "2019" can coincidentally substring-match an unrelated KB
        # entry whose alias happens to include that year (e.g. "Consumer
        # Protection Act 2019"), producing a meaningless false match.
        # Dates are kept as separate contextual info instead.
        name_entities = [e for e in raw_entities if e["chanakya_type"] != "DATE"]
        date_entities = sorted(set(e["text"] for e in raw_entities if e["chanakya_type"] == "DATE"))

        # Deduplicate while keeping the entity type of each unique mention
        unique_entities = {}
        for e in name_entities:
            unique_entities[e["text"]] = e["chanakya_type"]

        # STAGE 5-7: Knowledge Base Matching for every unique NAME entity found
        matched_entities = []
        connected_topics = set()

        for entity_text, entity_type in unique_entities.items():
            # LOCATION entities (India, Delhi, Mumbai, etc.) are too generic
            # for partial/fuzzy matching -- they coincidentally appear as
            # substrings inside many unrelated KB aliases (e.g. "India"
            # inside "Clean India Mission"), producing misleading matches.
            # Only exact matches are trusted for this entity type.
            is_location = (entity_type == "LOCATION")
            match_result = self.kb_matcher.match_entity(
                entity_text,
                allow_partial=not is_location,
                allow_fuzzy=not is_location
            )
            entity_info = {
                "mention": entity_text,
                "kb_match": None,
                "category": None
            }

            if match_result["matched_entry"]:
                kb_entry = match_result["matched_entry"]
                entity_info["kb_match"] = kb_entry["name"]
                entity_info["category"] = kb_entry["category"]
                entity_info["description"] = kb_entry["description"]
                entity_info["match_tier"] = match_result["match_tier"]

                # Collect the category as a "connected topic"
                connected_topics.add(kb_entry["category"])
                if "ministry" in kb_entry:
                    connected_topics.add(kb_entry["ministry"])

            matched_entities.append(entity_info)

        # STAGE 8: Assemble the 5 final outputs
        outputs = {
            "event_summary": self._format_summary(classified_sentences["event_summary"]),
            "why_in_news": self._format_summary(classified_sentences["why_news"]),
            "background": self._format_summary(classified_sentences["background"]),
            "connected_topics": sorted(list(connected_topics)),
            "important_entities": matched_entities,
            "dates_mentioned": date_entities,
        }

        return outputs

    def _format_summary(self, sentence_list: list) -> str:
        """Joins classified sentences into a readable summary string."""
        if not sentence_list:
            return "No relevant information found in this article for this section."
        return " ".join(sentence_list)


def print_report(outputs: dict):
    """Pretty-prints the 5 outputs in a readable report format."""
    print("=" * 75)
    print("1. EVENT SUMMARY")
    print("=" * 75)
    print(outputs["event_summary"])

    print("\n" + "=" * 75)
    print("2. WHY IS THIS IN THE NEWS?")
    print("=" * 75)
    print(outputs["why_in_news"])

    print("\n" + "=" * 75)
    print("3. BACKGROUND OF THE NEWS")
    print("=" * 75)
    print(outputs["background"])

    print("\n" + "=" * 75)
    print("4. CONNECTED TOPICS")
    print("=" * 75)
    for topic in outputs["connected_topics"]:
        print(f"  - {topic}")

    print("\n" + "=" * 75)
    print("5. IMPORTANT ENTITIES")
    print("=" * 75)
    for entity in outputs["important_entities"]:
        if entity["kb_match"]:
            print(f"  {entity['mention']:<30} -> {entity['kb_match']} ({entity['category']}) [{entity['match_tier']}]")
        else:
            print(f"  {entity['mention']:<30} -> (no KB match found)")

    if outputs.get("dates_mentioned"):
        print("\n  Dates mentioned in article (contextual, not KB-matched):")
        for date in outputs["dates_mentioned"]:
            print(f"    - {date}")


# ---------------------------------------------------------
# Quick manual test - full end-to-end pipeline run
# ---------------------------------------------------------
if __name__ == "__main__":
    pipeline = ChanakyaPipeline()

    sample_article = """
    Finance Minister Nirmala Sitharaman announced an expansion of the PM-KISAN
    scheme on Monday, increasing the annual benefit for small farmers. The
    Ministry of Finance said the change would affect over ten million
    farming families across the country. This decision comes as inflation
    concerns continue to affect rural households nationwide, and experts say
    it could significantly influence next year's agricultural output. The
    scheme was originally launched in 2019 as part of a broader push to
    support small and marginal farmers. A similar expansion was considered
    back in 2021 but was shelved due to budget constraints. The Reserve Bank
    of India welcomed the move, noting it could help stabilize rural demand.
    """

    print("PROCESSING SAMPLE ARTICLE...\n")
    outputs = pipeline.process_article(sample_article)
    print_report(outputs)
