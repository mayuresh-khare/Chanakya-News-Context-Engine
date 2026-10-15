
# CHANAKYA – AI News Context Engine

> Understand the news, not just read it.

CHANAKYA takes a news article and returns the context a reader needs to understand it. It is built on classical NLP techniques, not a chatbot or a black-box API.

It is **not** a chatbot, fake-news detector, sentiment analyzer or recommender. It does one job and returns exactly five outputs.

## Output

| # | Section | What it gives you |
|---|---------|-------------------|
| 1 | **Event Summary** | What happened |
| 2 | **Why is this in the News?** | The trigger: announcement, decision, deadline |
| 3 | **Background** | History and prior context |
| 4 | **Connected Topics** | Related schemes, ministries, bodies |
| 5 | **Important Entities** | People, organisations and places, linked to a knowledge base |

## Screenshots

| Home | Result |
|------|--------|
| ![Home](screenshots/home.png) | ![Result](screenshots/result.png) |

## How It Works

A 5-stage NLP pipeline:

```
Article → Clean → Understand → Extract → Classify → Enrich → 5-section report
```

| Stage | Techniques |
|-------|-----------|
| 1. Clean | Tokenisation, script validation (incl. Devanagari), stop-word removal, stemming vs lemmatisation, spelling correction (Levenshtein edit distance) |
| 2. Understand | N-gram language model (Laplace smoothing, perplexity), rule-based vs probabilistic POS tagging, CFG parsing (top-down and bottom-up) |
| 3. Extract | spaCy Named Entity Recognition, NLTK chunking, Lesk word-sense disambiguation (WordNet) |
| 4. Classify | TF-IDF + Multinomial Naive Bayes labels sentences as event, trigger or background (5-fold cross-validated) |
| 5. Enrich | Knowledge-base matching on 51 Indian-governance entries (schemes, acts, ministries, people, international organisations) using exact, partial and fuzzy tiers |

### Design decisions

- **Precision over recall.** Articles outside the knowledge base (cricket, tech, film) return zero matches instead of guesses.
- **Word-boundary matching**, so "MoD" does not match inside "Modi".
- **Type-aware matching.** Locations match exactly only, so "India" does not match "Clean India Mission". Partial matching is direction-aware, so "Bank of India" does not match "Reserve Bank of India".
- **Fuzzy guard.** Short mentions and large edit distances are rejected, so "UPI" does not match "PLI".
- **Majority-vote entity typing** across repeated mentions.

## Example

Input: a PM-KISAN budget article

| Entity | Type | Match |
|--------|------|-------|
| Nirmala Sitharaman | Person | exact |
| Ministry of Finance | Ministry | partial |
| Reserve Bank of India | Regulatory Body | partial |

Connected topics: Ministry, Person, Regulatory Body. Dates found: 2019, 2021.

## Tech Stack

Python · NLTK · spaCy · scikit-learn · Flask · SQLite · NumPy · Jinja2 · HTML/CSS/JS (dark/light theme)

## Project Structure

```
CHANAKYA/
├── app/                 # Flask app and templates
├── nlp/                 # Pipeline modules (one per NLP experiment)
│   ├── preprocessing.py
│   ├── normalization.py
│   ├── spelling_correction.py
│   ├── ngram_analysis.py
│   ├── pos_tagging.py
│   ├── parsing.py
│   ├── ner_chunking.py
│   ├── wsd.py
│   ├── classification.py
│   └── pipeline.py
├── kb/                  # Knowledge-base loader and matcher
├── data/knowledge_base/ # JSON knowledge base (51 entries)
├── static/              # CSS and theme toggle
├── db/                  # SQLite history
└── requirements.txt
```

## Run Locally

```bash
git clone https://github.com/mayuresh-khare/Chanakya-news-context-engine.git
cd chanakya-news-context-engine

python -m venv venv
venv\Scripts\activate          # Windows
# source venv/bin/activate     # macOS / Linux

pip install -r requirements.txt
python -m spacy download en_core_web_sm
python setup_nltk.py           # downloads required NLTK data

python app/app.py
```

Open http://127.0.0.1:5000

## Known Limitations

- The sentence classifier is trained on a small hand-labelled set (about 67% cross-validated accuracy).
- Lesk word-sense disambiguation sometimes picks the wrong sense.
- The knowledge base covers Indian governance only.
- The CFG parser is a teaching demo. The production pipeline uses spaCy's dependency parser.

## Roadmap

- Larger knowledge base and training data
- Transformer-based NER
- Hindi support (Devanagari validation already in place)
- Live news feeds

## Author

**Mayuresh Khare**
B.Tech Computer Engineering (Honors in AI & Data Science), Lokmanya Tilak College of Engineering, Navi Mumbai

Built as an NLP course mini-project covering preprocessing, stemming/lemmatisation, edit distance, n-grams, POS tagging, CFG parsing, NER/chunking, word-sense disambiguation and text classification.
