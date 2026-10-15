import nltk
import spacy

# Test NLTK
from nltk.tokenize import word_tokenize
print("NLTK Test:", word_tokenize("Hello CHANAKYA, let's build something great!"))

# Test spaCy
nlp = spacy.load('en_core_web_sm')
doc = nlp("Narendra Modi visited the United Nations in New York.")
print("spaCy Test:", [(ent.text, ent.label_) for ent in doc.ents])