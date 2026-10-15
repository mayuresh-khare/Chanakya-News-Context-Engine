from nlp.preprocessing import preprocess_article

mixed_text = "The Lok Sabha passed the विधेयक today after long debate."
result = preprocess_article(mixed_text)
print("Devanagari tokens found:", result["devanagari_tokens"])
print("Has Devanagari:", result["has_devanagari"])