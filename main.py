# TP5_nltk_spacy.py - NLP en français avec NLTK + POS tagging SpaCy

import string
import nltk
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
from nltk.probability import FreqDist
from nltk.util import ngrams
from nltk.tokenize import word_tokenize

# --------------------------
# 1️⃣ Télécharger les ressources NLTK
# --------------------------
nltk.download('stopwords')
nltk.download('punkt')

# --------------------------
# 2️⃣ Charger SpaCy pour POS tagging
# --------------------------
import spacy
# Si le modèle n'est pas installé :
# pip install spacy
# python -m spacy download fr_core_news_sm
nlp = spacy.load("fr_core_news_sm")

# --------------------------
# 3️⃣ Phrases à analyser
# --------------------------
phrases = [
    "Emmanuel Macron est président de la France depuis 2017.",
    "Paris est la capitale de la France.",
    "Apple a annoncé un nouvel iPhone à San Francisco.",
    
]
from nltk.tokenize import RegexpTokenizer

tokenizer = RegexpTokenizer(r'\w+')
# --------------------------
# 4️⃣ Tokenisation avec NLTK
# --------------------------
def nltk_tokenize(sentence):
    """
    Tokenisation en mots avec NLTK pour le français.
    """
      # Tokenisation basée sur des mots (ignore ponctuation)
    tokens = tokenizer.tokenize(sentence)
    return tokens

tokenized_phrases = [nltk_tokenize(p) for p in phrases]

print("=== 1. TOKENISATION ===\n")
for i, tokens in enumerate(tokenized_phrases, 1):
    print(f"S{i} tokens :", tokens)
print("\n")

# --------------------------
# 5️⃣ Suppression des stopwords
# --------------------------
stop_words = set(stopwords.words('french'))
cleaned_phrases = [[w for w in tokens if w.lower() not in stop_words and w not in string.punctuation]
                   for tokens in tokenized_phrases]

print("=== 2. SUPPRESSION STOPWORDS ===\n")
for i, tokens in enumerate(cleaned_phrases, 1):
    print(f"S{i} mots nettoyés :", tokens, "\n")

# --------------------------
# 6️⃣ Stemming avec NLTK
# --------------------------
stemmer = SnowballStemmer("french")
stemmed_phrases = [[stemmer.stem(w) for w in tokens] for tokens in cleaned_phrases]

print("=== 3. STEMMING ===\n")
for i, tokens in enumerate(stemmed_phrases, 1):
    print(f"S{i} stems :", tokens, "\n")

# --------------------------
# 7️⃣ POS tagging avec SpaCy
# --------------------------
pos_tagged_phrases = []
for tokens in cleaned_phrases:
    sentence = " ".join(tokens)
    doc = nlp(sentence)
    pos_tags = [(token.text, token.pos_) for token in doc]
    pos_tagged_phrases.append(pos_tags)

print("=== 4. POS TAGGING ===\n")
for i, pos_tokens in enumerate(pos_tagged_phrases, 1):
    print(f"S{i} POS tags :", pos_tokens)
print("\n")

# --------------------------
# 8️⃣ Fréquence des mots avec NLTK
# --------------------------
all_words = [w.lower() for tokens in cleaned_phrases for w in tokens]
freq_dist = FreqDist(all_words)

print("=== FREQUENCE DES MOTS ===\n")
print("Top 10 mots les plus fréquents :", freq_dist.most_common(10), "\n")

# --------------------------
# 9️⃣ Bigrams et Trigrams avec NLTK
# --------------------------
all_bigrams = list(ngrams(all_words, 2))
all_trigrams = list(ngrams(all_words, 3))

freq_bigrams = FreqDist(all_bigrams)
freq_trigrams = FreqDist(all_trigrams)

print("=== BIGRAMS ET TRIGRAMS ===\n")
print("Top 5 bigrams :", freq_bigrams.most_common(5))
print("Top 5 trigrams :", freq_trigrams.most_common(5))
