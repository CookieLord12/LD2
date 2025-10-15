import pandas as pd
import numpy as np
import string
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Importai ir duomenų įkėlimas
books = pd.read_csv("BooksDatasetClean.csv")

print(books.head())

#Duomenų valymas
books['Description'] = books['Description'].fillna('')
books['Category'] = books['Category'].fillna('')

books['text'] = books['Description'] + ' ' + books['Category']

def clean_text(text):
    text = text.lower()
    text = ''.join([ch for ch in text if ch not in string.punctuation])
    return text

books['clean_text'] = books['text'].apply(clean_text)

# TF–IDF vektorizacija
tfidf = TfidfVectorizer(stop_words='english', max_features=10000)
tfidf_matrix = tfidf.fit_transform(books['clean_text'])

# Kosinio panašumo matrica
cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

# Rekomendacijų funkcija
def recommend(title, n=5):
    idx = books[books['Title'].str.lower() == title.lower()].index
    if len(idx) == 0:
        print("Knyga nerasta. Bandyk su tiksliu pavadinimu.")
        return
    idx = idx[0]
    
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)[1:n+1]
    
    print(f"\nPanašios knygos į: '{books.iloc[idx]['Title']}'\n")
    for i, score in sim_scores:
        print(f"- {books.iloc[i]['Title']} ({books.iloc[i]['Authors']})  |  Similarity: {score:.2f}")