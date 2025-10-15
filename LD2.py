# ===============================================
# LD2: Knygų rekomendacijų sistema (TF–IDF + NearestNeighbors)
# Tikslas: pasiūlyti panašias knygas pagal tekstinį aprašymą
# ===============================================

import pandas as pd
import numpy as np
import string
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors
import warnings
warnings.filterwarnings("ignore")

# 1️⃣ Duomenų įkėlimas
print("📚 Įkeliami duomenys...")
books = pd.read_csv("BooksDatasetClean.csv")

# 2️⃣ Duomenų paruošimas ir apjungimas į vieną tekstinį lauką
print("🧹 Valomi ir apjungiami tekstai...")

for col in ['Title', 'Authors', 'Description', 'Category', 'Publisher']:
    books[col] = books[col].fillna('')

# Sujungiame kelis laukus į vieną bendrą tekstinį lauką
books['text'] = (
    books['Title'] + ' ' +
    books['Authors'] + ' ' +
    books['Description'] + ' ' +
    books['Category'] + ' ' +
    books['Publisher']
)

# Teksto valymo funkcija
def clean_text(text):
    text = text.lower()
    text = ''.join([ch for ch in text if ch not in string.punctuation])
    return text

books['clean_text'] = books['text'].apply(clean_text)

# 3️⃣ TF–IDF vektorizacija
print("🧠 Vektorizuojama (TF–IDF)...")
tfidf = TfidfVectorizer(stop_words='english', max_features=40000)
tfidf_matrix = tfidf.fit_transform(books['clean_text'])

print(f"✅ TF–IDF matricos forma: {tfidf_matrix.shape}")

# 4️⃣ Kaimynų paieška (cosine similarity)
print("🔍 Kuriamas panašumo modelis...")
nn = NearestNeighbors(metric='cosine', algorithm='brute')
nn.fit(tfidf_matrix)

# 5️⃣ Rekomendacijų funkcija
def recommend(book_title, n=5):
    idx = books[books['Title'].str.lower() == book_title.lower()].index
    if len(idx) == 0:
        print("⚠️ Knyga nerasta duomenų rinkinyje.")
        return
    idx = idx[0]

    # Randame artimiausius kaimynus
    distances, indices = nn.kneighbors(tfidf_matrix[idx], n_neighbors=n+1)

    print(f"\n📖 Panašios knygos į: '{books.iloc[idx]['Title']}'\n")
    for i, dist in zip(indices[0][1:], distances[0][1:]):
        title = books.iloc[i]['Title']
        author = books.iloc[i]['Authors']
        category = books.iloc[i]['Category']
        print(f"- {title} ({author}) | {category.strip()} | panašumas: {1 - dist:.3f}")

# 6️⃣ Testinis kvietimas
if __name__ == "__main__":
    print("\n✅ Duomenys įkelti. Gali naudoti funkciją recommend('Goat Brothers')\n")
    # Pvz.:
recommend("Goat Brothers")
