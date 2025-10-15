# ===============================================
# LD2: Knygų rekomendacijų sistema su filtravimu ir web sąsaja (Streamlit)
# ===============================================

import pandas as pd
import numpy as np
import string
import streamlit as st
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors
from sklearn.decomposition import TruncatedSVD

# 1️⃣ Duomenų įkėlimas
@st.cache_data
def load_data():
    df = pd.read_csv("BooksDatasetClean.csv")
    for col in ['Title', 'Authors', 'Description', 'Category', 'Publisher', 'Price Starting With ($)']:
        df[col] = df[col].fillna('')
    df['Price Starting With ($)'] = pd.to_numeric(df['Price Starting With ($)'], errors='coerce').fillna(0)
    df['text'] = (
        df['Title'] + ' ' + df['Authors'] + ' ' + df['Description'] +
        ' ' + df['Category'] + ' ' + df['Publisher']
    )
    def clean_text(t):
        t = t.lower()
        return ''.join(ch for ch in t if ch not in string.punctuation)
    df['clean_text'] = df['text'].apply(clean_text)
    return df

books = load_data()

# 2️⃣ TF–IDF + Nearest Neighbors
@st.cache_resource
def build_model(df):
    tfidf = TfidfVectorizer(stop_words='english', max_features=40000)
    tfidf_matrix = tfidf.fit_transform(df['clean_text'])
    nn = NearestNeighbors(metric='cosine', algorithm='brute')
    nn.fit(tfidf_matrix)
    return tfidf, tfidf_matrix, nn

tfidf, tfidf_matrix, nn = build_model(books)

# 3️⃣ Rekomendacijų funkcija su filtrais
def recommend_df(book_title, n=5, genre=None, price_min=None, price_max=None):
    idx = books[books['Title'].str.lower() == book_title.lower()].index
    if len(idx) == 0:
        st.warning("Knyga nerasta duomenų rinkinyje.")
        return pd.DataFrame()
    idx = idx[0]
    distances, indices = nn.kneighbors(tfidf_matrix[idx], n_neighbors=n+50)  # daugiau, kad būtų ką filtruoti
    rec_idx = indices[0][1:]
    rec_dist = distances[0][1:]
    sim = 1.0 - rec_dist
    df = books.iloc[rec_idx].copy()
    df['Similarity'] = np.round(sim, 3)

    # Filtrai
    if genre:
        df = df[df['Category'].str.contains(genre, case=False, na=False)]
    if price_min is not None and price_max is not None:
        df = df[(df['Price Starting With ($)'] >= price_min) & (df['Price Starting With ($)'] <= price_max)]

    return df.head(n)

# 4️⃣ 2D vizualizacija (PCA)
def plot_projection(title, df):
    idx = books[books['Title'].str.lower() == title.lower()].index
    if len(idx) == 0 or df.empty:
        return None
    idx = idx[0]
    indices = list(df.index[:5]) + [idx]
    subX = tfidf_matrix[indices, :]
    reducer = TruncatedSVD(n_components=2, random_state=42)
    X2 = reducer.fit_transform(subX)
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.scatter(X2[:-1, 0], X2[:-1, 1], c='blue', label='Rekomendacijos')
    ax.scatter(X2[-1, 0], X2[-1, 1], c='red', marker='*', s=200, label='Užklausa')
    for i, t in enumerate(list(df['Title'][:5]) + [books.iloc[idx]['Title']]):
        ax.text(X2[i, 0]+0.02, X2[i, 1], t[:30], fontsize=8)
    ax.legend()
    ax.set_title(f"2D PCA projekcija – '{books.iloc[idx]['Title']}'")
    st.pyplot(fig)

# 5️⃣ Web sąsaja
st.title("📚 Knygų rekomendacijų sistema (TF–IDF + Cosine Similarity)")
st.write("Rekomendacijos pagal panašumą tarp knygų aprašymų ir papildomus filtrus.")

user_title = st.text_input("Įveskite knygos pavadinimą:")
genres = sorted(list(set([g.strip() for g in books['Category'].dropna().unique() if isinstance(g, str)])))
genre_filter = st.selectbox("Pasirinkite žanrą (nebūtina):", ["Visi"] + genres)
price_min, price_max = st.slider("Kainos intervalas ($):", 0.0, float(books['Price Starting With ($)'].max()), (0.0, 50.0))

if user_title:
    genre_val = None if genre_filter == "Visi" else genre_filter
    recs = recommend_df(user_title, n=5, genre=genre_val, price_min=price_min, price_max=price_max)

    if not recs.empty:
        st.success(f"📖 Panašios knygos į: *{user_title}*")
        st.dataframe(recs[['Title', 'Authors', 'Category', 'Publisher', 'Price Starting With ($)', 'Similarity']])
        plot_projection(user_title, recs)
    else:
        st.warning("Nerasta rekomendacijų pagal pasirinktus filtrus.")