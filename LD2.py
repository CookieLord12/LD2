import pandas as pd
import numpy as np
import string
import streamlit as st
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors
from sklearn.decomposition import TruncatedSVD

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

@st.cache_resource
def build_model(df):
    tfidf = TfidfVectorizer(stop_words='english', max_features=40000)
    tfidf_matrix = tfidf.fit_transform(df['clean_text'])
    nn = NearestNeighbors(metric='cosine', algorithm='brute')
    nn.fit(tfidf_matrix)
    return tfidf, tfidf_matrix, nn

tfidf, tfidf_matrix, nn = build_model(books)

def suggest_titles(prefix, limit=10):
    prefix = prefix.lower().strip()
    if not prefix:
        return []
    matches = books[books['Title'].str.lower().str.contains(prefix, na=False)]
    return matches['Title'].head(limit).tolist()

def recommend_df(book_title, n=5, genre=None, price_min=None, price_max=None):
    idx = books[books['Title'].str.lower() == book_title.lower()].index
    if len(idx) == 0:
        st.warning("Knyga nerasta duomenų rinkinyje.")
        return pd.DataFrame()
    idx = idx[0]
    distances, indices = nn.kneighbors(tfidf_matrix[idx], n_neighbors=n+50)
    rec_idx = indices[0][1:]
    rec_dist = distances[0][1:]
    sim = 1.0 - rec_dist
    df = books.iloc[rec_idx].copy()
    df['Similarity'] = np.round(sim, 3)
    if genre:
        df = df[df['Category'].apply(lambda x: genre.lower() in [g.strip().lower() for g in x.split(',')])]
    if price_min is not None and price_max is not None:
        df = df[(df['Price Starting With ($)'] >= price_min) & (df['Price Starting With ($)'] <= price_max)]
    return df.head(n)

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

def plot_full_dataset(sample_size=3000):
    if len(books) > sample_size:
        sample = books.sample(sample_size, random_state=42)
        X_sample = tfidf.transform(sample['clean_text'])
    else:
        sample = books
        X_sample = tfidf_matrix

    reducer = TruncatedSVD(n_components=2, random_state=42)
    X2 = reducer.fit_transform(X_sample)

    categories = sample['Category'].apply(
        lambda x: x.split(',')[0].strip() if isinstance(x, str) and ',' in x else (x.strip() if isinstance(x, str) else 'Unknown')
    )

    cat_codes, cat_labels = pd.factorize(categories)

    fig, ax = plt.subplots(figsize=(9, 7))
    scatter = ax.scatter(X2[:, 0], X2[:, 1], c=cat_codes, cmap='tab20', s=10, alpha=0.6)

    unique_cats = pd.unique(categories)
    colors = [plt.cm.tab20(i / max(1, len(unique_cats)-1)) for i in range(len(unique_cats))]
    for cat, color in zip(unique_cats, colors):
        ax.scatter([], [], c=[color], label=cat, s=30)
    ax.legend(title="Žanrai", loc='upper right', fontsize=8)

    ax.set_title(f"Visų knygų 2D PCA projekcija (pavyzdys: {sample_size} iš {len(books)})")
    ax.set_xlabel("Component 1")
    ax.set_ylabel("Component 2")
    st.pyplot(fig)

st.title("📚 Knygų rekomendacijų sistema (TF–IDF + Cosine Similarity)")
st.write("Rekomendacijos pagal panašumą tarp knygų aprašymų ir papildomus filtrus.")

all_genres = []
for val in books['Category'].dropna():
    parts = [p.strip() for p in val.split(',') if p.strip()]
    all_genres.extend(parts)
unique_genres = sorted(set(all_genres))
genre_filter = st.selectbox("Pasirinkite žanrą (nebūtina):", ["Visi"] + unique_genres)
price_min, price_max = st.slider("Kainos intervalas ($):", 0.0, float(books['Price Starting With ($)'].max()), (0.0, 50.0))

partial_input = st.text_input("Įveskite knygos pavadinimą (dalį pavadinimo):")
suggested_titles = suggest_titles(partial_input)
if suggested_titles:
    selected_title = st.selectbox("Pasirinkite knygą iš pasiūlymų:", suggested_titles)
else:
    selected_title = None

if selected_title:
    genre_val = None if genre_filter == "Visi" else genre_filter
    recs = recommend_df(selected_title, n=5, genre=genre_val, price_min=price_min, price_max=price_max)
    if not recs.empty:
        st.success(f"📖 Panašios knygos į: *{selected_title}*")
        st.dataframe(recs[['Title', 'Authors', 'Category', 'Publisher', 'Price Starting With ($)', 'Similarity']])
        plot_projection(selected_title, recs)
    else:
        st.warning("Nerasta rekomendacijų pagal pasirinktus filtrus.")

st.divider()
st.header("🌍 Visų knygų vizualizacija")
sample_size = st.slider("Pasirinkite kiek knygų parodyti:", 1000, 8000, 3000, step=500)
plot_full_dataset(sample_size)
