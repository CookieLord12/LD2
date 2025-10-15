# ===============================================
# LD2: Knygų rekomendacijų sistema
# Metodai: TF–IDF + kosinis panašumas (NearestNeighbors)
# Rezultatų pateikimas: lentelė terminale (pandas)
# Vizualizacija: 2D projekcija (PCA/TruncatedSVD) su pažymėtomis panašiomis knygomis
# Papildomai: vartotojo įvestis terminale
# ===============================================

import pandas as pd
import numpy as np
import string
import warnings
warnings.filterwarnings("ignore")

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import TruncatedSVD
# t-SNE (pasirenkama; išjunk jei neturi paketo ar lėta)
try:
    from sklearn.manifold import TSNE
    HAS_TSNE = True
except Exception:
    HAS_TSNE = False

import matplotlib.pyplot as plt


# 1) Duomenų įkėlimas ir paruošimas
print("📚 Įkeliami duomenys...")
books = pd.read_csv("BooksDatasetClean.csv")

print("🧹 Valomi ir apjungiami tekstai...")
for col in ['Title', 'Authors', 'Description', 'Category', 'Publisher']:
    if col not in books.columns:
        books[col] = ''
    books[col] = books[col].fillna('')

books['text'] = (
    books['Title'] + ' ' +
    books['Authors'] + ' ' +
    books['Description'] + ' ' +
    books['Category'] + ' ' +
    books['Publisher']
)

def clean_text(text: str) -> str:
    text = text.lower()
    text = ''.join(ch for ch in text if ch not in string.punctuation)
    return text

books['clean_text'] = books['text'].apply(clean_text)


# 2) TF–IDF
print("🧠 Vektorizuojama (TF–IDF)...")
tfidf = TfidfVectorizer(stop_words='english', max_features=40000)
tfidf_matrix = tfidf.fit_transform(books['clean_text'])
print(f"✅ TF–IDF matricos forma: {tfidf_matrix.shape}")


# 3) Nearest Neighbors pagal kosinį atstumą
print("🔍 Kuriamas panašumo modelis...")
nn = NearestNeighbors(metric='cosine', algorithm='brute')
nn.fit(tfidf_matrix)

# Pagalbinė: grąžinti rekomendacijų DataFrame
def recommend_df(book_title: str, n: int = 5) -> pd.DataFrame:
    idx = books[books['Title'].str.lower() == book_title.lower()].index
    if len(idx) == 0:
        raise ValueError("⚠️ Knyga nerasta duomenų rinkinyje.")
    idx = idx[0]

    distances, indices = nn.kneighbors(tfidf_matrix[idx], n_neighbors=n+1)
    # Pirmas elementas – pati knyga, praleidžiam
    rec_idx = indices[0][1:]
    rec_dist = distances[0][1:]
    sim = 1.0 - rec_dist  # kosinio panašumo balas

    df = pd.DataFrame({
        "Title": books.iloc[rec_idx]['Title'].values,
        "Authors": books.iloc[rec_idx]['Authors'].values,
        "Category": books.iloc[rec_idx]['Category'].values,
        "Publisher": books.iloc[rec_idx]['Publisher'].values,
        "Similarity": np.round(sim, 3)
    })
    return df

# 4) Konsolinė recommend() funkcija (spausdina lentelę)
def recommend(book_title: str, n: int = 5):
    try:
        df = recommend_df(book_title, n=n)
    except ValueError as e:
        print(str(e))
        return

    print(f"\n📖 Panašios knygos į: '{book_title}'\n")
    # Gražiai atspausdinti lentelę terminale
    with pd.option_context('display.max_colwidth', 80, 'display.width', 140):
        print(df.to_string(index=False))


# 5) 2D projekcija (PCA/TruncatedSVD) su pažymėtais rekomenduotais kaimynais
def visualize_neighbors(book_title: str, n: int = 5, use_tsne: bool = False, random_state: int = 42):
    """
    Pavaizduoja 2D projekciją (numatytai: TruncatedSVD=PCA analogas) tik
    QUERY + REKOMENDACIJOS (kad būtų greita net su 100k įrašų).
    """
    idx = books[books['Title'].str.lower() == book_title.lower()].index
    if len(idx) == 0:
        print("⚠️ Knyga nerasta duomenų rinkinyje.")
        return
    idx = idx[0]

    # Rekomenduoti ir gauti indeksus
    distances, indices = nn.kneighbors(tfidf_matrix[idx], n_neighbors=n+1)
    rec_idx = indices[0]        # įskaitant pačią knygą
    labels = ["Query"] + [f"Rec{i}" for i in range(1, len(rec_idx))]

    # Ištraukiam tik šių eilučių TF–IDF
    subX = tfidf_matrix[rec_idx, :]

    # Projekcija į 2D:
    if use_tsne and HAS_TSNE:
        # t-SNE ant keliolikos vektorių bus greitas; perpaversime į tankų formatą
        X_dense = subX.toarray()
        reducer = TSNE(n_components=2, init='pca', learning_rate='auto', random_state=random_state, perplexity=min(5, len(rec_idx) - 1))
        X2 = reducer.fit_transform(X_dense)
        method_name = "t-SNE"
    else:
        # TruncatedSVD veikia tiesiogiai su sparse (PCA analogas)
        reducer = TruncatedSVD(n_components=2, random_state=random_state)
        X2 = reducer.fit_transform(subX)
        method_name = "PCA (TruncatedSVD)"

    # Braižymas
    plt.figure(figsize=(8, 6))
    # Query – didesnis ženkliukas
    plt.scatter(X2[0, 0], X2[0, 1], s=160, marker='*', label='Query', zorder=5)
    # Rekomendacijos
    if X2.shape[0] > 1:
        plt.scatter(X2[1:, 0], X2[1:, 1], s=60, marker='o', label='Recommendations', alpha=0.85)

    # Pridėti etiketes (pavadinimus), bet neperkraunant
    for i, idx_i in enumerate(rec_idx):
        title = books.iloc[idx_i]['Title']
        # Truputį patraukti tekstą nuo taško
        dx, dy = (0.01, 0.01) if i == 0 else (0.008, -0.008)
        plt.text(X2[i, 0] + dx, X2[i, 1] + dy,
                 (title[:40] + "…") if len(title) > 40 else title,
                 fontsize=8)

    plt.title(f"2D projekcija ({method_name}) – '{books.iloc[idx]['Title']}' ir {len(rec_idx)-1} rekomendacijų")
    plt.xlabel("Component 1")
    plt.ylabel("Component 2")
    plt.legend(loc='best')
    plt.tight_layout()
    plt.show()


# 6) Paleidimas iš terminalo
if __name__ == "__main__":
    print("\n✅ Duomenys įkelti. Gali naudoti recommend('______')")
    print("💡 Įvesk knygos pavadinimą (arba palik tuščią, kad išeitum).")

    while True:
        try:
            user_title = input("\n🔎 Knygos pavadinimas: ").strip()
        except EOFError:
            break

        if user_title == "":
            print("👋 Baigiame.")
            break

        # Lentelė su rezultatais
        recommend(user_title, n=5)

        # Vizualizacija (PCA/SVD pagal nutylėjimą; t-SNE gali būti lėtas)
        try:
            visualize_neighbors(user_title, n=5, use_tsne=False)
        except Exception as e:
            print(f"⚠️ Nepavyko nubraižyti: {e}")