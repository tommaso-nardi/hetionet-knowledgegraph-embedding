import pandas as pd
import networkx as nx
from node2vec import Node2Vec
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# Parametri embedding (per Kaggle)
EMBEDDING_DIM = 512
WALK_LENGTH = 40
NUM_WALKS = 20
P = 0.5
Q = 2

# Carica nodi Disease con categoria e archi
disease_df = pd.read_csv("/kaggle/input/diseases/disease_nodes_with_category.csv")
edges_df = pd.read_csv("/kaggle/input/diseases/edges_informative_all.csv")

# Costruisci grafo NetworkX
G = nx.MultiDiGraph()
for _, row in edges_df.dropna(subset=["source", "target"]).iterrows():
    G.add_edge(str(row["source"]), str(row["target"]), type=row["type"])

# Filtra solo i nodi Disease con categoria assegnata
disease_with_cat = disease_df[disease_df["category"].notna()].copy()

# Node2Vec
node2vec = Node2Vec(
    G,
    dimensions=EMBEDDING_DIM,
    walk_length=WALK_LENGTH,
    num_walks=NUM_WALKS,
    p=P,
    q=Q,
    workers=2
)
model = node2vec.fit(window=10, min_count=1, batch_words=4)

# Ottieni solo gli id per cui esiste l'embedding
valid_ids = [str(n) for n in disease_with_cat["id"] if str(n) in model.wv]
emb = np.array([model.wv[n] for n in valid_ids])

# Normalizza tutti gli embedding
scaler = StandardScaler()
embeddings_scaled = scaler.fit_transform(emb)

# Aggiungi la categoria
categories = disease_with_cat.set_index("id").loc[valid_ids, "category"].values

# Salva embedding normalizzati + categoria
emb_cols = [f"{i}" for i in range(emb.shape[1])]
emb_df = pd.DataFrame(embeddings_scaled, columns=emb_cols)
emb_df["category"] = categories
emb_df = emb_df.reset_index(drop=True)
emb_df.to_csv("disease_embedding_node.csv", index=False)
print("Salvato: disease_embedding_node_grandchildren.csv (embedding normalizzati + categoria)")

# PCA per ridurre la dimensionalità a 2
pca = PCA(n_components=2)
embeddings_2d = pca.fit_transform(embeddings_scaled)

emb_2d_df = pd.DataFrame(embeddings_2d, columns=['pca1', 'pca2'])
emb_2d_df['category'] = categories
emb_2d_df.to_csv("disease_embedding_node_pca2d.csv", index=False)
print("Salvato: disease_embedding_node_grandchildren_pca2d.csv (PCA 2D + categoria)")