import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from pykeen.pipeline import pipeline
from pykeen.triples import TriplesFactory

# Carica edges.csv come triple (head, rel, tail)
edges = pd.read_csv("/kaggle/input/diseases/edges_informative_hub.csv")
triples = [(str(h), str(r), str(t)) for h, r, t in edges[['source', 'type', 'target']].values]
tf_all = TriplesFactory.from_labeled_triples(np.array(triples))
tf_train, tf_test = tf_all.split([0.8, 0.2], random_state=42)

# Prepara TransE per Kaggle
results = pipeline(
    model='TransE',
    training=tf_train,
    testing=tf_test,
    model_kwargs={'embedding_dim': 512},
    training_kwargs={
        'num_epochs': 500,
        'batch_size': 512
    },
    random_seed=42,
    device='cuda'
)

# Prendi tutti i nodi Disease con categoria
nodes_df = pd.read_csv("/kaggle/input/diseases/disease_nodes_with_category.csv")
disease_ids = nodes_df['id'].astype(str).tolist()

# Ottieni embedding solo per i Disease con categoria
entity_embeddings = results.model.entity_representations[0](indices=None).detach().cpu().numpy()
entity_to_id = results.training.entity_to_id
embeddings = np.array([entity_embeddings[entity_to_id[n]] if n in entity_to_id else np.zeros(512) for n in disease_ids])

# Normalizza gli embedding
scaler = StandardScaler()
embeddings_scaled = scaler.fit_transform(embeddings)

# PCA per ridurre la dimensionalità a 2 (per visualizzazione)
pca = PCA(n_components=2)
embeddings_2d = pca.fit_transform(embeddings_scaled)

# Salva embedding e categoria
emb_df = pd.DataFrame(embeddings_scaled)
emb_df['category'] = nodes_df.set_index('id').loc[disease_ids, 'category'].values
emb_df.to_csv("disease_embedding_transe.csv", index=False)

# Salva anche la versione 2D per visualizzazione
emb_2d_df = pd.DataFrame(embeddings_2d, columns=['pca1', 'pca2'])
emb_2d_df['category'] = emb_df['category']
emb_2d_df.to_csv("disease_embedding_transe_pca2d.csv", index=False)

print("Salvati: disease_embedding_transe.csv (embedding TransE PyKEEN) e disease_embedding_transe_pca2d.csv (PCA 2D)")