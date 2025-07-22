import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from pykeen.pipeline import pipeline
from pykeen.triples import TriplesFactory

# Carica edges.csv come triple (head, rel, tail)
edges = pd.read_csv("work_on_graph/edges_informative_hub.csv")
triples = [(str(h), str(r), str(t)) for h, r, t in edges[['source', 'type', 'target']].values]
tf_all = TriplesFactory.from_labeled_triples(np.array(triples))
tf_train, tf_test = tf_all.split([0.8, 0.2], random_state=42)

# Prepara RotatE per Kaggle
results = pipeline(
    model='RotatE',
    training=tf_train,
    testing=tf_test,
    model_kwargs={'embedding_dim': 512},
    training_kwargs={'num_epochs': 500, 'batch_size': 512},
    random_seed=42,
    device='cuda'
)

# Prendi tutti i nodi Disease con categoria
nodes_df = pd.read_csv("work_on_graph/disease_nodes_with_category.csv")
disease_ids = nodes_df['id'].astype(str).tolist()
entity_embeddings = results.model.entity_representations[0](indices=None).detach().cpu().numpy()
entity_to_id = results.training.entity_to_id
embeddings = np.array([entity_embeddings[entity_to_id[n]] if n in entity_to_id else np.zeros(512) for n in disease_ids])

# Gestione dei valori complessi: estrai solo la parte reale degli embedding
embeddings = np.real(embeddings)

# Normalizza gli embedding
scaler = StandardScaler()
embeddings_scaled = scaler.fit_transform(embeddings)

pca = PCA(n_components=2)
embeddings_2d = pca.fit_transform(embeddings_scaled)
emb_df = pd.DataFrame(embeddings_scaled)
emb_df['category'] = nodes_df.set_index('id').loc[disease_ids, 'category'].values
emb_df.to_csv("disease_embedding_rotate.csv", index=False)

emb_2d_df = pd.DataFrame(embeddings_2d, columns=['pca1', 'pca2'])
emb_2d_df['category'] = emb_df['category']
emb_2d_df.to_csv("disease_embedding_rotate_pca2d.csv", index=False)

print("Salvati: disease_embedding_rotate.csv (embedding RotatE PyKEEN) e disease_embedding_rotate_pca2d.csv (PCA 2D)")
