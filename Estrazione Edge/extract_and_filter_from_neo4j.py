from neo4j import GraphDatabase
import pandas as pd
import networkx as nx
import os
from pronto import Ontology

# Configurazione connessione Neo4j
NEO4J_URI = "***"
NEO4J_USER = "***"
NEO4J_PASSWORD = "***"

# Cartella di output
os.makedirs("cartellaesempio", exist_ok=True)

# Funzione per estrarre nodi Disease
def get_disease_nodes(tx):
    query = """
    MATCH (d:Disease) RETURN d.identifier AS id, d.name AS name
    """
    return list(tx.run(query))

# Funzione per estrarre tutte le relazioni
def get_edges(tx):
    query = """
    MATCH (a)-[r]->(b) RETURN a.identifier AS source, b.identifier AS target, type(r) AS type
    """
    return list(tx.run(query))

# Funzione per estrarre antenati fino a una certa profondità
def get_ancestors_limited_depth(doid, onto, max_depth=3):
    ancestors = set()
    try:
        term = onto.get(doid)
    except Exception:
        return ancestors
    if not term:
        return ancestors
    queue = [(term, 0)]
    while queue:
        current, depth = queue.pop(0)
        if depth > 0:
            ancestors.add(current.id)
        if depth < max_depth:
            for parent in current.superclasses(with_self=False):
                queue.append((parent, depth + 1))
    return ancestors

CATEGORY_DOIDS = {
    'cancer': 'DOID:162',
    'gastrointestinal': 'DOID:77',
    'nervous_system': 'DOID:863',
    'immune_system': ['DOID:2914', 'DOID:17'],
    'mental_health': 'DOID:150',
}

def map_doid_to_category(doid, onto, category_doids):
    try:
        term = onto.get(doid)
    except Exception:
        term = None
    if term:
        ancestors = {a.id for a in term.superclasses(with_self=True)}
        for cat, cat_doids in category_doids.items():
            if isinstance(cat_doids, list):
                if any(d in ancestors for d in cat_doids):
                    return cat
            elif cat_doids in ancestors:
                return cat
    return None

if __name__ == "__main__":
    # Connessione a Neo4j
    driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
    with driver.session() as session:
        disease_nodes = session.read_transaction(get_disease_nodes)
        edges = session.read_transaction(get_edges)

    disease_df = pd.DataFrame(disease_nodes, columns=['id', 'name'])
    edges_df = pd.DataFrame(edges, columns=['source', 'target', 'type'])

    # Costruisci grafo completo
    G = nx.MultiDiGraph()
    for _, row in edges_df.dropna(subset=["source", "target"]).iterrows():
        G.add_edge(str(row["source"]), str(row["target"]), type=row["type"])

    # Calcola il grado di ogni nodo
    degree = dict(G.degree())
    HUB_THRESHOLD = 5000
    hub_nodes = {n for n, d in degree.items() if d > HUB_THRESHOLD}
    print(f"Nodi hub (grado > {HUB_THRESHOLD}): {len(hub_nodes)}")

    # Filtra gli edge: rimuovi quelli che coinvolgono almeno un hub
    filtered_edges = edges_df[~edges_df["source"].isin(hub_nodes) & ~edges_df["target"].isin(hub_nodes)].copy()
    print(f"Edge rimanenti dopo filtro: {len(filtered_edges)} su {len(edges_df)}")
    onto = Ontology("c:/Users/nunzi/Documents/fdsml/doid.obo")
    
    disease_df['category'] = disease_df['id'].apply(lambda doid: map_doid_to_category(doid, onto, CATEGORY_DOIDS))
    disease_df[['id', 'category']].to_csv("work_filtered/filtered_graph/disease_nodes_with_category.csv", index=False)
    disease_ancestors_real = []
    for doid in disease_df['id']:
        ancestors = get_ancestors_limited_depth(doid, onto, max_depth=5)
        for anc in ancestors:
            # Cerca tutti gli edge tra doid e anc fino ad una profondità (dopo filtro hub - INUTILIZZATO NEL PROGETTO FINALE)
            real_edges = filtered_edges[(filtered_edges['source'] == doid) & (filtered_edges['target'] == anc)]
            for _, edge_row in real_edges.iterrows():
                disease_ancestors_real.append({'source': doid, 'target': anc, 'type': edge_row['type']})
    ancestors_real_df = pd.DataFrame(disease_ancestors_real)
    print(f"Edge disease-ancestor reali trovati: {len(ancestors_real_df)}")
    ancestors_real_df.to_csv("work_filtered/filtered_graph/edges_ancestors_depth3.csv", index=False)


    # Effettua una ricerca SOLO per questi tipi di edge, individuati come i più informativi
    disease_ids = set(disease_df['id'])
    informative_types_out = [
        'ASSOCIATES_DaG', 'UPREGULATES_DuG', 'DOWNREGULATES_DdG', 'PRESENTS_DpS', 'LOCALIZES_DlA', 'RESEMBLES_DrD'
    ]
    informative_types_in = [
        'TREATS_CtD', 'RESEMBLES_DrD', 'PALLIATES_CpD'
    ]
    informative_edges_out = filtered_edges[(filtered_edges['source'].isin(disease_ids)) & (filtered_edges['type'].isin(informative_types_out))]
    informative_edges_in = filtered_edges[(filtered_edges['target'].isin(disease_ids)) & (filtered_edges['type'].isin(informative_types_in))]
    informative_edges = pd.concat([informative_edges_out, informative_edges_in], ignore_index=True)
    informative_edges.to_csv("work_filtered/filtered_graph/edges_informative.csv", index=False)
    print(f"Salvati {len(informative_edges)} edge informativi in edges_informative.csv")

    # Ottimizzazione: pre-calcola una mappa source -> lista di edge per non far andare in loop la ricerca per "step" (se trova altre malattie va in loop)
    from collections import defaultdict
    edges_by_source = defaultdict(list)
    for idx, row in filtered_edges.iterrows():
        edges_by_source[row['source']].append(idx)

    disease_grandchildren_edges = []
    for doid in disease_df['id']:
        # Trova tutti gli indici dei figli (primo livello)
        children_idx = edges_by_source.get(doid, [])
        if children_idx:
            children_df = filtered_edges.loc[children_idx]
            for _, child_row in children_df.iterrows():
                disease_grandchildren_edges.append({
                    'source': doid,
                    'target': child_row['target'],
                    'type': child_row['type'],
                    'level': 1
                })
                child = child_row['target']
                # Trova tutti gli indici dei nipoti (secondo livello)
                grandchildren_idx = edges_by_source.get(child, [])
                if grandchildren_idx:
                    grandchildren_df = filtered_edges.loc[grandchildren_idx]
                    for _, grandchild_row in grandchildren_df.iterrows():
                        disease_grandchildren_edges.append({
                            'source': child,
                            'target': grandchild_row['target'],
                            'type': grandchild_row['type'],
                            'level': 2
                        })
    # Rimuovi duplicati dagli edge raccolti
    disease_grandchildren_edges_df = pd.DataFrame(disease_grandchildren_edges).drop_duplicates()
    print(f"Edge generali disease->figlio e figlio->nipote trovati: {len(disease_grandchildren_edges_df)}")
    disease_grandchildren_edges_df = disease_grandchildren_edges_df.drop(columns=['level'], errors='ignore')
    print(f"Edge unici disease->figlio e figlio->nipote trovati: {len(disease_grandchildren_edges_df)}")
    disease_grandchildren_edges_df.to_csv("work_filtered/filtered_graph/edges_children_grandchildren.csv", index=False)

    # Salva solo le relazioni disease -> figlio
    disease_children_edges = [
        edge for edge in disease_grandchildren_edges
        if edge['level'] == 1
    ]
    disease_children_edges_df = pd.DataFrame(disease_children_edges).drop_duplicates()
    print(f"Edge generali disease->figlio trovati: {len(disease_children_edges_df)}")
    disease_children_edges_df = disease_children_edges_df.drop(columns=['level'], errors='ignore')
    print(f"Edge unici disease->figlio trovati: {len(disease_children_edges_df)}")
    disease_children_edges_df.to_csv("work_filtered/filtered_graph/edges_disease_children.csv", index=False)

    disease_df.to_csv("work_filtered/filtered_graph/disease_nodes_with_category.csv", index=False)
    edges_df.to_csv("work_filtered/filtered_graph/edges.csv", index=False)
    # Salva anche gli edge filtrati in origine per motivi di tracciabilità
    filtered_edges.to_csv("work_filtered/filtered_graph/edges_filtered.csv", index=False)
    print("Salvati: work_filtered/filtered_graph/disease_nodes.csv, edges_filtered.csv e edges_ancestors_depth3.csv (antenati fino a profondità 3)")
