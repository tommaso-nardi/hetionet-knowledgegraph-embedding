## FDSML - Classificazione di Malattie tramite Knowledge Graph Embedding

Sistema di machine learning per la classificazione automatica di malattie in categorie cliniche utilizzando tecniche di embedding su knowledge graph biomedici.

### Descrizione

Il progetto implementa una pipeline completa per classificare 136 malattie del database Hetionet in 5 categorie cliniche: cancer, immune_system, nervous_system, mental_health e gastrointestinal. Il sistema combina tre algoritmi di knowledge graph embedding (TransE, RotatE, Node2Vec) con tre classificatori ensemble (Random Forest, XGBoost, Gradient Boosting) per valutare diverse configurazioni metodologiche.

L'approccio si basa su tre strategie di filtraggio del grafo biomedico. La strategia informativi seleziona esclusivamente relazioni ad alto valore semantico, la strategia children include tutti i nodi direttamente connessi alle malattie, mentre la strategia grandchildren estende l'analisi a nodi raggiungibili con due salti di distanza. Ogni strategia produce sottografi con caratteristiche topologiche diverse che influenzano la qualità degli embedding generati.

### Organizzazione del Codice

Classificatori contiene l'implementazione dei tre algoritmi di classificazione ensemble utilizzati nel sistema. I file rf_classification.py, xgb_classification.py e gb_classification.py implementano rispettivamente Random Forest, XGBoost e Gradient Boosting, fornendo un'interfaccia unificata per la valutazione delle performance su diverse tipologie di embedding.

Estrazione Edge contiene il codice e i file necessari per la connessione con il database Neo4j contenente Hetionet e l'estrazione dei sottografi secondo le tre strategie implementate. Il file extract_and_filter_from_neo4j.py costituisce lo script principale che genera i file CSV contenenti le relazioni filtrate e la categorizzazione delle malattie. I file edges_informative.csv, edges_disease_children.csv ed edges_children_grandchildren.csv contengono rispettivamente le relazioni estratte per ciascuna strategia, mentre disease_nodes_with_category.csv contiene la mappatura delle malattie alle categorie cliniche fatta con Disease Ontology.

File Embeddings raccolgono le rappresentazioni vettoriali generate dai tre algoritmi di embedding per ciascuna strategia di filtraggio. Ogni file CSV contiene gli embedding a 512 dimensioni per le 136 malattie insieme alle relative etichette categoriali. La directory File PCA contiene le visualizzazioni bidimensionali ottenute tramite Principal Component Analysis, utilizzate per l'analisi qualitativa della separabilità delle categorie nello spazio degli embedding. La cartella Risultati documenta le metriche di performance ottenute dalle diverse configurazioni sperimentali.

### Approccio Metodologico

Il sistema valuta sistematicamente 27 configurazioni ottenute dalla combinazione di 3 algoritmi di embedding, 3 strategie di filtraggio e 3 classificatori supervisionati. TransE implementa un approccio traslazionale che modella le relazioni come traslazioni vettoriali nello spazio euclideo, mentre RotatE utilizza rotazioni nello spazio complesso per catturare pattern relazionali più sofisticati. Node2Vec applica tecniche di random walk per generare embedding basati su similarità strutturali nel grafo.

Gli algoritmi di embedding vengono applicati ai sottografi estratti secondo le tre strategie, producendo rappresentazioni vettoriali embedding che catturano diverse granularità di informazione relazionale. Le rappresentazioni generate vengono quindi utilizzate per addestrare i classificatori ensemble, permettendo una valutazione comparativa delle capacità discriminative di ciascuna combinazione metodologica.

### Risultati e Performance

Come ben visibile dalla cartella Risultati, la configurazione ottimale identificata combina RotatE con Random Forest e la strategia children, raggiungendo il 67.9% di accuratezza con un F1-score di 0.633.
L'analisi comparativa evidenzia che RotatE mantiene performance consistentemente elevate across tutte le strategie di filtraggio, mentre TransE raggiunge risultati competitivi principalmente con la strategia informativi, Node2Vec presenta invece limitazioni significative, suggerendo che gli approcci basati su random walk non sono adatti per questo tipo di task.
