import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Carica la PCA 2D degli embedding TransE
df = pd.read_csv("InserireFilePCA.csv")

plt.figure(figsize=(12, 8))
sns.scatterplot(
    data=df,
    x='pca1',
    y='pca2',
    hue='category',
    palette='tab10',
    alpha=0.85,
    s=80,
    edgecolor='k',
    linewidth=0.5
)
plt.title("PCA 2D Transe - Grandchildren", fontsize=18)
plt.xlabel("Componente Principale 1", fontsize=14)
plt.ylabel("Componente Principale 2", fontsize=14)
plt.legend(title="Categoria clinica", bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=12, title_fontsize=13)
plt.grid(True, linestyle='--', alpha=0.4)
plt.tight_layout()
plt.savefig("disease_embedding_transe_grandchildren_pca2d.jpg", dpi=300)
plt.show()
print("Figura salvata come disease_embedding_transe_grandchildren_pca2d.jpg")
