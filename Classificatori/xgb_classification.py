import pandas as pd
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import LabelEncoder
from imblearn.over_sampling import SMOTE

# Carica il dataset degli embedding
embedding_path = "work_on_graph/embeddings/disease_embedding_node_grandchildren.csv"
df = pd.read_csv(embedding_path)
X = df.drop("category", axis=1)
y = df["category"]

X = X.fillna(0)  # Sostituisci NaN con 0 nelle feature per evitare crash
y = y.fillna("Unknown")
y = LabelEncoder().fit_transform(y)

X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.2, random_state=42)
smote = SMOTE(random_state=42)
X_train, y_train = smote.fit_resample(X_train, y_train)


clf = XGBClassifier(
    learning_rate=0.01,
    max_depth=3,
    n_estimators=500,
    subsample=0.8,
    use_label_encoder=False,
    eval_metric='mlogloss',
    random_state=42
)
clf.fit(X_train, y_train)

# Valutazione su test set
y_pred = clf.predict(X_test)
print("\n=== Classification Report ===")
print(classification_report(y_test, y_pred))
print("\n=== Confusion Matrix ===")
print(confusion_matrix(y_test, y_pred))

# Metriche principali
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)

print(f"\nAccuracy: {accuracy:.3f}")
print(f"Precision: {precision:.3f}")
print(f"Recall: {recall:.3f}")
print(f"F1 Score: {f1:.3f}")

clf.save_model("xgboost_disease_classifier.json")
print("\nModello XGBoost salvato come xgboost_disease_classifier.json")
