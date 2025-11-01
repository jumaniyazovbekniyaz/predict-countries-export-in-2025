

import os
import sys
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import mean_squared_error, r2_score, classification_report, accuracy_score, f1_score, silhouette_score
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import joblib
import matplotlib.pyplot as plt

# Define paths relative to script directory
MODELS_DIR = 'models'

# в этом я создаю папку models
try:
    os.makedirs(MODELS_DIR, exist_ok=True)
except OSError as e:
    print(f"Cannot create directory {MODELS_DIR}: {e}", file=sys.stderr)
    sys.exit(1)

# 1) Загрузка
try:
    df = pd.read_csv('2020-2025.csv')
except FileNotFoundError:
    print("Error: File '2020-2025.csv' not found in the current directory.", file=sys.stderr)
    print("Please place '2020-2025.csv' in the same directory as this script.", file=sys.stderr)
    sys.exit(1)
except pd.errors.ParserError:
    print("Error: Could not parse '2020-2025.csv'. Check if the CSV is correctly formatted.", file=sys.stderr)
    sys.exit(1)

# Validate columns
expected_cols = ['Country', '2020', '2021', '2022', '2023', '2024', '2025']
if not all(col in df.columns for col in expected_cols):
    print(f"Error: CSV missing expected columns. Expected: {expected_cols}, Found: {df.columns.tolist()}", file=sys.stderr)
    sys.exit(1)

print("Shape:", df.shape)
print("Columns:", df.columns.tolist())
print("\nHead:")
print(df.head(8))
print("\nNulls per column:")
print(df.isnull().sum())

# 2) Приведение типов
years = ['2020', '2021', '2022', '2023', '2024', '2025']
for y in years:
    if y in df.columns:
        df[y] = pd.to_numeric(df[y], errors='coerce')

# Маски: полные 2020-2024 и наличие 2025
mask_2020_2024 = df[['2020', '2021', '2022', '2023', '2024']].notnull().all(axis=1)
mask_2025 = df['2025'].notnull()

df_train = df[mask_2020_2024 & mask_2025].reset_index(drop=True)
df_predict = df[mask_2020_2024 & (~mask_2025)].reset_index(drop=True)

if df_train.empty:
    print("Error: No rows with complete 2020-2025 data for training.", file=sys.stderr)
    sys.exit(1)

print("\nTrain rows (complete 2020-2025):", df_train.shape[0])
print("To predict rows (2020-2024 present, 2025 missing):", df_predict.shape[0])

FEATURES = ['2020', '2021', '2022', '2023', '2024']

# 3) Регрессия
print("\n--- Регрессия: предсказание 2025 ---")
X = df_train[FEATURES].values
y = df_train['2025'].values
countries = df_train['Country'].values

if np.any(np.isnan(X)) or np.any(np.isnan(y)):
    print("Error: NaN values in training data after preprocessing.", file=sys.stderr)
    sys.exit(1)

X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42)

scaler_reg = StandardScaler()
X_tr_s = scaler_reg.fit_transform(X_tr)
X_te_s = scaler_reg.transform(X_te)

lr = LinearRegression()
rf = RandomForestRegressor(n_estimators=200, random_state=42)

lr.fit(X_tr_s, y_tr)
rf.fit(X_tr_s, y_tr)

def metrics_reg(true, pred):
    mse = mean_squared_error(true, pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(true, pred)
    return mse, rmse, r2

pred_lr = lr.predict(X_te_s)
pred_rf = rf.predict(X_te_s)

mse_lr, rmse_lr, r2_lr = metrics_reg(y_te, pred_lr)
mse_rf, rmse_rf, r2_rf = metrics_reg(y_te, pred_rf)

print(f"LinearRegression -> MSE={mse_lr:.2f}, RMSE={rmse_lr:.2f}, R2={r2_lr:.3f}")
print(f"RandomForestRegressor -> MSE={mse_rf:.2f}, RMSE={rmse_rf:.2f}, R2={r2_rf:.3f}")

# Cross-val for RF
cv = cross_val_score(rf, scaler_reg.transform(X), y, cv=5, scoring='r2')
print("RF cross-val R2 (5-fold):", cv, "mean:", cv.mean())

# Save models
try:
    joblib.dump(rf, os.path.join(MODELS_DIR, "rf_regressor.joblib"))
    joblib.dump(lr, os.path.join(MODELS_DIR, "lr_regressor.joblib"))
    joblib.dump(scaler_reg, os.path.join(MODELS_DIR, "scaler_reg.joblib"))
    print("Saved regression models and scaler.")
except OSError as e:
    print(f"Error saving models to {MODELS_DIR}: {e}", file=sys.stderr)
    sys.exit(1)

# Predict 2025 for missing rows
if df_predict.shape[0] > 0:
    X_pred = df_predict[FEATURES].values
    X_pred_s = scaler_reg.transform(X_pred)
    pred_missing = rf.predict(X_pred_s)
    df_predict['predicted_2025'] = pred_missing
    try:
        df_predict.to_csv('predicted_2025_rf.csv', index=False)
        print("Saved predicted 2025 for missing countries to predicted_2025_rf.csv")
    except OSError as e:
        print(f"Error saving predictions to predicted_2025_rf.csv: {e}", file=sys.stderr)
        sys.exit(1)

# Simple plot: true vs pred for RF on test
plt.figure(figsize=(6, 4))
plt.scatter(y_te, pred_rf)
plt.xlabel("True 2025")
plt.ylabel("Predicted 2025 (RF)")
plt.title("True vs Predicted 2025 (RF)")
plt.grid(True)
try:
    plt.savefig('true_vs_pred_rf.png')
    print("Saved true vs predicted plot to true_vs_pred_rf.png")
except OSError as e:
    print(f"Error saving plot to true_vs_pred_rf.png: {e}", file=sys.stderr)
    sys.exit(1)
plt.close()

# 4) Классификация по 2025 (три класса по квантилям)
print("\n--- Классификация: классы по 2025 (0=low,1=mid,2=high) ---")
q1 = np.quantile(y, 1/3)
q2 = np.quantile(y, 2/3)
def label_val(v):
    if v <= q1:
        return 0
    elif v <= q2:
        return 1
    else:
        return 2
labels = np.array([label_val(v) for v in y])
class_counts = pd.Series(labels).value_counts()
print("Counts per class:", class_counts.to_dict())
if any(class_counts / len(labels) < 0.1):
    print("Warning: Imbalanced classes detected:", class_counts.to_dict())

Xc_tr, Xc_te, yc_tr, yc_te = train_test_split(X, labels, test_size=0.2, random_state=42, stratify=labels)

scaler_clf = StandardScaler()
Xc_tr_s = scaler_clf.fit_transform(Xc_tr)
Xc_te_s = scaler_clf.transform(Xc_te)

clf_rf = RandomForestClassifier(n_estimators=200, random_state=42)
clf_lr = LogisticRegression(max_iter=1000, random_state=42)

clf_rf.fit(Xc_tr_s, yc_tr)
clf_lr.fit(Xc_tr_s, yc_tr)

pred_clf_rf = clf_rf.predict(Xc_te_s)
pred_clf_lr = clf_lr.predict(Xc_te_s)

print("RF classifier -> acc:", accuracy_score(yc_te, pred_clf_rf), "f1_macro:", f1_score(yc_te, pred_clf_rf, average='macro'))
print("LR classifier -> acc:", accuracy_score(yc_te, pred_clf_lr), "f1_macro:", f1_score(yc_te, pred_clf_lr, average='macro'))
print("\nClassification report (RF):\n", classification_report(yc_te, pred_clf_rf))

# Save classifiers and scaler
try:
    joblib.dump(clf_rf, os.path.join(MODELS_DIR, "rf_classifier.joblib"))
    joblib.dump(clf_lr, os.path.join(MODELS_DIR, "lr_classifier.joblib"))
    joblib.dump(scaler_clf, os.path.join(MODELS_DIR, "scaler_clf.joblib"))
    print("Saved classifiers and scaler.")
except OSError as e:
    print(f"Error saving classifiers to {MODELS_DIR}: {e}", file=sys.stderr)
    sys.exit(1)

# Predict classes for rows with missing 2025 (if any)
if df_predict.shape[0] > 0:
    Xc_missing = df_predict[FEATURES].values
    Xc_missing_s = scaler_clf.transform(Xc_missing)
    cls_missing = clf_rf.predict(Xc_missing_s)
    df_predict['predicted_class'] = cls_missing
    try:
        df_predict.to_csv('predicted_2025_with_classes.csv', index=False)
        print("Saved predictions with classes to predicted_2025_with_classes.csv")
    except OSError as e:
        print(f"Error saving class predictions to predicted_2025_with_classes.csv: {e}", file=sys.stderr)
        sys.exit(1)

# 5) Кластеризация (KMeans) по 2020-2024
print("\n--- Кластеризация: KMeans на 2020-2024 ---")
df_clust = df[mask_2020_2024].reset_index(drop=True)
if df_clust.shape[0] < 10:
    print("Warning: Too few rows for clustering:", df_clust.shape[0])

Xcl = df_clust[FEATURES].values
scaler_clust = StandardScaler()
Xcl_s = scaler_clust.fit_transform(Xcl)

best_k = 2
best_score = -1
for k in range(2, 7):
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels_k = km.fit_predict(Xcl_s)
    score = silhouette_score(Xcl_s, labels_k)
    print(f"k={k}, silhouette={score:.4f}")
    if score > best_score:
        best_score = score
        best_k = k

print("Best k:", best_k, "score:", best_score)
kmeans = KMeans(n_clusters=best_k, random_state=42, n_init=10)
klabels = kmeans.fit_predict(Xcl_s)
df_clust['cluster'] = klabels

# PCA для визуализации
pca = PCA(n_components=2, random_state=42)
pcs = pca.fit_transform(Xcl_s)
df_clust['pc1'] = pcs[:, 0]
df_clust['pc2'] = pcs[:, 1]

print("Clusters counts:", pd.Series(klabels).value_counts().to_dict())

print("Explained variance ratio:", pca.explained_variance_ratio_)
print("Total explained variance:", pca.explained_variance_ratio_.sum())

# Save clustering artifacts
try:
    joblib.dump(kmeans, os.path.join(MODELS_DIR, "kmeans.joblib"))
    joblib.dump(scaler_clust, os.path.join(MODELS_DIR, "scaler_clust.joblib"))
    joblib.dump(pca, os.path.join(MODELS_DIR, "pca.joblib"))
    print("Saved clustering model and artifacts.")
except OSError as e:
    print(f"Error saving clustering artifacts to {MODELS_DIR}: {e}", file=sys.stderr)
    sys.exit(1)

# Save a small sample of cluster results
try:
    df_clust[['Country', 'cluster', 'pc1', 'pc2']].to_csv('cluster_results_sample.csv', index=False)
    print("Saved sample cluster results to cluster_results_sample.csv")
except OSError as e:
    print(f"Error saving cluster results to cluster_results_sample.csv: {e}", file=sys.stderr)
    sys.exit(1)

# 6) Streamlit skeleton
streamlit_code = f"""
import os
import joblib
import streamlit as st
import streamlit as st
import numpy as np
import joblib

MODELS_DIR = 'models'
scaler_reg = joblib.load(os.path.join(MODELS_DIR, "scaler_reg.joblib"))
rf = joblib.load(os.path.join(MODELS_DIR, "rf_regressor.joblib"))
scaler_clf = joblib.load(os.path.join(MODELS_DIR, "scaler_clf.joblib"))
clf_rf = joblib.load(os.path.join(MODELS_DIR, "rf_classifier.joblib"))
kmeans = joblib.load(os.path.join(MODELS_DIR, "kmeans.joblib"))
scaler_clust = joblib.load(os.path.join(MODELS_DIR, "scaler_clust.joblib"))

st.title("Diploma Project: Economic Indicator 2020-2025")
st.write("Input values for years 2020-2024")
vals = []
for y in ['2020', '2021', '2022', '2023', '2024']:
    v = st.number_input(f'Value {{y}}', value=0.0, format="%.2f")
    vals.append(v)

if st.button("Predict"):
    X = np.array(vals).reshape(1, -1)
    Xs_reg = scaler_reg.transform(X)
    pred2025 = rf.predict(Xs_reg)[0]
    st.write("Predicted 2025:", pred2025)
    Xs_clf = scaler_clf.transform(X)
    cls = clf_rf.predict(Xs_clf)[0]
    st.write("Predicted class (0=low, 1=mid, 2=high):", int(cls))
    Xs_clust = scaler_clust.transform(X)
    cl = kmeans.predict(Xs_clust)[0]
    st.write("Cluster:", int(cl))
"""

try:
    with open('app.py', 'w', encoding='UTF-8') as f:
        f.write(streamlit_code)
    print("Streamlit skeleton saved to app.py")
except OSError as e:
    print(f"Error: Could not write Streamlit app to app.py: {e}", file=sys.stderr)
    sys.exit(1)

print("\nFiles in models directory:")
try:
    print(os.listdir(MODELS_DIR))
except OSError as e:
    print(f"Error listing {MODELS_DIR}: {e}", file=sys.stderr)

print("\nDone. Все шаги завершены. Проверь текущую директорию для артефактов: predicted files, models/, app.py")



# 2.5) Визуализации: boxplot и корреляция
import seaborn as sns

# Boxplot по всем годам (распределения значений)
plt.figure(figsize=(10, 6))
sns.boxplot(data=df[years])
plt.title("Boxplot по годам (2020–2025)")
plt.xlabel("Год")
plt.ylabel("Значение показателя")
plt.grid(True, alpha=0.3)
try:
    plt.savefig('boxplot_years.png', bbox_inches='tight')
    print("Saved boxplot to boxplot_years.png")
except OSError as e:
    print(f"Error saving boxplot_years.png: {e}", file=sys.stderr)
plt.close()
















