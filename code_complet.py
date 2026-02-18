# ============================
#   IMPORT DES LIBRAIRIES
# ============================

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns          # pour la heatmap
import sklearn                 # pour KMeans via le wrapper

# ============================
#   FONCTIONS FOURNIES (PROF)
# ============================

def correlation_coefficient(df, col1, col2):
    """Calcule la corrélation entre deux colonnes en vérifiant qu'elles sont utilisables."""
    subdf = df[[col1, col2]].dropna()

    n = len(subdf)
    if n == 0:
        print("No data to compute correlation between", col1, "and", col2)
        return 0.0
    
    if (max(subdf[col1]) - min(subdf[col1])) < 0.001:
        print("Column", col1, "is constant, cannot compute correlation")
        return 0.0
    
    if (max(subdf[col2]) - min(subdf[col2])) < 0.001:
        print("Column", col2, "is constant, cannot compute correlation")
        return 0.0

    if subdf[col1].dtype not in [np.float64, np.int64]:
        print("Column", col1, "is not numeric, cannot compute correlation")
        return 0.0
    
    if subdf[col2].dtype not in [np.float64, np.int64]:
        print("Column", col2, "is not numeric, cannot compute correlation")
        return 0.0

    return np.corrcoef(subdf[col1], subdf[col2])[0, 1]


def correlated_subdataframe(df, threshold=0.0):
    """
    Garde uniquement les colonnes numériques qui ont au moins
    une corrélation au-dessus du seuil (en valeur absolue).
    """
    # Colonnes numériques uniquement
    df = df.select_dtypes(include=["int64", "float64"])

    # Supprimer les colonnes vides ou quasi constantes
    useful_columns = []
    for col in df.columns:
        if df[col].count() > 0 and (max(df[col]) - min(df[col])) > 0.001:
            useful_columns.append(col)
    df = df[useful_columns]

    # Matrice de corrélation
    corr = df.corr()

    # Garder seulement les colonnes qui corrèlent avec au moins une autre
    useful_columns = []
    for col in corr.columns:
        corr[col][col] = 0  # on met la diagonale à 0 pour ne pas la compter
        if max(corr[col]) >= threshold or min(corr[col]) <= -threshold:
            useful_columns.append(col)

    return df[useful_columns]


def fitting(df, x_name, y_name, degree, verbose=True):
    """
    Fit d'un polynôme entre deux colonnes (facultatif dans ce projet).
    """
    df = df[[x_name, y_name]].dropna()
    coeffs, residual, rank, singular_values, rcond = np.polyfit(
        df[x_name], df[y_name], degree, full=True
    )
    poly = np.poly1d(coeffs)
    if verbose and len(residual) > 0:
        print(poly, "/ MAE =", residual[0] / len(df))
    return poly

# ========= CLUSTERING (WRAPPER DU PROF) =========

def center_and_reduce(df, columns):
    """
    Centre et réduit les colonnes données (moyenne 0, écart-type 1).
    """
    for col in columns:
        avg_val = df[col].mean()
        std_val = df[col].std()
        df[col] = (df[col] - avg_val) / (std_val)
    return df[columns]


def kmeans(df, columns, n_clusters):
    """
    Applique KMeans aux colonnes données, ajoute une colonne 'kmeans'
    avec les labels de cluster, et retourne aussi les centres.
    """
    df = df[columns].dropna()
    km = sklearn.cluster.KMeans(n_clusters=n_clusters, random_state=314159265)
    km.fit(df)
    df["kmeans"] = km.labels_
    centers = pd.DataFrame(km.cluster_centers_, columns=columns)
    return df, centers


# ============================
#      CHARGEMENT DU CSV
# ============================

csv_path = r"C:\Users\PJ7\Desktop\project classe\crime2025.csv"

df = pd.read_csv(
    csv_path,
    parse_dates=["REPORT_DAT", "START_DATE", "END_DATE"],
    low_memory=False
)

print("===== Aperçu des données =====")
print(df.head(), "\n")

print("===== Infos générales =====")
print(df.info(), "\n")

print("===== Dimensions =====")
print(df.shape)    # (n_lignes, n_colonnes)


# ============================
#   ANALYSE DESCRIPTIVE
# ============================

print("\n===== Top 10 des types de crimes (OFFENSE) =====")
print(df["OFFENSE"].value_counts().head(10))

print("\n===== Nombre d'incidents par SHIFT =====")
print(df["SHIFT"].value_counts())

print("\n===== Nombre d'incidents par WARD =====")
print(df["WARD"].value_counts())

# Graphique : top 10 des crimes
top_offenses = df["OFFENSE"].value_counts().head(10)

plt.figure(figsize=(10, 5))
top_offenses.plot(kind="bar")
plt.title("Top 10 des types de crimes à Washington DC en 2025")
plt.xlabel("Type de crime (OFFENSE)")
plt.ylabel("Nombre d'incidents")
plt.tight_layout()
plt.show()


# ============================
#        CORRÉLATIONS
# ============================

print("\n===== ANALYSE DES CORRÉLATIONS =====")

# 1) Sous-dataframe avec colonnes numériques corrélées
df_corr = correlated_subdataframe(df, threshold=0.0)

print("\nColonnes numériques retenues pour la corrélation :")
print(list(df_corr.columns))

# 2) Matrice de corrélation numérique
corr_matrix = df_corr.corr()
print("\nMatrice de corrélation :")
print(corr_matrix)

# 3) Exemples de corrélation entre certaines colonnes
if "WARD" in df.columns and "LATITUDE" in df.columns:
    print("\nCorrélation WARD ↔ LATITUDE :",
          correlation_coefficient(df, "WARD", "LATITUDE"))

if "WARD" in df.columns and "LONGITUDE" in df.columns:
    print("Corrélation WARD ↔ LONGITUDE :",
          correlation_coefficient(df, "WARD", "LONGITUDE"))

if "X" in df.columns and "LONGITUDE" in df.columns:
    print("Corrélation X ↔ LONGITUDE :",
          correlation_coefficient(df, "X", "LONGITUDE"))

if "Y" in df.columns and "LATITUDE" in df.columns:
    print("Corrélation Y ↔ LATITUDE :",
          correlation_coefficient(df, "Y", "LATITUDE"))

# 4) Heatmap visuelle des corrélations
plt.figure(figsize=(10, 8))
sns.heatmap(
    corr_matrix,
    cmap="coolwarm",
    annot=True,
    fmt=".2f",
    linewidths=0.5
)
plt.title("Heatmap des corrélations des variables numériques (Crime 2025)")
plt.tight_layout()
plt.show()


# ============================
#           CLUSTERING
# ============================

print("\n===== CLUSTERING DES WARDS SELON LEUR PROFIL DE CRIME =====")

# 1) Tableau croisé : WARD vs OFFENSE
crime_matrix = pd.crosstab(df["WARD"], df["OFFENSE"])

print("\nAperçu du tableau croisé (WARD x OFFENSE) :")
print(crime_matrix.head())

# 2) Liste des colonnes à utiliser (tous les types de crimes)
crime_columns = list(crime_matrix.columns)

# 3) Centrer / réduire les colonnes (obligatoire avant KMeans)
crime_matrix_scaled = center_and_reduce(crime_matrix.copy(), crime_columns)

# 4) Appliquer KMeans via la fonction du prof
clustered_df, centers = kmeans(crime_matrix_scaled, crime_columns, n_clusters=3)

print("\nClusters assignés à chaque WARD :")
print(clustered_df["kmeans"])

# 5) Graphique : combien de Wards par cluster ?
plt.figure(figsize=(8, 5))
clustered_df["kmeans"].value_counts().sort_index().plot(kind="bar")
plt.title("Répartition des Wards par cluster (KMeans)")
plt.xlabel("Cluster")
plt.ylabel("Nombre de Wards")
plt.tight_layout()
plt.show()

print("\nCentres des clusters (dans l'espace des types de crimes, centrés/réduits) :")
print(centers)

print("\nAnalyse terminée.")
