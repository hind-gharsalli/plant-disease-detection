# plant-disease-detection
#  Détection et Classification Automatique des Maladies des Plantes
> Projet académique — Ingénierie de l'Image  
> Pipeline complet : Traitement d'image · Machine Learning · Deep Learning

##  Description

Ce projet implémente un système intelligent de détection et classification automatique des maladies foliaires à partir d'images. Il combine des techniques classiques de traitement d'image, du feature engineering manuel et des approches de Deep Learning, dans le but de comparer leurs performances sur le dataset **PlantVillage**.

---

##  Résultats Obtenus

| Modèle | Accuracy | F1-Score |
|--------|----------|----------|
| **SVM (RBF)** | **93.13%** | **93.06%** |
| Logistic Regression | 92.50% | 92.40% |
| Gradient Boosting | 90.62% | 90.69% |
| MobileNetV2 (Transfer Learning) | 88.75% | 88.55% |
| CNN from Scratch | 25.62% | 19.79% |

---

## Classes Étudiées

Les 4 classes utilisées appartiennent toutes à la plante **Tomate** :

- 🟢 `Tomato_healthy` — Feuille saine
- 🟡 `Tomato_Early_blight` — Alternariose (taches brunes concentriques)
- 🔵 `Tomato_Late_blight` — Mildiou tardif (lésions gris-verdâtres)
- 🟠 `Tomato_Leaf_Mold` — Moisissure foliaire (taches jaunes)

---

## Architecture du Pipeline

```
Images PlantVillage
        │
        ▼
┌─────────────────────┐
│   1. PRÉTRAITEMENT  │  Redimensionnement 128×128 · RGB→HSV · Filtres Gaussien/Médian
└─────────────────────┘
        │
        ▼
┌─────────────────────┐
│  2. SEGMENTATION    │  Sobel/Canny · Otsu · HSV masking · K-Means (k=3)
└─────────────────────┘
        │
        ▼
┌─────────────────────┐
│ 3. FEATURE EXTRACT. │  Couleur (198D) · Texture GLCM (40D) · Forme (6D) = 244D
└─────────────────────┘
        │           │
        ▼           ▼
┌──────────┐  ┌──────────────────┐
│    ML    │  │   DEEP LEARNING  │
│  SVM ✅  │  │ CNN / MobileNetV2│
└──────────┘  └──────────────────┘
        │           │
        └─────┬─────┘
              ▼
     Comparaison & Évaluation
```

---

##  Dataset

- **Nom** : PlantVillage Dataset
- **Source** : [Kaggle — mohitsingh1804/plantvillage](https://www.kaggle.com/datasets/mohitsingh1804/plantvillage)
- **Contenu** : ~54 000 images · 38 classes
- **Utilisé** : 800 images · 4 classes · 200 images/classe

---

## Technologies Utilisées

```
Python 3.10
├── OpenCV           — Prétraitement, segmentation, contours
├── scikit-image     — GLCM (texture), features de forme
├── scikit-learn     — SVM, Random Forest, Gradient Boosting, métriques
├── TensorFlow/Keras — CNN from scratch, MobileNetV2
├── matplotlib       — Visualisations
├── seaborn          — Matrice de confusion
└── NumPy / Pandas   — Manipulation des données
```

---

## Structure du Notebook

```
plant_disease_detection.ipynb
│
├── Section 0  — Setup & Imports
├── Section 1  — Exploration du Dataset
├── Section 2  — Prétraitement des Images
│              (redimensionnement, HSV, filtres, histogrammes)
├── Section 3  — Segmentation & Contours
│              (Sobel, Canny, Otsu, HSV, K-Means)
├── Section 4  — Extraction de Features
│              (couleur, GLCM, forme)
├── Section 5  — Classification ML
│              (SVM, LR, RF, GB, KNN + métriques)
├── Section 6  — Deep Learning
│   ├── 6.1   CNN from Scratch
│   └── 6.2   MobileNetV2 Transfer Learning
└── Section 7  — Comparaison Finale ML vs DL
```

---

##  Lancer le Projet

### Sur Kaggle (recommandé — 0 installation)

1. Aller sur [kaggle.com](https://kaggle.com) et se connecter
2. Ouvrir le notebook : **[lien vers votre notebook Kaggle ici]**
3. Cliquer sur **Copy & Edit**
4. Activer le GPU : *Settings → Accelerator → GPU T4*
5. **Run All**

### En local

```bash
# Cloner le repo
git clone https://github.com/VOTRE_USERNAME/plant-disease-detection.git
cd plant-disease-detection

# Installer les dépendances
pip install -r requirements.txt

# Lancer Jupyter
jupyter notebook plant_disease_detection.ipynb
```

---

## Features Extraites

| Famille | Méthode | Dimensions |
|---------|---------|-----------|
| Couleur | Histogrammes RGB + HSV normalisés (32 bins) + statistiques | 198 |
| Texture | GLCM — Contrast, Dissimilarity, Homogeneity, Energy, Correlation | 40 |
| Forme | Surface, Périmètre, Circularité, Excentricité, Extent, Solidity | 6 |
| **TOTAL** | | **244** |

---

## Points Clés du Projet

- **Pourquoi SVM > CNN scratch ?** Avec seulement 200 images/classe, le feature engineering manuel capture les discriminants visuels plus efficacement qu'un CNN entraîné from scratch qui nécessite des dizaines de milliers d'exemples.

- **Pourquoi HSV > RGB ?** L'espace HSV sépare la teinte (H) de la luminosité (V), rendant l'analyse des couleurs pathologiques robuste aux variations d'éclairage.

- **Transfer Learning** : MobileNetV2 pré-entraîné sur ImageNet transfère des features génériques (bords, textures) utiles pour les feuilles, atteignant 88.75% sans données supplémentaires.

---

## Environnement

| Paramètre | Valeur |
|-----------|--------|
| Plateforme | Kaggle Notebooks |
| GPU | NVIDIA T4 (gratuit) |
| Dataset path | `/kaggle/input/datasets/mohitsingh1804/plantvillage/PlantVillage/train` |
| Taille des images | 128 × 128 px |
| Train / Test split | 80% / 20% (stratifié) |

---

##  Licence

Projet académique — Usage éducatif uniquement.  
Dataset : [PlantVillage](https://www.kaggle.com/datasets/mohitsingh1804/plantvillage) — Licence publique Kaggle.
