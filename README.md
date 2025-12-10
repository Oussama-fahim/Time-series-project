# 🕰️ Projet de Prévision de Séries Temporelles Avancées

<div align="center">

![Python](https://img.shields.io/badge/Python-3.9%2B-blue)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange)
![PyTorch](https://img.shields.io/badge/PyTorch-1.x-red)
![Status](https://img.shields.io/badge/Status-Complété-green)

*Une solution complète pour la prévision de séries temporelles utilisant des méthodes statistiques classiques et des architectures de deep learning*

[Vue d'ensemble](#-vue-densemble) • [Structure](#-structure-du-projet) • [Méthodologie](#-méthodologie) • [Installation](#-installation) • [Résultats](#-résultats)

</div>

---

## 📋 Table des Matières

- [Vue d'ensemble](#-vue-densemble)
- [Structure du Projet](#-structure-du-projet)
- [Objectifs](#-objectifs)
- [Description des Données](#-description-des-données)
- [Méthodologie](#-méthodologie)
- [Installation et Exécution](#-installation-et-exécution)
- [Résultats](#-résultats)
- [Points Forts](#-points-forts)
- [Auteur](#-auteur)

---

## 🌟 Vue d'ensemble

Ce projet se concentre sur la construction de modèles de prévision avancés utilisant des méthodes statistiques classiques et des architectures de deep learning. Il comprend toutes les étapes d'un pipeline moderne de séries temporelles, de l'exploration des données à l'ingénierie des caractéristiques, la modélisation, l'évaluation et la prévision future.

Deux workflows principaux sont inclus :

### 📓 Deep_Time_Series.ipynb
Un pipeline complet de deep learning pour la prévision de séries temporelles génériques utilisant des architectures telles que LSTM, GRU, CNN-1D et modèles hybrides.

### 📊 M5_Forecasting.ipynb
Une solution complète pour le challenge M5 de prévision des ventes Walmart, combinant la prévision hiérarchique, les caractéristiques exogènes et les modèles de deep learning.

---

## 📁 Structure du Projet

```
Projet_Forecasting_Series_Temporelles/
│
├── 📓 Deep_Time_Series.ipynb              # Pipeline deep learning pour séries temporelles génériques
├── 📊 M5_Forecasting.ipynb                # Solution complète du challenge M5 Walmart
├── 📋 README.md                           # Documentation du projet
│
├── 📂 data/                               # Dossiers de données
│   ├── 📁 raw/                            # Données originales
│   └── 📁 processed/                      # Données nettoyées et transformées
│
├── 📂 models/                             # Modèles sauvegardés (optionnel)
└── 📂 results/                            # Résultats et visualisations (optionnel)
```

---

## 🎯 Objectifs

### Principaux Objectifs
1. **Construire des systèmes de prévision robustes** pour les séries temporelles multivariées ou hiérarchiques
2. **Comparer les approches classiques et de deep learning**
3. **Ingénierie des caractéristiques** améliorant la puissance prédictive
4. **Optimiser les modèles** avec des techniques avancées
5. **Évaluer** avec des métriques standards de l'industrie

### Comparaison des Modèles
- **Approches Classiques** : ARIMA, SARIMA, ETS, Prophet
- **Architectures Deep Learning** : LSTM, GRU, Encoder-Decoder, CNN-1D

---

## 📊 Description des Données

### 1. 📈 Dataset de Séries Temporelles Génériques
Utilisé pour explorer les architectures profondes :
- **value** : mesures observées
- **timestamp** : index temporel
- **Variables exogènes optionnelles** selon l'expérience

### 2. 🛒 Dataset M5 Forecasting
Fourni par Walmart :
- **30,490 produits hiérarchiques**
- **1,941 jours de ventes**
- **Variables exogènes** :
  - Prix
  - Événements calendaires
  - Événements SNAP
  - Différences au niveau des états

---

## 🔧 Méthodologie

### 1. Prétraitement des Données
- Gestion des timestamps manquants
- Remplissage des valeurs manquantes avec imputation contextuelle
- Normalisation : MinMaxScaler / StandardScaler
- Traitement des valeurs aberrantes via IQR ou z-score
- Test de stationnarité :
  - Test ADF
  - Test KPSS
  - Transformations Box-Cox optionnelles

### 2. Ingénierie des Caractéristiques
**Caractéristiques temporelles :**
- Retards : t-1, t-7, t-28
- Statistiques glissantes : moyenne, écart-type, min, max
- Caractéristiques basées sur le temps :
  - Jour de la semaine, mois, année
  - Indicateurs de vacances
- Séries de Fourier (pour les saisonnalités > 365 jours)

**Caractéristiques spécifiques M5 :**
- Volatilité des prix
- Indicateurs de promotion

### 3. Modélisation

#### 🧮 Modèles Classiques
| Modèle | Cas d'Utilisation |
|--------|-------------------|
| ARIMA / SARIMA | Données stationnaires ou saisonnières |
| Lissage Exponentiel | Tendances et modèles saisonniers |
| Prophet | Séries temporelles commerciales |

#### 🧠 Modèles Deep Learning
| Modèle | Description |
|--------|-------------|
| LSTM | Capture les dépendances temporelles longues |
| GRU | Variante plus rapide et efficace du LSTM |
| Conv1D | Extrait les caractéristiques temporelles locales |
| Encoder-Decoder | Prévision séquence-à-séquence |
| N-BEATS | Architecture profonde avancée pour la prévision univariée |
| Transformers | Méthode SOTA pour les dépendances à longue portée |

### 4. Évaluation
**Métriques selon le dataset :**

**Séries Temporelles Génériques :**
- MAE (Erreur Absolue Moyenne)
- RMSE (Racine de l'Erreur Quadratique Moyenne)
- MAPE (Erreur en Pourcentage Absolue Moyenne)
- sMAPE (Erreur Symétrique en Pourcentage Absolue Moyenne)

**Challenge M5 :**
- WRMSSE (Erreur Quadratique Moyenne Redimensionnée Pondérée)

**Visualisations :**
- Tableaux d'erreur
- Graphiques Prédictions vs Réelles
- Analyses des résidus

---

## 🚀 Installation et Exécution

### Prérequis
- Python 3.9+
- Matériel recommandé : GPU pour les modèles de deep learning

### Installation des Dépendances
```bash
# Créer un environnement virtuel
python -m venv venv

# Activer l'environnement
# Windows
venv\Scripts\activate
# macOS/Linux
source venv/bin/activate

# Installer les packages requis
pip install numpy pandas matplotlib seaborn scikit-learn statsmodels prophet
pip install tensorflow torch torchvision torchaudio
pip install jupyter notebook
```

### Exécution des Notebooks
```bash
# Lancer Jupyter
jupyter notebook

# Ouvrir dans l'ordre :
# 1. Deep_Time_Series.ipynb
# 2. M5_Forecasting.ipynb
```

---

## 📈 Résultats

### Découvertes Deep Learning
- **LSTM et GRU** fournissent des résultats solides pour la prévision à moyen terme
- **CNN-1D** améliore les performances pour les séries à mémoire courte et bruyante
- **Transformers** excellent dans la prévision de longues séquences

### Résultats M5 Forecasting
- L'ingénierie des caractéristiques améliore considérablement la précision
- Le modèle le plus performant combine :
  - Encodeur-décodeur de deep learning
  - Réconciliation hiérarchique
  - Caractéristiques calendaires et de prix

---

## 💪 Points Forts du Projet

### ✅ Structure Professionnelle
- Pipeline de prévision extrêmement bien structuré
- Exploration et modélisation de qualité professionnelle
- Intégration complète des techniques statistiques et de deep learning

### ✅ Reproductibilité et Praticité
- Workflow détaillé et reproductible
- Prêt pour le déploiement (modèles exportés + scripts)
- Documentation claire et complète

---

## 👨‍💻 Auteur

**Oussama Fahim**  
*Étudiant en intelligence artificielle*  
**ENSAM – Université Moulay Ismail**  
Meknès, Maroc

### 🔗 Contact
- **Email** : Oussamafahim2017@gmail.com
- **Téléphone** : +212 645 468 306
- **GitHub** : [github.com/oussama-fahim](https://github.com/oussama-fahim)

---

<div align="center">

**"La prévision est très difficile, surtout si elle concerne l'avenir."**  
*– Niels Bohr*

</div>
