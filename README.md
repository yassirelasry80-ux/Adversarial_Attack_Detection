# 🛡️ Système de Détection d'Attaques Adversariales sur Images Médicales

Projet complet d'apprentissage fédéré avec détection d'attaques adversariales (FGSM et PGD) sur des radiographies thoraciques.

## 📋 Table des Matières

- [Architecture](#architecture)
- [Prérequis](#prérequis)
- [Installation](#installation)
- [Structure du Projet](#structure-du-projet)
- [Utilisation](#utilisation)
- [Résultats](#résultats)

## 🏗️ Architecture

Le système implémente l'architecture suivante:

```
Hôpitaux (n datasets) → Détection d'attaques → Données propres → Apprentissage fédéré → Modèle global
                              ↓
                    Modèle pré-entraîné
                              ↓
                    Détecteur d'attaques
```

### Composants principaux:

1. **Collecte de données multi-sources**: Simulation de N hôpitaux avec leurs datasets
2. **Détection d'attaques**: Deep Learning pour identifier FGSM et PGD
3. **Filtrage des données**: Suppression des exemples adversariaux
4. **Apprentissage fédéré**: FedAvg pour l'agrégation des modèles locaux
5. **Modèle central**: Classification NORMAL vs PNEUMONIA

## 🔧 Prérequis

### Matériel
- **GPU**: RTX 4060 8GB (ou supérieur)
- **RAM**: 16GB recommandé
- **Stockage**: 5GB minimum

### Logiciels
- Windows 10/11
- Python 3.8+
- CUDA 11.8+ (pour GPU)
- Compte Kaggle (pour télécharger le dataset)

## 📥 Installation

### 1. Cloner ou créer le projet

Créez un dossier pour votre projet et copiez-y tous les fichiers.

### 2. Créer un environnement virtuel

```bash
python -m venv venv
venv\Scripts\activate
```

### 3. Installer les dépendances

```bash
pip install -r requirements.txt
```

### 4. Configurer Kaggle API

1. Allez sur [kaggle.com/account](https://www.kaggle.com/account)
2. Créez un nouveau token API (bouton "Create New API Token")
3. Placez le fichier `kaggle.json` dans: `C:\Users\<VotreNom>\.kaggle\`
4. Assurez-vous que le fichier a les permissions appropriées

### 5. Accepter les règles du dataset

Allez sur [kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia) et cliquez sur "Download" pour accepter les règles.

## 📁 Structure du Projet

```
projet/
├── app/                         # Code source du projet
│   ├── attacks/                 # Modules d'attaques
│   │   └── adversarial.py       # Implémentation FGSM et PGD
│   ├── data/                    # Gestion des données
│   │   ├── downloader.py        # Téléchargement du dataset
│   │   └── loader.py            # Chargement et préparation
│   ├── federated/               # Apprentissage fédéré
│   │   └── learning.py          # Logique d'entraînement fédéré
│   ├── models/                  # Architectures Deep Learning
│   │   ├── classifier.py        # Modèle de classification CNN
│   │   └── detector.py          # Détecteur d'attaques
│   └── config.py                # Configuration globale
├── main.py                      # Script principal
├── inference.py                 # Inférence et visualisation
├── requirements.txt             # Dépendances
└── README.md                    # Ce fichier
```

## 🚀 Utilisation

### Étape 1: Télécharger le dataset

```bash
python download_data.py
```

Cela téléchargera ~1.2GB de données depuis Kaggle.

### Étape 2: Lancer l'entraînement complet

```bash
python main.py
```

Le script exécutera automatiquement:
1. ✅ Chargement des données
2. ✅ Création des datasets fédérés (4 hôpitaux par défaut)
3. ✅ Génération d'attaques adversariales (FGSM et PGD)
4. ✅ Entraînement du détecteur d'attaques
5. ✅ Filtrage des données empoisonnées
6. ✅ Apprentissage fédéré (5 rounds par défaut)
7. ✅ Évaluation finale du modèle

### Étape 3: Tester l'inférence

```bash
python inference.py
```

Ou utilisez le code suivant pour vos propres images:

```python
from inference import InferenceSystem

# Créer le système
inference = InferenceSystem()

# Prédire sur une image
result = inference.predict_single_image("chemin/vers/image.jpg")

# Afficher le résultat
print(f"Prédiction: {result['prediction']}")
print(f"Confiance: {result['confidence']*100:.2f}%")
print(f"Attaque détectée: {result['is_adversarial']}")

# Visualiser
inference.visualize_prediction("chemin/vers/image.jpg", result)
```

## ⚙️ Configuration

Modifiez `config.py` pour ajuster les paramètres:

```python
# Paramètres du modèle
BATCH_SIZE = 16          # Réduire si manque de mémoire GPU
EPOCHS = 10
LEARNING_RATE = 0.001

# Paramètres fédérés
NUM_HOSPITALS = 4        # Nombre d'hôpitaux simulés
FEDERATED_ROUNDS = 5     # Nombre de rounds fédérés

# Paramètres d'attaques
EPSILON_FGSM = 0.03      # Intensité FGSM
EPSILON_PGD = 0.03       # Intensité PGD
PGD_ITERATIONS = 10      # Itérations PGD

# Détection
DETECTION_THRESHOLD = 0.15  # Seuil de détection
```

## 📊 Résultats Attendus

### Performance du modèle
- **Accuracy baseline**: ~85-90% sur données propres
- **Robustesse**: Détection de 70-85% des attaques adversariales

### Fichiers générés
- `poison_detector.pth`: Modèle de détection d'attaques
- `global_model_final.pth`: Modèle fédéré final

### Temps d'exécution (RTX 4060)
- Téléchargement: ~5-10 minutes
- Entraînement complet: ~30-45 minutes
- Inférence: <1 seconde par image

## 🔍 Détails Techniques

### Attaques Adversariales

**FGSM (Fast Gradient Sign Method)**
```python
perturbation = epsilon * sign(∇_x Loss(model(x), y))
x_adv = x + perturbation
```

**PGD (Projected Gradient Descent)**
```python
for i in range(iterations):
    x = x + alpha * sign(∇_x Loss(model(x), y))
    x = clip(x, x_original - epsilon, x_original + epsilon)
```

### Apprentissage Fédéré

**FedAvg Algorithm**
```
Pour chaque round:
  1. Distribuer le modèle global aux hôpitaux
  2. Entraîner localement sur les données de chaque hôpital
  3. Agréger: w_global = (1/N) * Σ w_local_i
```

## 🛠️ Dépannage

### Erreur de mémoire GPU

Réduisez `BATCH_SIZE` dans `config.py`:
```python
BATCH_SIZE = 8  # ou 4
```

### Dataset non trouvé

Vérifiez:
1. Fichier `kaggle.json` dans `C:\Users\<VotreNom>\.kaggle\`
2. Règles du dataset acceptées sur Kaggle
3. Connexion internet stable

### Erreur CUDA

Installez PyTorch avec CUDA:
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

## 📈 Améliorations Possibles

1. **Modèles**: Tester ResNet50, EfficientNet
2. **Attaques**: Ajouter C&W, DeepFool
3. **Fédéré**: Implémenter FedProx, FedBN
4. **Privacy**: Ajouter Differential Privacy
5. **Datasets**: Tester sur d'autres modalités médicales

## 📝 Citation

Si vous utilisez ce code, veuillez citer:

```bibtex
@software{adversarial_detection_federated,
  title={Adversarial Attack Detection in Federated Medical Imaging},
  year={2024},
  author={Your Name}
}
```

## 📄 Licence

Ce projet est fourni à des fins éducatives. Le dataset Chest X-Ray est soumis à sa propre licence sur Kaggle.

## 🤝 Contribution

Les contributions sont les bienvenues! Pour contribuer:
1. Fork le projet
2. Créez une branche (`git checkout -b feature/amelioration`)
3. Commit vos changements
4. Push vers la branche
5. Ouvrez une Pull Request

## 📧 Contact

Pour questions et support, ouvrez une issue sur GitHub.

---

**Note**: Ce projet est optimisé pour RTX 4060 8GB. Pour des GPUs avec moins de mémoire, ajustez les paramètres dans `config.py`.