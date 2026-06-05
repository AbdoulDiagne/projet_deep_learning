# Projet ADL

Bonjour et bienvenue dans notre projet ! Ce fichier `README.md` a pour but de vous présenter simplement notre travail sur l'alignement éthique des modèles de langage et de vous expliquer comment exécuter notre code pas à pas.

---

## 1. C'est quoi ce projet ?

L'objectif de ce projet est d'apprendre à un petit modèle d'Intelligence Artificielle (ici Qwen2.5-1.5B-Instruct: un modèle de 1,5 milliard de paramètres) à **mieux respecter les valeurs éthiques humaines**. 


# Partie 2 : Alignement par Renforcement (RLHF)
*Responsable : [Ton Nom / Prénom]*

Pour y parvenir sous des contraintes techniques réalistes (comme l'utilisation de cartes graphiques gratuites sur Google Colab), nous avons implémenté la méthode **RLHF** (Reinforcement Learning from Human Feedback ou *Apprentissage par Renforcement à partir de Commentaires Humains*).

Notre pipeline de travail se divise en deux grandes étapes :
1. **Le Modèle de Récompense (Reward Model) :** On apprend d'abord à un modèle à faire la différence entre une "bonne" réponse éthique et une "mauvaise" réponse, à partir d'un jeu de données de préférences humaines.
2. **L'Optimisation par Renforcement (PPO) :** On utilise ensuite ce modèle de récompense comme un "professeur" (ou un arbitre). Le modèle de langage va générer des réponses, recevoir une note de l'arbitre, et modifier ses poids pour essayer d'obtenir les meilleures notes possibles à l'avenir.

Toutes nos versions sont testées à la fin sur un benchmark officiel appelé **ETHICS**, qui mesure la précision (*accuracy*) du modèle sur des questions de justice, de bon sens et de moralité.

---

### 2.1 Installation de l'environnement (Spécifique RLHF)

Pour installer proprement toutes les dépendances de cette partie sans conflits de versions (notamment avec NumPy 2.0 ou les packages expérimentaux préinstallés de Colab), exécutez ces commandes dans votre terminal ou dans une cellule Colab avant de démarrer :

```bash
# 1. Nettoyage des packages conflictuels préinstallés par Colab
pip uninstall -y torchtao numpy pyarrow google-colab opencv-python

# 2. Installation forcée de la base stable requise
pip install "numpy<2" "pyarrow>=14.0.0"

# 3. Installation automatique de toutes les dépendances du projet
pip install -r requirements.txt --ignore-installed pyarrow

### 2.2 Commandes d'Entraînement

Toute la configuration (taille des lots, taux d'apprentissage, dossiers de sauvegarde) est centralisée dans le fichier `configs/rlhf.yaml`.

#### Étape 1 : Entraîner le Reward Model (L'Arbitre)
Cette commande entraîne la tête de classification du modèle pour lui apprendre à noter le caractère acceptable ou inacceptable d'une réponse.
```bash
python scripts/train_reward_model.py --config configs/rlhf.yaml
### Étape 2 : Lancer l'optimisation par renforcement (PPO)
python scripts/train_ppo.py --config configs/rlhf.yaml

### Tester le modèle de base d'origine (Baseline)
HF_DATASETS_TRUST_REMOTE_CODE=1 python scripts/evaluate_ethics.py --config configs/rlhf.yaml --model baseline

#### Tester notre modèle aligné par renforcement (PPO)
HF_DATASETS_TRUST_REMOTE_CODE=1 python scripts/evaluate_ethics.py --config configs/rlhf.yaml --model aligned