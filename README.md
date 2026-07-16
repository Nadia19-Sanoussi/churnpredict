# ChurnPredict — Application Streamlit

Système intelligent de prédiction du churn et de recommandation de rétention
client (e-commerce & SaaS), basé sur un pipeline scikit-learn réel.

## Contenu du dossier

```
churnpredict_app.py               → application Streamlit (point d'entrée unique)
churn_pipeline.pkl                → pipeline sklearn entraîné (Imputer → Scaler → RandomForest)
churn_pipeline_metadonnees.pkl    → métadonnées du modèle (seuil, AUC, variables)
assets/
  illustration_accueil_dark.png   → illustration page d'accueil (thème sombre)
  illustration_accueil_light.png  → illustration page d'accueil (thème clair)
requirements.txt                  → dépendances Python
.env.example                      → modèle de fichier pour la clé Gemini / Ollama
```

La base SQLite `churnpredict_historique.db` est créée automatiquement au
premier lancement, dans le même dossier que `churnpredict_app.py`.

## Installation

```bash
pip install -r requirements.txt
```

## Lancement

```bash
streamlit run churnpredict_app.py
```

## Navigation (5 écrans)

- **Accueil** — présentation du système et des colonnes attendues
- **Tableau de bord** — vue d'ensemble de la dernière analyse (KPIs, graphiques)
- **Analyse CSV** — import d'un fichier, liste des clients avec score et segment ;
  cliquer sur un client ouvre sa page de détail (informations, recommandations
  métier, recommandation IA à la demande)
- **Saisie manuelle** — prédiction instantanée pour un client unique
- **Historique** — toutes les analyses passées : renommer, supprimer, télécharger
  le rapport PDF de chacune

## Recommandations IA — Ollama (local) ou Gemini (cloud)

ChurnPredict tente automatiquement, dans cet ordre :
**Ollama (local) → Gemini (cloud) → recommandations par règles métier.**

### Ollama (recommandé, gratuit, sans quota)

```bash
ollama pull llama3.2
ollama list
```

Aucune configuration supplémentaire n'est nécessaire : ChurnPredict détecte
automatiquement Ollama sur `http://localhost:11434`.

### Gemini (alternative cloud)

1. Obtenez une clé sur https://aistudio.google.com/app/apikey
2. Copiez `.env.example` vers `.env` et renseignez `GOOGLE_API_KEY`.

Si aucun des deux moteurs n'est disponible, l'utilisateur voit toujours un
message clair et non technique : « Les recommandations IA sont momentanément
indisponibles. Les recommandations basées sur les règles métier restent
disponibles. » — jamais de détail technique (URL, code d'erreur, trace).

## Rapport PDF

Depuis la page Analyse CSV ou l'Historique, un rapport PDF complet peut être
téléchargé : informations générales, répartition par segment, tableau des
clients, et recommandations détaillées pour les clients à risque élevé. Conçu
pour être partagé directement à un responsable.

## Colonnes attendues pour l'import CSV/Excel

| Variable             | Description                                      |
|----------------------|---------------------------------------------------|
| recence_jours        | Jours depuis la dernière interaction              |
| frequence_activite   | Nombre d'interactions sur la période               |
| engagement_temps     | Minutes de session sur la période                  |
| anciennete_mois      | Ancienneté du client (mois)                        |
| valeur_client        | Montant moyen / valeur vie client (CLV)            |
| satisfaction_client  | Score de satisfaction (1 à 5)                      |

Vos noms de colonnes peuvent différer : un mapping automatique est proposé,
ajustable manuellement à l'écran.

## Thème clair / sombre

L'interface suit automatiquement le thème choisi via le menu en haut à droite
de Streamlit (Settings → Choose app theme).
