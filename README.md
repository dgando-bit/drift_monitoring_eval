# Bike Sharing – Surveillance de la dérive des données avec Evidently

## Présentation

Utilisation de evidently pour surveiller la dérive des données de partage de vélos. 
Le jeu de données couvre 2 années complètes (de janvier 2011 à décembre 2012) et nous ne travaillerons que sur les mois de janvier et février.
Le jeu de données : https://archive.ics.uci.edu/static/public/275/bike+sharing+dataset.zip

---

## Structure du projet

```
.
├── bike_sharing_monitoring.py   # Script principal du pipeline de surveillance
├── requirements.txt             # Dépendances Python
└── README.md                    # Ce fichier
```
---

## Commande d'exécution

```bash
pip install -r requirements.txt
python bike_sharing_monitoring.py
```

Puis lancez l'interface Evidently pour visualiser tous les rapports :

```bash
evidently ui --workspace bike_sharing_workspace --port 8080
```

Ouvrez votre navigateur à l'adresse **http://localhost:8080** et naviguez vers le projet **Bike Sharing – Drift Monitoring**.

---

## Rapports générés (une seule exécution, un seul projet)

| # | Rapport | Preset Evidently |
|---|---------|-----------------|
| 1 | Validation du modèle (train vs test – janvier) | `RegressionPreset` |
| 2 | Modèle de production – Performance de référence (janvier complet) | `RegressionPreset` |
| 3 | Dérive hebdomadaire – Semaine 1 | `RegressionPreset` |
| 4 | Dérive hebdomadaire – Semaine 2 | `RegressionPreset` |
| 5 | Dérive hebdomadaire – Semaine 3 | `RegressionPreset` |
| 6 | Dérive de la cible – Pire semaine | `TargetDriftPreset` |
| 7 | Dérive des données – Semaine 3 (variables numériques uniquement) | `DataDriftPreset` |

---

## Analyse & Réponses aux questions

### Après l'étape 4 – Qu'est-ce qui a changé au cours des semaines 1, 2 et 3 ?

Les performances du modèle se dégradent progressivement au fil des semaines :

- **Semaine 1 ('2011-01-29 00:00:00' : '2011-02-07 23:00:00'):** RMSE = 21,04 — performances proches de la référence janvier, dérive faible.
- **Semaine 2 ('2011-02-07 00:00:00' : '2011-02-14 23:00:00'):** RMSE = 22,56 — légère dégradation, le modèle commence à sous-estimer la demande.
- **Semaine 3 ('2011-02-15 00:00:00' : '2011-02-21 23:00:00'):** RMSE = 37,90 — forte dégradation, la demande réelle dépasse systématiquement les prédictions.

La dérive est cumulative et s'accélère : la semaine 3 est la plus critique.

---

### Après l'étape 5 – Cause racine de la dérive (conclusion basée sur les données)

Le rapport `TargetDriftPreset` sur la semaine 3 confirme une dérive de la variable cible (`cnt`) : la demande réelle de vélos a augmenté en février au-delà de ce que le modèle entraîné sur janvier peut anticiper. Les prédictions restent calées sur la distribution de janvier et ne suivent plus la demande réelle.

**Cause racine : un concept drift saisonnier**, probablement lié à la hausse des températures et à une adoption croissante du service.

---

### Après l'étape 6 – Quelle stratégie appliquer ?

Le rapport `DataDriftPreset` sur la semaine 3 montre que les features météorologiques (`temp`, `atemp`) ont significativement dérivé, tandis que `hr` et `weekday` restent stables.

Face à ce **concept drift saisonnier**, deux stratégies prioritaires sont recommandées :

1. **Réentraînement par fenêtre glissante** : réentraîner le modèle chaque semaine sur les 4 à 6 dernières semaines pour rester aligné avec l'évolution de la demande.
2. **Surveillance avec alertes** : déclencher automatiquement un réentraînement dès que le RMSE ou le score de dérive dépasse un seuil prédéfini.

