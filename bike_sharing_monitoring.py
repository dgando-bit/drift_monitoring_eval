import datetime
import io
import warnings
import zipfile

import numpy as np
import pandas as pd
import requests
from sklearn import ensemble, model_selection

from evidently.report import Report
from evidently.metric_preset import DataDriftPreset, RegressionPreset, TargetDriftPreset
from evidently.pipeline.column_mapping import ColumnMapping
from evidently.ui.workspace import Workspace

warnings.filterwarnings("ignore")
warnings.simplefilter("ignore")


# -------------Constants--------------------------------------------------------------

TARGET = "cnt"
PREDICTION = "prediction"
NUMERICAL_FEATURES = ["temp", "atemp", "hum", "windspeed", "mnth", "hr", "weekday"]
CATEGORICAL_FEATURES = ["season", "holiday", "workingday"]

WEEK_PERIODS = {
    "week_1": ("2011-01-29 00:00:00", "2011-02-07 23:00:00"),
    "week_2": ("2011-02-07 00:00:00", "2011-02-14 23:00:00"),
    "week_3": ("2011-02-15 00:00:00", "2011-02-21 23:00:00"),
}

WORKSPACE_PATH = "bike_sharing_workspace"
PROJECT_NAME = "Bike Sharing - Drift Monitoring"
PROJECT_DESCRIPTION = (
    "Monitoring de la dérive des données de partage de vélos "
    " en utilisant Evidently."
)



# ---------------------------------------------------------------------------
# Step 1 – Ingérer ou lire les données, preprocessing, et entraînement du modèle de validation
# ---------------------------------------------------------------------------

def _fetch_data() -> pd.DataFrame:
    """Download and extract the UCI Bike Sharing dataset (hourly)."""
    url = "https://archive.ics.uci.edu/static/public/275/bike+sharing+dataset.zip"
    content = requests.get(url, verify=False).content
    with zipfile.ZipFile(io.BytesIO(content)) as arc:
        raw_data = pd.read_csv(
            arc.open("hour.csv"), header=0, sep=",", parse_dates=["dteday"]
        )
    return raw_data


def _process_data(raw_data: pd.DataFrame) -> pd.DataFrame:
    """Set a DatetimeIndex combining date and hour columns."""
    raw_data.index = raw_data.apply(
        lambda row: datetime.datetime.combine(
            row.dteday.date(), datetime.time(row.hr)
        ),
        axis=1,
    )
    return raw_data


def load_data() -> pd.DataFrame:
    """Fetch and pre-process the bike sharing dataset."""
    print("[1/7] Fetching data from UCI repository...")
    raw_data = _process_data(_fetch_data())
    print(
        f"      Dataset loaded: {len(raw_data):,} rows  |  "
        f"range: {raw_data.index.min()} -> {raw_data.index.max()}"
    )
    return raw_data


### --- Column mapping helper ---
def _build_column_mapping(
    target: str = TARGET,
    prediction: str = PREDICTION,
    numerical_features: list = None,
    categorical_features: list = None,
) -> ColumnMapping:
    """Return a configured ColumnMapping object for Evidently."""
    if numerical_features is None:
        numerical_features = NUMERICAL_FEATURES
    if categorical_features is None:
        categorical_features = CATEGORICAL_FEATURES

    cm = ColumnMapping()
    cm.target = target
    cm.prediction = prediction
    cm.numerical_features = numerical_features
    cm.categorical_features = categorical_features
    return cm


### Entrainer le modèle
def step1_train_validation_model(
    raw_data: pd.DataFrame,
) -> tuple:
    """
    Diviser les données en ensembles d'entraînement et de test (70/30),
    Entraîner un RandomForestRegressor sur les données d'entraînement et de test des données de janvier 2011,
    """
    print("[2/7] Step 1 - Modèle de validation entraîné sur les données de janvier 2011...")

    reference_jan11 = raw_data.loc["2011-01-01 00:00:00":"2011-01-28 23:00:00"]
    features = NUMERICAL_FEATURES + CATEGORICAL_FEATURES

    X_train, X_test, y_train, y_test = model_selection.train_test_split(
        reference_jan11[features],
        reference_jan11[TARGET],
        test_size=0.3,
        random_state=42,
    )

    regressor = ensemble.RandomForestRegressor(random_state=0, n_estimators=50)
    regressor.fit(X_train, y_train)

    preds_train = regressor.predict(X_train)
    preds_test = regressor.predict(X_test)

    X_train = X_train.copy()
    X_train["target"] = y_train
    X_train["prediction"] = preds_train

    X_test = X_test.copy()
    X_test["target"] = y_test
    X_test["prediction"] = preds_test

    print(f"      Train size: {len(X_train):,}  |  Test size: {len(X_test):,}")
    return regressor, X_train, X_test


# ---------------------------------------------------------------------------
# Step 2 – Rapport de la validation du modèle (train vs test)
# ---------------------------------------------------------------------------

def step2_model_validation_report(
    workspace: Workspace,
    project,
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
) -> None:
    """
    Créez et enregistrez un rapport RegressionPreset en utilisant les données d'entraînement comme référence
    et les données de test comme données actuelles.
    """
    print("[3/7] Step 2 - Génération du rapport de validation du modèle...")

    # The validation split uses 'target' / 'prediction' column names
    column_mapping = ColumnMapping()
    column_mapping.target = "target"
    column_mapping.prediction = "prediction"
    column_mapping.numerical_features = NUMERICAL_FEATURES
    column_mapping.categorical_features = CATEGORICAL_FEATURES

    regression_performance_report = Report(metrics=[RegressionPreset()])
    regression_performance_report.run(
        reference_data=X_train.sort_index(),
        current_data=X_test.sort_index(),
        column_mapping=column_mapping,
    )

    workspace.add_report(project.id, regression_performance_report)
    print("      Rapport de validation enregistré dans le workspace.")


# ---------------------------------------------------------------------------
# Step 3 – Construire le modèle de production sur l'ensemble des données de janvier.
# ---------------------------------------------------------------------------

def step3_build_production_model(
    raw_data: pd.DataFrame,
) -> tuple:
    """
    Réentraîner le regressor sur l'ensemble des données de janvier 2011 (modèle de production).
    """
    print("[4/7] Step 3 - Construire le modèle de production sur l'ensemble des données de janvier...")

    reference_jan11 = raw_data.loc["2011-01-01 00:00:00":"2011-01-28 23:00:00"].copy()
    features = NUMERICAL_FEATURES + CATEGORICAL_FEATURES

    regressor = ensemble.RandomForestRegressor(random_state=0, n_estimators=50)
    regressor.fit(reference_jan11[features], reference_jan11[TARGET])

    reference_jan11["prediction"] = regressor.predict(reference_jan11[features])

    print(f"      Modèle de production entraîné sur {len(reference_jan11):,} lignes.")
    return regressor, reference_jan11


# ---------------------------------------------------------------------------
# Step 4 – Weekly drift reports for February 2011
# rapports de suivi pour les semaines 1, 2 et 3 de février 
# ---------------------------------------------------------------------------


def step4_weekly_drift_reports(
    workspace: Workspace,
    project,
    raw_data: pd.DataFrame,
    regressor: ensemble.RandomForestRegressor,
    reference_jan11: pd.DataFrame,
) -> dict:
    """
    Générer des rapports de drift RegressionPreset pour les semaines 1, 2 et 3 de février 2011.
    """
    print("[5/7] Step 4 - Génération des rapports de drift hebdomadaires (Semaines 1-3)...")

    column_mapping = _build_column_mapping()
    features = NUMERICAL_FEATURES + CATEGORICAL_FEATURES
    week_rmse = {}

    for week_label, (start, end) in WEEK_PERIODS.items():
        current_week = raw_data.loc[start:end].copy()
        current_week["prediction"] = regressor.predict(current_week[features])

        weekly_report = Report(metrics=[RegressionPreset()])
        weekly_report.run(
            reference_data=reference_jan11,
            current_data=current_week,
            column_mapping=column_mapping,
        )
        workspace.add_report(project.id, weekly_report)

        # Extraire RMSE pour identifier la semaine la plus problématique
        report_dict = weekly_report.as_dict()
        rmse_value = float("inf")
        for metric in report_dict.get("metrics", []):
            result = metric.get("result", {})
            current_metrics = result.get("current", {})
            if "rmse" in current_metrics:
                rmse_value = current_metrics["rmse"]
                break

        week_rmse[week_label] = rmse_value
        print(f"      {week_label}: RMSE = {rmse_value:.2f}")

    return week_rmse


# ---------------------------------------------------------------------------
# Step 5 – Analyse du drift de la cible sur la semaine la plus problématique
# ---------------------------------------------------------------------------

def step5_target_drift_report(
    workspace: Workspace,
    project,
    raw_data: pd.DataFrame,
    regressor: ensemble.RandomForestRegressor,
    reference_jan11: pd.DataFrame,
    week_rmse: dict,
) -> str:
    """
    Exécuter un rapport TargetDriftPreset sur la semaine avec le plus haut RMSE.

    """
    worst_week = max(week_rmse, key=lambda k: week_rmse[k])
    start, end = WEEK_PERIODS[worst_week]
    print(
        f"[6/7] Step 5 - Rapport de drift de la cible pour la semaine la plus problématique: "
        f"{worst_week} ({start} -> {end})..."
    )

    features = NUMERICAL_FEATURES + CATEGORICAL_FEATURES
    current_worst = raw_data.loc[start:end].copy()
    current_worst["prediction"] = regressor.predict(current_worst[features])

    column_mapping = _build_column_mapping()

    target_drift_report = Report(metrics=[TargetDriftPreset()])
    target_drift_report.run(
        reference_data=reference_jan11,
        current_data=current_worst,
        column_mapping=column_mapping,
    )
    workspace.add_report(project.id, target_drift_report)
    print(f"      Rapport de drift de la cible sauvegardé pour {worst_week}.")
    return worst_week


# ---------------------------------------------------------------------------
# Step 6 – Analyse du drift de données sur la semaine 3 (features numériques uniquement)
# ---------------------------------------------------------------------------


def step6_data_drift_report(
    workspace: Workspace,
    project,
    raw_data: pd.DataFrame,
    regressor: ensemble.RandomForestRegressor,
    reference_jan11: pd.DataFrame,
) -> None:
    """
    Run a DataDriftPreset report on week 3 using only numerical features.
    Categorical features are excluded from the drift analysis per the spec.
    """
    print("[7/7] Step 6 - Rapport de drift de données pour la semaine 3 (features numériques uniquement)...")

    start, end = WEEK_PERIODS["week_3"]
    features = NUMERICAL_FEATURES + CATEGORICAL_FEATURES
    current_week3 = raw_data.loc[start:end].copy()
    current_week3["prediction"] = regressor.predict(current_week3[features])

    column_mapping_drift = ColumnMapping()
    column_mapping_drift.target = TARGET
    column_mapping_drift.prediction = PREDICTION
    column_mapping_drift.numerical_features = NUMERICAL_FEATURES
    column_mapping_drift.categorical_features = []  # numerical only

    data_drift_report = Report(metrics=[DataDriftPreset()])
    data_drift_report.run(
        reference_data=reference_jan11,
        current_data=current_week3,
        column_mapping=column_mapping_drift,
    )
    workspace.add_report(project.id, data_drift_report)
    print("      Rapport de drift de données sauvegardé pour la semaine 3 (features numériques uniquement).")


# ---------------------------------------------------------------------------
# Workspace & project helpers
# ---------------------------------------------------------------------------

def _get_or_create_project(workspace: Workspace, name: str, description: str):
    for project in workspace.list_projects():
        if project.name == name:
            print(f"      Réutilisation du projet existant: '{name}' (id={project.id})")
            return project

    project = workspace.create_project(name)
    project.description = description
    project.save()
    print(f"      Création du nouveau projet: '{name}' (id={project.id})")
    return project


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    print("=" * 65)
    print("  Bike Sharing - Evidently Drift Monitoring Pipeline")
    # print("  evidently==0.6.7")
    # print("=" * 65)

    # Initialisation du workspace et projet
    workspace = Workspace.create(WORKSPACE_PATH)
    project = _get_or_create_project(workspace, PROJECT_NAME, PROJECT_DESCRIPTION)

    # Chargement des données
    raw_data = load_data()

    # Step 1 - Entraînement du modèle de validation
    _regressor_val, X_train, X_test = step1_train_validation_model(raw_data)

    # Step 2 - Rapport de validation du modèle (entraînement vs test)
    step2_model_validation_report(workspace, project, X_train, X_test)

    # Step 3 - Construction du modèle de production sur les données de janvier complètes
    regressor_prod, reference_jan11 = step3_build_production_model(raw_data)

    # Step 4 - Rapports de drift hebdomadaires (semaines 1, 2, 3)
    week_rmse = step4_weekly_drift_reports(
        workspace, project, raw_data, regressor_prod, reference_jan11
    )

    # Step 5 - Drift de la cible la pire semaine (la semaine avec le RMSE le plus élevé)
    worst_week = step5_target_drift_report(
        workspace, project, raw_data, regressor_prod, reference_jan11, week_rmse
    )

    # Step 6 - Drift des données sur la semaine 3 (features numériques uniquement)
    step6_data_drift_report(
        workspace, project, raw_data, regressor_prod, reference_jan11
    )

    print("=" * 65)
    print("  Pipeline complete!  Tous les 7 rapports sauvegardés dans le workspace.")
    print(f"  Semaine la plus problématique identifiée: {worst_week}")
    print()
    print("  Lancez l'interface utilisateur Evidently avec:")
    print(f"    evidently ui --workspace {WORKSPACE_PATH}")
    print("  Puis ouvrez http://localhost:8000 dans votre navigateur.")
    print("=" * 65)


if __name__ == "__main__":
    main()
