"""
Gradient Boosting Win Probability
Returning Production Experiment - Model 1

Experiment question
-------------------
Does adding core returning production information to the established
Gradient Boosting Model 5 feature space improve out-of-sample
win-probability predictions?

Experiment Model 1
------------------
Baseline:
    Exact 310-feature space used by Gradient Boosting Model 5

Added features:
    16 core returning-production PPA/usage features
    8 features for home team
    8 features for away team

Total predictors:
    326

Excluded from this experiment:
    - Returning-production missingness indicators
    - Percentage-PPA features
    - Returning-production x gamesBefore interactions

Model:
    GradientBoostingClassifier

Hyperparameters:
    n_estimators     = 200
    learning_rate    = 0.03
    max_depth        = 4
    min_samples_leaf = 10
    subsample        = 0.75
    random_state     = 42

Preprocessing:
    Median imputation

Temporal split:
    Train      = 2015-2022
    Validation = 2023-2024
    Test       = 2025

Primary evaluation metric:
    Log Loss

Secondary evaluation metrics:
    Brier Score
    ROC AUC

Outputs:
    model.joblib
    feature_list.csv
    training_summary.csv
    validation_predictions.csv
    test_predictions.csv
"""

from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from sklearn.ensemble import GradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    brier_score_loss,
    log_loss,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline


# =============================================================================
# CONFIGURATION
# =============================================================================

RANDOM_STATE = 42

# Model 5 hyperparameters
N_ESTIMATORS = 200
LEARNING_RATE = 0.03
MAX_DEPTH = 4
MIN_SAMPLES_LEAF = 10
SUBSAMPLE = 0.75

# Feature counts
BASELINE_FEATURE_COUNT = 310
RETURNING_FEATURE_COUNT = 16
EXPECTED_FEATURE_COUNT = BASELINE_FEATURE_COUNT + RETURNING_FEATURE_COUNT

TARGET = "win_home"
GAME_ID = "gameId"
SEASON = "season"

TRAIN_SEASONS = list(range(2015, 2023))
VALIDATION_SEASONS = [2023, 2024]
TEST_SEASONS = [2025]

ALL_SEASONS = TRAIN_SEASONS + VALIDATION_SEASONS + TEST_SEASONS


# =============================================================================
# PATHS
# =============================================================================

SCRIPT_DIR = Path(__file__).resolve().parent

PROJECT_ROOT = SCRIPT_DIR

while PROJECT_ROOT.name != "College_Football_Prediction":
    if PROJECT_ROOT.parent == PROJECT_ROOT:
        raise RuntimeError(
            "Could not locate project root 'College_Football_Prediction'."
        )
    PROJECT_ROOT = PROJECT_ROOT.parent

# Existing Model 5 model directory
MODEL_5_DIR = (
    PROJECT_ROOT
    / "models"
    / "win_probability"
    / "gradient_boosting"
    / "model_5"
)

MODEL_5_FEATURE_LIST_PATH = MODEL_5_DIR / "feature_list.csv"

# Existing baseline model-input files
BASELINE_INPUT_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "model_inputs"
    / "win_probability"
)

# Audited returning-production feature files
RETURNING_FEATURE_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "features"
    / "win_probability"
    / "returning_production"
)

# Experiment output directory
OUTPUT_DIR = (
    PROJECT_ROOT
    / "models"
    / "win_probability"
    / "gradient_boosting"
    / "returning_production"
    / "model_1"
)


# =============================================================================
# RETURNING-PRODUCTION FEATURES
# =============================================================================
#
# Experiment Model 1 intentionally uses only the underlying PPA and usage
# measures. Percentage-PPA measures are excluded from the first experiment
# because their audited distributions contain extreme values caused by
# negative or near-zero denominators.
#
# Missingness indicators and gamesBefore interactions are also intentionally
# excluded so their incremental value can be evaluated separately later.
# =============================================================================

RETURNING_FEATURES = [
    "home_returning_total_ppa",
    "away_returning_total_ppa",

    "home_returning_passing_ppa",
    "away_returning_passing_ppa",

    "home_returning_receiving_ppa",
    "away_returning_receiving_ppa",

    "home_returning_rushing_ppa",
    "away_returning_rushing_ppa",

    "home_returning_usage",
    "away_returning_usage",

    "home_returning_passing_usage",
    "away_returning_passing_usage",

    "home_returning_receiving_usage",
    "away_returning_receiving_usage",

    "home_returning_rushing_usage",
    "away_returning_rushing_usage",
]


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def print_section(title):
    """Print a formatted section heading."""
    print()
    print("=" * 80)
    print(title)
    print("=" * 80)


def validate_required_file(path):
    """Raise a clear error if a required file does not exist."""
    if not path.exists():
        raise FileNotFoundError(
            f"\nRequired file not found:\n{path}\n"
        )


def load_model_5_feature_list():
    """
    Load the exact 310-feature list used by Gradient Boosting Model 5.

    This guarantees that Experiment Model 1 uses precisely the same
    baseline feature space rather than reconstructing it independently.
    """

    validate_required_file(MODEL_5_FEATURE_LIST_PATH)

    feature_df = pd.read_csv(MODEL_5_FEATURE_LIST_PATH)

    if "feature" not in feature_df.columns:
        raise ValueError(
            "Model 5 feature_list.csv must contain a 'feature' column."
        )

    features = feature_df["feature"].astype(str).tolist()

    if len(features) != BASELINE_FEATURE_COUNT:
        raise ValueError(
            f"Model 5 feature list contains {len(features)} features. "
            f"Expected {BASELINE_FEATURE_COUNT}."
        )

    if len(set(features)) != len(features):
        duplicates = (
            pd.Series(features)
            [pd.Series(features).duplicated()]
            .tolist()
        )

        raise ValueError(
            f"Duplicate features found in Model 5 feature list: {duplicates}"
        )

    forbidden = {TARGET, GAME_ID, SEASON}

    leakage = sorted(set(features) & forbidden)

    if leakage:
        raise ValueError(
            f"Model 5 feature list contains forbidden columns: {leakage}"
        )

    return features


def validate_returning_feature_definition():
    """Validate the Experiment Model 1 returning feature specification."""

    if len(RETURNING_FEATURES) != RETURNING_FEATURE_COUNT:
        raise ValueError(
            f"Returning feature definition contains "
            f"{len(RETURNING_FEATURES)} features. "
            f"Expected {RETURNING_FEATURE_COUNT}."
        )

    if len(set(RETURNING_FEATURES)) != len(RETURNING_FEATURES):
        raise ValueError(
            "Duplicate returning-production features detected."
        )

    forbidden = {TARGET, GAME_ID, SEASON}

    leakage = sorted(set(RETURNING_FEATURES) & forbidden)

    if leakage:
        raise ValueError(
            f"Returning feature list contains forbidden columns: {leakage}"
        )


def load_returning_season(season):
    """
    Load one audited returning-production feature file.
    """

    path = (
        RETURNING_FEATURE_DIR
        / f"returning_features_{season}.csv"
    )

    validate_required_file(path)

    df = pd.read_csv(path)

    if df.empty:
        raise ValueError(
            f"Returning feature file is empty for season {season}: {path}"
        )

    if SEASON not in df.columns:
        raise ValueError(
            f"Returning feature file for {season} does not contain "
            f"'{SEASON}'."
        )

    if GAME_ID not in df.columns:
        raise ValueError(
            f"Returning feature file for {season} does not contain "
            f"'{GAME_ID}'."
        )

    actual_seasons = df[SEASON].dropna().unique().tolist()

    if actual_seasons != [season]:
        raise ValueError(
            f"Season mismatch in {path}.\n"
            f"Expected season: {season}\n"
            f"Found seasons: {actual_seasons}"
        )

    if df[GAME_ID].isna().any():
        raise ValueError(
            f"Missing gameId values found in returning feature file "
            f"for {season}."
        )

    if df[GAME_ID].duplicated().any():
        duplicate_ids = (
            df.loc[df[GAME_ID].duplicated(), GAME_ID]
            .head(10)
            .tolist()
        )

        raise ValueError(
            f"Duplicate gameId values found in returning feature file "
            f"for {season}. Examples: {duplicate_ids}"
        )

    return df


def load_baseline_split(split_name):
    """
    Load an existing baseline model-input split.

    These files are used only to verify that the returning-production
    feature layer preserves the exact same games and ordering as the
    established Model 5 input data.
    """

    path = BASELINE_INPUT_DIR / f"{split_name}.csv"

    validate_required_file(path)

    df = pd.read_csv(path)

    required = {GAME_ID, SEASON, TARGET}

    missing = sorted(required - set(df.columns))

    if missing:
        raise ValueError(
            f"Baseline {split_name}.csv is missing required columns: "
            f"{missing}"
        )

    return df


def load_data(baseline_features):
    """
    Load all returning-production feature files and construct
    train/validation/test datasets.

    The exact Model 5 feature list is selected from the expanded
    returning-production data, then the 16 experiment features are added.
    """

    print_section("LOADING DATA")

    season_frames = []

    for season in ALL_SEASONS:
        print(f"Loading returning features for {season}...")

        df = load_returning_season(season)

        missing_features = sorted(
            set(baseline_features + RETURNING_FEATURES)
            - set(df.columns)
        )

        if missing_features:
            raise ValueError(
                f"Season {season} is missing required features:\n"
                f"{missing_features}"
            )

        season_frames.append(df)

        print(f"  Rows: {len(df):,}")
        print(f"  Columns: {len(df):,}")

    combined = pd.concat(
        season_frames,
        axis=0,
        ignore_index=True,
    )

    print()
    print(f"Combined rows: {len(combined):,}")
    print(f"Combined columns: {len(combined.columns):,}")

    # -------------------------------------------------------------------------
    # Verify total feature space
    # -------------------------------------------------------------------------

    experiment_features = baseline_features + RETURNING_FEATURES

    if len(experiment_features) != EXPECTED_FEATURE_COUNT:
        raise ValueError(
            f"Experiment feature list contains "
            f"{len(experiment_features)} features. "
            f"Expected {EXPECTED_FEATURE_COUNT}."
        )

    if len(set(experiment_features)) != len(experiment_features):
        raise ValueError(
            "Duplicate feature names exist in Experiment Model 1."
        )

    overlap = sorted(
        set(baseline_features) & set(RETURNING_FEATURES)
    )

    if overlap:
        raise ValueError(
            "Returning-production features overlap with the Model 5 "
            f"baseline feature space:\n{overlap}"
        )

    # -------------------------------------------------------------------------
    # Verify exact game alignment against existing Model 5 inputs
    # -------------------------------------------------------------------------

    print_section("VERIFYING GAME ALIGNMENT")

    baseline_splits = {
        "train": load_baseline_split("train"),
        "validation": load_baseline_split("validation"),
        "test": load_baseline_split("test"),
    }

    split_seasons = {
        "train": TRAIN_SEASONS,
        "validation": VALIDATION_SEASONS,
        "test": TEST_SEASONS,
    }

    split_frames = {}

    for split_name, seasons in split_seasons.items():

        experiment_df = combined[
            combined[SEASON].isin(seasons)
        ].copy()

        baseline_df = baseline_splits[split_name]

        experiment_ids = experiment_df[GAME_ID].tolist()
        baseline_ids = baseline_df[GAME_ID].tolist()

        if len(experiment_df) != len(baseline_df):
            raise ValueError(
                f"{split_name.capitalize()} row count mismatch.\n"
                f"Returning feature data: {len(experiment_df):,}\n"
                f"Baseline Model 5 data: {len(baseline_df):,}"
            )

        if experiment_ids != baseline_ids:
            raise ValueError(
                f"{split_name.capitalize()} gameId ordering does not "
                "exactly match the Model 5 baseline input."
            )

        if not (
            experiment_df[SEASON].to_numpy()
            == baseline_df[SEASON].to_numpy()
        ).all():
            raise ValueError(
                f"{split_name.capitalize()} season ordering does not "
                "match the Model 5 baseline input."
            )

        if not (
            experiment_df[TARGET].to_numpy()
            == baseline_df[TARGET].to_numpy()
        ).all():
            raise ValueError(
                f"{split_name.capitalize()} target values do not "
                "match the Model 5 baseline input."
            )

        split_frames[split_name] = experiment_df

        print(
            f"✓ {split_name.capitalize()} aligned: "
            f"{len(experiment_df):,} games"
        )

    return (
        split_frames["train"],
        split_frames["validation"],
        split_frames["test"],
        experiment_features,
    )


def validate_targets(df, split_name):
    """Validate target values."""

    if df[TARGET].isna().any():
        raise ValueError(
            f"{split_name}: target contains missing values."
        )

    unique_values = sorted(df[TARGET].unique().tolist())

    if not set(unique_values).issubset({0, 1}):
        raise ValueError(
            f"{split_name}: target must contain only 0/1. "
            f"Found: {unique_values}"
        )


def validate_split_seasons(df, expected_seasons, split_name):
    """Validate temporal split membership."""

    actual = sorted(df[SEASON].unique().tolist())

    if actual != sorted(expected_seasons):
        raise ValueError(
            f"{split_name}: incorrect seasons.\n"
            f"Expected: {expected_seasons}\n"
            f"Found: {actual}"
        )


def validate_game_ids(train, validation, test):
    """Verify game IDs are unique and disjoint across splits."""

    train_ids = set(train[GAME_ID])
    validation_ids = set(validation[GAME_ID])
    test_ids = set(test[GAME_ID])

    if len(train_ids) != len(train):
        raise ValueError("Duplicate game IDs found in training data.")

    if len(validation_ids) != len(validation):
        raise ValueError(
            "Duplicate game IDs found in validation data."
        )

    if len(test_ids) != len(test):
        raise ValueError("Duplicate game IDs found in test data.")

    if train_ids & validation_ids:
        raise ValueError(
            "Game IDs overlap between training and validation."
        )

    if train_ids & test_ids:
        raise ValueError(
            "Game IDs overlap between training and test."
        )

    if validation_ids & test_ids:
        raise ValueError(
            "Game IDs overlap between validation and test."
        )


def validate_features(train, validation, test, features):
    """
    Validate the complete experiment feature space.
    """

    print_section("VALIDATING FEATURE SPACE")

    if len(features) != EXPECTED_FEATURE_COUNT:
        raise ValueError(
            f"Expected {EXPECTED_FEATURE_COUNT} features, "
            f"found {len(features)}."
        )

    for split_name, df in [
        ("Training", train),
        ("Validation", validation),
        ("Test", test),
    ]:

        missing = sorted(set(features) - set(df.columns))

        if missing:
            raise ValueError(
                f"{split_name} data is missing features:\n{missing}"
            )

        nonnumeric = [
            feature
            for feature in features
            if not pd.api.types.is_numeric_dtype(df[feature])
        ]

        if nonnumeric:
            raise ValueError(
                f"{split_name} contains nonnumeric features:\n"
                f"{nonnumeric}"
            )

        print(
            f"✓ {split_name}: "
            f"{len(features)} numeric predictors available"
        )

    # Confirm no identifiers or target slipped into the predictor space.
    forbidden = {GAME_ID, SEASON, TARGET}

    leakage = sorted(set(features) & forbidden)

    if leakage:
        raise ValueError(
            f"Forbidden columns found in feature space: {leakage}"
        )

    print("✓ No identifier or target leakage detected")


def print_missingness(train, validation, test, features):
    """Print missing-value counts before model preprocessing."""

    print_section("MISSINGNESS CHECK")

    for split_name, df in [
        ("Training", train),
        ("Validation", validation),
        ("Test", test),
    ]:

        missing_count = int(
            df[features].isna().sum().sum()
        )

        missing_features = int(
            df[features].isna().any().sum()
        )

        print(
            f"{split_name}:"
            f" {missing_count:,} missing values"
            f" across {missing_features} features"
        )


def build_model():
    """
    Build the Experiment Model 1 pipeline.

    The imputer is inside the pipeline so medians are learned using
    training data only.
    """

    return Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(strategy="median"),
            ),
            (
                "classifier",
                GradientBoostingClassifier(
                    n_estimators=N_ESTIMATORS,
                    learning_rate=LEARNING_RATE,
                    max_depth=MAX_DEPTH,
                    min_samples_leaf=MIN_SAMPLES_LEAF,
                    subsample=SUBSAMPLE,
                    random_state=RANDOM_STATE,
                ),
            ),
        ]
    )


def calculate_metrics(y_true, probabilities):
    """Calculate classification and probability metrics."""

    predictions = (
        probabilities >= 0.5
    ).astype(int)

    return {
        "accuracy": accuracy_score(
            y_true,
            predictions,
        ),
        "balanced_accuracy": balanced_accuracy_score(
            y_true,
            predictions,
        ),
        "precision": precision_score(
            y_true,
            predictions,
            zero_division=0,
        ),
        "recall": recall_score(
            y_true,
            predictions,
            zero_division=0,
        ),
        "roc_auc": roc_auc_score(
            y_true,
            probabilities,
        ),
        "log_loss": log_loss(
            y_true,
            probabilities,
        ),
        "brier_score": brier_score_loss(
            y_true,
            probabilities,
        ),
    }


def create_predictions(df, probabilities):
    """Create standardized prediction output."""

    predictions = (
        probabilities >= 0.5
    ).astype(int)

    return pd.DataFrame(
        {
            GAME_ID: df[GAME_ID].values,
            SEASON: df[SEASON].values,
            TARGET: df[TARGET].values,
            "predicted_probability_home_win": probabilities,
            "predicted_home_win": predictions,
        }
    )


def save_feature_list(features):
    """Save the complete 326-feature experiment feature list."""

    output_path = OUTPUT_DIR / "feature_list.csv"

    feature_df = pd.DataFrame(
        {
            "feature": features
        }
    )

    feature_df.to_csv(
        output_path,
        index=False,
    )


def save_training_summary(
    train_metrics,
    validation_metrics,
    test_metrics,
    train,
    validation,
    test,
):
    """Save experiment metadata and evaluation metrics."""

    summary_rows = [
        {
            "experiment": "returning_production_model_1",
            "model_family": "gradient_boosting",
            "baseline_model": "model_5",
            "feature_definition": (
                "Model 5 310 features + core returning PPA/usage"
            ),
            "baseline_feature_count": BASELINE_FEATURE_COUNT,
            "returning_feature_count": RETURNING_FEATURE_COUNT,
            "total_feature_count": EXPECTED_FEATURE_COUNT,
            "excluded_missingness_indicators": True,
            "excluded_percentage_ppa": True,
            "excluded_games_before_interactions": True,
            "random_state": RANDOM_STATE,
            "n_estimators": N_ESTIMATORS,
            "learning_rate": LEARNING_RATE,
            "max_depth": MAX_DEPTH,
            "min_samples_leaf": MIN_SAMPLES_LEAF,
            "subsample": SUBSAMPLE,
            "train_start_season": min(TRAIN_SEASONS),
            "train_end_season": max(TRAIN_SEASONS),
            "validation_start_season": min(VALIDATION_SEASONS),
            "validation_end_season": max(VALIDATION_SEASONS),
            "test_season": TEST_SEASONS[0],
            "train_rows": len(train),
            "validation_rows": len(validation),
            "test_rows": len(test),
            "train_accuracy": train_metrics["accuracy"],
            "train_balanced_accuracy": train_metrics[
                "balanced_accuracy"
            ],
            "train_precision": train_metrics["precision"],
            "train_recall": train_metrics["recall"],
            "train_roc_auc": train_metrics["roc_auc"],
            "train_log_loss": train_metrics["log_loss"],
            "train_brier_score": train_metrics["brier_score"],
            "validation_accuracy": validation_metrics["accuracy"],
            "validation_balanced_accuracy": validation_metrics[
                "balanced_accuracy"
            ],
            "validation_precision": validation_metrics["precision"],
            "validation_recall": validation_metrics["recall"],
            "validation_roc_auc": validation_metrics["roc_auc"],
            "validation_log_loss": validation_metrics["log_loss"],
            "validation_brier_score": validation_metrics[
                "brier_score"
            ],
            "test_accuracy": test_metrics["accuracy"],
            "test_balanced_accuracy": test_metrics[
                "balanced_accuracy"
            ],
            "test_precision": test_metrics["precision"],
            "test_recall": test_metrics["recall"],
            "test_roc_auc": test_metrics["roc_auc"],
            "test_log_loss": test_metrics["log_loss"],
            "test_brier_score": test_metrics["brier_score"],
        }
    ]

    summary_df = pd.DataFrame(summary_rows)

    summary_df.to_csv(
        OUTPUT_DIR / "training_summary.csv",
        index=False,
    )


def print_metrics(name, metrics):
    """Print evaluation metrics."""

    print()
    print(name)

    print(f"  Accuracy:          {metrics['accuracy']:.6f}")
    print(
        f"  Balanced Accuracy: "
        f"{metrics['balanced_accuracy']:.6f}"
    )
    print(f"  Precision:         {metrics['precision']:.6f}")
    print(f"  Recall:            {metrics['recall']:.6f}")
    print(f"  ROC AUC:           {metrics['roc_auc']:.6f}")
    print(f"  Log Loss:          {metrics['log_loss']:.6f}")
    print(
        f"  Brier Score:       "
        f"{metrics['brier_score']:.6f}"
    )


# =============================================================================
# MAIN
# =============================================================================

def main():

    print()
    print("=" * 80)
    print(
        "GRADIENT BOOSTING WIN PROBABILITY - "
        "RETURNING PRODUCTION EXPERIMENT MODEL 1"
    )
    print("=" * 80)

    print()
    print("Experiment Model 1 definition:")
    print()
    print("    Baseline feature space:")
    print("        Exact Gradient Boosting Model 5 feature space")
    print(f"        {BASELINE_FEATURE_COUNT} features")
    print()
    print("    Added returning-production features:")
    print("        Core PPA + usage")
    print(f"        {RETURNING_FEATURE_COUNT} features")
    print()
    print("    Total predictors:")
    print(f"        {EXPECTED_FEATURE_COUNT}")
    print()
    print("    Excluded:")
    print("        Missingness indicators")
    print("        Percentage-PPA features")
    print("        Returning x gamesBefore interactions")
    print()
    print("    Hyperparameter tuning:")
    print("        NONE")
    print()
    print("    Hyperparameters:")
    print(f"        n_estimators     = {N_ESTIMATORS}")
    print(f"        learning_rate    = {LEARNING_RATE}")
    print(f"        max_depth        = {MAX_DEPTH}")
    print(f"        min_samples_leaf = {MIN_SAMPLES_LEAF}")
    print(f"        subsample        = {SUBSAMPLE}")
    print()
    print("    Training:")
    print("        2015-2022")
    print()
    print("    Validation:")
    print("        2023-2024")
    print()
    print("    Test:")
    print("        2025")

    # -------------------------------------------------------------------------
    # Setup
    # -------------------------------------------------------------------------

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    validate_returning_feature_definition()

    # -------------------------------------------------------------------------
    # Load exact Model 5 feature list
    # -------------------------------------------------------------------------

    print_section("LOADING MODEL 5 BASELINE FEATURE LIST")

    baseline_features = load_model_5_feature_list()

    print(
        f"✓ Loaded {len(baseline_features)} "
        "baseline Model 5 features"
    )

    # -------------------------------------------------------------------------
    # Load data
    # -------------------------------------------------------------------------

    (
        train,
        validation,
        test,
        experiment_features,
    ) = load_data(baseline_features)

    # -------------------------------------------------------------------------
    # Validate targets
    # -------------------------------------------------------------------------

    print_section("VALIDATING TARGETS")

    validate_targets(
        train,
        "Training",
    )

    validate_targets(
        validation,
        "Validation",
    )

    validate_targets(
        test,
        "Test",
    )

    print("✓ Training target validated")
    print("✓ Validation target validated")
    print("✓ Test target validated")

    # -------------------------------------------------------------------------
    # Validate temporal splits
    # -------------------------------------------------------------------------

    print_section("VALIDATING TEMPORAL SPLITS")

    validate_split_seasons(
        train,
        TRAIN_SEASONS,
        "Training",
    )

    validate_split_seasons(
        validation,
        VALIDATION_SEASONS,
        "Validation",
    )

    validate_split_seasons(
        test,
        TEST_SEASONS,
        "Test",
    )

    print(
        f"✓ Training seasons: "
        f"{min(TRAIN_SEASONS)}-{max(TRAIN_SEASONS)}"
    )

    print(
        f"✓ Validation seasons: "
        f"{min(VALIDATION_SEASONS)}-{max(VALIDATION_SEASONS)}"
    )

    print(
        f"✓ Test season: "
        f"{TEST_SEASONS[0]}"
    )

    # -------------------------------------------------------------------------
    # Validate game IDs
    # -------------------------------------------------------------------------

    print_section("VALIDATING GAME IDS")

    validate_game_ids(
        train,
        validation,
        test,
    )

    print("✓ Game IDs unique within all splits")
    print("✓ Game IDs disjoint across all splits")

    # -------------------------------------------------------------------------
    # Validate feature space
    # -------------------------------------------------------------------------

    validate_features(
        train,
        validation,
        test,
        experiment_features,
    )

    # -------------------------------------------------------------------------
    # Missingness
    # -------------------------------------------------------------------------

    print_missingness(
        train,
        validation,
        test,
        experiment_features,
    )

    # -------------------------------------------------------------------------
    # Build X/y
    # -------------------------------------------------------------------------

    X_train = train[experiment_features].copy()
    y_train = train[TARGET].copy()

    X_validation = validation[experiment_features].copy()
    y_validation = validation[TARGET].copy()

    X_test = test[experiment_features].copy()
    y_test = test[TARGET].copy()

    # -------------------------------------------------------------------------
    # Build model
    # -------------------------------------------------------------------------

    print_section("BUILDING MODEL")

    model = build_model()

    print("GradientBoostingClassifier configuration:")
    print(f"  n_estimators     = {N_ESTIMATORS}")
    print(f"  learning_rate    = {LEARNING_RATE}")
    print(f"  max_depth        = {MAX_DEPTH}")
    print(f"  min_samples_leaf = {MIN_SAMPLES_LEAF}")
    print(f"  subsample        = {SUBSAMPLE}")
    print(f"  random_state     = {RANDOM_STATE}")
    print()
    print("Preprocessing:")
    print("  SimpleImputer(strategy='median')")
    print("  Imputer fitted on training data only")

    # -------------------------------------------------------------------------
    # Train
    # -------------------------------------------------------------------------

    print_section("TRAINING")

    print(
        f"Training on {len(X_train):,} games "
        f"with {X_train.shape[1]} predictors..."
    )

    model.fit(
        X_train,
        y_train,
    )

    print("✓ Model training complete")

    # -------------------------------------------------------------------------
    # Predictions
    # -------------------------------------------------------------------------

    print_section("GENERATING PREDICTIONS")

    train_probabilities = model.predict_proba(
        X_train
    )[:, 1]

    validation_probabilities = model.predict_proba(
        X_validation
    )[:, 1]

    test_probabilities = model.predict_proba(
        X_test
    )[:, 1]

    # Guard against invalid probability values.
    for name, probabilities in [
        ("Training", train_probabilities),
        ("Validation", validation_probabilities),
        ("Test", test_probabilities),
    ]:

        if not np.isfinite(probabilities).all():
            raise ValueError(
                f"{name} predictions contain non-finite values."
            )

        if (
            (probabilities < 0).any()
            or (probabilities > 1).any()
        ):
            raise ValueError(
                f"{name} predictions contain values outside [0, 1]."
            )

    print("✓ Training probabilities generated")
    print("✓ Validation probabilities generated")
    print("✓ Test probabilities generated")

    # -------------------------------------------------------------------------
    # Metrics
    # -------------------------------------------------------------------------

    print_section("EVALUATION")

    train_metrics = calculate_metrics(
        y_train,
        train_probabilities,
    )

    validation_metrics = calculate_metrics(
        y_validation,
        validation_probabilities,
    )

    test_metrics = calculate_metrics(
        y_test,
        test_probabilities,
    )

    print_metrics(
        "TRAINING",
        train_metrics,
    )

    print_metrics(
        "VALIDATION",
        validation_metrics,
    )

    print_metrics(
        "TEST",
        test_metrics,
    )

    # -------------------------------------------------------------------------
    # Save artifacts
    # -------------------------------------------------------------------------

    print_section("SAVING ARTIFACTS")

    model_path = OUTPUT_DIR / "model.joblib"

    joblib.dump(
        model,
        model_path,
    )

    print(f"✓ Saved: {model_path}")

    save_feature_list(
        experiment_features
    )

    print(
        f"✓ Saved: "
        f"{OUTPUT_DIR / 'feature_list.csv'}"
    )

    validation_predictions = create_predictions(
        validation,
        validation_probabilities,
    )

    validation_predictions.to_csv(
        OUTPUT_DIR / "validation_predictions.csv",
        index=False,
    )

    print(
        f"✓ Saved: "
        f"{OUTPUT_DIR / 'validation_predictions.csv'}"
    )

    test_predictions = create_predictions(
        test,
        test_probabilities,
    )

    test_predictions.to_csv(
        OUTPUT_DIR / "test_predictions.csv",
        index=False,
    )

    print(
        f"✓ Saved: "
        f"{OUTPUT_DIR / 'test_predictions.csv'}"
    )

    save_training_summary(
        train_metrics=train_metrics,
        validation_metrics=validation_metrics,
        test_metrics=test_metrics,
        train=train,
        validation=validation,
        test=test,
    )

    print(
        f"✓ Saved: "
        f"{OUTPUT_DIR / 'training_summary.csv'}"
    )

    # -------------------------------------------------------------------------
    # Final summary
    # -------------------------------------------------------------------------

    print_section("EXPERIMENT MODEL 1 COMPLETE")

    print()
    print("Feature space:")
    print(f"  Model 5 baseline:       {BASELINE_FEATURE_COUNT}")
    print(f"  Returning production:   {RETURNING_FEATURE_COUNT}")
    print(f"  Total predictors:       {EXPECTED_FEATURE_COUNT}")

    print()
    print("Primary results:")
    print(
        f"  Validation Log Loss: "
        f"{validation_metrics['log_loss']:.6f}"
    )

    print(
        f"  Test Log Loss:       "
        f"{test_metrics['log_loss']:.6f}"
    )

    print()
    print("Secondary probability metrics:")
    print(
        f"  Validation Brier:    "
        f"{validation_metrics['brier_score']:.6f}"
    )

    print(
        f"  Test Brier:          "
        f"{test_metrics['brier_score']:.6f}"
    )

    print()
    print("ROC AUC:")
    print(
        f"  Validation:          "
        f"{validation_metrics['roc_auc']:.6f}"
    )

    print(
        f"  Test:                "
        f"{test_metrics['roc_auc']:.6f}"
    )

    print()
    print(f"Output directory:")
    print(f"  {OUTPUT_DIR}")

    print()
    print("=" * 80)


if __name__ == "__main__":
    main()