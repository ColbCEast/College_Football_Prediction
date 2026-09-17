"""
Gradient Boosting Win Probability - Returning Production Model 3

Experiment 3: Phase-Aware Relative Returning Production

Definition:
    - Exact 310-feature Model 5 baseline
    - 8 relative returning-production features
    - Relative features are weighted by season phase:
        0-2 games played  -> 1.00
        3-5 games played  -> 0.67
        6+ games played   -> 0.33

    - gamesBefore_min = min(gamesBefore_home, gamesBefore_away)

    - Exact Model 5 hyperparameters
    - Temporal train/validation/test split
    - Training-only median imputation

No hyperparameter tuning is performed.

Temporal split:
    Train      = 2015-2022
    Validation = 2023-2024
    Test       = 2025
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
# PATHS
# =============================================================================

# train.py:
# src/modeling/problems/win_probability/gradient_boosting/
# returning_production/model_3/train.py
#
# parents[7] = project root
ROOT = Path(__file__).resolve().parents[7]

MODEL_5_DIR = (
    ROOT
    / "models"
    / "win_probability"
    / "gradient_boosting"
    / "model_5"
)

OUTPUT_DIR = (
    ROOT
    / "models"
    / "win_probability"
    / "gradient_boosting"
    / "returning_production"
    / "model_3"
)

MODEL_INPUT_DIR = (
    ROOT
    / "data"
    / "processed"
    / "model_inputs"
    / "win_probability"
)

RETURNING_FEATURE_DIR = (
    ROOT
    / "data"
    / "processed"
    / "features"
    / "win_probability"
    / "returning_production"
)


# =============================================================================
# CONFIGURATION
# =============================================================================

TRAIN_SEASONS = list(range(2015, 2023))
VALIDATION_SEASONS = [2023, 2024]
TEST_SEASONS = [2025]

TARGET = "win_home"
GAME_ID = "gameId"
SEASON = "season"

# Exact Model 5 hyperparameters
MODEL_PARAMS = {
    "n_estimators": 200,
    "learning_rate": 0.03,
    "max_depth": 4,
    "min_samples_leaf": 10,
    "subsample": 0.75,
    "random_state": 42,
}

# Fixed a priori phase weights.
PHASE_WEIGHTS = {
    "early": 1.00,
    "mid": 0.67,
    "late": 0.33,
}


# =============================================================================
# FEATURE DEFINITIONS
# =============================================================================

RELATIVE_FEATURES = [
    "returning_total_ppa_diff",
    "returning_passing_ppa_diff",
    "returning_receiving_ppa_diff",
    "returning_rushing_ppa_diff",
    "returning_usage_diff",
    "returning_passing_usage_diff",
    "returning_receiving_usage_diff",
    "returning_rushing_usage_diff",
]

PHASE_FEATURES = [
    "returning_total_ppa_diff_phase",
    "returning_passing_ppa_diff_phase",
    "returning_receiving_ppa_diff_phase",
    "returning_rushing_ppa_diff_phase",
    "returning_usage_diff_phase",
    "returning_passing_usage_diff_phase",
    "returning_receiving_usage_diff_phase",
    "returning_rushing_usage_diff_phase",
]

EXPECTED_PHASE_COLUMNS = [
    "gamesBefore_home",
    "gamesBefore_away",
]

RETURNING_BASE_COLUMNS = [
    "returning_total_ppa",
    "returning_passing_ppa",
    "returning_receiving_ppa",
    "returning_rushing_ppa",
    "returning_usage",
    "returning_passing_usage",
    "returning_receiving_usage",
    "returning_rushing_usage",
]


# =============================================================================
# HELPERS
# =============================================================================

def load_csv(path):
    """Load a CSV and fail clearly if it does not exist."""
    if not path.exists():
        raise FileNotFoundError(f"Required file not found:\n{path}")

    return pd.read_csv(path)


def load_model_5_features():
    """Load the exact 310-feature Model 5 feature list."""

    feature_path = MODEL_5_DIR / "feature_list.csv"

    if not feature_path.exists():
        raise FileNotFoundError(
            f"Model 5 feature list not found:\n{feature_path}"
        )

    feature_df = pd.read_csv(feature_path)

    if "feature" in feature_df.columns:
        features = feature_df["feature"].tolist()

    elif "feature_name" in feature_df.columns:
        features = feature_df["feature_name"].tolist()

    elif feature_df.shape[1] == 1:
        features = feature_df.iloc[:, 0].tolist()

    else:
        raise ValueError(
            "Could not identify the feature-name column in "
            "Model 5 feature_list.csv."
        )

    features = [str(feature) for feature in features]

    if len(features) != 310:
        raise ValueError(
            f"Expected exactly 310 Model 5 features, "
            f"found {len(features)}."
        )

    if len(set(features)) != len(features):
        raise ValueError(
            "Model 5 feature list contains duplicate feature names."
        )

    return features


def get_split_path(split_name):
    """Return the model-input path for a split."""

    valid_splits = {
        "train": MODEL_INPUT_DIR / "train.csv",
        "validation": MODEL_INPUT_DIR / "validation.csv",
        "test": MODEL_INPUT_DIR / "test.csv",
    }

    if split_name not in valid_splits:
        raise ValueError(f"Unknown split: {split_name}")

    return valid_splits[split_name]


def load_split(split_name):
    """
    Load each model-input split exactly once.

    This is important because train.csv already contains all
    2015-2022 training seasons, and validation.csv already contains
    2023-2024. They must NOT be loaded once per season.
    """

    path = get_split_path(split_name)
    df = load_csv(path)

    if GAME_ID not in df.columns:
        raise ValueError(
            f"{GAME_ID} is missing from {split_name}.csv."
        )

    if TARGET not in df.columns:
        raise ValueError(
            f"{TARGET} is missing from {split_name}.csv."
        )

    if SEASON not in df.columns:
        raise ValueError(
            f"{SEASON} is missing from {split_name}.csv."
        )

    if df[GAME_ID].duplicated().any():
        duplicate_count = int(df[GAME_ID].duplicated().sum())

        raise ValueError(
            f"{split_name}.csv contains {duplicate_count} duplicate game IDs."
        )

    return df


def load_returning_features(season):
    """Load the game-level returning-production features for one season."""

    path = RETURNING_FEATURE_DIR / f"returning_features_{season}.csv"

    df = load_csv(path)

    if GAME_ID not in df.columns:
        raise ValueError(
            f"{GAME_ID} is missing from returning feature file:\n{path}"
        )

    if df[GAME_ID].duplicated().any():
        duplicate_count = int(df[GAME_ID].duplicated().sum())

        raise ValueError(
            f"Season {season}: returning feature file contains "
            f"{duplicate_count} duplicate game IDs."
        )

    return df


def assign_phase(games_before_min):
    """
    Assign season phase.

    Phase definitions:
        0-2 -> early
        3-5 -> mid
        6+  -> late
    """

    if pd.isna(games_before_min):
        raise ValueError(
            "gamesBefore_min contains missing values."
        )

    if games_before_min <= 2:
        return "early"

    if games_before_min <= 5:
        return "mid"

    return "late"


def add_relative_features(df):
    """
    Construct the eight relative returning-production differences.

    Relative feature:
        home value - away value
    """

    df = df.copy()

    required_columns = []

    for feature in RETURNING_BASE_COLUMNS:
        required_columns.extend(
            [
                f"home_{feature}",
                f"away_{feature}",
            ]
        )

    missing = [
        column
        for column in required_columns
        if column not in df.columns
    ]

    if missing:
        raise ValueError(
            "Missing returning-production columns required to construct "
            f"relative features:\n{missing}"
        )

    for feature in RETURNING_BASE_COLUMNS:
        diff_name = f"{feature}_diff"

        df[diff_name] = (
            df[f"home_{feature}"]
            - df[f"away_{feature}"]
        )

    return df


def add_phase_features(df):
    """
    Construct phase-aware relative returning-production features.

    gamesBefore_min:
        min(gamesBefore_home, gamesBefore_away)

    Phase weights:
        0-2 -> 1.00
        3-5 -> 0.67
        6+  -> 0.33
    """

    df = df.copy()

    missing = [
        column
        for column in EXPECTED_PHASE_COLUMNS
        if column not in df.columns
    ]

    if missing:
        raise ValueError(
            "Missing gamesBefore columns required for phase assignment:\n"
            f"{missing}"
        )

    df["gamesBefore_min"] = (
        df[EXPECTED_PHASE_COLUMNS]
        .min(axis=1)
    )

    if df["gamesBefore_min"].isna().any():
        missing_count = int(
            df["gamesBefore_min"].isna().sum()
        )

        raise ValueError(
            f"gamesBefore_min contains {missing_count} missing values."
        )

    df["season_phase"] = (
        df["gamesBefore_min"]
        .apply(assign_phase)
    )

    df["phase_weight"] = (
        df["season_phase"]
        .map(PHASE_WEIGHTS)
    )

    if df["phase_weight"].isna().any():
        raise ValueError(
            "One or more rows failed to receive a valid phase weight."
        )

    for relative_feature, phase_feature in zip(
        RELATIVE_FEATURES,
        PHASE_FEATURES,
    ):
        df[phase_feature] = (
            df[relative_feature]
            * df["phase_weight"]
        )

    return df


def merge_returning_features(model_df):
    """
    Merge returning-production features onto a model-input split.

    The model-input split is loaded once. Returning data is loaded
    separately by season and merged only onto the corresponding games.
    """

    model_df = model_df.copy()

    season_values = sorted(
        model_df[SEASON]
        .dropna()
        .unique()
        .tolist()
    )

    merged_frames = []

    for season in season_values:

        season = int(season)

        season_games = model_df[
            model_df[SEASON] == season
        ].copy()

        returning_df = load_returning_features(season)

        returning_columns = [
            GAME_ID,
            "gamesBefore_home",
            "gamesBefore_away",
        ]

        returning_feature_columns = [
            column
            for column in returning_df.columns
            if column.startswith("home_returning_")
            or column.startswith("away_returning_")
        ]

        returning_columns.extend(
            returning_feature_columns
        )

        returning_columns = list(
            dict.fromkeys(returning_columns)
        )

        missing_returning = [
            column
            for column in returning_columns
            if column not in returning_df.columns
        ]

        if missing_returning:
            raise ValueError(
                f"Season {season}: returning feature file is missing:\n"
                f"{missing_returning}"
            )

        returning_subset = returning_df[
            returning_columns
        ].copy()

        merged = season_games.merge(
            returning_subset,
            on=GAME_ID,
            how="left",
            validate="one_to_one",
            suffixes=("", "_returning"),
        )

        if len(merged) != len(season_games):
            raise ValueError(
                f"Season {season}: row count changed during merge."
            )

        if merged[GAME_ID].isna().any():
            raise ValueError(
                f"Season {season}: missing game IDs after merge."
            )

        merged_frames.append(merged)

    result = pd.concat(
        merged_frames,
        ignore_index=True,
    )

    if len(result) != len(model_df):
        raise ValueError(
            "Total row count changed after merging returning features."
        )

    return result


def prepare_split(model_df):
    """
    Prepare one train/validation/test split.
    """

    df = merge_returning_features(model_df)

    df = add_relative_features(df)

    df = add_phase_features(df)

    return df


def evaluate_predictions(y_true, probabilities):
    """Calculate evaluation metrics."""

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


def save_feature_list(features):
    """Save the final ordered feature list."""

    feature_df = pd.DataFrame(
        {
            "feature": features,
            "feature_index": range(len(features)),
        }
    )

    feature_df.to_csv(
        OUTPUT_DIR / "feature_list.csv",
        index=False,
    )


# =============================================================================
# MAIN
# =============================================================================

def main():

    print("=" * 80)
    print(
        "GRADIENT BOOSTING WIN PROBABILITY - "
        "RETURNING PRODUCTION MODEL 3"
    )
    print("=" * 80)
    print()
    print("Experiment 3: Phase-Aware Relative Returning Production")
    print()
    print("Definition:")
    print("  Baseline: Exact Model 5 310-feature space")
    print("  Added:    8 phase-aware relative returning features")
    print()
    print("Season phases:")
    print("  0-2 games  -> weight 1.00")
    print("  3-5 games  -> weight 0.67")
    print("  6+ games   -> weight 0.33")
    print()
    print("Hyperparameters:")

    for key, value in MODEL_PARAMS.items():
        print(f"  {key}: {value}")

    print()

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    # =========================================================================
    # LOAD EXACT MODEL 5 FEATURE LIST
    # =========================================================================

    baseline_features = load_model_5_features()

    print(
        f"Loaded Model 5 feature list: "
        f"{len(baseline_features)} features"
    )

    # =========================================================================
    # LOAD EACH SPLIT EXACTLY ONCE
    # =========================================================================

    print()
    print("Loading model-input splits...")

    train_raw = load_split("train")
    validation_raw = load_split("validation")
    test_raw = load_split("test")

    print(
        f"  Train input rows:      {len(train_raw):,}"
    )
    print(
        f"  Validation input rows: {len(validation_raw):,}"
    )
    print(
        f"  Test input rows:       {len(test_raw):,}"
    )

    # =========================================================================
    # VALIDATE EXPECTED TEMPORAL SPLITS
    # =========================================================================

    actual_train_seasons = sorted(
        train_raw[SEASON].unique().tolist()
    )

    actual_validation_seasons = sorted(
        validation_raw[SEASON].unique().tolist()
    )

    actual_test_seasons = sorted(
        test_raw[SEASON].unique().tolist()
    )

    if actual_train_seasons != TRAIN_SEASONS:
        raise ValueError(
            "Unexpected training seasons.\n"
            f"Expected: {TRAIN_SEASONS}\n"
            f"Found:    {actual_train_seasons}"
        )

    if actual_validation_seasons != VALIDATION_SEASONS:
        raise ValueError(
            "Unexpected validation seasons.\n"
            f"Expected: {VALIDATION_SEASONS}\n"
            f"Found:    {actual_validation_seasons}"
        )

    if actual_test_seasons != TEST_SEASONS:
        raise ValueError(
            "Unexpected test seasons.\n"
            f"Expected: {TEST_SEASONS}\n"
            f"Found:    {actual_test_seasons}"
        )

    # =========================================================================
    # PREPARE SPLITS
    # =========================================================================

    print()
    print("Preparing training data...")

    train_df = prepare_split(
        train_raw
    )

    print("Preparing validation data...")

    validation_df = prepare_split(
        validation_raw
    )

    print("Preparing test data...")

    test_df = prepare_split(
        test_raw
    )

    # =========================================================================
    # VERIFY ROW COUNTS
    # =========================================================================

    expected_counts = {
        "train": len(train_raw),
        "validation": len(validation_raw),
        "test": len(test_raw),
    }

    actual_counts = {
        "train": len(train_df),
        "validation": len(validation_df),
        "test": len(test_df),
    }

    for split_name in expected_counts:

        if actual_counts[split_name] != expected_counts[split_name]:
            raise ValueError(
                f"{split_name}: row count changed during preparation. "
                f"Expected {expected_counts[split_name]:,}, "
                f"found {actual_counts[split_name]:,}."
            )

    # =========================================================================
    # FINAL FEATURE DEFINITION
    # =========================================================================

    final_features = (
        baseline_features
        + PHASE_FEATURES
    )

    if len(final_features) != 318:
        raise ValueError(
            f"Expected 318 total predictors, "
            f"found {len(final_features)}."
        )

    if len(set(final_features)) != len(final_features):
        raise ValueError(
            "Final feature list contains duplicate feature names."
        )

    missing_features = [
        feature
        for feature in final_features
        if feature not in train_df.columns
    ]

    if missing_features:
        raise ValueError(
            "Final feature list contains features missing from "
            "training data:\n"
            f"{missing_features}"
        )

    # =========================================================================
    # EXTRACT X / y
    # =========================================================================

    X_train = train_df[
        final_features
    ].copy()

    y_train = train_df[
        TARGET
    ].copy()

    X_validation = validation_df[
        final_features
    ].copy()

    y_validation = validation_df[
        TARGET
    ].copy()

    X_test = test_df[
        final_features
    ].copy()

    y_test = test_df[
        TARGET
    ].copy()

    # =========================================================================
    # TARGET VALIDATION
    # =========================================================================

    for name, y in [
        ("train", y_train),
        ("validation", y_validation),
        ("test", y_test),
    ]:

        unique_values = sorted(
            y.dropna().unique().tolist()
        )

        if unique_values != [0, 1]:
            raise ValueError(
                f"{name} target contains unexpected values: "
                f"{unique_values}"
            )

    # =========================================================================
    # PHASE DISTRIBUTIONS
    # =========================================================================

    print()
    print("Season-phase distributions:")
    print()

    for name, df in [
        ("Train", train_df),
        ("Validation", validation_df),
        ("Test", test_df),
    ]:

        phase_counts = (
            df["season_phase"]
            .value_counts()
            .reindex(
                ["early", "mid", "late"],
                fill_value=0,
            )
        )

        print(f"{name}:")
        print(
            f"  Early (0-2): "
            f"{phase_counts['early']:,}"
        )
        print(
            f"  Mid   (3-5): "
            f"{phase_counts['mid']:,}"
        )
        print(
            f"  Late    (6+): "
            f"{phase_counts['late']:,}"
        )

    # =========================================================================
    # MISSINGNESS
    # =========================================================================

    print()
    print("Missingness before imputation:")
    print(
        f"  Train:      "
        f"{int(X_train.isna().sum().sum()):,}"
    )
    print(
        f"  Validation: "
        f"{int(X_validation.isna().sum().sum()):,}"
    )
    print(
        f"  Test:       "
        f"{int(X_test.isna().sum().sum()):,}"
    )

    # =========================================================================
    # MODEL PIPELINE
    # =========================================================================

    model = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(
                    strategy="median"
                ),
            ),
            (
                "model",
                GradientBoostingClassifier(
                    **MODEL_PARAMS
                ),
            ),
        ]
    )

    # =========================================================================
    # TRAIN
    # =========================================================================

    print()
    print("Training Model 3...")
    print(
        f"  Training rows:   {len(X_train):,}"
    )
    print(
        f"  Validation rows: {len(X_validation):,}"
    )
    print(
        f"  Test rows:       {len(X_test):,}"
    )
    print(
        f"  Predictors:      {len(final_features):,}"
    )

    model.fit(
        X_train,
        y_train,
    )

    print("Training complete.")

    # =========================================================================
    # PREDICTIONS
    # =========================================================================

    train_probabilities = (
        model.predict_proba(X_train)[:, 1]
    )

    validation_probabilities = (
        model.predict_proba(X_validation)[:, 1]
    )

    test_probabilities = (
        model.predict_proba(X_test)[:, 1]
    )

    # =========================================================================
    # EVALUATION
    # =========================================================================

    train_metrics = evaluate_predictions(
        y_train,
        train_probabilities,
    )

    validation_metrics = evaluate_predictions(
        y_validation,
        validation_probabilities,
    )

    test_metrics = evaluate_predictions(
        y_test,
        test_probabilities,
    )

    # =========================================================================
    # PRINT RESULTS
    # =========================================================================

    print()
    print("=" * 80)
    print("MODEL 3 RESULTS")
    print("=" * 80)

    for name, metrics in [
        ("TRAIN", train_metrics),
        ("VALIDATION", validation_metrics),
        ("TEST", test_metrics),
    ]:

        print()
        print(name)
        print("-" * 40)

        for metric, value in metrics.items():
            print(
                f"{metric:20s}: {value:.6f}"
            )

    # =========================================================================
    # SAVE PREDICTIONS
    # =========================================================================

    def create_prediction_df(
        df,
        y,
        probabilities,
    ):

        return pd.DataFrame(
            {
                GAME_ID: df[GAME_ID].values,
                SEASON: df[SEASON].values,
                TARGET: y.values,
                "predicted_probability": probabilities,
                "predicted_class": (
                    probabilities >= 0.5
                ).astype(int),
                "season_phase": df[
                    "season_phase"
                ].values,
                "gamesBefore_min": df[
                    "gamesBefore_min"
                ].values,
                "phase_weight": df[
                    "phase_weight"
                ].values,
            }
        )

    train_predictions = create_prediction_df(
        train_df,
        y_train,
        train_probabilities,
    )

    validation_predictions = create_prediction_df(
        validation_df,
        y_validation,
        validation_probabilities,
    )

    test_predictions = create_prediction_df(
        test_df,
        y_test,
        test_probabilities,
    )

    train_predictions.to_csv(
        OUTPUT_DIR / "train_predictions.csv",
        index=False,
    )

    validation_predictions.to_csv(
        OUTPUT_DIR / "validation_predictions.csv",
        index=False,
    )

    test_predictions.to_csv(
        OUTPUT_DIR / "test_predictions.csv",
        index=False,
    )

    # =========================================================================
    # FEATURE IMPORTANCE
    # =========================================================================

    gb_model = model.named_steps["model"]

    feature_importance = (
        pd.DataFrame(
            {
                "feature": final_features,
                "importance": (
                    gb_model.feature_importances_
                ),
            }
        )
        .sort_values(
            "importance",
            ascending=False,
        )
    )

    feature_importance.to_csv(
        OUTPUT_DIR / "feature_importance.csv",
        index=False,
    )

    # =========================================================================
    # TRAINING SUMMARY
    # =========================================================================

    summary_rows = []

    for split_name, seasons, rows, metrics in [
        (
            "train",
            "2015-2022",
            len(train_df),
            train_metrics,
        ),
        (
            "validation",
            "2023-2024",
            len(validation_df),
            validation_metrics,
        ),
        (
            "test",
            "2025",
            len(test_df),
            test_metrics,
        ),
    ]:

        summary_rows.append(
            {
                "split": split_name,
                "seasons": seasons,
                "rows": rows,
                "features": len(final_features),
                **metrics,
            }
        )

    training_summary = pd.DataFrame(
        summary_rows
    )

    training_summary.to_csv(
        OUTPUT_DIR / "training_summary.csv",
        index=False,
    )

    # =========================================================================
    # SAVE MODEL
    # =========================================================================

    joblib.dump(
        model,
        OUTPUT_DIR / "model.joblib",
    )

    # =========================================================================
    # SAVE FEATURE LIST
    # =========================================================================

    save_feature_list(
        final_features
    )

    # =========================================================================
    # FINAL SUMMARY
    # =========================================================================

    print()
    print("=" * 80)
    print("FILES SAVED")
    print("=" * 80)
    print("Output directory:")
    print(f"  {OUTPUT_DIR}")
    print()
    print("  model.joblib")
    print("  feature_list.csv")
    print("  training_summary.csv")
    print("  train_predictions.csv")
    print("  validation_predictions.csv")
    print("  test_predictions.csv")
    print("  feature_importance.csv")
    print()
    print("Model 3 training completed successfully.")
    print("=" * 80)


if __name__ == "__main__":
    main()