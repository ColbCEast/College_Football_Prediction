"""
GRADIENT BOOSTING WIN PROBABILITY
RETURNING PRODUCTION - EXPERIMENT 2

Experiment 2:
Relative Returning Production

Purpose:
    Test whether returning production is more useful when represented
    as the home team's returning production minus the away team's
    returning production, rather than as separate home/away features.

Model definition:
    - Complete Model 5 predictive-safe baseline feature space
    - 310 baseline predictors
    - 8 relative returning production predictors
    - Total: 318 predictors
    - Same hyperparameters as Gradient Boosting Model 5
    - Same temporal train/validation/test splits
    - Median imputation fitted on training data only
    - No test-set tuning

Relative returning features:
    returning_total_ppa_diff
    returning_passing_ppa_diff
    returning_receiving_ppa_diff
    returning_rushing_ppa_diff
    returning_usage_diff
    returning_passing_usage_diff
    returning_receiving_usage_diff
    returning_rushing_usage_diff

Output:
    models/win_probability/gradient_boosting/returning_production/model_2/
"""

from pathlib import Path
import warnings

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


warnings.filterwarnings("ignore")


# =============================================================================
# PATHS
# =============================================================================

PROJECT_ROOT = Path(__file__).resolve().parents[7]

MODEL_INPUT_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "model_inputs"
    / "win_probability"
)

RETURNING_FEATURE_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "features"
    / "win_probability"
    / "returning_production"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "models"
    / "win_probability"
    / "gradient_boosting"
    / "returning_production"
    / "model_2"
)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# =============================================================================
# EXPERIMENT DEFINITION
# =============================================================================

MODEL_NAME = "Gradient Boosting Returning Production - Experiment 2"

RANDOM_STATE = 42

# Exact Model 5 hyperparameters
MODEL_PARAMS = {
    "n_estimators": 200,
    "learning_rate": 0.03,
    "max_depth": 4,
    "min_samples_leaf": 10,
    "subsample": 0.75,
    "random_state": RANDOM_STATE,
}


# =============================================================================
# RELATIVE RETURNING FEATURES
# =============================================================================

RETURNING_FEATURE_PAIRS = {
    "returning_total_ppa_diff": (
        "home_returning_total_ppa",
        "away_returning_total_ppa",
    ),
    "returning_passing_ppa_diff": (
        "home_returning_passing_ppa",
        "away_returning_passing_ppa",
    ),
    "returning_receiving_ppa_diff": (
        "home_returning_receiving_ppa",
        "away_returning_receiving_ppa",
    ),
    "returning_rushing_ppa_diff": (
        "home_returning_rushing_ppa",
        "away_returning_rushing_ppa",
    ),
    "returning_usage_diff": (
        "home_returning_usage",
        "away_returning_usage",
    ),
    "returning_passing_usage_diff": (
        "home_returning_passing_usage",
        "away_returning_passing_usage",
    ),
    "returning_receiving_usage_diff": (
        "home_returning_receiving_usage",
        "away_returning_receiving_usage",
    ),
    "returning_rushing_usage_diff": (
        "home_returning_rushing_usage",
        "away_returning_rushing_usage",
    ),
}

RELATIVE_RETURNING_FEATURES = list(RETURNING_FEATURE_PAIRS.keys())


# =============================================================================
# EXPECTED MODEL 5 BASELINE FEATURE LIST
# =============================================================================

def load_model_5_feature_list():
    """
    Load the exact 310-feature list used by Gradient Boosting Model 5.

    Model 5 is the baseline for all returning-production experiments.
    """

    model_5_feature_path = (
        PROJECT_ROOT
        / "models"
        / "win_probability"
        / "gradient_boosting"
        / "model_5"
        / "feature_list.csv"
    )

    if not model_5_feature_path.exists():
        raise FileNotFoundError(
            f"Model 5 feature list not found:\n"
            f"  {model_5_feature_path}\n\n"
            "Run Gradient Boosting Model 5 before Experiment 2."
        )

    feature_df = pd.read_csv(model_5_feature_path)

    if "feature" in feature_df.columns:
        features = feature_df["feature"].tolist()
    else:
        # Fall back to the first column if the feature list uses
        # a different column name.
        features = feature_df.iloc[:, 0].tolist()

    features = [str(feature) for feature in features]

    if len(features) != 310:
        raise ValueError(
            f"Expected exactly 310 Model 5 baseline features, "
            f"found {len(features)}."
        )

    if len(set(features)) != len(features):
        raise ValueError("Model 5 feature list contains duplicate features.")

    return features


# =============================================================================
# DATA LOADING
# =============================================================================

def load_model_input(split):
    """
    Load the original Model 5 model-input dataset.

    These files contain the complete 310-feature predictive-safe
    baseline space.
    """

    path = MODEL_INPUT_DIR / f"{split}.csv"

    if not path.exists():
        raise FileNotFoundError(
            f"Model input file not found:\n  {path}"
        )

    return pd.read_csv(path)


def load_returning_features(year):
    """
    Load the processed returning-production feature dataset.

    These files contain:
        - original 448 final features
        - returning-production features
        - missingness indicators
        - gamesBefore interactions
    """

    path = RETURNING_FEATURE_DIR / f"returning_features_{year}.csv"

    if not path.exists():
        raise FileNotFoundError(
            f"Returning feature file not found:\n  {path}"
        )

    return pd.read_csv(path)


# =============================================================================
# RETURNING FEATURE CONSTRUCTION
# =============================================================================

def build_relative_returning_features(df, year):
    """
    Construct the eight relative returning-production features.

    Each feature is:

        home returning value - away returning value

    No information is learned from the data here. This is a deterministic
    transformation of already-created predictive-safe features.
    """

    required_columns = set()

    for home_col, away_col in RETURNING_FEATURE_PAIRS.values():
        required_columns.add(home_col)
        required_columns.add(away_col)

    missing = sorted(required_columns - set(df.columns))

    if missing:
        raise ValueError(
            f"Year {year} is missing required returning-production "
            f"columns:\n  {missing}"
        )

    result = pd.DataFrame(index=df.index)

    for diff_name, (home_col, away_col) in RETURNING_FEATURE_PAIRS.items():
        result[diff_name] = df[home_col] - df[away_col]

    # These are differences of already-imputed returning features,
    # so no new missing values should be introduced.
    if result.isna().any().any():
        missing_counts = result.isna().sum()
        missing_counts = missing_counts[missing_counts > 0]

        raise ValueError(
            f"Relative returning features contain missing values "
            f"in year {year}:\n{missing_counts}"
        )

    return result


# =============================================================================
# BUILD EXPERIMENT DATASETS
# =============================================================================

def prepare_split(split, years, baseline_features):
    """
    Prepare one model split.

    Loads:
        - Model 5 baseline model-input data
        - Returning-production feature files

    Returns:
        X
        y
        game_ids
        seasons
    """

    model_df = load_model_input(split)

    # -------------------------------------------------------------------------
    # Validate baseline features
    # -------------------------------------------------------------------------

    missing_baseline = sorted(
        set(baseline_features) - set(model_df.columns)
    )

    if missing_baseline:
        raise ValueError(
            f"{split} dataset is missing Model 5 baseline features:\n"
            f"{missing_baseline}"
        )

    if "win_home" not in model_df.columns:
        raise ValueError(
            f"{split} dataset does not contain target column 'win_home'."
        )

    if "gameId" not in model_df.columns:
        raise ValueError(
            f"{split} dataset does not contain 'gameId'."
        )

    if "season" not in model_df.columns:
        raise ValueError(
            f"{split} dataset does not contain 'season'."
        )

    # -------------------------------------------------------------------------
    # Build relative returning features by season
    # -------------------------------------------------------------------------

    returning_parts = []

    for year in years:

        season_model = model_df[model_df["season"] == year].copy()

        if season_model.empty:
            continue

        returning_df = load_returning_features(year)

        if "gameId" not in returning_df.columns:
            raise ValueError(
                f"Returning feature file for {year} does not contain gameId."
            )

        if returning_df["gameId"].duplicated().any():
            duplicates = returning_df.loc[
                returning_df["gameId"].duplicated(keep=False),
                "gameId",
            ].tolist()

            raise ValueError(
                f"Duplicate gameId values found in returning features "
                f"for {year}: {duplicates[:10]}"
            )

        relative_features = build_relative_returning_features(
            returning_df,
            year,
        )

        relative_features.insert(
            0,
            "gameId",
            returning_df["gameId"].values,
        )

        # Verify exact game coverage.
        model_ids = set(season_model["gameId"])
        returning_ids = set(relative_features["gameId"])

        missing_from_returning = model_ids - returning_ids
        extra_in_returning = returning_ids - model_ids

        if missing_from_returning:
            raise ValueError(
                f"Year {year}: {len(missing_from_returning)} game IDs "
                f"from model input are missing from returning features."
            )

        if extra_in_returning:
            raise ValueError(
                f"Year {year}: {len(extra_in_returning)} game IDs "
                f"exist in returning features but not model input."
            )

        # Preserve model-input order.
        relative_features = (
            season_model[["gameId"]]
            .merge(
                relative_features,
                on="gameId",
                how="left",
                validate="one_to_one",
            )
        )

        if len(relative_features) != len(season_model):
            raise ValueError(
                f"Year {year}: row count changed during returning "
                f"feature merge."
            )

        returning_parts.append(relative_features)

    if not returning_parts:
        raise ValueError(
            f"No returning-production data found for {split}."
        )

    relative_all = pd.concat(
        returning_parts,
        ignore_index=True,
    )

    # -------------------------------------------------------------------------
    # Align relative features to model input order
    # -------------------------------------------------------------------------

    relative_all = (
        model_df[["gameId"]]
        .merge(
            relative_all,
            on="gameId",
            how="left",
            validate="one_to_one",
        )
    )

    if len(relative_all) != len(model_df):
        raise ValueError(
            f"{split}: final row count changed after feature alignment."
        )

    # -------------------------------------------------------------------------
    # Construct final feature matrix
    # -------------------------------------------------------------------------

    baseline_df = model_df[baseline_features].copy()

    relative_df = relative_all[RELATIVE_RETURNING_FEATURES].copy()

    X = pd.concat(
        [baseline_df, relative_df],
        axis=1,
    )

    # Ensure no duplicate feature names.
    if X.columns.duplicated().any():
        duplicates = X.columns[X.columns.duplicated()].tolist()

        raise ValueError(
            f"{split}: duplicate predictor names detected:\n"
            f"{duplicates}"
        )

    # Verify exact expected feature count.
    expected_count = len(baseline_features) + len(
        RELATIVE_RETURNING_FEATURES
    )

    if X.shape[1] != expected_count:
        raise ValueError(
            f"{split}: expected {expected_count} predictors, "
            f"found {X.shape[1]}."
        )

    # -------------------------------------------------------------------------
    # Validate target
    # -------------------------------------------------------------------------

    y = model_df["win_home"].copy()

    if y.isna().any():
        raise ValueError(
            f"{split}: target contains missing values."
        )

    if not set(y.unique()).issubset({0, 1}):
        raise ValueError(
            f"{split}: target contains values other than 0/1: "
            f"{sorted(y.unique())}"
        )

    # -------------------------------------------------------------------------
    # Validate relative features
    # -------------------------------------------------------------------------

    relative_missing = X[RELATIVE_RETURNING_FEATURES].isna().sum()

    if relative_missing.sum() > 0:
        raise ValueError(
            f"{split}: relative returning features contain missing values:\n"
            f"{relative_missing[relative_missing > 0]}"
        )

    return (
        X,
        y,
        model_df["gameId"].copy(),
        model_df["season"].copy(),
    )


# =============================================================================
# METRICS
# =============================================================================

def calculate_metrics(y_true, probabilities, predictions):
    """
    Calculate standard classification and probability metrics.
    """

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


# =============================================================================
# MAIN
# =============================================================================

def main():

    print("=" * 80)
    print("GRADIENT BOOSTING WIN PROBABILITY - RETURNING PRODUCTION")
    print("EXPERIMENT 2: RELATIVE RETURNING PRODUCTION")
    print("=" * 80)

    print()
    print("Experiment definition:")
    print("  Baseline: Gradient Boosting Model 5")
    print("  Baseline feature space: 310 predictors")
    print("  Added feature space: 8 relative returning features")
    print("  Total predictors: 318")
    print("  Representation: home returning value - away returning value")
    print("  Hyperparameters: exact Model 5 configuration")
    print("  Preprocessing: median imputation")
    print("  Imputation fit: training data only")
    print("  Tuning: none")
    print()

    # =========================================================================
    # LOAD MODEL 5 FEATURE LIST
    # =========================================================================

    baseline_features = load_model_5_feature_list()

    print(f"Loaded Model 5 baseline features: {len(baseline_features)}")
    print(
        f"Relative returning features: "
        f"{len(RELATIVE_RETURNING_FEATURES)}"
    )
    print(
        f"Total Experiment 2 predictors: "
        f"{len(baseline_features) + len(RELATIVE_RETURNING_FEATURES)}"
    )

    # =========================================================================
    # PREPARE DATA
    # =========================================================================

    print()
    print("Preparing training data...")

    X_train, y_train, train_ids, train_seasons = prepare_split(
        "train",
        range(2015, 2023),
        baseline_features,
    )

    print("Preparing validation data...")

    X_validation, y_validation, validation_ids, validation_seasons = (
        prepare_split(
            "validation",
            range(2023, 2025),
            baseline_features,
        )
    )

    print("Preparing test data...")

    X_test, y_test, test_ids, test_seasons = prepare_split(
        "test",
        range(2025, 2026),
        baseline_features,
    )

    # =========================================================================
    # VERIFY SPLITS
    # =========================================================================

    print()
    print("Dataset summary:")
    print(
        f"  Train:      {X_train.shape[0]:,} rows | "
        f"{X_train.shape[1]:,} predictors"
    )
    print(
        f"  Validation: {X_validation.shape[0]:,} rows | "
        f"{X_validation.shape[1]:,} predictors"
    )
    print(
        f"  Test:       {X_test.shape[0]:,} rows | "
        f"{X_test.shape[1]:,} predictors"
    )

    if X_train.shape[1] != 318:
        raise ValueError(
            f"Expected 318 predictors, found {X_train.shape[1]}."
        )

    if list(X_train.columns) != list(X_validation.columns):
        raise ValueError(
            "Training and validation feature columns do not match."
        )

    if list(X_train.columns) != list(X_test.columns):
        raise ValueError(
            "Training and test feature columns do not match."
        )

    # =========================================================================
    # IDENTIFIER / TARGET LEAKAGE CHECK
    # =========================================================================

    forbidden_columns = {
        "gameId",
        "season",
        "win_home",
    }

    leakage_columns = sorted(
        forbidden_columns.intersection(X_train.columns)
    )

    if leakage_columns:
        raise ValueError(
            f"Identifier/target leakage detected in predictors:\n"
            f"{leakage_columns}"
        )

    # =========================================================================
    # MISSINGNESS SUMMARY
    # =========================================================================

    print()
    print("Missingness before model imputation:")

    for name, X in [
        ("Train", X_train),
        ("Validation", X_validation),
        ("Test", X_test),
    ]:
        total_missing = int(X.isna().sum().sum())
        features_missing = int((X.isna().sum() > 0).sum())

        print(
            f"  {name:<12} "
            f"{total_missing:,} missing values across "
            f"{features_missing:,} features"
        )

    # =========================================================================
    # BUILD MODEL PIPELINE
    # =========================================================================

    print()
    print("Building model pipeline...")

    model = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(
                    strategy="median",
                ),
            ),
            (
                "classifier",
                GradientBoostingClassifier(
                    **MODEL_PARAMS,
                ),
            ),
        ]
    )

    # =========================================================================
    # TRAIN
    # =========================================================================

    print("Training Experiment 2...")
    print()

    model.fit(
        X_train,
        y_train,
    )

    print("Training complete.")

    # =========================================================================
    # PREDICTIONS
    # =========================================================================

    train_probabilities = model.predict_proba(X_train)[:, 1]
    validation_probabilities = model.predict_proba(X_validation)[:, 1]
    test_probabilities = model.predict_proba(X_test)[:, 1]

    train_predictions = (
        train_probabilities >= 0.5
    ).astype(int)

    validation_predictions = (
        validation_probabilities >= 0.5
    ).astype(int)

    test_predictions = (
        test_probabilities >= 0.5
    ).astype(int)

    # =========================================================================
    # METRICS
    # =========================================================================

    train_metrics = calculate_metrics(
        y_train,
        train_probabilities,
        train_predictions,
    )

    validation_metrics = calculate_metrics(
        y_validation,
        validation_probabilities,
        validation_predictions,
    )

    test_metrics = calculate_metrics(
        y_test,
        test_probabilities,
        test_predictions,
    )

    # =========================================================================
    # PRINT METRICS
    # =========================================================================

    print()
    print("=" * 80)
    print("MODEL PERFORMANCE")
    print("=" * 80)

    for split_name, metrics in [
        ("TRAIN", train_metrics),
        ("VALIDATION", validation_metrics),
        ("TEST", test_metrics),
    ]:

        print()
        print(split_name)

        for metric_name, value in metrics.items():
            print(
                f"  {metric_name:<20}: {value:.6f}"
            )

    # =========================================================================
    # FEATURE IMPORTANCE
    # =========================================================================

    classifier = model.named_steps["classifier"]

    importances = classifier.feature_importances_

    feature_importance_df = pd.DataFrame(
        {
            "feature": X_train.columns,
            "importance": importances,
        }
    ).sort_values(
        "importance",
        ascending=False,
    )

    feature_importance_df["rank"] = (
        np.arange(1, len(feature_importance_df) + 1)
    )

    feature_importance_df = feature_importance_df[
        ["rank", "feature", "importance"]
    ]

    # =========================================================================
    # SAVE MODEL
    # =========================================================================

    model_path = OUTPUT_DIR / "model.joblib"

    joblib.dump(
        model,
        model_path,
    )

    # =========================================================================
    # SAVE FEATURE LIST
    # =========================================================================

    feature_list_df = pd.DataFrame(
        {
            "feature": X_train.columns,
        }
    )

    feature_list_df.to_csv(
        OUTPUT_DIR / "feature_list.csv",
        index=False,
    )

    # =========================================================================
    # SAVE FEATURE IMPORTANCE
    # =========================================================================

    feature_importance_df.to_csv(
        OUTPUT_DIR / "feature_importance.csv",
        index=False,
    )

    # =========================================================================
    # SAVE TRAINING SUMMARY
    # =========================================================================

    training_summary = pd.DataFrame(
        [
            {
                "model": MODEL_NAME,
                "experiment": "Experiment 2",
                "baseline_model": "Gradient Boosting Model 5",
                "feature_space": "Model 5 + relative returning production",
                "baseline_features": len(baseline_features),
                "returning_features": len(
                    RELATIVE_RETURNING_FEATURES
                ),
                "total_features": X_train.shape[1],
                "train_rows": len(X_train),
                "validation_rows": len(X_validation),
                "test_rows": len(X_test),
                "train_start_season": int(train_seasons.min()),
                "train_end_season": int(train_seasons.max()),
                "validation_start_season": int(
                    validation_seasons.min()
                ),
                "validation_end_season": int(
                    validation_seasons.max()
                ),
                "test_start_season": int(test_seasons.min()),
                "test_end_season": int(test_seasons.max()),
                "n_estimators": MODEL_PARAMS["n_estimators"],
                "learning_rate": MODEL_PARAMS["learning_rate"],
                "max_depth": MODEL_PARAMS["max_depth"],
                "min_samples_leaf": MODEL_PARAMS["min_samples_leaf"],
                "subsample": MODEL_PARAMS["subsample"],
                "random_state": MODEL_PARAMS["random_state"],
                "imputation_strategy": "median",
                "test_log_loss": test_metrics["log_loss"],
                "test_brier_score": test_metrics["brier_score"],
                "test_roc_auc": test_metrics["roc_auc"],
                "test_accuracy": test_metrics["accuracy"],
                "test_balanced_accuracy": test_metrics[
                    "balanced_accuracy"
                ],
                "test_precision": test_metrics["precision"],
                "test_recall": test_metrics["recall"],
            }
        ]
    )

    training_summary.to_csv(
        OUTPUT_DIR / "training_summary.csv",
        index=False,
    )

    # =========================================================================
    # SAVE VALIDATION PREDICTIONS
    # =========================================================================

    validation_prediction_df = pd.DataFrame(
        {
            "gameId": validation_ids.values,
            "season": validation_seasons.values,
            "win_home": y_validation.values,
            "predicted_probability_home_win": (
                validation_probabilities
            ),
            "predicted_home_win": validation_predictions,
        }
    )

    validation_prediction_df.to_csv(
        OUTPUT_DIR / "validation_predictions.csv",
        index=False,
    )

    # =========================================================================
    # SAVE TEST PREDICTIONS
    # =========================================================================

    test_prediction_df = pd.DataFrame(
        {
            "gameId": test_ids.values,
            "season": test_seasons.values,
            "win_home": y_test.values,
            "predicted_probability_home_win": test_probabilities,
            "predicted_home_win": test_predictions,
        }
    )

    test_prediction_df.to_csv(
        OUTPUT_DIR / "test_predictions.csv",
        index=False,
    )

    # =========================================================================
    # SAVE RELATIVE FEATURE SUMMARY
    # =========================================================================

    relative_feature_summary = pd.DataFrame(
        {
            "feature": RELATIVE_RETURNING_FEATURES,
            "home_feature": [
                pair[0]
                for pair in RETURNING_FEATURE_PAIRS.values()
            ],
            "away_feature": [
                pair[1]
                for pair in RETURNING_FEATURE_PAIRS.values()
            ],
            "importance": [
                feature_importance_df.loc[
                    feature_importance_df["feature"] == feature,
                    "importance",
                ].iloc[0]
                for feature in RELATIVE_RETURNING_FEATURES
            ],
        }
    ).sort_values(
        "importance",
        ascending=False,
    )

    relative_feature_summary.to_csv(
        OUTPUT_DIR / "relative_feature_importance.csv",
        index=False,
    )

    # =========================================================================
    # FINAL SUMMARY
    # =========================================================================

    print()
    print("=" * 80)
    print("EXPERIMENT 2 COMPLETE")
    print("=" * 80)

    print()
    print("Relative returning features:")

    for feature in RELATIVE_RETURNING_FEATURES:
        print(f"  - {feature}")

    print()
    print("Test performance:")
    print(
        f"  Log Loss:       {test_metrics['log_loss']:.6f}"
    )
    print(
        f"  Brier Score:    {test_metrics['brier_score']:.6f}"
    )
    print(
        f"  ROC AUC:        {test_metrics['roc_auc']:.6f}"
    )
    print(
        f"  Accuracy:       {test_metrics['accuracy']:.6f}"
    )

    print()
    print("Output directory:")
    print(f"  {OUTPUT_DIR}")

    print()
    print("Files created:")
    print("  model.joblib")
    print("  feature_list.csv")
    print("  feature_importance.csv")
    print("  relative_feature_importance.csv")
    print("  training_summary.csv")
    print("  validation_predictions.csv")
    print("  test_predictions.csv")

    print()
    print("=" * 80)


if __name__ == "__main__":
    main()