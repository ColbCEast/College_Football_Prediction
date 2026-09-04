"""
Audit Random Forest Model 2
============================

Purpose
-------
Perform a diagnostic and stability audit of the trained Random Forest
Model 2.

Model 2:
    60 predictors

Feature groups:
    - Core strength
    - Recent form
    - Trend
    - Prior strength of schedule
    - Offensive efficiency
    - Defensive efficiency

The 16 engineered Trend/SOS features are stored in the enhanced logistic
feature files and are merged onto the generic win-probability model-input
splits by gameId, exactly as performed during Model 2 training.

Temporal split
--------------
Training:
    2015-2022

Validation:
    2023-2024

Test:
    2025

Audit principles
----------------
- The saved Model 2 is loaded; the model is NOT retrained.
- The shared model-input files are NOT modified.
- Enhanced features are reconstructed using the same merge procedure as
  Model 2 training.
- The saved Model 2 feature list is used as the authoritative predictor
  list.
- Validation performance is used for diagnostic analysis.
- Test performance is reported as final out-of-sample evaluation.
- The test set is NOT used to modify the feature set.
- Permutation importance is calculated on validation data.
- Calibration is evaluated using fixed probability bins.
- Prediction stability is summarized through probability percentiles.
- Feature importance is reported both individually and by feature family.

Outputs
-------
Saved to:

    models/win_probability/random_forest/model_2/audit/

Files:
    audit_summary.csv
    feature_importance.csv
    permutation_importance.csv
    feature_importance_by_family.csv
    calibration.csv
    season_performance.csv
    prediction_stability.csv
"""

from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from sklearn.inspection import permutation_importance
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    brier_score_loss,
    confusion_matrix,
    log_loss,
    precision_score,
    recall_score,
    roc_auc_score,
)


# ============================================================================
# CONFIGURATION
# ============================================================================

RANDOM_STATE = 42

TARGET_COLUMN = "win_home"
GAME_ID_COLUMN = "gameId"
SEASON_COLUMN = "season"

TRAIN_YEARS = list(range(2015, 2023))
VALIDATION_YEARS = [2023, 2024]
TEST_YEARS = [2025]

PERMUTATION_REPEATS = 10

CALIBRATION_BIN_WIDTH = 0.05


# ============================================================================
# PROJECT PATHS
# ============================================================================

# audit.py is located at:
#
# src/
#   modeling/
#     problems/
#       win_probability/
#         random_forest/
#           model_2/
#             audit.py
#
# parents[6] = project root

SCRIPT_PATH = Path(__file__).resolve()

PROJECT_ROOT = SCRIPT_PATH.parents[6]


MODEL_INPUT_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "model_inputs"
    / "win_probability"
)

ENHANCED_FEATURE_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "features"
    / "win_probability"
    / "logistic_regression"
    / "enhanced"
)


TRAIN_PATH = (
    MODEL_INPUT_DIR
    / "train.csv"
)

VALIDATION_PATH = (
    MODEL_INPUT_DIR
    / "validation.csv"
)

TEST_PATH = (
    MODEL_INPUT_DIR
    / "test.csv"
)


MODEL_DIR = (
    PROJECT_ROOT
    / "models"
    / "win_probability"
    / "random_forest"
    / "model_2"
)


MODEL_PATH = (
    MODEL_DIR
    / "model.joblib"
)

FEATURE_LIST_PATH = (
    MODEL_DIR
    / "feature_list.csv"
)


AUDIT_DIR = (
    MODEL_DIR
    / "audit"
)


# ============================================================================
# MODEL 2 FEATURE DEFINITIONS
# ============================================================================

# These definitions mirror train.py exactly.

CORE_STRENGTH_FEATURES = [
    "homePregameElo",
    "awayPregameElo",

    "winPctBefore_home",
    "winPctBefore_away",

    "pointDifferentialBefore_home",
    "pointDifferentialBefore_away",

    "pointDifferentialAvgBefore_home",
    "pointDifferentialAvgBefore_away",
]


RECENT_FORM_FEATURES = [
    "pointDifferentialAvgLast3_home",
    "pointDifferentialAvgLast3_away",

    "pointDifferentialAvgLast5_home",
    "pointDifferentialAvgLast5_away",

    "pointsForAvgLast5_home",
    "pointsForAvgLast5_away",

    "pointsAgainstAvgLast5_home",
    "pointsAgainstAvgLast5_away",
]


TREND_FEATURES = [
    "pointsForTrend_home",
    "pointsForTrend_away",

    "pointsAgainstTrend_home",
    "pointsAgainstTrend_away",

    "pointDifferentialTrend_home",
    "pointDifferentialTrend_away",

    "totalYardsTrend_home",
    "totalYardsTrend_away",

    "netPassingYardsTrend_home",
    "netPassingYardsTrend_away",

    "winPctTrend_home",
    "winPctTrend_away",
]


SOS_FEATURES = [
    "priorSOSWinPct_home",
    "priorSOSWinPct_away",

    "priorSOSPointDiff_home",
    "priorSOSPointDiff_away",
]


OFFENSIVE_FEATURES = [
    "home_pregame_offense_successRate",
    "away_pregame_offense_successRate",

    "home_pregame_offense_ppa",
    "away_pregame_offense_ppa",

    "yardsPerPassAttemptBefore_home",
    "yardsPerPassAttemptBefore_away",

    "yardsPerRushAttemptBefore_home",
    "yardsPerRushAttemptBefore_away",
]


DEFENSIVE_FEATURES = [
    "home_pregame_defense_successRate",
    "away_pregame_defense_successRate",

    "home_pregame_defense_ppa",
    "away_pregame_defense_ppa",
]


MODEL_2_FEATURES = (
    CORE_STRENGTH_FEATURES
    + RECENT_FORM_FEATURES
    + TREND_FEATURES
    + SOS_FEATURES
    + OFFENSIVE_FEATURES
    + DEFENSIVE_FEATURES
)


# ============================================================================
# PRINT HELPERS
# ============================================================================

def print_section(title):
    print()
    print("=" * 80)
    print(title)
    print("=" * 80)


# ============================================================================
# DATA LOADING
# ============================================================================

def load_split(path, name):
    """Load a generic win-probability modeling split."""

    if not path.exists():
        raise FileNotFoundError(
            f"{name} split does not exist:\n"
            f"  {path}"
        )

    df = pd.read_csv(path)

    print(
        f"{name:<12}: "
        f"{len(df):,} rows × {len(df.columns):,} columns"
    )

    return df


def load_enhanced_features(year):
    """
    Load the enhanced feature file for a given season.

    This mirrors train.py.
    """

    path = (
        ENHANCED_FEATURE_DIR
        / f"logistic_features_{year}.csv"
    )

    if not path.exists():
        raise FileNotFoundError(
            f"Enhanced feature file does not exist for {year}:\n"
            f"  {path}"
        )

    df = pd.read_csv(path)

    if GAME_ID_COLUMN not in df.columns:
        raise ValueError(
            f"Enhanced feature file for {year} does not contain "
            f"'{GAME_ID_COLUMN}':\n"
            f"  {path}"
        )

    return df


# ============================================================================
# MERGE ENHANCED FEATURES
# ============================================================================

def add_enhanced_features(df, split_name):
    """
    Merge the 16 engineered Trend/SOS features onto the generic
    model-input dataframe.

    This reproduces the exact merge logic used by train.py.
    """

    print_section(
        f"ADDING ENHANCED FEATURES — {split_name.upper()}"
    )

    seasons = sorted(
        df[SEASON_COLUMN].unique()
    )

    print(
        f"Seasons found: {seasons}"
    )

    enhanced_frames = []

    required_columns = (
        [GAME_ID_COLUMN]
        + TREND_FEATURES
        + SOS_FEATURES
    )

    for year in seasons:

        enhanced = load_enhanced_features(
            int(year)
        )

        missing = [
            column
            for column in required_columns
            if column not in enhanced.columns
        ]

        if missing:
            raise ValueError(
                f"Enhanced feature file for {year} is missing "
                f"required Model 2 engineered features:\n"
                + "\n".join(
                    f"  {column}"
                    for column in missing
                )
            )

        enhanced = enhanced[
            required_columns
        ].copy()

        if enhanced[GAME_ID_COLUMN].duplicated().any():

            duplicate_count = (
                enhanced[GAME_ID_COLUMN]
                .duplicated()
                .sum()
            )

            raise ValueError(
                f"Enhanced feature file for {year} contains "
                f"{duplicate_count} duplicate game IDs."
            )

        enhanced_frames.append(
            enhanced
        )

        print(
            f"  {int(year)}: "
            f"{len(enhanced):,} enhanced rows loaded"
        )

    enhanced_all = pd.concat(
        enhanced_frames,
        ignore_index=True,
    )

    print()
    print(
        f"Total enhanced rows: "
        f"{len(enhanced_all):,}"
    )

    if enhanced_all[GAME_ID_COLUMN].duplicated().any():

        duplicates = (
            enhanced_all[GAME_ID_COLUMN]
            .duplicated()
            .sum()
        )

        raise ValueError(
            f"Enhanced feature dataset contains "
            f"{duplicates} duplicate game IDs."
        )

    before_rows = len(df)

    merged = df.merge(
        enhanced_all,
        on=GAME_ID_COLUMN,
        how="left",
        validate="one_to_one",
        indicator=True,
    )

    unmatched = (
        merged["_merge"] != "both"
    ).sum()

    if unmatched > 0:

        missing_ids = (
            merged.loc[
                merged["_merge"] != "both",
                GAME_ID_COLUMN
            ]
            [GAME_ID_COLUMN]
            .head(20)
            .tolist()
        )

        raise ValueError(
            f"{unmatched} games in the {split_name} split "
            f"could not be matched to enhanced features.\n"
            f"Example game IDs: {missing_ids}"
        )

    merged = merged.drop(
        columns=["_merge"]
    )

    if len(merged) != before_rows:

        raise ValueError(
            f"Row count changed during enhanced feature merge: "
            f"{before_rows:,} → {len(merged):,}"
        )

    print()
    print(
        "Enhanced features merged successfully."
    )

    print(
        f"  Rows:          {len(merged):,}"
    )

    print(
        f"  Added features: "
        f"{len(TREND_FEATURES) + len(SOS_FEATURES)}"
    )

    return merged


# ============================================================================
# DATASET VALIDATION
# ============================================================================

def validate_splits(train, validation, test):

    print_section(
        "VALIDATING MODELING SPLITS"
    )

    required = {
        TARGET_COLUMN,
        GAME_ID_COLUMN,
        SEASON_COLUMN,
    }

    for name, df in [
        ("Training", train),
        ("Validation", validation),
        ("Test", test),
    ]:

        missing = (
            required
            - set(df.columns)
        )

        if missing:
            raise ValueError(
                f"{name} is missing required columns: "
                f"{sorted(missing)}"
            )

        if df[TARGET_COLUMN].isna().any():
            raise ValueError(
                f"{name} contains missing target values."
            )

        if df[GAME_ID_COLUMN].isna().any():
            raise ValueError(
                f"{name} contains missing game IDs."
            )

        if df[GAME_ID_COLUMN].duplicated().any():
            raise ValueError(
                f"{name} contains duplicate game IDs."
            )

        target_values = sorted(
            df[TARGET_COLUMN]
            .astype(int)
            .unique()
            .tolist()
        )

        if target_values != [0, 1]:
            raise ValueError(
                f"{name} has unexpected target values: "
                f"{target_values}"
            )

    train_ids = set(
        train[GAME_ID_COLUMN]
    )

    validation_ids = set(
        validation[GAME_ID_COLUMN]
    )

    test_ids = set(
        test[GAME_ID_COLUMN]
    )

    if train_ids & validation_ids:
        raise ValueError(
            "Training and validation game IDs overlap."
        )

    if train_ids & test_ids:
        raise ValueError(
            "Training and test game IDs overlap."
        )

    if validation_ids & test_ids:
        raise ValueError(
            "Validation and test game IDs overlap."
        )

    train_seasons = sorted(
        train[SEASON_COLUMN].unique()
    )

    validation_seasons = sorted(
        validation[SEASON_COLUMN].unique()
    )

    test_seasons = sorted(
        test[SEASON_COLUMN].unique()
    )

    print()
    print(
        f"Training seasons:   {train_seasons}"
    )

    print(
        f"Validation seasons: {validation_seasons}"
    )

    print(
        f"Test seasons:       {test_seasons}"
    )

    if train_seasons != TRAIN_YEARS:
        raise ValueError(
            f"Unexpected training seasons: "
            f"{train_seasons}"
        )

    if validation_seasons != VALIDATION_YEARS:
        raise ValueError(
            f"Unexpected validation seasons: "
            f"{validation_seasons}"
        )

    if test_seasons != TEST_YEARS:
        raise ValueError(
            f"Unexpected test seasons: "
            f"{test_seasons}"
        )

    print()
    print(
        "Dataset validation passed."
    )


# ============================================================================
# FEATURE LIST LOADING
# ============================================================================

def load_feature_list():
    """
    Load the saved Model 2 feature list.

    The saved feature list is treated as the authoritative record of
    the predictors used by the trained model.
    """

    print_section(
        "LOADING MODEL 2 FEATURE LIST"
    )

    if not FEATURE_LIST_PATH.exists():
        raise FileNotFoundError(
            f"Model 2 feature list does not exist:\n"
            f"  {FEATURE_LIST_PATH}"
        )

    feature_df = pd.read_csv(
        FEATURE_LIST_PATH
    )

    if "feature" not in feature_df.columns:
        raise ValueError(
            "Model 2 feature list does not contain "
            "a 'feature' column."
        )

    features = (
        feature_df["feature"]
        .dropna()
        .astype(str)
        .tolist()
    )

    if len(features) != len(set(features)):
        duplicates = sorted(
            {
                feature
                for feature in features
                if features.count(feature) > 1
            }
        )

        raise ValueError(
            "Saved Model 2 feature list contains duplicate features:\n"
            + "\n".join(
                f"  {feature}"
                for feature in duplicates
            )
        )

    print(
        f"Feature count: {len(features)}"
    )

    print()

    if len(features) != len(MODEL_2_FEATURES):

        print(
            "WARNING:"
        )

        print(
            f"  Saved feature list contains {len(features)} features."
        )

        print(
            f"  train.py Model 2 definition contains "
            f"{len(MODEL_2_FEATURES)} features."
        )

        print(
            "  The saved feature list will be used as the "
            "authoritative list for auditing the trained model."
        )

    return features


# ============================================================================
# FEATURE VALIDATION
# ============================================================================

def validate_features(
    train,
    validation,
    test,
    features,
):

    print_section(
        "VALIDATING MODEL 2 FEATURES"
    )

    print(
        f"Model 2 predictors: {len(features)}"
    )

    for name, df in [
        ("Training", train),
        ("Validation", validation),
        ("Test", test),
    ]:

        missing = [
            feature
            for feature in features
            if feature not in df.columns
        ]

        if missing:
            raise ValueError(
                f"{name} is missing Model 2 features:\n"
                + "\n".join(
                    f"  {feature}"
                    for feature in missing
                )
            )

    print()
    print(
        "Feature validation passed."
    )

    print()
    print(
        "Feature groups:"
    )

    print(
        f"  Core strength:        "
        f"{len(CORE_STRENGTH_FEATURES)}"
    )

    print(
        f"  Recent form:          "
        f"{len(RECENT_FORM_FEATURES)}"
    )

    print(
        f"  Trend:                "
        f"{len(TREND_FEATURES)}"
    )

    print(
        f"  Prior SOS:            "
        f"{len(SOS_FEATURES)}"
    )

    print(
        f"  Offensive efficiency: "
        f"{len(OFFENSIVE_FEATURES)}"
    )

    print(
        f"  Defensive efficiency: "
        f"{len(DEFENSIVE_FEATURES)}"
    )

    print()
    print(
        f"Total expected predictors: "
        f"{len(MODEL_2_FEATURES)}"
    )

    print(
        f"Total saved predictors:    "
        f"{len(features)}"
    )


# ============================================================================
# PREPARE FEATURES
# ============================================================================

def prepare_features(df, features):

    X = df[
        features
    ].copy()

    for column in features:

        X[column] = pd.to_numeric(
            X[column],
            errors="coerce",
        )

    return X


# ============================================================================
# MODEL LOADING
# ============================================================================

def load_model():

    print_section(
        "LOADING TRAINED MODEL"
    )

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model file does not exist:\n"
            f"  {MODEL_PATH}"
        )

    print(
        f"Model: {MODEL_PATH}"
    )

    model = joblib.load(
        MODEL_PATH
    )

    print(
        "Model loaded successfully."
    )

    return model


# ============================================================================
# MISSINGNESS
# ============================================================================

def calculate_missingness(
    X_train,
    X_validation,
    X_test,
    features,
):

    rows = []

    for feature in features:

        rows.append(
            {
                "feature": feature,

                "train_missing_pct":
                    X_train[feature].isna().mean() * 100,

                "validation_missing_pct":
                    X_validation[feature].isna().mean() * 100,

                "test_missing_pct":
                    X_test[feature].isna().mean() * 100,
            }
        )

    return (
        pd.DataFrame(rows)
        .sort_values(
            "train_missing_pct",
            ascending=False,
        )
        .reset_index(drop=True)
    )


def print_missingness(
    X_train,
    X_validation,
    X_test,
    features,
):

    print_section(
        "FEATURE MISSINGNESS"
    )

    missingness = calculate_missingness(
        X_train,
        X_validation,
        X_test,
        features,
    )

    nonzero = missingness[
        missingness["train_missing_pct"] > 0
    ]

    if len(nonzero) == 0:

        print(
            "No missing predictor values in training."
        )

        return missingness

    print(
        "Training missingness:"
    )

    for _, row in nonzero.iterrows():

        print(
            f"  {row['feature']:<55}"
            f"{row['train_missing_pct']:>7.2f}%"
        )

    print()
    print(
        f"Features with training missing values: "
        f"{len(nonzero)}"
    )

    return missingness


# ============================================================================
# METRICS
# ============================================================================

def calculate_metrics(
    y_true,
    probability,
):

    prediction = (
        probability >= 0.50
    ).astype(int)

    return {
        "log_loss":
            log_loss(
                y_true,
                probability,
            ),

        "brier_score":
            brier_score_loss(
                y_true,
                probability,
            ),

        "roc_auc":
            roc_auc_score(
                y_true,
                probability,
            ),

        "accuracy":
            accuracy_score(
                y_true,
                prediction,
            ),

        "balanced_accuracy":
            balanced_accuracy_score(
                y_true,
                prediction,
            ),

        "precision":
            precision_score(
                y_true,
                prediction,
                zero_division=0,
            ),

        "recall":
            recall_score(
                y_true,
                prediction,
                zero_division=0,
            ),
    }


def print_metrics(
    name,
    y_true,
    probability,
):

    metrics = calculate_metrics(
        y_true,
        probability,
    )

    prediction = (
        probability >= 0.50
    ).astype(int)

    matrix = confusion_matrix(
        y_true,
        prediction,
    )

    print_section(
        f"{name.upper()} PERFORMANCE"
    )

    print(
        f"Log Loss:            "
        f"{metrics['log_loss']:.6f}"
    )

    print(
        f"Brier Score:         "
        f"{metrics['brier_score']:.6f}"
    )

    print(
        f"ROC AUC:             "
        f"{metrics['roc_auc']:.6f}"
    )

    print(
        f"Accuracy:            "
        f"{metrics['accuracy']:.4%}"
    )

    print(
        f"Balanced Accuracy:   "
        f"{metrics['balanced_accuracy']:.4%}"
    )

    print(
        f"Precision:           "
        f"{metrics['precision']:.4%}"
    )

    print(
        f"Recall:              "
        f"{metrics['recall']:.4%}"
    )

    print()
    print(
        "Confusion Matrix:"
    )

    print(
        "                 Predicted"
    )

    print(
        "                 Away   Home"
    )

    print(
        f"Actual Away      "
        f"{matrix[0, 0]:5d}  "
        f"{matrix[0, 1]:5d}"
    )

    print(
        f"Actual Home      "
        f"{matrix[1, 0]:5d}  "
        f"{matrix[1, 1]:5d}"
    )

    return metrics, matrix


# ============================================================================
# PREDICTION DISTRIBUTION
# ============================================================================

def calculate_prediction_distribution(
    probability,
):

    percentiles = [
        1,
        5,
        10,
        25,
        50,
        75,
        90,
        95,
        99,
    ]

    result = {
        "mean": np.mean(probability),
        "std": np.std(probability),
        "min": np.min(probability),
        "max": np.max(probability),
    }

    for percentile in percentiles:

        result[
            f"p{percentile:02d}"
        ] = np.percentile(
            probability,
            percentile,
        )

    return result


def print_prediction_distribution(
    name,
    probability,
):

    distribution = calculate_prediction_distribution(
        probability
    )

    print()
    print(name)

    print(
        f"  Mean:   {distribution['mean']:.6f}"
    )

    print(
        f"  Std:    {distribution['std']:.6f}"
    )

    print(
        f"  P01:    {distribution['p01']:.6f}"
    )

    print(
        f"  P05:    {distribution['p05']:.6f}"
    )

    print(
        f"  P10:    {distribution['p10']:.6f}"
    )

    print(
        f"  P25:    {distribution['p25']:.6f}"
    )

    print(
        f"  P50:    {distribution['p50']:.6f}"
    )

    print(
        f"  P75:    {distribution['p75']:.6f}"
    )

    print(
        f"  P90:    {distribution['p90']:.6f}"
    )

    print(
        f"  P95:    {distribution['p95']:.6f}"
    )

    print(
        f"  P99:    {distribution['p99']:.6f}"
    )

    print(
        f"  Min:    {distribution['min']:.6f}"
    )

    print(
        f"  Max:    {distribution['max']:.6f}"
    )

    return distribution


# ============================================================================
# FEATURE IMPORTANCE
# ============================================================================

def calculate_impurity_importance(
    model,
    features,
):

    # Model 2 was trained as:
    #
    # Pipeline(
    #     imputer,
    #     random_forest
    # )
    #
    # Therefore the RandomForestClassifier is the "model" step.

    if not hasattr(model, "named_steps"):
        raise ValueError(
            "Expected Model 2 to be a sklearn Pipeline."
        )

    if "model" not in model.named_steps:
        raise ValueError(
            "Expected Model 2 pipeline to contain a "
            "'model' step."
        )

    random_forest = model.named_steps["model"]

    if not hasattr(
        random_forest,
        "feature_importances_",
    ):
        raise ValueError(
            "Random Forest model does not contain "
            "'feature_importances_'."
        )

    importance = random_forest.feature_importances_

    if len(importance) != len(features):
        raise ValueError(
            f"Feature importance count ({len(importance)}) "
            f"does not match feature count ({len(features)})."
        )

    result = pd.DataFrame(
        {
            "feature": features,
            "impurity_importance": importance,
        }
    )

    result = result.sort_values(
        "impurity_importance",
        ascending=False,
    ).reset_index(drop=True)

    result.insert(
        0,
        "importance_rank",
        np.arange(
            1,
            len(result) + 1,
        ),
    )

    return result


def print_impurity_importance(
    importance,
):

    print_section(
        "IMPURITY-BASED FEATURE IMPORTANCE"
    )

    print(
        "Top 20 features:"
    )

    print()

    for _, row in importance.head(20).iterrows():

        print(
            f"{int(row['importance_rank']):>3}. "
            f"{row['feature']:<55}"
            f"{row['impurity_importance']:.6f}"
        )


# ============================================================================
# PERMUTATION IMPORTANCE
# ============================================================================

def calculate_permutation_importance(
    model,
    X_validation,
    y_validation,
    features,
):

    print_section(
        "PERMUTATION IMPORTANCE"
    )

    print(
        "Calculating validation-set permutation importance..."
    )

    print(
        f"Repeats: {PERMUTATION_REPEATS}"
    )

    print(
        "Scoring: neg_log_loss"
    )

    print(
        "This measures the impact of each feature on "
        "out-of-sample probabilistic performance."
    )

    result = permutation_importance(
        model,
        X_validation,
        y_validation,
        scoring="neg_log_loss",
        n_repeats=PERMUTATION_REPEATS,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )

    importance = pd.DataFrame(
        {
            "feature": features,

            "permutation_importance_mean":
                result.importances_mean,

            "permutation_importance_std":
                result.importances_std,
        }
    )

    importance = importance.sort_values(
        "permutation_importance_mean",
        ascending=False,
    ).reset_index(drop=True)

    importance.insert(
        0,
        "importance_rank",
        np.arange(
            1,
            len(importance) + 1,
        ),
    )

    print()
    print(
        "Top 20 features:"
    )

    print()

    for _, row in importance.head(20).iterrows():

        print(
            f"{int(row['importance_rank']):>3}. "
            f"{row['feature']:<55}"
            f"{row['permutation_importance_mean']:.6f} "
            f"+/- "
            f"{row['permutation_importance_std']:.6f}"
        )

    return importance


# ============================================================================
# FEATURE FAMILY CLASSIFICATION
# ============================================================================

def classify_feature_family(
    feature,
):

    feature_lower = feature.lower()

    if "trend" in feature_lower:
        return "Trend"

    if "priorsos" in feature_lower:
        return "Prior SOS"

    if "pregameelo" in feature_lower:
        return "Elo"

    if (
        "last3" in feature_lower
        or "last5" in feature_lower
    ):
        return "Recent Form"

    if (
        "successrate" in feature_lower
        or "_ppa" in feature_lower
    ):

        if "defense" in feature_lower:
            return "Defensive Efficiency"

        return "Offensive Efficiency"

    if (
        "yardsperpassattempt" in feature_lower
        or "yardsperrushattempt" in feature_lower
    ):
        return "Offensive Efficiency"

    if (
        "pointdifferential" in feature_lower
        or "winpct" in feature_lower
        or "pointsfor" in feature_lower
        or "pointsagainst" in feature_lower
    ):
        return "Core Strength"

    return "Other"


def add_feature_family(
    importance_df,
):

    result = importance_df.copy()

    result["feature_family"] = (
        result["feature"]
        .apply(classify_feature_family)
    )

    return result


def calculate_family_importance(
    importance_df,
):

    result = (
        importance_df
        .groupby(
            "feature_family",
            as_index=False,
        )
        .agg(
            feature_count=(
                "feature",
                "count",
            ),

            total_importance=(
                "impurity_importance",
                "sum",
            ),

            mean_importance=(
                "impurity_importance",
                "mean",
            ),

            max_importance=(
                "impurity_importance",
                "max",
            ),
        )
    )

    result["importance_share"] = (
        result["total_importance"]
        / result["total_importance"].sum()
    )

    result = result.sort_values(
        "total_importance",
        ascending=False,
    ).reset_index(drop=True)

    result.insert(
        0,
        "family_rank",
        np.arange(
            1,
            len(result) + 1,
        ),
    )

    return result


def print_family_importance(
    family_importance,
):

    print_section(
        "FEATURE IMPORTANCE BY FAMILY"
    )

    for _, row in family_importance.iterrows():

        print(
            f"{int(row['family_rank']):>2}. "
            f"{row['feature_family']:<25}"
            f"{int(row['feature_count']):>3} features  "
            f"total={row['total_importance']:.6f}  "
            f"share={row['importance_share']:.2%}"
        )


# ============================================================================
# CALIBRATION
# ============================================================================

def calculate_calibration(
    y_true,
    probability,
):

    bins = np.arange(
        0.0,
        1.0 + CALIBRATION_BIN_WIDTH,
        CALIBRATION_BIN_WIDTH,
    )

    rows = []

    for lower in bins[:-1]:

        upper = lower + CALIBRATION_BIN_WIDTH

        if upper >= 1.0:

            mask = (
                (probability >= lower)
                & (probability <= upper)
            )

        else:

            mask = (
                (probability >= lower)
                & (probability < upper)
            )

        count = mask.sum()

        if count == 0:
            continue

        actual_rate = np.mean(
            y_true[mask]
        )

        predicted_rate = np.mean(
            probability[mask]
        )

        rows.append(
            {
                "bin_lower": lower,
                "bin_upper": upper,
                "count": count,
                "mean_predicted_probability":
                    predicted_rate,
                "actual_home_win_rate":
                    actual_rate,
                "calibration_error":
                    actual_rate - predicted_rate,
                "absolute_calibration_error":
                    abs(
                        actual_rate
                        - predicted_rate
                    ),
            }
        )

    result = pd.DataFrame(
        rows
    )

    if len(result) > 0:

        result["weighted_absolute_calibration_error"] = (
            result["absolute_calibration_error"]
            * result["count"]
            / result["count"].sum()
        )

    return result


def print_calibration(
    name,
    calibration,
):

    print_section(
        f"{name.upper()} CALIBRATION"
    )

    print(
        "Probability bins:"
    )

    print()

    print(
        f"{'Bin':<15}"
        f"{'N':>6}"
        f"{'Predicted':>14}"
        f"{'Actual':>14}"
        f"{'Error':>14}"
    )

    print("-" * 63)

    for _, row in calibration.iterrows():

        bin_label = (
            f"{row['bin_lower']:.2f}-"
            f"{row['bin_upper']:.2f}"
        )

        print(
            f"{bin_label:<15}"
            f"{int(row['count']):>6}"
            f"{row['mean_predicted_probability']:>14.4f}"
            f"{row['actual_home_win_rate']:>14.4f}"
            f"{row['calibration_error']:>14.4f}"
        )

    if len(calibration) > 0:

        weighted_mae = (
            calibration[
                "weighted_absolute_calibration_error"
            ].sum()
        )

        print()
        print(
            f"Weighted absolute calibration error: "
            f"{weighted_mae:.6f}"
        )


# ============================================================================
# SEASON PERFORMANCE
# ============================================================================

def calculate_season_performance(
    df,
    probability,
):

    rows = []

    seasons = sorted(
        df[SEASON_COLUMN].unique()
    )

    for season in seasons:

        mask = (
            df[SEASON_COLUMN]
            .values
            == season
        )

        y_true = (
            df.loc[
                mask,
                TARGET_COLUMN,
            ]
            .astype(int)
            .values
        )

        season_probability = probability[
            mask
        ]

        metrics = calculate_metrics(
            y_true,
            season_probability,
        )

        rows.append(
            {
                "season": int(season),
                "games": len(y_true),
                "home_win_rate": np.mean(y_true),
                "log_loss": metrics["log_loss"],
                "brier_score": metrics["brier_score"],
                "roc_auc": metrics["roc_auc"],
                "accuracy": metrics["accuracy"],
                "balanced_accuracy":
                    metrics["balanced_accuracy"],
            }
        )

    return pd.DataFrame(rows)


def print_season_performance(
    season_performance,
):

    print_section(
        "SEASON-BY-SEASON PERFORMANCE"
    )

    print(
        f"{'Season':<10}"
        f"{'Games':>7}"
        f"{'Home Win':>11}"
        f"{'Log Loss':>12}"
        f"{'Brier':>10}"
        f"{'AUC':>10}"
        f"{'Accuracy':>12}"
    )

    print("-" * 72)

    for _, row in season_performance.iterrows():

        print(
            f"{int(row['season']):<10}"
            f"{int(row['games']):>7}"
            f"{row['home_win_rate']:>11.4f}"
            f"{row['log_loss']:>12.6f}"
            f"{row['brier_score']:>10.6f}"
            f"{row['roc_auc']:>10.6f}"
            f"{row['accuracy']:>12.4%}"
        )


# ============================================================================
# PREDICTION STABILITY
# ============================================================================

def calculate_prediction_stability(
    validation_probability,
    test_probability,
):

    percentiles = [
        1,
        5,
        10,
        25,
        50,
        75,
        90,
        95,
        99,
    ]

    rows = []

    for split_name, probability in [
        ("validation", validation_probability),
        ("test", test_probability),
    ]:

        row = {
            "split": split_name,
            "mean": np.mean(probability),
            "std": np.std(probability),
            "min": np.min(probability),
            "max": np.max(probability),
        }

        for percentile in percentiles:

            row[
                f"p{percentile:02d}"
            ] = np.percentile(
                probability,
                percentile,
            )

        rows.append(row)

    return pd.DataFrame(rows)


def print_prediction_stability(
    stability,
):

    print_section(
        "PREDICTION STABILITY"
    )

    for _, row in stability.iterrows():

        print(
            f"{row['split'].capitalize()}"
        )

        print(
            f"  Mean: {row['mean']:.6f}"
        )

        print(
            f"  Std:  {row['std']:.6f}"
        )

        print(
            f"  P01:  {row['p01']:.6f}"
        )

        print(
            f"  P05:  {row['p05']:.6f}"
        )

        print(
            f"  P10:  {row['p10']:.6f}"
        )

        print(
            f"  P25:  {row['p25']:.6f}"
        )

        print(
            f"  P50:  {row['p50']:.6f}"
        )

        print(
            f"  P75:  {row['p75']:.6f}"
        )

        print(
            f"  P90:  {row['p90']:.6f}"
        )

        print(
            f"  P95:  {row['p95']:.6f}"
        )

        print(
            f"  P99:  {row['p99']:.6f}"
        )

        print(
            f"  Min:  {row['min']:.6f}"
        )

        print(
            f"  Max:  {row['max']:.6f}"
        )

        print()


# ============================================================================
# AUDIT SUMMARY
# ============================================================================

def create_audit_summary(
    validation_metrics,
    test_metrics,
    validation_distribution,
    test_distribution,
    feature_count,
):

    return pd.DataFrame(
        [
            {
                "model":
                    "random_forest_model_2",

                "feature_count":
                    feature_count,

                "validation_log_loss":
                    validation_metrics["log_loss"],

                "validation_brier_score":
                    validation_metrics["brier_score"],

                "validation_auc":
                    validation_metrics["roc_auc"],

                "validation_accuracy":
                    validation_metrics["accuracy"],

                "validation_balanced_accuracy":
                    validation_metrics["balanced_accuracy"],

                "validation_precision":
                    validation_metrics["precision"],

                "validation_recall":
                    validation_metrics["recall"],

                "validation_probability_mean":
                    validation_distribution["mean"],

                "validation_probability_std":
                    validation_distribution["std"],

                "validation_probability_min":
                    validation_distribution["min"],

                "validation_probability_max":
                    validation_distribution["max"],

                "test_log_loss":
                    test_metrics["log_loss"],

                "test_brier_score":
                    test_metrics["brier_score"],

                "test_auc":
                    test_metrics["roc_auc"],

                "test_accuracy":
                    test_metrics["accuracy"],

                "test_balanced_accuracy":
                    test_metrics["balanced_accuracy"],

                "test_precision":
                    test_metrics["precision"],

                "test_recall":
                    test_metrics["recall"],

                "test_probability_mean":
                    test_distribution["mean"],

                "test_probability_std":
                    test_distribution["std"],

                "test_probability_min":
                    test_distribution["min"],

                "test_probability_max":
                    test_distribution["max"],
            }
        ]
    )


# ============================================================================
# SAVE OUTPUTS
# ============================================================================

def save_audit_outputs(
    audit_summary,
    feature_importance,
    permutation_importance_df,
    family_importance,
    validation_calibration,
    test_calibration,
    validation_season_performance,
    test_season_performance,
    prediction_stability,
    missingness,
):

    AUDIT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    audit_summary.to_csv(
        AUDIT_DIR
        / "audit_summary.csv",
        index=False,
    )

    feature_importance.to_csv(
        AUDIT_DIR
        / "feature_importance.csv",
        index=False,
    )

    permutation_importance_df.to_csv(
        AUDIT_DIR
        / "permutation_importance.csv",
        index=False,
    )

    family_importance.to_csv(
        AUDIT_DIR
        / "feature_importance_by_family.csv",
        index=False,
    )

    # Combine validation and test calibration into one file.
    validation_calibration = (
        validation_calibration.copy()
    )

    test_calibration = (
        test_calibration.copy()
    )

    validation_calibration.insert(
        0,
        "split",
        "validation",
    )

    test_calibration.insert(
        0,
        "split",
        "test",
    )

    calibration = pd.concat(
        [
            validation_calibration,
            test_calibration,
        ],
        ignore_index=True,
    )

    calibration.to_csv(
        AUDIT_DIR
        / "calibration.csv",
        index=False,
    )

    validation_season_performance = (
        validation_season_performance.copy()
    )

    validation_season_performance.insert(
        0,
        "split",
        "validation",
    )

    test_season_performance = (
        test_season_performance.copy()
    )

    test_season_performance.insert(
        0,
        "split",
        "test",
    )

    season_performance = pd.concat(
        [
            validation_season_performance,
            test_season_performance,
        ],
        ignore_index=True,
    )

    season_performance.to_csv(
        AUDIT_DIR
        / "season_performance.csv",
        index=False,
    )

    prediction_stability.to_csv(
        AUDIT_DIR
        / "prediction_stability.csv",
        index=False,
    )

    missingness.to_csv(
        AUDIT_DIR
        / "missingness.csv",
        index=False,
    )

    print_section(
        "AUDIT OUTPUTS SAVED"
    )

    output_files = [
        "audit_summary.csv",
        "feature_importance.csv",
        "permutation_importance.csv",
        "feature_importance_by_family.csv",
        "calibration.csv",
        "season_performance.csv",
        "prediction_stability.csv",
        "missingness.csv",
    ]

    for filename in output_files:

        print(
            f"  {AUDIT_DIR / filename}"
        )


# ============================================================================
# MAIN
# ============================================================================

def main():

    print()
    print("=" * 80)
    print(
        "RANDOM FOREST MODEL 2 — "
        "DIAGNOSTIC / STABILITY AUDIT"
    )
    print("=" * 80)

    # ========================================================================
    # PROJECT INFORMATION
    # ========================================================================

    print()
    print(
        "Project root:"
    )

    print(
        f"  {PROJECT_ROOT}"
    )

    print()
    print(
        "Model directory:"
    )

    print(
        f"  {MODEL_DIR}"
    )

    print()
    print(
        "Audit directory:"
    )

    print(
        f"  {AUDIT_DIR}"
    )

    # ========================================================================
    # LOAD DATA
    # ========================================================================

    print_section(
        "LOADING BASE MODEL INPUTS"
    )

    train = load_split(
        TRAIN_PATH,
        "Training",
    )

    validation = load_split(
        VALIDATION_PATH,
        "Validation",
    )

    test = load_split(
        TEST_PATH,
        "Test",
    )

    # ========================================================================
    # VALIDATE BASE SPLITS
    # ========================================================================

    validate_splits(
        train,
        validation,
        test,
    )

    # ========================================================================
    # LOAD SAVED MODEL
    # ========================================================================

    model = load_model()

    # ========================================================================
    # LOAD SAVED FEATURE LIST
    # ========================================================================

    features = load_feature_list()

    # ========================================================================
    # RECONSTRUCT MODEL 2 INPUTS
    # ========================================================================

    train = add_enhanced_features(
        train,
        "Training",
    )

    validation = add_enhanced_features(
        validation,
        "Validation",
    )

    test = add_enhanced_features(
        test,
        "Test",
    )

    # ========================================================================
    # VALIDATE FINAL FEATURES
    # ========================================================================

    validate_features(
        train,
        validation,
        test,
        features,
    )

    # ========================================================================
    # PREPARE X / Y
    # ========================================================================

    print_section(
        "PREPARING MODEL 2 DATA"
    )

    X_train = prepare_features(
        train,
        features,
    )

    X_validation = prepare_features(
        validation,
        features,
    )

    X_test = prepare_features(
        test,
        features,
    )

    y_train = (
        train[TARGET_COLUMN]
        .astype(int)
        .values
    )

    y_validation = (
        validation[TARGET_COLUMN]
        .astype(int)
        .values
    )

    y_test = (
        test[TARGET_COLUMN]
        .astype(int)
        .values
    )

    print(
        f"Training:   {X_train.shape}"
    )

    print(
        f"Validation: {X_validation.shape}"
    )

    print(
        f"Test:       {X_test.shape}"
    )

    # ========================================================================
    # MISSINGNESS
    # ========================================================================

    missingness = print_missingness(
        X_train,
        X_validation,
        X_test,
        features,
    )

    # ========================================================================
    # GENERATE PREDICTIONS
    # ========================================================================

    print_section(
        "GENERATING PREDICTIONS"
    )

    validation_probability = (
        model
        .predict_proba(
            X_validation
        )[:, 1]
    )

    test_probability = (
        model
        .predict_proba(
            X_test
        )[:, 1]
    )

    print(
        "Validation predictions generated."
    )

    print(
        "Test predictions generated."
    )

    # ========================================================================
    # PERFORMANCE
    # ========================================================================

    validation_metrics, validation_matrix = (
        print_metrics(
            "Validation",
            y_validation,
            validation_probability,
        )
    )

    test_metrics, test_matrix = (
        print_metrics(
            "Test",
            y_test,
            test_probability,
        )
    )

    # ========================================================================
    # PREDICTION DISTRIBUTION
    # ========================================================================

    print_section(
        "PREDICTION DISTRIBUTION"
    )

    validation_distribution = (
        print_prediction_distribution(
            "Validation",
            validation_probability,
        )
    )

    test_distribution = (
        print_prediction_distribution(
            "Test",
            test_probability,
        )
    )

    # ========================================================================
    # IMPURITY IMPORTANCE
    # ========================================================================

    feature_importance = (
        calculate_impurity_importance(
            model,
            features,
        )
    )

    print_impurity_importance(
        feature_importance,
    )

    # ========================================================================
    # PERMUTATION IMPORTANCE
    # ========================================================================

    permutation_importance_df = (
        calculate_permutation_importance(
            model,
            X_validation,
            y_validation,
            features,
        )
    )

    # ========================================================================
    # FEATURE FAMILY IMPORTANCE
    # ========================================================================

    feature_importance_with_family = (
        add_feature_family(
            feature_importance,
        )
    )

    family_importance = (
        calculate_family_importance(
            feature_importance_with_family,
        )
    )

    print_family_importance(
        family_importance,
    )

    # ========================================================================
    # CALIBRATION
    # ========================================================================

    validation_calibration = (
        calculate_calibration(
            y_validation,
            validation_probability,
        )
    )

    test_calibration = (
        calculate_calibration(
            y_test,
            test_probability,
        )
    )

    print_calibration(
        "Validation",
        validation_calibration,
    )

    print_calibration(
        "Test",
        test_calibration,
    )

    # ========================================================================
    # SEASON PERFORMANCE
    # ========================================================================

    validation_season_performance = (
        calculate_season_performance(
            validation,
            validation_probability,
        )
    )

    test_season_performance = (
        calculate_season_performance(
            test,
            test_probability,
        )
    )

    print_season_performance(
        validation_season_performance,
    )

    print_season_performance(
        test_season_performance,
    )

    # ========================================================================
    # PREDICTION STABILITY
    # ========================================================================

    prediction_stability = (
        calculate_prediction_stability(
            validation_probability,
            test_probability,
        )
    )

    print_prediction_stability(
        prediction_stability,
    )

    # ========================================================================
    # AUDIT SUMMARY
    # ========================================================================

    audit_summary = create_audit_summary(
        validation_metrics,
        test_metrics,
        validation_distribution,
        test_distribution,
        len(features),
    )

    # ========================================================================
    # SAVE OUTPUTS
    # ========================================================================

    save_audit_outputs(
        audit_summary,
        feature_importance_with_family,
        permutation_importance_df,
        family_importance,
        validation_calibration,
        test_calibration,
        validation_season_performance,
        test_season_performance,
        prediction_stability,
        missingness,
    )

    # ========================================================================
    # FINAL SUMMARY
    # ========================================================================

    print_section(
        "RANDOM FOREST MODEL 2 — AUDIT COMPLETE"
    )

    print()
    print(
        f"Model 2 predictors: "
        f"{len(features)}"
    )

    print()
    print(
        "Validation:"
    )

    print(
        f"  Log Loss:    "
        f"{validation_metrics['log_loss']:.6f}"
    )

    print(
        f"  Brier Score: "
        f"{validation_metrics['brier_score']:.6f}"
    )

    print(
        f"  ROC AUC:     "
        f"{validation_metrics['roc_auc']:.6f}"
    )

    print(
        f"  Accuracy:    "
        f"{validation_metrics['accuracy']:.4%}"
    )

    print()
    print(
        "Test:"
    )

    print(
        f"  Log Loss:    "
        f"{test_metrics['log_loss']:.6f}"
    )

    print(
        f"  Brier Score: "
        f"{test_metrics['brier_score']:.6f}"
    )

    print(
        f"  ROC AUC:     "
        f"{test_metrics['roc_auc']:.6f}"
    )

    print(
        f"  Accuracy:    "
        f"{test_metrics['accuracy']:.4%}"
    )

    print()
    print(
        "Audit complete."
    )

    print()
    print(
        "The test set should not be used to modify the Model 2 "
        "feature set."
    )

    print()
    print(
        "Review the saved audit outputs before proceeding "
        "to Model 3."
    )


if __name__ == "__main__":
    main()