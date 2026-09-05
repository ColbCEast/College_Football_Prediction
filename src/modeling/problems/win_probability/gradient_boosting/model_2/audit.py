"""
Gradient Boosting Win Probability - Model 2 Audit
=================================================

Audits the compact 44-feature Gradient Boosting Model 2.

Model 2 uses the same temporal evaluation framework as Model 1,
but reduces the predictor space from 310 features to 44 selected
features, including 16 engineered trend/SOS features.

The audit is diagnostic only:
    - Does not retrain the model
    - Does not tune hyperparameters
    - Does not modify predictions
    - Uses the fixed temporal validation/test splits

Diagnostics:
    1. Dataset and prediction validation
    2. Enhanced feature construction validation
    3. Saved model / pipeline validation
    4. Feature missingness
    5. Impurity feature importance
    6. Validation permutation importance
    7. Feature-family importance
    8. Calibration
    9. Test-season performance
    10. Prediction stability
    11. Audit summary

Primary metric:
    Log Loss

Secondary metrics:
    Brier Score
    ROC AUC
    Accuracy
    Balanced Accuracy
    Precision
    Recall
"""

from pathlib import Path
import warnings

import joblib
import numpy as np
import pandas as pd

from sklearn.ensemble import GradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    brier_score_loss,
    log_loss,
    precision_score,
    recall_score,
    roc_auc_score,
)


# =============================================================================
# PATHS
# =============================================================================

PROJECT_ROOT = Path(__file__).resolve().parents[6]

MODEL_DIR = (
    PROJECT_ROOT
    / "models"
    / "win_probability"
    / "gradient_boosting"
    / "model_2"
)

AUDIT_DIR = MODEL_DIR / "audit"

MODEL_PATH = MODEL_DIR / "model.joblib"
FEATURE_LIST_PATH = MODEL_DIR / "feature_list.csv"
TRAINING_SUMMARY_PATH = MODEL_DIR / "training_summary.csv"

TRAIN_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "model_inputs"
    / "win_probability"
    / "train.csv"
)

VALIDATION_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "model_inputs"
    / "win_probability"
    / "validation.csv"
)

TEST_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "model_inputs"
    / "win_probability"
    / "test.csv"
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


# =============================================================================
# EXPECTED MODEL CONFIGURATION
# =============================================================================

EXPECTED_FEATURE_COUNT = 44

EXPECTED_PARAMS = {
    "n_estimators": 500,
    "learning_rate": 0.05,
    "max_depth": 3,
    "min_samples_leaf": 5,
    "subsample": 1.0,
    "random_state": 42,
}


# =============================================================================
# ENGINEERED FEATURES
# =============================================================================

ENGINEERED_FEATURES = [
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
    "priorSOSWinPct_home",
    "priorSOSWinPct_away",
    "priorSOSPointDiff_home",
    "priorSOSPointDiff_away",
]


# =============================================================================
# MODEL 2 FEATURE GROUPS
# =============================================================================

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


FEATURE_FAMILIES = {
    "Core Strength": CORE_STRENGTH_FEATURES,
    "Recent Form": RECENT_FORM_FEATURES,
    "Trend": TREND_FEATURES,
    "Prior SOS": SOS_FEATURES,
    "Offensive Efficiency": OFFENSIVE_FEATURES,
    "Defensive Efficiency": DEFENSIVE_FEATURES,
}


# =============================================================================
# HELPERS
# =============================================================================

def print_header(title):
    print("\n" + "=" * 78)
    print(title)
    print("=" * 78)


def validate_columns(df, required_columns, dataset_name):
    missing = [
        column
        for column in required_columns
        if column not in df.columns
    ]

    if missing:
        raise ValueError(
            f"{dataset_name} is missing {len(missing)} required columns:\n"
            + "\n".join(f"  - {column}" for column in missing)
        )


def calculate_metrics(y_true, probabilities, threshold=0.5):
    predictions = (probabilities >= threshold).astype(int)

    return {
        "log_loss": log_loss(y_true, probabilities),
        "brier_score": brier_score_loss(
            y_true,
            probabilities,
        ),
        "roc_auc": roc_auc_score(
            y_true,
            probabilities,
        ),
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
    }


def assign_feature_family(feature):
    for family, features in FEATURE_FAMILIES.items():
        if feature in features:
            return family

    return "Other"


def load_enhanced_features_for_seasons(seasons):
    """
    Load the enhanced feature files used by Model 2 training.

    Only gameId plus the 16 engineered features are retained.
    """

    frames = []

    for season in seasons:
        path = (
            ENHANCED_FEATURE_DIR
            / f"logistic_features_{season}.csv"
        )

        if not path.exists():
            raise FileNotFoundError(
                f"Enhanced feature file not found for season "
                f"{season}: {path}"
            )

        df = pd.read_csv(path)

        required = ["gameId"] + ENGINEERED_FEATURES

        validate_columns(
            df,
            required,
            f"Enhanced feature file for {season}",
        )

        df = df[required].copy()

        frames.append(df)

        print(
            f"  Loaded {season}: "
            f"{len(df):,} rows"
        )

    enhanced = pd.concat(
        frames,
        ignore_index=True,
    )

    duplicate_game_ids = enhanced["gameId"].duplicated().sum()

    if duplicate_game_ids > 0:
        raise ValueError(
            f"Enhanced feature data contains "
            f"{duplicate_game_ids} duplicate gameId values."
        )

    return enhanced


def merge_enhanced_features(base_df, enhanced_df, dataset_name):
    """
    Merge engineered features onto the base model-input dataset.
    """

    if "gameId" not in base_df.columns:
        raise ValueError(
            f"{dataset_name} is missing gameId."
        )

    base_game_ids = set(base_df["gameId"])
    enhanced_game_ids = set(enhanced_df["gameId"])

    missing_enhanced = base_game_ids - enhanced_game_ids

    if missing_enhanced:
        sample = list(missing_enhanced)[:10]

        raise ValueError(
            f"{dataset_name}: {len(missing_enhanced)} gameIds "
            f"are missing from the enhanced feature data. "
            f"Examples: {sample}"
        )

    merged = base_df.merge(
        enhanced_df,
        on="gameId",
        how="left",
        validate="one_to_one",
    )

    if len(merged) != len(base_df):
        raise ValueError(
            f"{dataset_name}: row count changed after "
            f"enhanced feature merge."
        )

    return merged


# =============================================================================
# INITIALIZATION
# =============================================================================

print_header(
    "GRADIENT BOOSTING WIN PROBABILITY - MODEL 2 AUDIT"
)

print("Loading model and datasets...")

if not MODEL_PATH.exists():
    raise FileNotFoundError(
        f"Model not found: {MODEL_PATH}"
    )

if not FEATURE_LIST_PATH.exists():
    raise FileNotFoundError(
        f"Feature list not found: {FEATURE_LIST_PATH}"
    )

if not TRAIN_PATH.exists():
    raise FileNotFoundError(
        f"Training data not found: {TRAIN_PATH}"
    )

if not VALIDATION_PATH.exists():
    raise FileNotFoundError(
        f"Validation data not found: {VALIDATION_PATH}"
    )

if not TEST_PATH.exists():
    raise FileNotFoundError(
        f"Test data not found: {TEST_PATH}"
    )

model = joblib.load(MODEL_PATH)

train = pd.read_csv(TRAIN_PATH)
validation = pd.read_csv(VALIDATION_PATH)
test = pd.read_csv(TEST_PATH)

feature_list_df = pd.read_csv(FEATURE_LIST_PATH)

print("Model and datasets loaded successfully.")


# =============================================================================
# DATASET VALIDATION
# =============================================================================

print_header("DATASET VALIDATION")

print(f"Training shape:   {train.shape}")
print(f"Validation shape: {validation.shape}")
print(f"Test shape:       {test.shape}")

EXPECTED_TRAIN_ROWS = 6432
EXPECTED_VALIDATION_ROWS = 1741
EXPECTED_TEST_ROWS = 888

if len(train) != EXPECTED_TRAIN_ROWS:
    raise ValueError(
        f"Unexpected training row count: {len(train)} "
        f"(expected {EXPECTED_TRAIN_ROWS})"
    )

if len(validation) != EXPECTED_VALIDATION_ROWS:
    raise ValueError(
        f"Unexpected validation row count: {len(validation)} "
        f"(expected {EXPECTED_VALIDATION_ROWS})"
    )

if len(test) != EXPECTED_TEST_ROWS:
    raise ValueError(
        f"Unexpected test row count: {len(test)} "
        f"(expected {EXPECTED_TEST_ROWS})"
    )

print("Row counts validated.")

for dataset_name, df in [
    ("Training", train),
    ("Validation", validation),
    ("Test", test),
]:
    if "win_home" not in df.columns:
        raise ValueError(
            f"{dataset_name} is missing target column "
            f"'win_home'."
        )

    unique_targets = set(
        df["win_home"].dropna().unique()
    )

    if not unique_targets.issubset({0, 1}):
        raise ValueError(
            f"{dataset_name} contains unexpected target "
            f"values: {unique_targets}"
        )

print("Target columns validated.")

train_years = sorted(train["season"].unique())
validation_years = sorted(validation["season"].unique())
test_years = sorted(test["season"].unique())

print(f"Training seasons:   {train_years}")
print(f"Validation seasons: {validation_years}")
print(f"Test seasons:       {test_years}")

if train_years != list(range(2015, 2023)):
    raise ValueError(
        f"Unexpected training seasons: {train_years}"
    )

if validation_years != [2023, 2024]:
    raise ValueError(
        f"Unexpected validation seasons: "
        f"{validation_years}"
    )

if test_years != [2025]:
    raise ValueError(
        f"Unexpected test seasons: {test_years}"
    )

print("Temporal split validated.")


# =============================================================================
# ENHANCED FEATURE MERGE
# =============================================================================

print_header("ENHANCED FEATURE VALIDATION")

all_seasons = (
    list(range(2015, 2023))
    + [2023, 2024]
    + [2025]
)

print(
    "Loading enhanced trend/SOS features..."
)

enhanced_features = load_enhanced_features_for_seasons(
    all_seasons
)

print(
    f"\nCombined enhanced feature shape: "
    f"{enhanced_features.shape}"
)

print(
    f"Engineered features available: "
    f"{len(ENGINEERED_FEATURES)}"
)

train = merge_enhanced_features(
    train,
    enhanced_features,
    "Training",
)

validation = merge_enhanced_features(
    validation,
    enhanced_features,
    "Validation",
)

test = merge_enhanced_features(
    test,
    enhanced_features,
    "Test",
)

print("\nEnhanced features merged successfully.")

print(
    f"Training shape after merge:   {train.shape}"
)

print(
    f"Validation shape after merge: {validation.shape}"
)

print(
    f"Test shape after merge:       {test.shape}"
)

validate_columns(
    train,
    MODEL_2_FEATURES,
    "Training dataset after enhanced merge",
)

validate_columns(
    validation,
    MODEL_2_FEATURES,
    "Validation dataset after enhanced merge",
)

validate_columns(
    test,
    MODEL_2_FEATURES,
    "Test dataset after enhanced merge",
)

print("\nAll 44 Model 2 features are present.")


# =============================================================================
# FEATURE LIST VALIDATION
# =============================================================================

print_header("FEATURE LIST VALIDATION")

if "feature" not in feature_list_df.columns:
    raise ValueError(
        "feature_list.csv does not contain the expected "
        "'feature' column."
    )

saved_features = (
    feature_list_df["feature"]
    .dropna()
    .tolist()
)

if saved_features != MODEL_2_FEATURES:
    saved_set = set(saved_features)
    expected_set = set(MODEL_2_FEATURES)

    missing = sorted(
        expected_set - saved_set
    )

    extra = sorted(
        saved_set - expected_set
    )

    print(
        "WARNING: Saved feature list differs "
        "from expected Model 2 feature list."
    )

    if missing:
        print("\nMissing from saved feature list:")

        for feature in missing:
            print(f"  - {feature}")

    if extra:
        print(
            "\nUnexpected features in saved feature list:"
        )

        for feature in extra:
            print(f"  - {feature}")

    raise ValueError(
        "Model 2 feature list validation failed."
    )

print(
    f"Model 2 feature count: "
    f"{len(saved_features)}"
)

if len(saved_features) != EXPECTED_FEATURE_COUNT:
    raise ValueError(
        f"Expected {EXPECTED_FEATURE_COUNT} features, "
        f"found {len(saved_features)}."
    )

print("Feature list validated successfully.")

print("\nFeature groups:")

for family, features in FEATURE_FAMILIES.items():
    print(
        f"  {family}: {len(features)}"
    )

print(
    f"  Total: {len(MODEL_2_FEATURES)}"
)


# =============================================================================
# MODEL / PIPELINE VALIDATION
# =============================================================================

print_header("MODEL / PIPELINE VALIDATION")

print(
    f"Model type: {type(model).__name__}"
)

if not hasattr(model, "named_steps"):
    raise ValueError(
        "Loaded model does not contain named pipeline steps."
    )

print("\nPipeline steps:")

for name, step in model.named_steps.items():
    print(
        f"  {name}: {type(step).__name__}"
    )

if "imputer" not in model.named_steps:
    raise ValueError(
        "Expected pipeline step 'imputer' not found."
    )

if "classifier" not in model.named_steps:
    raise ValueError(
        "Expected pipeline step 'classifier' not found."
    )

imputer = model.named_steps["imputer"]
classifier = model.named_steps["classifier"]

if not isinstance(imputer, SimpleImputer):
    raise ValueError(
        "Pipeline imputer is not SimpleImputer."
    )

if imputer.strategy != "median":
    raise ValueError(
        f"Expected median imputation, "
        f"found: {imputer.strategy}"
    )

if not isinstance(
    classifier,
    GradientBoostingClassifier,
):
    raise ValueError(
        "Pipeline classifier is not "
        "GradientBoostingClassifier."
    )

print("\nImputer:")
print(
    f"  Strategy: {imputer.strategy}"
)

print("\nGradient Boosting configuration:")

for parameter, expected_value in EXPECTED_PARAMS.items():

    actual_value = classifier.get_params()[
        parameter
    ]

    status = (
        "[OK]"
        if actual_value == expected_value
        else "[MISMATCH]"
    )

    print(
        f"  {parameter}: "
        f"{actual_value} {status}"
    )

    if actual_value != expected_value:
        raise ValueError(
            f"Unexpected classifier parameter "
            f"{parameter}: {actual_value} "
            f"(expected {expected_value})"
        )

print(
    "\nModel pipeline validated successfully."
)


# =============================================================================
# PREPARE DATA
# =============================================================================

X_train = train[
    MODEL_2_FEATURES
].copy()

y_train = train[
    "win_home"
].copy()

X_validation = validation[
    MODEL_2_FEATURES
].copy()

y_validation = validation[
    "win_home"
].copy()

X_test = test[
    MODEL_2_FEATURES
].copy()

y_test = test[
    "win_home"
].copy()


# =============================================================================
# FEATURE MISSINGNESS
# =============================================================================

print_header("FEATURE MISSINGNESS")

missingness = (
    X_train
    .isna()
    .mean()
    .sort_values(
        ascending=False
    )
    * 100
)

print(
    "Training missingness by feature:"
)

for feature, pct in missingness.items():

    if pct > 0:
        print(
            f"  {feature}: {pct:.2f}%"
        )

print(
    f"\nFeatures with missing values: "
    f"{(missingness > 0).sum()} / "
    f"{len(missingness)}"
)

print(
    "\nTotal validation missing values:",
    X_validation.isna().sum().sum(),
)

print(
    "Total test missing values:",
    X_test.isna().sum().sum(),
)

print(
    "\nNote: The saved pipeline performs "
    "median imputation using training-set "
    "statistics."
)


# =============================================================================
# PREDICTIONS
# =============================================================================

print_header("GENERATING PREDICTIONS")

with warnings.catch_warnings():

    warnings.simplefilter("ignore")

    validation_probabilities = (
        model.predict_proba(
            X_validation
        )[:, 1]
    )

    test_probabilities = (
        model.predict_proba(
            X_test
        )[:, 1]
    )

print(
    "Predictions generated successfully."
)


# =============================================================================
# PREDICTION VALIDATION
# =============================================================================

if not np.isfinite(
    validation_probabilities
).all():
    raise ValueError(
        "Validation predictions contain "
        "non-finite values."
    )

if not np.isfinite(
    test_probabilities
).all():
    raise ValueError(
        "Test predictions contain "
        "non-finite values."
    )

if (
    (validation_probabilities < 0).any()
    or
    (validation_probabilities > 1).any()
):
    raise ValueError(
        "Validation probabilities outside [0, 1]."
    )

if (
    (test_probabilities < 0).any()
    or
    (test_probabilities > 1).any()
):
    raise ValueError(
        "Test probabilities outside [0, 1]."
    )

print(
    "Prediction values validated."
)


# =============================================================================
# PERFORMANCE
# =============================================================================

print_header("MODEL PERFORMANCE")

validation_metrics = calculate_metrics(
    y_validation,
    validation_probabilities,
)

test_metrics = calculate_metrics(
    y_test,
    test_probabilities,
)

print("\nValidation performance:")

for metric, value in validation_metrics.items():

    if metric == "accuracy":
        print(
            f"  {metric}: {value:.4%}"
        )
    else:
        print(
            f"  {metric}: {value:.6f}"
        )

print("\nTest performance:")

for metric, value in test_metrics.items():

    if metric == "accuracy":
        print(
            f"  {metric}: {value:.4%}"
        )
    else:
        print(
            f"  {metric}: {value:.6f}"
        )


# =============================================================================
# IMPURITY FEATURE IMPORTANCE
# =============================================================================

print_header("IMPURITY FEATURE IMPORTANCE")

classifier_feature_importance = (
    classifier.feature_importances_
)

if len(classifier_feature_importance) != (
    len(MODEL_2_FEATURES)
):
    raise ValueError(
        "Feature importance length does not "
        "match Model 2 feature count."
    )

feature_importance = pd.DataFrame(
    {
        "feature": MODEL_2_FEATURES,
        "importance": (
            classifier_feature_importance
        ),
    }
).sort_values(
    "importance",
    ascending=False,
)

feature_importance["rank"] = range(
    1,
    len(feature_importance) + 1,
)

feature_importance = feature_importance[
    [
        "rank",
        "feature",
        "importance",
    ]
]

print("\nTop 20 features:")

print(
    feature_importance
    .head(20)
    .to_string(index=False)
)


# =============================================================================
# PERMUTATION IMPORTANCE
# =============================================================================

print_header(
    "VALIDATION PERMUTATION IMPORTANCE"
)

print(
    "Calculating permutation importance "
    "using validation Log Loss..."
)

with warnings.catch_warnings():

    warnings.simplefilter("ignore")

    permutation = permutation_importance(
        model,
        X_validation,
        y_validation,
        scoring="neg_log_loss",
        n_repeats=10,
        random_state=42,
        n_jobs=-1,
    )

permutation_importance_df = pd.DataFrame(
    {
        "feature": MODEL_2_FEATURES,
        "importance_mean": (
            permutation.importances_mean
        ),
        "importance_std": (
            permutation.importances_std
        ),
    }
).sort_values(
    "importance_mean",
    ascending=False,
)

permutation_importance_df["rank"] = range(
    1,
    len(permutation_importance_df) + 1,
)

permutation_importance_df = (
    permutation_importance_df[
        [
            "rank",
            "feature",
            "importance_mean",
            "importance_std",
        ]
    ]
)

print(
    "\nTop 20 validation permutation features:"
)

print(
    permutation_importance_df
    .head(20)
    .to_string(index=False)
)


# =============================================================================
# FEATURE FAMILY IMPORTANCE
# =============================================================================

print_header(
    "FEATURE FAMILY IMPORTANCE"
)

feature_importance_family = (
    feature_importance.copy()
)

feature_importance_family["family"] = (
    feature_importance_family[
        "feature"
    ].apply(assign_feature_family)
)

family_importance = (
    feature_importance_family
    .groupby(
        "family",
        as_index=False,
    )["importance"]
    .sum()
    .sort_values(
        "importance",
        ascending=False,
    )
)

family_importance[
    "importance_pct"
] = (
    family_importance[
        "importance"
    ]
    / family_importance[
        "importance"
    ].sum()
    * 100
)

print(
    family_importance
    .to_string(index=False)
)


# =============================================================================
# CALIBRATION
# =============================================================================

print_header("CALIBRATION")

calibration_df = pd.DataFrame(
    {
        "actual": (
            y_validation.to_numpy()
        ),
        "predicted_probability": (
            validation_probabilities
        ),
    }
)

calibration_df["bin"] = pd.cut(
    calibration_df[
        "predicted_probability"
    ],
    bins=np.arange(
        0,
        1.05,
        0.05,
    ),
    include_lowest=True,
)

calibration = (
    calibration_df
    .groupby(
        "bin",
        observed=False,
    )
    .agg(
        count=(
            "actual",
            "size",
        ),
        mean_predicted_probability=(
            "predicted_probability",
            "mean",
        ),
        actual_win_rate=(
            "actual",
            "mean",
        ),
    )
    .reset_index()
)

calibration[
    "calibration_error"
] = (
    calibration[
        "mean_predicted_probability"
    ]
    - calibration[
        "actual_win_rate"
    ]
)

calibration[
    "absolute_calibration_error"
] = (
    calibration[
        "calibration_error"
    ].abs()
)

print(
    calibration.to_string(
        index=False
    )
)


# =============================================================================
# TEST SEASON PERFORMANCE
# =============================================================================

print_header(
    "TEST SEASON PERFORMANCE"
)

season_results = []

test_with_predictions = test.copy()

test_with_predictions[
    "predicted_probability"
] = test_probabilities

for season, group in (
    test_with_predictions
    .groupby("season")
):

    y_true = group[
        "win_home"
    ]

    probabilities = group[
        "predicted_probability"
    ]

    predictions = (
        probabilities >= 0.5
    ).astype(int)

    season_results.append(
        {
            "season": season,
            "games": len(group),
            "home_win_rate": (
                y_true.mean()
            ),
            "mean_predicted_probability": (
                probabilities.mean()
            ),
            "log_loss": log_loss(
                y_true,
                probabilities,
            ),
            "brier_score": (
                brier_score_loss(
                    y_true,
                    probabilities,
                )
            ),
            "roc_auc": roc_auc_score(
                y_true,
                probabilities,
            ),
            "accuracy": accuracy_score(
                y_true,
                predictions,
            ),
            "balanced_accuracy": (
                balanced_accuracy_score(
                    y_true,
                    predictions,
                )
            ),
        }
    )

season_performance = pd.DataFrame(
    season_results
)

print(
    season_performance.to_string(
        index=False
    )
)


# =============================================================================
# PREDICTION STABILITY
# =============================================================================

print_header(
    "PREDICTION STABILITY"
)

stability = pd.DataFrame(
    {
        "dataset": [
            "validation",
            "test",
        ],
        "mean": [
            validation_probabilities.mean(),
            test_probabilities.mean(),
        ],
        "std": [
            validation_probabilities.std(),
            test_probabilities.std(),
        ],
        "min": [
            validation_probabilities.min(),
            test_probabilities.min(),
        ],
        "p01": [
            np.percentile(
                validation_probabilities,
                1,
            ),
            np.percentile(
                test_probabilities,
                1,
            ),
        ],
        "p05": [
            np.percentile(
                validation_probabilities,
                5,
            ),
            np.percentile(
                test_probabilities,
                5,
            ),
        ],
        "p10": [
            np.percentile(
                validation_probabilities,
                10,
            ),
            np.percentile(
                test_probabilities,
                10,
            ),
        ],
        "p25": [
            np.percentile(
                validation_probabilities,
                25,
            ),
            np.percentile(
                test_probabilities,
                25,
            ),
        ],
        "median": [
            np.median(
                validation_probabilities
            ),
            np.median(
                test_probabilities
            ),
        ],
        "p75": [
            np.percentile(
                validation_probabilities,
                75,
            ),
            np.percentile(
                test_probabilities,
                75,
            ),
        ],
        "p90": [
            np.percentile(
                validation_probabilities,
                90,
            ),
            np.percentile(
                test_probabilities,
                90,
            ),
        ],
        "p95": [
            np.percentile(
                validation_probabilities,
                95,
            ),
            np.percentile(
                test_probabilities,
                95,
            ),
        ],
        "p99": [
            np.percentile(
                validation_probabilities,
                99,
            ),
            np.percentile(
                test_probabilities,
                99,
            ),
        ],
        "max": [
            validation_probabilities.max(),
            test_probabilities.max(),
        ],
    }
)

print(
    stability.to_string(
        index=False
    )
)


# =============================================================================
# EXTREME PROBABILITY RATES
# =============================================================================

validation_extreme_low = (
    validation_probabilities < 0.10
).mean()

validation_extreme_high = (
    validation_probabilities > 0.90
).mean()

test_extreme_low = (
    test_probabilities < 0.10
).mean()

test_extreme_high = (
    test_probabilities > 0.90
).mean()

print(
    "\nExtreme probability rates:"
)

print(
    f"Validation P < 0.10: "
    f"{validation_extreme_low:.2%}"
)

print(
    f"Validation P > 0.90: "
    f"{validation_extreme_high:.2%}"
)

print(
    f"Test P < 0.10:       "
    f"{test_extreme_low:.2%}"
)

print(
    f"Test P > 0.90:       "
    f"{test_extreme_high:.2%}"
)


# =============================================================================
# AUDIT SUMMARY
# =============================================================================

print_header(
    "AUDIT SUMMARY"
)

summary = pd.DataFrame(
    [
        {
            "model": (
                "Gradient Boosting Model 2"
            ),
            "feature_count": (
                EXPECTED_FEATURE_COUNT
            ),
            "train_rows": len(train),
            "validation_rows": len(
                validation
            ),
            "test_rows": len(test),
            "validation_log_loss": (
                validation_metrics[
                    "log_loss"
                ]
            ),
            "validation_brier_score": (
                validation_metrics[
                    "brier_score"
                ]
            ),
            "validation_roc_auc": (
                validation_metrics[
                    "roc_auc"
                ]
            ),
            "validation_accuracy": (
                validation_metrics[
                    "accuracy"
                ]
            ),
            "test_log_loss": (
                test_metrics[
                    "log_loss"
                ]
            ),
            "test_brier_score": (
                test_metrics[
                    "brier_score"
                ]
            ),
            "test_roc_auc": (
                test_metrics[
                    "roc_auc"
                ]
            ),
            "test_accuracy": (
                test_metrics[
                    "accuracy"
                ]
            ),
            "validation_mean_probability": (
                validation_probabilities.mean()
            ),
            "test_mean_probability": (
                test_probabilities.mean()
            ),
            "validation_std_probability": (
                validation_probabilities.std()
            ),
            "test_std_probability": (
                test_probabilities.std()
            ),
        }
    ]
)

print(
    summary.to_string(
        index=False
    )
)


# =============================================================================
# SAVE AUDIT OUTPUTS
# =============================================================================

print_header(
    "SAVING AUDIT OUTPUTS"
)

AUDIT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

output_files = {
    "audit_summary.csv": summary,
    "feature_importance.csv": (
        feature_importance
    ),
    "permutation_importance.csv": (
        permutation_importance_df
    ),
    "feature_importance_by_family.csv": (
        family_importance
    ),
    "calibration.csv": calibration,
    "season_performance.csv": (
        season_performance
    ),
    "prediction_stability.csv": (
        stability
    ),
}

for filename, dataframe in (
    output_files.items()
):

    output_path = (
        AUDIT_DIR / filename
    )

    dataframe.to_csv(
        output_path,
        index=False,
    )

    print(
        f"Saved: {output_path}"
    )


# =============================================================================
# COMPLETION
# =============================================================================

print_header(
    "AUDIT COMPLETE"
)

print(
    "Gradient Boosting Model 2 audit "
    "completed successfully."
)

print(
    "\nAudit directory:"
)

print(AUDIT_DIR)

print(
    "\nGenerated files:"
)

for filename in output_files:
    print(f"  - {filename}")

print(
    "\nNo model training or tuning was performed."
)

print(
    "The test set was used only for "
    "final diagnostic evaluation."
)