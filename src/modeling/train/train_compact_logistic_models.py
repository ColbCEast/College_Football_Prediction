"""
Train and evaluate compact logistic regression models.

Models
------
Model 1:
    Core matchup-strength differences.

Model 2:
    Model 1 + recent-form differences.

Model 3:
    Model 2 + prior strength-of-schedule differences.

Temporal split
--------------
Training:
    2015-2022

Validation:
    2023-2024

Test:
    2025

The test season is used only once for final evaluation.
"""

from pathlib import Path
import warnings

import joblib
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    brier_score_loss,
    log_loss,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


# ============================================================================
# CONFIGURATION
# ============================================================================

RANDOM_STATE = 42

TARGET_COLUMN = "win_home"

TRAIN_YEARS = list(range(2015, 2023))
VALIDATION_YEARS = [2023, 2024]
TEST_YEARS = [2025]

PROJECT_ROOT = Path(__file__).resolve().parents[3]

INPUT_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "logistic_enhanced_features"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "models"
    / "compact_logistic"
)

MODEL_DIR = OUTPUT_DIR / "models"
PREDICTION_DIR = OUTPUT_DIR / "predictions"

SUMMARY_PATH = OUTPUT_DIR / "compact_logistic_comparison.csv"


# ============================================================================
# FEATURE DEFINITIONS
# ============================================================================

MODEL_1_FEATURES = {
    "winPctDiff": (
        "winPctBefore_home",
        "winPctBefore_away",
        1,
    ),

    "pointDifferentialAvgDiff": (
        "pointDifferentialAvgBefore_home",
        "pointDifferentialAvgBefore_away",
        1,
    ),

    "pointsForAvgDiff": (
        "pointsForAvgBefore_home",
        "pointsForAvgBefore_away",
        1,
    ),

    "pointsAgainstAvgDiff": (
        "pointsAgainstAvgBefore_home",
        "pointsAgainstAvgBefore_away",
        -1,
    ),

    "yardsPerPassAttemptDiff": (
        "yardsPerPassAttemptBefore_home",
        "yardsPerPassAttemptBefore_away",
        1,
    ),

    "yardsPerRushAttemptDiff": (
        "yardsPerRushAttemptBefore_home",
        "yardsPerRushAttemptBefore_away",
        1,
    ),
}


MODEL_2_FEATURES = {
    "pointsForTrendDiff": (
        "pointsForTrend_home",
        "pointsForTrend_away",
        1,
    ),

    "pointsAgainstTrendDiff": (
        "pointsAgainstTrend_home",
        "pointsAgainstTrend_away",
        -1,
    ),

    "pointDifferentialTrendDiff": (
        "pointDifferentialTrend_home",
        "pointDifferentialTrend_away",
        1,
    ),

    "totalYardsTrendDiff": (
        "totalYardsTrend_home",
        "totalYardsTrend_away",
        1,
    ),

    "netPassingYardsTrendDiff": (
        "netPassingYardsTrend_home",
        "netPassingYardsTrend_away",
        1,
    ),

    "winPctTrendDiff": (
        "winPctTrend_home",
        "winPctTrend_away",
        1,
    ),
}


MODEL_3_FEATURES = {
    "priorSOSWinPctDiff": (
        "priorSOSWinPct_home",
        "priorSOSWinPct_away",
        1,
    ),

    "priorSOSPointDiffDiff": (
        "priorSOSPointDiff_home",
        "priorSOSPointDiff_away",
        1,
    ),
}


# ============================================================================
# MODEL DEFINITIONS
# ============================================================================

MODEL_FEATURES = {
    "Model 1": {
        **MODEL_1_FEATURES,
    },

    "Model 2": {
        **MODEL_1_FEATURES,
        **MODEL_2_FEATURES,
    },

    "Model 3": {
        **MODEL_1_FEATURES,
        **MODEL_2_FEATURES,
        **MODEL_3_FEATURES,
    },
}


# ============================================================================
# OUTPUT / PRINT HELPERS
# ============================================================================

def print_section(title):
    print()
    print("=" * 78)
    print(title)
    print("=" * 78)


def fail(message):
    raise ValueError(f"\nVALIDATION FAILED:\n{message}")


# ============================================================================
# DATA LOADING
# ============================================================================

def load_season(year):
    """
    Load one enhanced feature dataset.
    """

    path = INPUT_DIR / f"logistic_features_{year}.csv"

    if not path.exists():
        fail(
            f"Missing enhanced feature file for {year}:\n"
            f"{path}"
        )

    df = pd.read_csv(path)

    if TARGET_COLUMN not in df.columns:
        fail(
            f"{year}: target column '{TARGET_COLUMN}' "
            f"is missing."
        )

    return df


def load_all_data():
    """
    Load all seasons and combine them into one dataframe.
    """

    print_section("LOADING ENHANCED FEATURE DATA")

    years = (
        TRAIN_YEARS
        + VALIDATION_YEARS
        + TEST_YEARS
    )

    frames = []

    for year in years:

        df = load_season(year)

        df["model_year"] = year

        frames.append(df)

        print(
            f"{year}: "
            f"{len(df):,} rows × "
            f"{len(df.columns):,} columns"
        )

    data = pd.concat(
        frames,
        ignore_index=True,
    )

    print()
    print(
        f"Combined data: "
        f"{len(data):,} rows × "
        f"{len(data.columns):,} columns"
    )

    return data


# ============================================================================
# FEATURE CONSTRUCTION
# ============================================================================

def construct_compact_features(
    data,
    feature_definition,
):
    """
    Construct home-minus-away matchup features.

    The third element of each definition tuple determines the sign:

        +1:
            home - away

        -1:
            away - home

    This guarantees that a positive feature value consistently means
    "better for the home team."
    """

    result = pd.DataFrame(
        index=data.index
    )

    for feature_name, definition in feature_definition.items():

        home_column = definition[0]
        away_column = definition[1]
        direction = definition[2]

        if home_column not in data.columns:
            fail(
                f"Missing source feature: "
                f"{home_column}"
            )

        if away_column not in data.columns:
            fail(
                f"Missing source feature: "
                f"{away_column}"
            )

        home = pd.to_numeric(
            data[home_column],
            errors="coerce",
        )

        away = pd.to_numeric(
            data[away_column],
            errors="coerce",
        )

        difference = (
            home - away
        )

        result[feature_name] = (
            difference * direction
        )

    return result


# ============================================================================
# FEATURE VALIDATION
# ============================================================================

def validate_feature_sets(data):
    """
    Confirm all source columns required by Models 1-3 exist.
    """

    print_section("VALIDATING MODEL FEATURE DEFINITIONS")

    for model_name, feature_definition in MODEL_FEATURES.items():

        constructed = construct_compact_features(
            data,
            feature_definition,
        )

        expected_count = len(
            feature_definition
        )

        actual_count = len(
            constructed.columns
        )

        if actual_count != expected_count:
            fail(
                f"{model_name}: expected "
                f"{expected_count} features but constructed "
                f"{actual_count}."
            )

        print(
            f"{model_name}: "
            f"{actual_count} features — VALID"
        )


# ============================================================================
# TEMPORAL SPLIT
# ============================================================================

def split_data(data):
    """
    Create the fixed temporal train/validation/test split.
    """

    print_section("TEMPORAL SPLIT")

    train = data[
        data["model_year"].isin(TRAIN_YEARS)
    ].copy()

    validation = data[
        data["model_year"].isin(VALIDATION_YEARS)
    ].copy()

    test = data[
        data["model_year"].isin(TEST_YEARS)
    ].copy()

    print(
        f"Training seasons:   "
        f"{TRAIN_YEARS[0]}-{TRAIN_YEARS[-1]}"
    )

    print(
        f"Validation seasons: "
        f"{VALIDATION_YEARS}"
    )

    print(
        f"Test seasons:       "
        f"{TEST_YEARS}"
    )

    print()

    print(
        f"Training rows:   {len(train):,}"
    )

    print(
        f"Validation rows: {len(validation):,}"
    )

    print(
        f"Test rows:       {len(test):,}"
    )

    if train.empty:
        fail("Training dataset is empty.")

    if validation.empty:
        fail("Validation dataset is empty.")

    if test.empty:
        fail("Test dataset is empty.")

    return train, validation, test


# ============================================================================
# MODEL PIPELINE
# ============================================================================

def build_pipeline(feature_names):
    """
    Create the standardized logistic regression pipeline.

    Missing values are median-imputed using training data only.
    """

    preprocessing = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(
                    strategy="median"
                ),
            ),
            (
                "scaler",
                StandardScaler(),
            ),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "numeric",
                preprocessing,
                feature_names,
            ),
        ],
        remainder="drop",
    )

    model = LogisticRegression(
        max_iter=2000,
        random_state=RANDOM_STATE,
    )

    pipeline = Pipeline(
        steps=[
            (
                "preprocessor",
                preprocessor,
            ),
            (
                "model",
                model,
            ),
        ]
    )

    return pipeline


# ============================================================================
# METRICS
# ============================================================================

def calculate_metrics(
    y_true,
    probability,
):
    """
    Calculate classification and probability metrics.
    """

    prediction = (
        probability >= 0.5
    ).astype(int)

    return {
        "auc": roc_auc_score(
            y_true,
            probability,
        ),

        "log_loss": log_loss(
            y_true,
            probability,
        ),

        "brier_score": brier_score_loss(
            y_true,
            probability,
        ),

        "accuracy": accuracy_score(
            y_true,
            prediction,
        ),

        "home_win_rate": float(
            np.mean(y_true)
        ),

        "mean_predicted_home_win_prob": float(
            np.mean(probability)
        ),
    }


# ============================================================================
# COEFFICIENT EXTRACTION
# ============================================================================

def extract_coefficients(
    pipeline,
    feature_names,
):
    """
    Extract logistic regression coefficients after preprocessing.

    Because all compact features are standardized, coefficients represent
    the change in log-odds associated with a one-standard-deviation increase
    in the matchup feature.
    """

    model = pipeline.named_steps[
        "model"
    ]

    coefficients = (
        model.coef_[0]
    )

    rows = []

    for feature, coefficient in zip(
        feature_names,
        coefficients,
    ):

        rows.append(
            {
                "feature": feature,
                "coefficient_per_1sd": float(
                    coefficient
                ),
                "odds_ratio_per_1sd": float(
                    np.exp(coefficient)
                ),
            }
        )

    return pd.DataFrame(rows)


# ============================================================================
# MODEL TRAINING
# ============================================================================

def train_model(
    model_name,
    feature_definition,
    train,
    validation,
    test,
):
    """
    Train one compact logistic model and evaluate it on validation and test.
    """

    print_section(
        f"TRAINING {model_name.upper()}"
    )

    feature_names = list(
        feature_definition.keys()
    )

    print(
        f"Feature count: {len(feature_names)}"
    )

    print(
        "Features:"
    )

    for feature in feature_names:
        print(f"  {feature}")

    # ----------------------------------------------------------------------
    # Construct matchup features
    # ----------------------------------------------------------------------

    X_train = construct_compact_features(
        train,
        feature_definition,
    )

    X_validation = construct_compact_features(
        validation,
        feature_definition,
    )

    X_test = construct_compact_features(
        test,
        feature_definition,
    )

    y_train = train[
        TARGET_COLUMN
    ].astype(int)

    y_validation = validation[
        TARGET_COLUMN
    ].astype(int)

    y_test = test[
        TARGET_COLUMN
    ].astype(int)

    # ----------------------------------------------------------------------
    # Missingness diagnostics
    # ----------------------------------------------------------------------

    print()

    print(
        "Training missingness:"
    )

    for feature in feature_names:

        pct = (
            X_train[feature]
            .isna()
            .mean()
            * 100
        )

        print(
            f"  {feature:<35} "
            f"{pct:7.3f}%"
        )

    # ----------------------------------------------------------------------
    # Build and fit model
    # ----------------------------------------------------------------------

    pipeline = build_pipeline(
        feature_names
    )

    pipeline.fit(
        X_train,
        y_train,
    )

    # ----------------------------------------------------------------------
    # Predictions
    # ----------------------------------------------------------------------

    train_probability = pipeline.predict_proba(
        X_train
    )[:, 1]

    validation_probability = (
        pipeline.predict_proba(
            X_validation
        )[:, 1]
    )

    test_probability = (
        pipeline.predict_proba(
            X_test
        )[:, 1]
    )

    # ----------------------------------------------------------------------
    # Metrics
    # ----------------------------------------------------------------------

    train_metrics = calculate_metrics(
        y_train,
        train_probability,
    )

    validation_metrics = calculate_metrics(
        y_validation,
        validation_probability,
    )

    test_metrics = calculate_metrics(
        y_test,
        test_probability,
    )

    print()
    print(
        "PERFORMANCE"
    )

    metrics_table = pd.DataFrame(
        [
            {
                "dataset": "train",
                **train_metrics,
            },
            {
                "dataset": "validation",
                **validation_metrics,
            },
            {
                "dataset": "test",
                **test_metrics,
            },
        ]
    )

    print(
        metrics_table.to_string(
            index=False,
            formatters={
                "auc": "{:.4f}".format,
                "log_loss": "{:.4f}".format,
                "brier_score": "{:.4f}".format,
                "accuracy": "{:.4f}".format,
                "home_win_rate": "{:.4f}".format,
                "mean_predicted_home_win_prob":
                    "{:.4f}".format,
            },
        )
    )

    # ----------------------------------------------------------------------
    # Coefficients
    # ----------------------------------------------------------------------

    coefficients = extract_coefficients(
        pipeline,
        feature_names,
    )

    print()
    print(
        "COEFFICIENTS"
    )

    print(
        coefficients.to_string(
            index=False,
            formatters={
                "coefficient_per_1sd":
                    "{:.4f}".format,
                "odds_ratio_per_1sd":
                    "{:.4f}".format,
            },
        )
    )

    # ----------------------------------------------------------------------
    # Save model
    # ----------------------------------------------------------------------

    MODEL_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    model_path = (
        MODEL_DIR
        / f"{model_name.lower().replace(' ', '_')}.joblib"
    )

    joblib.dump(
        pipeline,
        model_path,
    )

    print()
    print(
        f"Saved model: {model_path}"
    )

    # ----------------------------------------------------------------------
    # Save predictions
    # ----------------------------------------------------------------------

    PREDICTION_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    prediction_rows = []

    for split_name, df, probability in [
        (
            "train",
            train,
            train_probability,
        ),
        (
            "validation",
            validation,
            validation_probability,
        ),
        (
            "test",
            test,
            test_probability,
        ),
    ]:

        prediction = (
            probability >= 0.5
        ).astype(int)

        output = pd.DataFrame(
            {
                "model": model_name,
                "dataset": split_name,
                "season": df["model_year"].values,
                "actual_win_home": df[
                    TARGET_COLUMN
                ].values,
                "predicted_home_win_probability":
                    probability,
                "predicted_win_home":
                    prediction,
            }
        )

        prediction_rows.append(
            output
        )

    predictions = pd.concat(
        prediction_rows,
        ignore_index=True,
    )

    prediction_path = (
        PREDICTION_DIR
        / f"{model_name.lower().replace(' ', '_')}_predictions.csv"
    )

    predictions.to_csv(
        prediction_path,
        index=False,
    )

    print(
        f"Saved predictions: "
        f"{prediction_path}"
    )

    # ----------------------------------------------------------------------
    # Summary record
    # ----------------------------------------------------------------------

    summary = {
        "model": model_name,
        "feature_count": len(feature_names),

        "train_rows": len(train),
        "validation_rows": len(validation),
        "test_rows": len(test),

        "train_auc": train_metrics["auc"],
        "validation_auc":
            validation_metrics["auc"],
        "test_auc": test_metrics["auc"],

        "train_log_loss":
            train_metrics["log_loss"],
        "validation_log_loss":
            validation_metrics["log_loss"],
        "test_log_loss":
            test_metrics["log_loss"],

        "train_brier_score":
            train_metrics["brier_score"],
        "validation_brier_score":
            validation_metrics["brier_score"],
        "test_brier_score":
            test_metrics["brier_score"],

        "train_accuracy":
            train_metrics["accuracy"],
        "validation_accuracy":
            validation_metrics["accuracy"],
        "test_accuracy":
            test_metrics["accuracy"],
    }

    return summary, coefficients


# ============================================================================
# MAIN
# ============================================================================

def main():

    warnings.filterwarnings(
        "ignore",
        category=FutureWarning,
    )

    print_section(
        "COMPACT LOGISTIC REGRESSION MODELING"
    )

    print(
        f"Project root:      {PROJECT_ROOT}"
    )

    print(
        f"Input directory:   {INPUT_DIR}"
    )

    print(
        f"Output directory:  {OUTPUT_DIR}"
    )

    # ----------------------------------------------------------------------
    # Load
    # ----------------------------------------------------------------------

    data = load_all_data()

    # ----------------------------------------------------------------------
    # Validate feature definitions
    # ----------------------------------------------------------------------

    validate_feature_sets(
        data
    )

    # ----------------------------------------------------------------------
    # Validate target
    # ----------------------------------------------------------------------

    print_section(
        "VALIDATING TARGET"
    )

    if data[
        TARGET_COLUMN
    ].isna().any():

        fail(
            "Target contains missing values."
        )

    unique_target = sorted(
        data[
            TARGET_COLUMN
        ].unique()
    )

    if unique_target != [0, 1]:

        fail(
            f"Unexpected target values: "
            f"{unique_target}"
        )

    print(
        f"Target: {TARGET_COLUMN}"
    )

    print(
        f"Unique values: {unique_target}"
    )

    print(
        f"Home wins: "
        f"{int(data[TARGET_COLUMN].sum()):,}"
    )

    print(
        f"Away wins: "
        f"{int((data[TARGET_COLUMN] == 0).sum()):,}"
    )

    # ----------------------------------------------------------------------
    # Split
    # ----------------------------------------------------------------------

    train, validation, test = split_data(
        data
    )

    # ----------------------------------------------------------------------
    # Train Models 1-3
    # ----------------------------------------------------------------------

    summaries = []
    coefficient_results = {}

    for model_name, feature_definition in (
        MODEL_FEATURES.items()
    ):

        summary, coefficients = train_model(
            model_name,
            feature_definition,
            train,
            validation,
            test,
        )

        summaries.append(
            summary
        )

        coefficient_results[
            model_name
        ] = coefficients

    # ----------------------------------------------------------------------
    # Comparison
    # ----------------------------------------------------------------------

    print_section(
        "MODEL COMPARISON"
    )

    comparison = pd.DataFrame(
        summaries
    )

    comparison = comparison.sort_values(
        "validation_log_loss",
        ascending=True,
    )

    print(
        comparison.to_string(
            index=False,
            formatters={
                column: "{:.4f}".format
                for column in comparison.columns
                if column not in [
                    "model",
                    "feature_count",
                    "train_rows",
                    "validation_rows",
                    "test_rows",
                ]
            },
        )
    )

    # ----------------------------------------------------------------------
    # Save comparison
    # ----------------------------------------------------------------------

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    comparison.to_csv(
        SUMMARY_PATH,
        index=False,
    )

    print()
    print(
        f"Saved comparison: "
        f"{SUMMARY_PATH}"
    )

    # ----------------------------------------------------------------------
    # Validation-based recommendation
    # ----------------------------------------------------------------------

    best_model = (
        comparison
        .sort_values(
            "validation_log_loss",
            ascending=True,
        )
        .iloc[0]
    )

    print_section(
        "VALIDATION-BASED MODEL RANKING"
    )

    print(
        f"Best validation log loss: "
        f"{best_model['model']}"
    )

    print(
        f"Validation log loss: "
        f"{best_model['validation_log_loss']:.4f}"
    )

    print(
        f"Validation AUC: "
        f"{best_model['validation_auc']:.4f}"
    )

    print()
    print(
        "IMPORTANT:"
    )

    print(
        "The 2025 test results are reported for "
        "final evaluation only."
    )

    print(
        "The test results should not be used to "
        "select between Models 1-3."
    )

    print()
    print_section(
        "COMPACT LOGISTIC REGRESSION MODELING "
        "COMPLETED SUCCESSFULLY"
    )


if __name__ == "__main__":
    main()