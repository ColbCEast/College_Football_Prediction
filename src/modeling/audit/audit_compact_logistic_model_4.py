"""
Diagnostic and stability audit for Compact Logistic Regression Model 4.

Model 4
-------
21 compact matchup features:

Core matchup strength:
    winPctDiff
    pointDifferentialAvgDiff
    pointsForAvgDiff
    pointsAgainstAvgDiff
    yardsPerPassAttemptDiff
    yardsPerRushAttemptDiff

Recent form:
    pointsForTrendDiff
    pointsAgainstTrendDiff
    pointDifferentialTrendDiff
    totalYardsTrendDiff
    netPassingYardsTrendDiff
    winPctTrendDiff

Prior strength of schedule:
    priorSOSWinPctDiff
    priorSOSPointDiffDiff

Additional matchup features:
    turnoversAvgDiff
    thirdDownPctDiff
    sacksAvgDiff
    completionPctDiff
    totalYardsAvgDiff
    possessionSecondsAvgDiff
    penaltyYardsAvgDiff

Temporal design
---------------
Training:
    2015-2022

Validation:
    2023-2024

Test:
    2025

Important
---------
The compact matchup dataset already contains the engineered matchup
differences. This audit therefore uses those 21 columns directly.

The 2025 test season is NEVER used for model selection.

The audit performs diagnostic/stability analyses only. It does not
re-tune Model 4 or search for a better feature specification.

Important implementation detail
--------------------------------
Some early seasons contain features that are entirely missing for that
season. In particular, sacksAvgDiff is entirely missing in some early
seasons.

For season-specific diagnostic models, SimpleImputer is therefore
configured with:

    keep_empty_features=True

This preserves the full 21-feature structure rather than silently
dropping an all-missing feature. Such a feature receives a coefficient
of zero in that diagnostic model because it contains no information
within that season.

This does NOT modify the saved production Model 4.
"""

from pathlib import Path
import warnings

import joblib
import numpy as np
import pandas as pd

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
GAME_ID_COLUMN = "gameId"

TRAIN_YEARS = list(range(2015, 2023))
VALIDATION_YEARS = [2023, 2024]
TEST_YEARS = [2025]

ALL_YEARS = (
    TRAIN_YEARS
    + VALIDATION_YEARS
    + TEST_YEARS
)

# Number of bootstrap samples used for coefficient stability.
N_BOOTSTRAPS = 500

# Minimum observations required for a season-specific model.
MIN_SEASON_ROWS = 100

# Correlation threshold used only for diagnostics.
HIGH_CORRELATION_THRESHOLD = 0.80

# Coefficients with absolute magnitude below this value are treated as
# effectively zero for sign-stability summaries.
NEAR_ZERO_COEFFICIENT_THRESHOLD = 0.01

PROJECT_ROOT = Path(__file__).resolve().parents[3]

INPUT_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "logistic_matchup_features"
)

MODEL_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "models"
    / "compact_logistic"
    / "models"
)

AUDIT_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "models"
    / "compact_logistic"
    / "audit"
)

MODEL_3_PATH = MODEL_DIR / "model_3.joblib"
MODEL_4_PATH = MODEL_DIR / "model_4.joblib"


# ============================================================================
# MODEL 4 FEATURE DEFINITIONS
# ============================================================================

MODEL_1_FEATURES = [
    "winPctDiff",
    "pointDifferentialAvgDiff",
    "pointsForAvgDiff",
    "pointsAgainstAvgDiff",
    "yardsPerPassAttemptDiff",
    "yardsPerRushAttemptDiff",
]

MODEL_2_ADDITIONS = [
    "pointsForTrendDiff",
    "pointsAgainstTrendDiff",
    "pointDifferentialTrendDiff",
    "totalYardsTrendDiff",
    "netPassingYardsTrendDiff",
    "winPctTrendDiff",
]

MODEL_3_ADDITIONS = [
    "priorSOSWinPctDiff",
    "priorSOSPointDiffDiff",
]

MODEL_4_ADDITIONS = [
    "turnoversAvgDiff",
    "thirdDownPctDiff",
    "sacksAvgDiff",
    "completionPctDiff",
    "totalYardsAvgDiff",
    "possessionSecondsAvgDiff",
    "penaltyYardsAvgDiff",
]

MODEL_2_FEATURES = (
    MODEL_1_FEATURES
    + MODEL_2_ADDITIONS
)

MODEL_3_FEATURES = (
    MODEL_2_FEATURES
    + MODEL_3_ADDITIONS
)

MODEL_4_FEATURES = (
    MODEL_3_FEATURES
    + MODEL_4_ADDITIONS
)


# ============================================================================
# PRINT / ERROR HELPERS
# ============================================================================

def print_section(title):
    print()
    print("=" * 78)
    print(title)
    print("=" * 78)


def print_subsection(title):
    print()
    print(title)
    print("-" * len(title))


def fail(message):
    raise ValueError(
        f"\nVALIDATION FAILED:\n{message}"
    )


# ============================================================================
# DATA LOADING
# ============================================================================

def load_season(year):
    """
    Load one compact matchup-feature dataset.
    """

    path = (
        INPUT_DIR
        / f"compact_matchup_features_{year}.csv"
    )

    if not path.exists():

        candidates = sorted(
            INPUT_DIR.glob(f"*{year}.csv")
        )

        if len(candidates) == 1:
            path = candidates[0]

        elif len(candidates) == 0:
            fail(
                f"Missing compact matchup feature file for {year}.\n"
                f"Expected:\n{path}"
            )

        else:
            fail(
                f"Could not uniquely identify the compact matchup "
                f"feature file for {year}.\n"
                f"Candidates:\n"
                + "\n".join(
                    str(p)
                    for p in candidates
                )
            )

    df = pd.read_csv(path)

    if TARGET_COLUMN not in df.columns:
        fail(
            f"{year}: missing target column "
            f"'{TARGET_COLUMN}'."
        )

    if GAME_ID_COLUMN not in df.columns:
        fail(
            f"{year}: missing game ID column "
            f"'{GAME_ID_COLUMN}'."
        )

    return df


def load_all_data():
    """
    Load all seasons and combine them.
    """

    print_section(
        "LOADING COMPACT MATCHUP FEATURE DATA"
    )

    frames = []

    for year in ALL_YEARS:

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
# DATASET VALIDATION
# ============================================================================

def validate_dataset(data):
    """
    Validate the compact matchup dataset.
    """

    print_section(
        "VALIDATING DATASET"
    )

    # ------------------------------------------------------------------
    # Game ID uniqueness
    # ------------------------------------------------------------------

    duplicate_ids = data[
        GAME_ID_COLUMN
    ].duplicated(
        keep=False
    )

    if duplicate_ids.any():

        duplicate_count = (
            data.loc[
                duplicate_ids,
                GAME_ID_COLUMN,
            ]
            .nunique()
        )

        fail(
            f"Found {duplicate_count:,} duplicated "
            f"game IDs."
        )

    print(
        "Game ID uniqueness: VALID"
    )

    # ------------------------------------------------------------------
    # Feature existence
    # ------------------------------------------------------------------

    missing_features = [
        feature
        for feature in MODEL_4_FEATURES
        if feature not in data.columns
    ]

    if missing_features:

        fail(
            "Missing Model 4 features:\n"
            + "\n".join(
                f"  {feature}"
                for feature in missing_features
            )
        )

    print(
        f"Model 4 features: "
        f"{len(MODEL_4_FEATURES)}"
    )

    print(
        "Required features: VALID"
    )

    # ------------------------------------------------------------------
    # Target
    # ------------------------------------------------------------------

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
        "Target: VALID"
    )


# ============================================================================
# MODEL LOADING
# ============================================================================

def load_trained_models():
    """
    Load Model 3 and Model 4 for comparison diagnostics.
    """

    print_section(
        "LOADING TRAINED MODELS"
    )

    if not MODEL_3_PATH.exists():

        fail(
            f"Model 3 not found:\n"
            f"{MODEL_3_PATH}"
        )

    if not MODEL_4_PATH.exists():

        fail(
            f"Model 4 not found:\n"
            f"{MODEL_4_PATH}"
        )

    model_3 = joblib.load(
        MODEL_3_PATH
    )

    model_4 = joblib.load(
        MODEL_4_PATH
    )

    print(
        f"Loaded Model 3: {MODEL_3_PATH}"
    )

    print(
        f"Loaded Model 4: {MODEL_4_PATH}"
    )

    return model_3, model_4


# ============================================================================
# FEATURE CONSTRUCTION
# ============================================================================

def construct_features(
    data,
    feature_names,
):
    """
    Return already-engineered compact matchup features.

    The input dataset is logistic_matchup_features, so the matchup
    differences already exist. We do NOT reconstruct them from
    *_home / *_away columns.
    """

    missing_features = [
        feature
        for feature in feature_names
        if feature not in data.columns
    ]

    if missing_features:

        fail(
            "Missing compact matchup features:\n"
            + "\n".join(
                f"  {feature}"
                for feature in missing_features
            )
        )

    X = data[
        feature_names
    ].copy()

    for feature in feature_names:

        X[feature] = pd.to_numeric(
            X[feature],
            errors="coerce",
        )

    return X


# ============================================================================
# MODEL PIPELINE
# ============================================================================

def build_audit_pipeline():
    """
    Build the diagnostic logistic regression pipeline.

    This intentionally mirrors the production Model 4 preprocessing:

        median imputation
        standardization
        logistic regression

    keep_empty_features=True is required for season-specific audits
    because some early seasons contain features that are entirely
    missing. Without this option, SimpleImputer removes those features,
    causing the coefficient vector to have fewer than 21 values.
    """

    return Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(
                    strategy="median",
                    keep_empty_features=True,
                ),
            ),
            (
                "scaler",
                StandardScaler(),
            ),
            (
                "model",
                LogisticRegression(
                    max_iter=2000,
                    random_state=RANDOM_STATE,
                ),
            ),
        ]
    )


# ============================================================================
# COEFFICIENT EXTRACTION
# ============================================================================

def extract_pipeline_coefficients(
    pipeline,
    feature_names,
):
    """
    Extract coefficients from a standardized logistic pipeline.

    Validates that the model contains exactly the expected number of
    coefficients.

    Because build_audit_pipeline() uses keep_empty_features=True,
    all-missing diagnostic features remain in the transformed matrix.
    """

    model = pipeline.named_steps[
        "model"
    ]

    coefficients = np.asarray(
        model.coef_[0],
        dtype=float,
    )

    expected_count = len(
        feature_names
    )

    actual_count = len(
        coefficients
    )

    if actual_count != expected_count:

        fail(
            "Diagnostic model coefficient count does not match "
            "the expected feature count.\n"
            f"Expected: {expected_count}\n"
            f"Actual:   {actual_count}\n"
            f"Features: {feature_names}"
        )

    return pd.Series(
        coefficients,
        index=feature_names,
        dtype=float,
    )


def extract_saved_model_coefficients(
    pipeline,
    feature_names,
):
    """
    Extract coefficients from a saved compact logistic pipeline.

    The production training pipeline has:

        preprocessor
            -> numeric Pipeline
                -> imputer
                -> scaler
        model

    The saved model is expected to contain one coefficient per
    Model 3 / Model 4 feature.
    """

    model = pipeline.named_steps[
        "model"
    ]

    coefficients = np.asarray(
        model.coef_[0],
        dtype=float,
    )

    expected_count = len(
        feature_names
    )

    actual_count = len(
        coefficients
    )

    if actual_count != expected_count:

        fail(
            "Saved model coefficient count does not "
            "match expected feature count.\n"
            f"Expected: {expected_count}\n"
            f"Actual:   {actual_count}"
        )

    return pd.Series(
        coefficients,
        index=feature_names,
        dtype=float,
    )


# ============================================================================
# SEASON-BY-SEASON COEFFICIENT STABILITY
# ============================================================================

def train_season_specific_models(
    data,
):
    """
    Fit Model 4 independently within each season.

    This is a diagnostic only. These models are not used for prediction
    or model selection.

    A feature that is entirely missing in a particular season is retained
    by the diagnostic pipeline and receives a zero coefficient because
    it contains no information within that season.
    """

    print_section(
        "SEASON-BY-SEASON COEFFICIENT STABILITY"
    )

    results = []

    for year in ALL_YEARS:

        season = data[
            data["model_year"] == year
        ].copy()

        print(
            f"\n{year}: "
            f"{len(season):,} rows"
        )

        if len(season) < MIN_SEASON_ROWS:

            print(
                f"  SKIPPED — fewer than "
                f"{MIN_SEASON_ROWS} rows"
            )

            continue

        X = construct_features(
            season,
            MODEL_4_FEATURES,
        )

        y = season[
            TARGET_COLUMN
        ].astype(int)

        # --------------------------------------------------------------
        # Report all-missing features in this season
        # --------------------------------------------------------------

        all_missing = [
            feature
            for feature in MODEL_4_FEATURES
            if X[feature].notna().sum() == 0
        ]

        if all_missing:

            print(
                "  All-missing features retained "
                "for diagnostic model:"
            )

            for feature in all_missing:

                print(
                    f"    {feature}"
                )

        # --------------------------------------------------------------
        # Target validation
        # --------------------------------------------------------------

        if y.nunique() < 2:

            print(
                "  SKIPPED — target has fewer "
                "than two classes"
            )

            continue

        # --------------------------------------------------------------
        # Fit
        # --------------------------------------------------------------

        pipeline = build_audit_pipeline()

        pipeline.fit(
            X,
            y,
        )

        # --------------------------------------------------------------
        # Extract coefficients
        # --------------------------------------------------------------

        coefficients = (
            extract_pipeline_coefficients(
                pipeline,
                MODEL_4_FEATURES,
            )
        )

        # --------------------------------------------------------------
        # Store
        # --------------------------------------------------------------

        for feature in MODEL_4_FEATURES:

            coefficient = (
                coefficients[feature]
            )

            results.append(
                {
                    "season": year,
                    "feature": feature,
                    "coefficient":
                        float(coefficient),
                    "absolute_coefficient":
                        float(abs(coefficient)),
                    "sign":
                        float(np.sign(coefficient)),
                    "feature_all_missing":
                        feature in all_missing,
                }
            )

    result_df = pd.DataFrame(
        results
    )

    if result_df.empty:

        fail(
            "No season-specific models "
            "were successfully fitted."
        )

    # ------------------------------------------------------------------
    # Save
    # ------------------------------------------------------------------

    AUDIT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    path = (
        AUDIT_DIR
        / "season_coefficient_stability.csv"
    )

    result_df.to_csv(
        path,
        index=False,
    )

    print()
    print(
        f"Saved season coefficient audit: "
        f"{path}"
    )

    return result_df


# ============================================================================
# COEFFICIENT STABILITY SUMMARY
# ============================================================================

def summarize_coefficient_stability(
    season_coefficients,
):
    """
    Summarize coefficient behavior across seasons.
    """

    print_section(
        "COEFFICIENT STABILITY SUMMARY"
    )

    rows = []

    for feature in MODEL_4_FEATURES:

        subset = season_coefficients[
            season_coefficients["feature"]
            == feature
        ]

        coefficients = subset[
            "coefficient"
        ].dropna().to_numpy()

        nonzero = coefficients[
            np.abs(coefficients)
            >= NEAR_ZERO_COEFFICIENT_THRESHOLD
        ]

        if len(nonzero) > 0:

            positive_pct = (
                np.mean(nonzero > 0)
                * 100
            )

            negative_pct = (
                np.mean(nonzero < 0)
                * 100
            )

        else:

            positive_pct = np.nan
            negative_pct = np.nan

        rows.append(
            {
                "feature": feature,
                "seasons": len(coefficients),
                "mean_coefficient":
                    np.mean(coefficients),
                "std_coefficient":
                    (
                        np.std(
                            coefficients,
                            ddof=1,
                        )
                        if len(coefficients) > 1
                        else np.nan
                    ),
                "min_coefficient":
                    np.min(coefficients),
                "max_coefficient":
                    np.max(coefficients),
                "positive_pct":
                    positive_pct,
                "negative_pct":
                    negative_pct,
                "near_zero_pct":
                    (
                        np.mean(
                            np.abs(coefficients)
                            < NEAR_ZERO_COEFFICIENT_THRESHOLD
                        )
                        * 100
                    ),
            }
        )

    result = pd.DataFrame(
        rows
    )

    print(
        result.to_string(
            index=False,
            formatters={
                "mean_coefficient":
                    "{:.4f}".format,
                "std_coefficient":
                    "{:.4f}".format,
                "min_coefficient":
                    "{:.4f}".format,
                "max_coefficient":
                    "{:.4f}".format,
                "positive_pct":
                    "{:.1f}%".format,
                "negative_pct":
                    "{:.1f}%".format,
                "near_zero_pct":
                    "{:.1f}%".format,
            },
        )
    )

    path = (
        AUDIT_DIR
        / "coefficient_stability_summary.csv"
    )

    result.to_csv(
        path,
        index=False,
    )

    print()
    print(
        f"Saved coefficient stability summary: "
        f"{path}"
    )

    return result


# ============================================================================
# SIGN STABILITY
# ============================================================================

def analyze_sign_stability(
    season_coefficients,
):
    """
    Analyze coefficient sign consistency across seasons.
    """

    print_section(
        "COEFFICIENT SIGN STABILITY"
    )

    rows = []

    for feature in MODEL_4_FEATURES:

        subset = season_coefficients[
            season_coefficients["feature"]
            == feature
        ].copy()

        coefficients = subset[
            "coefficient"
        ].dropna().to_numpy()

        meaningful = coefficients[
            np.abs(coefficients)
            >= NEAR_ZERO_COEFFICIENT_THRESHOLD
        ]

        if len(meaningful) == 0:

            positive_pct = np.nan
            negative_pct = np.nan
            zero_pct = 100.0

        else:

            positive_pct = (
                np.mean(meaningful > 0)
                * 100
            )

            negative_pct = (
                np.mean(meaningful < 0)
                * 100
            )

            zero_pct = (
                np.mean(
                    np.abs(coefficients)
                    < NEAR_ZERO_COEFFICIENT_THRESHOLD
                )
                * 100
            )

        rows.append(
            {
                "feature": feature,
                "seasons": len(coefficients),
                "positive_pct":
                    positive_pct,
                "negative_pct":
                    negative_pct,
                "near_zero_pct":
                    zero_pct,
                "dominant_sign":
                    (
                        "positive"
                        if positive_pct > negative_pct
                        else "negative"
                    )
                    if pd.notna(positive_pct)
                    else "near_zero",
            }
        )

    result = pd.DataFrame(
        rows
    )

    print(
        result.to_string(
            index=False,
            formatters={
                "positive_pct":
                    "{:.1f}%".format,
                "negative_pct":
                    "{:.1f}%".format,
                "near_zero_pct":
                    "{:.1f}%".format,
            },
        )
    )

    path = (
        AUDIT_DIR
        / "coefficient_sign_stability.csv"
    )

    result.to_csv(
        path,
        index=False,
    )

    print()
    print(
        f"Saved sign stability audit: "
        f"{path}"
    )

    return result


# ============================================================================
# SAVED MODEL COEFFICIENT COMPARISON
# ============================================================================

def analyze_saved_model_coefficients(
    model_3,
    model_4,
):
    """
    Compare saved Model 3 and Model 4 coefficients.
    """

    print_section(
        "SAVED MODEL COEFFICIENT COMPARISON"
    )

    model_3_coefficients = (
        extract_saved_model_coefficients(
            model_3,
            MODEL_3_FEATURES,
        )
    )

    model_4_coefficients = (
        extract_saved_model_coefficients(
            model_4,
            MODEL_4_FEATURES,
        )
    )

    rows = []

    for feature in MODEL_4_FEATURES:

        coefficient_4 = (
            model_4_coefficients[
                feature
            ]
        )

        if feature in model_3_coefficients.index:

            coefficient_3 = (
                model_3_coefficients[
                    feature
                ]
            )

        else:

            coefficient_3 = 0.0

        rows.append(
            {
                "feature": feature,
                "model_3_coefficient":
                    coefficient_3,
                "model_4_coefficient":
                    coefficient_4,
                "coefficient_change":
                    coefficient_4
                    - coefficient_3,
                "absolute_change":
                    abs(
                        coefficient_4
                        - coefficient_3
                    ),
                "model_3_odds_ratio":
                    np.exp(
                        coefficient_3
                    ),
                "model_4_odds_ratio":
                    np.exp(
                        coefficient_4
                    ),
            }
        )

    comparison = pd.DataFrame(
        rows
    )

    print(
        comparison.to_string(
            index=False,
            formatters={
                column: "{:.4f}".format
                for column in comparison.columns
                if column != "feature"
            },
        )
    )

    path = (
        AUDIT_DIR
        / "model_3_vs_model_4_coefficients.csv"
    )

    comparison.to_csv(
        path,
        index=False,
    )

    print()
    print(
        f"Saved Model 3 vs Model 4 coefficients: "
        f"{path}"
    )

    return comparison


# ============================================================================
# FEATURE CORRELATION
# ============================================================================

def analyze_feature_correlations(
    data,
):
    """
    Analyze correlations among Model 4 matchup features.

    This is descriptive only.
    """

    print_section(
        "MODEL 4 FEATURE CORRELATION AUDIT"
    )

    X = construct_features(
        data,
        MODEL_4_FEATURES,
    )

    correlation = X.corr(
        method="pearson"
    )

    pairs = []

    for i, feature_a in enumerate(
        MODEL_4_FEATURES
    ):

        for j in range(
            i + 1,
            len(MODEL_4_FEATURES),
        ):

            feature_b = (
                MODEL_4_FEATURES[j]
            )

            value = correlation.loc[
                feature_a,
                feature_b,
            ]

            if pd.notna(value):

                pairs.append(
                    {
                        "feature_1":
                            feature_a,
                        "feature_2":
                            feature_b,
                        "correlation":
                            value,
                        "absolute_correlation":
                            abs(value),
                    }
                )

    pair_df = pd.DataFrame(
        pairs
    )

    pair_df = pair_df.sort_values(
        "absolute_correlation",
        ascending=False,
    )

    high_corr = pair_df[
        pair_df[
            "absolute_correlation"
        ]
        >= HIGH_CORRELATION_THRESHOLD
    ]

    print()
    print(
        f"Feature pairs with "
        f"|correlation| >= "
        f"{HIGH_CORRELATION_THRESHOLD:.2f}: "
        f"{len(high_corr)}"
    )

    if not high_corr.empty:

        print()
        print(
            high_corr.to_string(
                index=False,
                formatters={
                    "correlation":
                        "{:.4f}".format,
                    "absolute_correlation":
                        "{:.4f}".format,
                },
            )
        )

    else:

        print(
            "No highly correlated feature pairs "
            "at the selected threshold."
        )

    path = (
        AUDIT_DIR
        / "model_4_feature_correlations.csv"
    )

    pair_df.to_csv(
        path,
        index=False,
    )

    print()
    print(
        f"Saved feature correlation audit: "
        f"{path}"
    )

    return pair_df


# ============================================================================
# MISSINGNESS AUDIT
# ============================================================================

def analyze_missingness(
    data,
):
    """
    Analyze missingness by feature and season.
    """

    print_section(
        "MODEL 4 MISSINGNESS AUDIT"
    )

    rows = []

    for year in ALL_YEARS:

        season = data[
            data["model_year"] == year
        ]

        for feature in MODEL_4_FEATURES:

            missing_count = int(
                season[feature]
                .isna()
                .sum()
            )

            missing_pct = (
                missing_count
                / len(season)
                * 100
            )

            rows.append(
                {
                    "season": year,
                    "feature": feature,
                    "rows": len(season),
                    "missing_count":
                        missing_count,
                    "missing_pct":
                        missing_pct,
                    "all_missing":
                        missing_count == len(season),
                }
            )

    result = pd.DataFrame(
        rows
    )

    summary = (
        result
        .groupby("feature")
        .agg(
            mean_missing_pct=(
                "missing_pct",
                "mean",
            ),
            min_missing_pct=(
                "missing_pct",
                "min",
            ),
            max_missing_pct=(
                "missing_pct",
                "max",
            ),
            seasons_all_missing=(
                "all_missing",
                "sum",
            ),
        )
        .reset_index()
    )

    print(
        summary.to_string(
            index=False,
            formatters={
                "mean_missing_pct":
                    "{:.3f}%".format,
                "min_missing_pct":
                    "{:.3f}%".format,
                "max_missing_pct":
                    "{:.3f}%".format,
            },
        )
    )

    result_path = (
        AUDIT_DIR
        / "model_4_missingness_by_season.csv"
    )

    summary_path = (
        AUDIT_DIR
        / "model_4_missingness_summary.csv"
    )

    result.to_csv(
        result_path,
        index=False,
    )

    summary.to_csv(
        summary_path,
        index=False,
    )

    print()
    print(
        f"Saved seasonal missingness: "
        f"{result_path}"
    )

    print(
        f"Saved missingness summary: "
        f"{summary_path}"
    )

    return result, summary


# ============================================================================
# TEMPORAL PERFORMANCE AUDIT
# ============================================================================

def evaluate_model_by_season(
    model,
    data,
    feature_names,
    model_name,
):
    """
    Evaluate a saved model separately on every season.

    This is an audit of temporal performance, not model selection.
    """

    print_section(
        f"{model_name.upper()} SEASON-BY-SEASON PERFORMANCE"
    )

    rows = []

    for year in ALL_YEARS:

        season = data[
            data["model_year"] == year
        ].copy()

        X = construct_features(
            season,
            feature_names,
        )

        y = season[
            TARGET_COLUMN
        ].astype(int)

        probability = model.predict_proba(
            X
        )[:, 1]

        prediction = (
            probability >= 0.5
        ).astype(int)

        row = {
            "model": model_name,
            "season": year,
            "rows": len(season),
            "auc": np.nan,
            "log_loss": np.nan,
            "brier_score": np.nan,
            "accuracy": np.nan,
            "home_win_rate":
                np.mean(y),
            "mean_predicted_home_win_prob":
                np.mean(probability),
        }

        if y.nunique() >= 2:

            row["auc"] = (
                roc_auc_score(
                    y,
                    probability,
                )
            )

        row["log_loss"] = (
            log_loss(
                y,
                probability,
            )
        )

        row["brier_score"] = (
            brier_score_loss(
                y,
                probability,
            )
        )

        row["accuracy"] = (
            accuracy_score(
                y,
                prediction,
            )
        )

        rows.append(row)

    result = pd.DataFrame(
        rows
    )

    print(
        result.to_string(
            index=False,
            formatters={
                "auc":
                    "{:.4f}".format,
                "log_loss":
                    "{:.4f}".format,
                "brier_score":
                    "{:.4f}".format,
                "accuracy":
                    "{:.4f}".format,
                "home_win_rate":
                    "{:.4f}".format,
                "mean_predicted_home_win_prob":
                    "{:.4f}".format,
            },
        )
    )

    filename = (
        f"{model_name.lower().replace(' ', '_')}"
        "_season_performance.csv"
    )

    path = (
        AUDIT_DIR
        / filename
    )

    result.to_csv(
        path,
        index=False,
    )

    print()
    print(
        f"Saved seasonal performance: {path}"
    )

    return result


# ============================================================================
# BOOTSTRAP COEFFICIENT STABILITY
# ============================================================================

def bootstrap_coefficients(
    data,
):
    """
    Bootstrap the 2015-2022 training data and estimate coefficient
    distributions.

    This assesses how sensitive Model 4 coefficients are to sampling
    variation within the historical training period.

    The bootstrap operates on the complete training period, where all
    Model 4 features have observed values somewhere in the sample.
    """

    print_section(
        "BOOTSTRAP COEFFICIENT STABILITY"
    )

    training = data[
        data["model_year"].isin(
            TRAIN_YEARS
        )
    ].copy()

    X = construct_features(
        training,
        MODEL_4_FEATURES,
    )

    y = training[
        TARGET_COLUMN
    ].astype(int)

    rng = np.random.default_rng(
        RANDOM_STATE
    )

    coefficient_samples = []

    n_rows = len(training)

    print(
        f"Training rows: {n_rows:,}"
    )

    print(
        f"Bootstrap samples: "
        f"{N_BOOTSTRAPS:,}"
    )

    for iteration in range(
        N_BOOTSTRAPS
    ):

        indices = rng.integers(
            0,
            n_rows,
            size=n_rows,
        )

        X_bootstrap = X.iloc[
            indices
        ]

        y_bootstrap = y.iloc[
            indices
        ]

        if y_bootstrap.nunique() < 2:

            continue

        pipeline = build_audit_pipeline()

        pipeline.fit(
            X_bootstrap,
            y_bootstrap,
        )

        coefficients = (
            extract_pipeline_coefficients(
                pipeline,
                MODEL_4_FEATURES,
            )
        )

        row = {
            feature:
                coefficients[feature]
            for feature in MODEL_4_FEATURES
        }

        row[
            "bootstrap_iteration"
        ] = iteration

        coefficient_samples.append(
            row
        )

    bootstrap_df = pd.DataFrame(
        coefficient_samples
    )

    if bootstrap_df.empty:

        fail(
            "Bootstrap coefficient audit "
            "produced no valid samples."
        )

    summary_rows = []

    for feature in MODEL_4_FEATURES:

        values = bootstrap_df[
            feature
        ].to_numpy()

        lower = np.percentile(
            values,
            2.5,
        )

        upper = np.percentile(
            values,
            97.5,
        )

        summary_rows.append(
            {
                "feature": feature,
                "bootstrap_mean":
                    np.mean(values),
                "bootstrap_std":
                    np.std(
                        values,
                        ddof=1,
                    ),
                "bootstrap_median":
                    np.median(values),
                "ci_2_5_pct":
                    lower,
                "ci_97_5_pct":
                    upper,
                "positive_pct":
                    np.mean(values > 0)
                    * 100,
                "negative_pct":
                    np.mean(values < 0)
                    * 100,
                "ci_crosses_zero":
                    (
                        lower <= 0
                        <= upper
                    ),
            }
        )

    summary = pd.DataFrame(
        summary_rows
    )

    print()
    print(
        summary.to_string(
            index=False,
            formatters={
                column: "{:.4f}".format
                for column in summary.columns
                if column != "feature"
                and column != "ci_crosses_zero"
            },
        )
    )

    bootstrap_path = (
        AUDIT_DIR
        / "bootstrap_coefficients.csv"
    )

    summary_path = (
        AUDIT_DIR
        / "bootstrap_coefficient_summary.csv"
    )

    bootstrap_df.to_csv(
        bootstrap_path,
        index=False,
    )

    summary.to_csv(
        summary_path,
        index=False,
    )

    print()
    print(
        f"Saved bootstrap samples: "
        f"{bootstrap_path}"
    )

    print(
        f"Saved bootstrap summary: "
        f"{summary_path}"
    )

    return bootstrap_df, summary


# ============================================================================
# PREDICTION STABILITY
# ============================================================================

def analyze_prediction_stability(
    model,
    data,
):
    """
    Analyze prediction probability distributions by temporal split.
    """

    print_section(
        "PREDICTION STABILITY"
    )

    rows = []

    for split_name, years in [
        (
            "train",
            TRAIN_YEARS,
        ),
        (
            "validation",
            VALIDATION_YEARS,
        ),
        (
            "test",
            TEST_YEARS,
        ),
    ]:

        subset = data[
            data["model_year"].isin(
                years
            )
        ].copy()

        X = construct_features(
            subset,
            MODEL_4_FEATURES,
        )

        y = subset[
            TARGET_COLUMN
        ].astype(int)

        probability = model.predict_proba(
            X
        )[:, 1]

        prediction = (
            probability >= 0.5
        ).astype(int)

        rows.append(
            {
                "dataset": split_name,
                "rows": len(subset),
                "mean_probability":
                    np.mean(probability),
                "std_probability":
                    np.std(
                        probability,
                        ddof=1,
                    ),
                "min_probability":
                    np.min(probability),
                "p05_probability":
                    np.percentile(
                        probability,
                        5,
                    ),
                "median_probability":
                    np.median(probability),
                "p95_probability":
                    np.percentile(
                        probability,
                        95,
                    ),
                "max_probability":
                    np.max(probability),
                "predicted_home_win_rate":
                    np.mean(prediction),
                "actual_home_win_rate":
                    np.mean(y),
            }
        )

    result = pd.DataFrame(
        rows
    )

    print(
        result.to_string(
            index=False,
            formatters={
                column: "{:.4f}".format
                for column in result.columns
                if column not in [
                    "dataset",
                    "rows",
                ]
            },
        )
    )

    path = (
        AUDIT_DIR
        / "model_4_prediction_stability.csv"
    )

    result.to_csv(
        path,
        index=False,
    )

    print()
    print(
        f"Saved prediction stability: {path}"
    )

    return result


# ============================================================================
# CALIBRATION AUDIT
# ============================================================================

def calibration_by_bin(
    y_true,
    probability,
    n_bins=10,
):
    """
    Calculate reliability statistics across probability bins.
    """

    frame = pd.DataFrame(
        {
            "actual": np.asarray(
                y_true
            ),
            "probability": np.asarray(
                probability
            ),
        }
    )

    bins = np.linspace(
        0,
        1,
        n_bins + 1,
    )

    frame[
        "bin"
    ] = pd.cut(
        frame["probability"],
        bins=bins,
        include_lowest=True,
        labels=False,
    )

    rows = []

    for bin_id in range(
        n_bins
    ):

        subset = frame[
            frame["bin"] == bin_id
        ]

        if subset.empty:
            continue

        rows.append(
            {
                "bin": bin_id + 1,
                "rows": len(subset),
                "mean_predicted_probability":
                    subset[
                        "probability"
                    ].mean(),
                "observed_home_win_rate":
                    subset[
                        "actual"
                    ].mean(),
                "calibration_error":
                    (
                        subset[
                            "actual"
                        ].mean()
                        - subset[
                            "probability"
                        ].mean()
                    ),
            }
        )

    return pd.DataFrame(
        rows
    )


def analyze_calibration(
    model,
    data,
):
    """
    Evaluate calibration separately on validation and test.
    """

    print_section(
        "CALIBRATION AUDIT"
    )

    all_results = []

    for split_name, years in [
        (
            "validation",
            VALIDATION_YEARS,
        ),
        (
            "test",
            TEST_YEARS,
        ),
    ]:

        subset = data[
            data["model_year"].isin(
                years
            )
        ].copy()

        X = construct_features(
            subset,
            MODEL_4_FEATURES,
        )

        y = subset[
            TARGET_COLUMN
        ].astype(int)

        probability = model.predict_proba(
            X
        )[:, 1]

        calibration = calibration_by_bin(
            y,
            probability,
            n_bins=10,
        )

        calibration.insert(
            0,
            "dataset",
            split_name,
        )

        all_results.append(
            calibration
        )

        print()
        print(
            split_name.upper()
        )

        print(
            calibration.to_string(
                index=False,
                formatters={
                    "mean_predicted_probability":
                        "{:.4f}".format,
                    "observed_home_win_rate":
                        "{:.4f}".format,
                    "calibration_error":
                        "{:.4f}".format,
                },
            )
        )

    result = pd.concat(
        all_results,
        ignore_index=True,
    )

    path = (
        AUDIT_DIR
        / "model_4_calibration.csv"
    )

    result.to_csv(
        path,
        index=False,
    )

    print()
    print(
        f"Saved calibration audit: {path}"
    )

    return result


# ============================================================================
# TEMPORAL COEFFICIENT DRIFT
# ============================================================================

def analyze_temporal_coefficient_drift(
    season_coefficients,
):
    """
    Compare early, middle, and late historical coefficient behavior.
    """

    print_section(
        "TEMPORAL COEFFICIENT DRIFT"
    )

    available_seasons = sorted(
        season_coefficients[
            "season"
        ].unique()
    )

    groups = {
        "early":
            [
                year
                for year in available_seasons
                if year <= 2017
            ],

        "middle":
            [
                year
                for year in available_seasons
                if 2018 <= year <= 2020
            ],

        "late":
            [
                year
                for year in available_seasons
                if year >= 2021
            ],
    }

    rows = []

    for period, years in groups.items():

        subset = season_coefficients[
            season_coefficients[
                "season"
            ].isin(years)
        ]

        for feature in MODEL_4_FEATURES:

            values = subset.loc[
                subset["feature"]
                == feature,
                "coefficient",
            ].dropna().to_numpy()

            if len(values) == 0:
                continue

            rows.append(
                {
                    "period": period,
                    "seasons":
                        ",".join(
                            map(
                                str,
                                years,
                            )
                        ),
                    "feature": feature,
                    "mean_coefficient":
                        np.mean(values),
                    "std_coefficient":
                        (
                            np.std(
                                values,
                                ddof=1,
                            )
                            if len(values) > 1
                            else np.nan
                        ),
                }
            )

    result = pd.DataFrame(
        rows
    )

    print(
        result.to_string(
            index=False,
            formatters={
                "mean_coefficient":
                    "{:.4f}".format,
                "std_coefficient":
                    "{:.4f}".format,
            },
        )
    )

    path = (
        AUDIT_DIR
        / "temporal_coefficient_drift.csv"
    )

    result.to_csv(
        path,
        index=False,
    )

    print()
    print(
        f"Saved temporal coefficient drift: "
        f"{path}"
    )

    return result


# ============================================================================
# MODEL 3 → MODEL 4 VALIDATION IMPROVEMENT
# ============================================================================

def analyze_model_3_to_model_4_improvement(
    model_3,
    model_4,
    data,
):
    """
    Reproduce the Model 3 vs Model 4 validation comparison.

    This confirms that the Model 4 improvement remains visible when the
    saved models are evaluated directly.
    """

    print_section(
        "MODEL 3 → MODEL 4 VALIDATION COMPARISON"
    )

    validation = data[
        data["model_year"].isin(
            VALIDATION_YEARS
        )
    ].copy()

    y = validation[
        TARGET_COLUMN
    ].astype(int)

    X3 = construct_features(
        validation,
        MODEL_3_FEATURES,
    )

    X4 = construct_features(
        validation,
        MODEL_4_FEATURES,
    )

    probability_3 = model_3.predict_proba(
        X3
    )[:, 1]

    probability_4 = model_4.predict_proba(
        X4
    )[:, 1]

    metrics_3 = {
        "model": "Model 3",
        "auc":
            roc_auc_score(
                y,
                probability_3,
            ),
        "log_loss":
            log_loss(
                y,
                probability_3,
            ),
        "brier_score":
            brier_score_loss(
                y,
                probability_3,
            ),
        "accuracy":
            accuracy_score(
                y,
                (
                    probability_3 >= 0.5
                ).astype(int),
            ),
    }

    metrics_4 = {
        "model": "Model 4",
        "auc":
            roc_auc_score(
                y,
                probability_4,
            ),
        "log_loss":
            log_loss(
                y,
                probability_4,
            ),
        "brier_score":
            brier_score_loss(
                y,
                probability_4,
            ),
        "accuracy":
            accuracy_score(
                y,
                (
                    probability_4 >= 0.5
                ).astype(int),
            ),
    }

    result = pd.DataFrame(
        [
            metrics_3,
            metrics_4,
        ]
    )

    print(
        result.to_string(
            index=False,
            formatters={
                "auc":
                    "{:.4f}".format,
                "log_loss":
                    "{:.4f}".format,
                "brier_score":
                    "{:.4f}".format,
                "accuracy":
                    "{:.4f}".format,
            },
        )
    )

    model_3_row = result[
        result["model"] == "Model 3"
    ].iloc[0]

    model_4_row = result[
        result["model"] == "Model 4"
    ].iloc[0]

    log_loss_change = (
        model_4_row["log_loss"]
        - model_3_row["log_loss"]
    )

    auc_change = (
        model_4_row["auc"]
        - model_3_row["auc"]
    )

    brier_change = (
        model_4_row["brier_score"]
        - model_3_row["brier_score"]
    )

    print()
    print(
        "Model 3 → Model 4 validation change:"
    )

    print(
        f"  Log loss change: "
        f"{log_loss_change:+.4f}"
    )

    print(
        f"  AUC change:      "
        f"{auc_change:+.4f}"
    )

    print(
        f"  Brier change:    "
        f"{brier_change:+.4f}"
    )

    if log_loss_change < 0:

        print()
        print(
            "Model 4 improves validation log loss "
            "relative to Model 3."
        )

    else:

        print()
        print(
            "Model 4 does NOT improve validation log loss "
            "relative to Model 3."
        )

    path = (
        AUDIT_DIR
        / "model_3_vs_model_4_validation.csv"
    )

    result.to_csv(
        path,
        index=False,
    )

    print()
    print(
        f"Saved validation comparison: "
        f"{path}"
    )

    return result


# ============================================================================
# OVERALL AUDIT SUMMARY
# ============================================================================

def create_audit_summary(
    season_coefficients,
    sign_stability,
    bootstrap_summary,
    correlation_pairs,
    missingness_summary,
    validation_comparison,
):
    """
    Create a high-level diagnostic audit summary.

    Statuses are diagnostic indicators, not formal statistical
    acceptance tests.
    """

    print_section(
        "OVERALL AUDIT SUMMARY"
    )

    rows = []

    # ------------------------------------------------------------------
    # Season coefficient stability
    # ------------------------------------------------------------------

    unstable_features = []

    for feature in MODEL_4_FEATURES:

        subset = sign_stability[
            sign_stability["feature"]
            == feature
        ]

        if subset.empty:
            continue

        row = subset.iloc[0]

        positive_pct = row[
            "positive_pct"
        ]

        negative_pct = row[
            "negative_pct"
        ]

        if (
            pd.notna(positive_pct)
            and pd.notna(negative_pct)
            and positive_pct > 0
            and negative_pct > 0
        ):

            unstable_features.append(
                feature
            )

    rows.append(
        {
            "audit":
                "coefficient_sign_stability",
            "status":
                (
                    "REVIEW"
                    if unstable_features
                    else "PASS"
                ),
            "details":
                (
                    f"{len(unstable_features)} features "
                    "show both positive and negative "
                    "coefficients across seasons."
                    if unstable_features
                    else
                    "No feature shows meaningful sign "
                    "reversals across seasons."
                ),
        }
    )

    # ------------------------------------------------------------------
    # Bootstrap stability
    # ------------------------------------------------------------------

    bootstrap_cross_zero = (
        bootstrap_summary[
            bootstrap_summary[
                "ci_crosses_zero"
            ]
        ]["feature"]
        .tolist()
    )

    rows.append(
        {
            "audit":
                "bootstrap_coefficient_stability",
            "status":
                (
                    "REVIEW"
                    if bootstrap_cross_zero
                    else "PASS"
                ),
            "details":
                (
                    f"{len(bootstrap_cross_zero)} Model 4 "
                    "coefficient 95% bootstrap intervals "
                    "cross zero."
                    if bootstrap_cross_zero
                    else
                    "No Model 4 coefficient 95% bootstrap "
                    "interval crosses zero."
                ),
        }
    )

    # ------------------------------------------------------------------
    # Correlation
    # ------------------------------------------------------------------

    high_corr_pairs = correlation_pairs[
        correlation_pairs[
            "absolute_correlation"
        ]
        >= HIGH_CORRELATION_THRESHOLD
    ]

    rows.append(
        {
            "audit":
                "feature_correlation",
            "status":
                (
                    "REVIEW"
                    if not high_corr_pairs.empty
                    else "PASS"
                ),
            "details":
                (
                    f"{len(high_corr_pairs)} feature pairs "
                    f"have absolute correlation >= "
                    f"{HIGH_CORRELATION_THRESHOLD:.2f}."
                ),
        }
    )

    # ------------------------------------------------------------------
    # Missingness
    # ------------------------------------------------------------------

    max_missing = (
        missingness_summary[
            "max_missing_pct"
        ].max()
    )

    rows.append(
        {
            "audit":
                "feature_missingness",
            "status":
                (
                    "REVIEW"
                    if max_missing > 30
                    else "PASS"
                ),
            "details":
                (
                    f"Maximum observed seasonal "
                    f"missingness: "
                    f"{max_missing:.2f}%."
                ),
        }
    )

    # ------------------------------------------------------------------
    # Validation improvement
    # ------------------------------------------------------------------

    model_3_row = validation_comparison[
        validation_comparison[
            "model"
        ] == "Model 3"
    ].iloc[0]

    model_4_row = validation_comparison[
        validation_comparison[
            "model"
        ] == "Model 4"
    ].iloc[0]

    validation_improved = (
        model_4_row["log_loss"]
        < model_3_row["log_loss"]
    )

    rows.append(
        {
            "audit":
                "model_3_to_model_4_validation",
            "status":
                (
                    "PASS"
                    if validation_improved
                    else "REVIEW"
                ),
            "details":
                (
                    f"Validation log loss changed "
                    f"from "
                    f"{model_3_row['log_loss']:.4f} "
                    f"to "
                    f"{model_4_row['log_loss']:.4f}."
                ),
        }
    )

    result = pd.DataFrame(
        rows
    )

    print(
        result.to_string(
            index=False
        )
    )

    path = (
        AUDIT_DIR
        / "audit_summary.csv"
    )

    result.to_csv(
        path,
        index=False,
    )

    print()
    print(
        f"Saved audit summary: {path}"
    )

    return result


# ============================================================================
# MAIN
# ============================================================================

def main():

    warnings.filterwarnings(
        "ignore",
        category=FutureWarning,
    )

    print_section(
        "COMPACT LOGISTIC REGRESSION MODEL 4 "
        "— DIAGNOSTIC / STABILITY AUDIT"
    )

    print(
        f"Project root:   {PROJECT_ROOT}"
    )

    print(
        f"Input directory: {INPUT_DIR}"
    )

    print(
        f"Model directory: {MODEL_DIR}"
    )

    print(
        f"Audit directory: {AUDIT_DIR}"
    )

    # ------------------------------------------------------------------
    # Temporal design
    # ------------------------------------------------------------------

    print_section(
        "TEMPORAL DESIGN"
    )

    print(
        "----------------"
    )

    print(
        f"Training:   "
        f"{TRAIN_YEARS[0]}-{TRAIN_YEARS[-1]}"
    )

    print(
        f"Validation: "
        f"{VALIDATION_YEARS}"
    )

    print(
        f"Test:       "
        f"{TEST_YEARS}"
    )

    print()
    print(
        f"Model 4 feature count: "
        f"{len(MODEL_4_FEATURES)}"
    )

    print()
    print(
        "Model 4 additions:"
    )

    for feature in MODEL_4_ADDITIONS:

        print(
            f"  + {feature}"
        )

    # ------------------------------------------------------------------
    # Output directory
    # ------------------------------------------------------------------

    AUDIT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    # ------------------------------------------------------------------
    # Load data
    # ------------------------------------------------------------------

    data = load_all_data()

    # ------------------------------------------------------------------
    # Validate dataset
    # ------------------------------------------------------------------

    validate_dataset(
        data
    )

    # ------------------------------------------------------------------
    # Load saved models
    # ------------------------------------------------------------------

    model_3, model_4 = (
        load_trained_models()
    )

    # ------------------------------------------------------------------
    # Season coefficient stability
    # ------------------------------------------------------------------

    season_coefficients = (
        train_season_specific_models(
            data
        )
    )

    coefficient_summary = (
        summarize_coefficient_stability(
            season_coefficients
        )
    )

    sign_stability = (
        analyze_sign_stability(
            season_coefficients
        )
    )

    # ------------------------------------------------------------------
    # Saved coefficient comparison
    # ------------------------------------------------------------------

    model_comparison = (
        analyze_saved_model_coefficients(
            model_3,
            model_4,
        )
    )

    # ------------------------------------------------------------------
    # Correlation audit
    # ------------------------------------------------------------------

    correlation_pairs = (
        analyze_feature_correlations(
            data
        )
    )

    # ------------------------------------------------------------------
    # Missingness audit
    # ------------------------------------------------------------------

    (
        missingness_by_season,
        missingness_summary,
    ) = analyze_missingness(
        data
    )

    # ------------------------------------------------------------------
    # Season-by-season performance
    # ------------------------------------------------------------------

    model_4_season_performance = (
        evaluate_model_by_season(
            model_4,
            data,
            MODEL_4_FEATURES,
            "Model 4",
        )
    )

    # ------------------------------------------------------------------
    # Bootstrap stability
    # ------------------------------------------------------------------

    (
        bootstrap_coefficients_df,
        bootstrap_summary,
    ) = bootstrap_coefficients(
        data
    )

    # ------------------------------------------------------------------
    # Prediction stability
    # ------------------------------------------------------------------

    prediction_stability = (
        analyze_prediction_stability(
            model_4,
            data,
        )
    )

    # ------------------------------------------------------------------
    # Calibration
    # ------------------------------------------------------------------

    calibration = (
        analyze_calibration(
            model_4,
            data,
        )
    )

    # ------------------------------------------------------------------
    # Temporal coefficient drift
    # ------------------------------------------------------------------

    temporal_drift = (
        analyze_temporal_coefficient_drift(
            season_coefficients
        )
    )

    # ------------------------------------------------------------------
    # Model 3 → Model 4 validation comparison
    # ------------------------------------------------------------------

    validation_comparison = (
        analyze_model_3_to_model_4_improvement(
            model_3,
            model_4,
            data,
        )
    )

    # ------------------------------------------------------------------
    # Overall summary
    # ------------------------------------------------------------------

    audit_summary = (
        create_audit_summary(
            season_coefficients,
            sign_stability,
            bootstrap_summary,
            correlation_pairs,
            missingness_summary,
            validation_comparison,
        )
    )

    # ------------------------------------------------------------------
    # Final interpretation
    # ------------------------------------------------------------------

    print_section(
        "FINAL AUDIT INTERPRETATION"
    )

    print(
        "Model 4 has completed the diagnostic and "
        "stability audit."
    )

    print()
    print(
        "The audit evaluates:"
    )

    print(
        "  • Season-by-season coefficient stability"
    )

    print(
        "  • Coefficient sign stability"
    )

    print(
        "  • Bootstrap coefficient stability"
    )

    print(
        "  • Model 3 → Model 4 coefficient changes"
    )

    print(
        "  • Feature correlation / multicollinearity "
        "diagnostics"
    )

    print(
        "  • Missingness stability"
    )

    print(
        "  • Season-by-season predictive performance"
    )

    print(
        "  • Prediction probability stability"
    )

    print(
        "  • Validation/test calibration"
    )

    print(
        "  • Temporal coefficient drift"
    )

    print(
        "  • Validation comparison against Model 3"
    )

    print()
    print(
        "IMPORTANT:"
    )

    print(
        "The season-specific and bootstrap models are "
        "diagnostic models only."
    )

    print(
        "They are not used to select or retune Model 4."
    )

    print(
        "The 2025 test season remains a final evaluation "
        "set and is not used for model selection."
    )

    print()
    print(
        "All-missing features in individual seasons are "
        "retained in the diagnostic coefficient vector."
    )

    print(
        "They receive a zero coefficient for that season "
        "because they contain no observed information."
    )

    print_section(
        "COMPACT LOGISTIC REGRESSION MODEL 4 "
        "— AUDIT COMPLETED SUCCESSFULLY"
    )


if __name__ == "__main__":
    main()