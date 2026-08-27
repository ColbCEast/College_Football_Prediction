"""
Audit Logistic Regression Features
==================================

Purpose
-------
Diagnose the predictors currently being supplied to the baseline logistic
regression model.

This audit is intentionally focused on statistical interpretation and
feature quality. It does NOT rebuild matchup features and does NOT repeat
the project's already-completed leakage or temporal-safety audits.

The audit investigates:

1. Actual model predictor structure
2. Feature missingness
3. Missingness patterns
4. Univariate relationship with win_home
5. Expected vs observed feature direction
6. Quantile win-rate relationships
7. Feature distributions
8. Highly correlated predictors
9. Home-vs-away paired feature interpretation
10. Diagnostic summary

Model target
------------
win_home

Target interpretation
---------------------
1 = home team wins
0 = away team wins

Current temporal split
----------------------
Training:   2015-2022
Validation: 2023-2024
Test:       2025

Important
---------
This script audits the ACTUAL predictors present in the temporal modeling
datasets. It does not assume that any particular feature naming convention,
such as "matchup_*", exists.

The baseline logistic regression currently excludes:

- gameId
- season
- startDate
- win_home

Everything else present in the training split is treated as a candidate
predictor, matching the baseline model's feature-selection logic.
"""

from pathlib import Path
import math

import numpy as np
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    log_loss,
    roc_auc_score,
)
from sklearn.preprocessing import StandardScaler


# ============================================================================
# CONFIGURATION
# ============================================================================

PROJECT_ROOT = Path(__file__).resolve().parents[3]

MODEL_DATA_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "modeling"
)

TRAIN_PATH = (
    MODEL_DATA_DIR
    / "logistic_regression_train.csv"
)

VALIDATION_PATH = (
    MODEL_DATA_DIR
    / "logistic_regression_validation.csv"
)

TEST_PATH = (
    MODEL_DATA_DIR
    / "logistic_regression_test.csv"
)

TARGET_COLUMN = "win_home"

IDENTIFIER_COLUMNS = [
    "gameId",
    "season",
]

DATE_COLUMNS = [
    "startDate",
]

TRAIN_YEARS = list(range(2015, 2023))
VALIDATION_YEARS = [2023, 2024]
TEST_YEARS = [2025]

RANDOM_STATE = 42

# Minimum number of non-missing observations required for a univariate
# logistic regression.
MIN_UNIVARIATE_N = 100

# Correlation threshold used for the redundancy audit.
HIGH_CORRELATION_THRESHOLD = 0.90

# Number of highest/lowest quantile relationships to display.
TOP_QUANTILE_RESULTS = 30


# ============================================================================
# GENERAL HELPERS
# ============================================================================

def print_section(title):
    """Print a standardized section header."""

    print()
    print("=" * 70)
    print(title)
    print("=" * 70)


def print_subsection(title):
    """Print a standardized subsection header."""

    print()
    print("-" * 70)
    print(title)
    print("-" * 70)


def load_split(path, name):
    """Load a temporal modeling split."""

    if not path.exists():
        raise FileNotFoundError(
            f"{name} split does not exist:\n{path}"
        )

    df = pd.read_csv(path)

    print(
        f"{name:<12}: "
        f"{len(df):,} rows × {len(df.columns):,} columns"
    )

    return df


# ============================================================================
# SPLIT LOADING
# ============================================================================

def load_temporal_data():
    """Load the existing train, validation, and test splits."""

    print_section("LOADING TEMPORAL DATA")

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

    return train, validation, test


# ============================================================================
# BASIC STRUCTURE
# ============================================================================

def identify_predictors(train):
    """
    Identify predictors exactly as the baseline logistic regression does.

    Excluded:
        - target
        - identifiers
        - raw date columns
    """

    excluded = (
        set(IDENTIFIER_COLUMNS)
        | set(DATE_COLUMNS)
        | {TARGET_COLUMN}
    )

    predictors = [
        column
        for column in train.columns
        if column not in excluded
    ]

    if not predictors:
        raise ValueError(
            "No predictor columns were identified."
        )

    return predictors


def audit_predictor_structure(
    train,
    validation,
    test,
    predictors,
):
    """Describe the actual predictor structure."""

    print_section("MODEL PREDICTOR STRUCTURE")

    print(
        f"Total columns in modeling data: "
        f"{len(train.columns):,}"
    )

    print(
        f"Target column: {TARGET_COLUMN}"
    )

    print(
        f"Predictor columns: {len(predictors):,}"
    )

    print(
        f"Excluded columns: "
        f"{len(train.columns) - len(predictors):,}"
    )

    # ------------------------------------------------------------------------
    # Column consistency
    # ------------------------------------------------------------------------

    train_columns = set(train.columns)
    validation_columns = set(validation.columns)
    test_columns = set(test.columns)

    train_only = sorted(
        train_columns - validation_columns
    )

    validation_only = sorted(
        validation_columns - train_columns
    )

    test_only = sorted(
        test_columns - train_columns
    )

    if train_only:
        print("\nWARNING: Training-only columns:")
        for column in train_only:
            print(f"  {column}")

    if validation_only:
        print("\nWARNING: Validation-only columns:")
        for column in validation_only:
            print(f"  {column}")

    if test_only:
        print("\nWARNING: Test-only columns:")
        for column in test_only:
            print(f"  {column}")

    if not train_only and not validation_only and not test_only:
        print(
            "Column structure is consistent across "
            "training, validation, and test."
        )

    # ------------------------------------------------------------------------
    # Predictor types
    # ------------------------------------------------------------------------

    numeric = [
        column
        for column in predictors
        if pd.api.types.is_numeric_dtype(train[column])
    ]

    categorical = [
        column
        for column in predictors
        if column not in numeric
    ]

    print()
    print(f"Numeric predictors:     {len(numeric):,}")
    print(f"Categorical predictors: {len(categorical):,}")

    if categorical:
        print()
        print("Categorical predictors:")
        for column in categorical:
            print(f"  {column}")


# ============================================================================
# TARGET
# ============================================================================

def validate_target(train, validation, test):
    """Confirm the target is usable for the statistical diagnostics."""

    print_section("TARGET SUMMARY")

    for name, df in [
        ("Training", train),
        ("Validation", validation),
        ("Test", test),
    ]:

        if TARGET_COLUMN not in df.columns:
            raise ValueError(
                f"{name} is missing target column "
                f"'{TARGET_COLUMN}'."
            )

        missing = df[TARGET_COLUMN].isna().sum()

        if missing:
            raise ValueError(
                f"{name} contains {missing:,} missing target values."
            )

        values = sorted(
            pd.Series(df[TARGET_COLUMN]).unique()
        )

        print(f"{name}:")
        print(f"  Home wins:  {int(df[TARGET_COLUMN].sum()):,}")
        print(
            f"  Away wins:  "
            f"{int((df[TARGET_COLUMN] == 0).sum()):,}"
        )
        print(
            f"  Home win rate: "
            f"{df[TARGET_COLUMN].mean():.4f}"
        )
        print(f"  Values: {values}")
        print()

    return train[TARGET_COLUMN]


# ============================================================================
# EXPECTED FEATURE DIRECTION
# ============================================================================

def classify_feature_direction(feature_name):
    """
    Determine the expected relationship between a feature and home-team
    win probability.

    This function is deliberately conservative.

    It returns:
        +1 = higher feature value should favor home wins
        -1 = higher feature value should hurt home wins
         0 = no reliable directional expectation
        None = not applicable / categorical

    The direction is determined from the feature's meaning and whether the
    variable represents the home or away team.

    This is an interpretive expectation, not a hypothesis test.
    """

    name = feature_name.lower()

    # ------------------------------------------------------------------------
    # Determine whether this is a home or away feature.
    # ------------------------------------------------------------------------

    is_home = name.endswith("_home")
    is_away = name.endswith("_away")

    if not is_home and not is_away:
        return 0

    # Remove the home/away suffix for semantic classification.
    base = name.rsplit("_", 1)[0]

    # ------------------------------------------------------------------------
    # Variables where higher values generally indicate better performance.
    # ------------------------------------------------------------------------

    positive_metrics = [
        "winsbefore",
        "winpctbefore",
        "winpctlast3",
        "winpctlast5",
        "pointsfor",
        "pointsforavg",
        "pointdifferential",
        "pointdifferentialavg",
        "rushingyards",
        "rushingyardsavg",
        "rushingtds",
        "rushingtdsavg",
        "netpassingyards",
        "netpassingyardsavg",
        "passingtds",
        "passingtdsavg",
        "completions",
        "completionsavg",
        "completionpct",
        "yardsperpass",
        "yardsperpassattempt",
        "yardsperrushattempt",
        "totalyards",
        "totalyardsavg",
        "firstdowns",
        "firstdownsavg",
        "thirddownconversions",
        "thirddownconversionsavg",
        "thirddownpct",
        "fourthdownconversions",
        "fourthdownconversionsavg",
        "fourthdownpct",
        "sacks",
        "sacksavg",
        "qbhurries",
        "qbhurriesavg",
        "passesdeflected",
        "passesdeflectedavg",
        "tacklesforloss",
        "tacklesforlossavg",
        "defensivetds",
        "defensivetdsavg",
        "offense_ppa",
        "offense_successrate",
        "offense_explosiveness",
        "defense_ppa",
        "defense_successrate",
        "defense_explosiveness",
    ]

    # ------------------------------------------------------------------------
    # Variables where higher values generally indicate worse performance.
    # ------------------------------------------------------------------------

    negative_metrics = [
        "pointsagainst",
        "pointsagainstavg",
        "penalties",
        "penaltiesavg",
        "penaltyyards",
        "penaltyyardsavg",
        "turnovers",
        "turnoversavg",
        "fumbleslost",
        "fumbleslostavg",
        "interceptions",
        "interceptionsavg",
    ]

    # ------------------------------------------------------------------------
    # Some variable names can contain these substrings.
    # Use exact/substring semantic matching conservatively.
    # ------------------------------------------------------------------------

    expected_for_team = None

    for metric in positive_metrics:
        if metric in base:
            expected_for_team = 1
            break

    if expected_for_team is None:
        for metric in negative_metrics:
            if metric in base:
                expected_for_team = -1
                break

    if expected_for_team is None:
        return 0

    # ------------------------------------------------------------------------
    # Reverse the expected direction for away-team predictors.
    # ------------------------------------------------------------------------

    if is_home:
        return expected_for_team

    return -expected_for_team


def direction_label(direction):
    """Convert numeric direction code to a readable label."""

    if direction == 1:
        return "Positive"

    if direction == -1:
        return "Negative"

    return "No clear expectation"


# ============================================================================
# MISSINGNESS
# ============================================================================

def missingness_audit(train, predictors):
    """
    Audit missing values in the training data.

    Missingness is examined against win_home because systematic missingness
    can affect univariate interpretation and model coefficients.
    """

    print_section("MISSINGNESS AUDIT")

    y = train[TARGET_COLUMN]

    rows = []

    for feature in predictors:

        missing_mask = train[feature].isna()

        missing_count = int(missing_mask.sum())

        if missing_count == 0:
            continue

        missing_pct = missing_count / len(train)

        missing_home_win_rate = (
            y[missing_mask].mean()
            if missing_count > 0
            else np.nan
        )

        observed_mask = ~missing_mask

        observed_home_win_rate = (
            y[observed_mask].mean()
            if observed_mask.sum() > 0
            else np.nan
        )

        rate_difference = (
            missing_home_win_rate
            - observed_home_win_rate
        )

        rows.append(
            {
                "feature": feature,
                "missing_count": missing_count,
                "missing_pct": missing_pct,
                "missing_home_win_rate": missing_home_win_rate,
                "observed_home_win_rate": observed_home_win_rate,
                "missing_vs_observed_win_rate_diff": rate_difference,
            }
        )

    if not rows:
        print("No missing predictor values found in training data.")
        return pd.DataFrame()

    result = pd.DataFrame(rows)

    result = result.sort_values(
        [
            "missing_pct",
            "missing_vs_observed_win_rate_diff",
        ],
        ascending=[False, False],
    )

    print(
        result.to_string(
            index=False,
            formatters={
                "missing_pct": "{:.4f}".format,
                "missing_home_win_rate": "{:.4f}".format,
                "observed_home_win_rate": "{:.4f}".format,
                "missing_vs_observed_win_rate_diff": "{:.4f}".format,
            },
        )
    )

    return result


# ============================================================================
# MISSINGNESS PATTERNS
# ============================================================================

def missingness_pattern_audit(train, predictors):
    """
    Identify groups of features that have identical missingness patterns.

    This is useful for determining whether many apparently different
    statistics are missing for the same underlying reason.
    """

    print_section("MISSINGNESS PATTERN AUDIT")

    missing_features = [
        feature
        for feature in predictors
        if train[feature].isna().any()
    ]

    if not missing_features:
        print("No missingness patterns to evaluate.")
        return []

    pattern_groups = {}

    for feature in missing_features:

        pattern = tuple(
            train[feature].isna().astype(np.int8).tolist()
        )

        pattern_groups.setdefault(
            pattern,
            []
        ).append(feature)

    groups = [
        features
        for features in pattern_groups.values()
        if len(features) > 1
    ]

    groups.sort(
        key=len,
        reverse=True,
    )

    print(
        f"Features with missing values: "
        f"{len(missing_features):,}"
    )

    print(
        f"Identical missingness pattern groups: "
        f"{len(groups):,}"
    )

    for i, features in enumerate(groups, start=1):

        print()
        print(
            f"Group {i} "
            f"({len(features)} features):"
        )

        for feature in features:
            print(f"  {feature}")

    return groups


# ============================================================================
# UNIVARIATE LOGISTIC REGRESSION
# ============================================================================

def fit_univariate_logistic(x, y):
    """
    Fit a one-feature standardized logistic regression.

    The feature is standardized using only the available observations.

    Returns:
        coefficient per 1 SD
        odds ratio per 1 SD
        AUC
        log loss
    """

    valid = (
        x.notna()
        & y.notna()
        & np.isfinite(x)
    )

    x_valid = x.loc[valid].astype(float)
    y_valid = y.loc[valid].astype(int)

    if len(x_valid) < MIN_UNIVARIATE_N:
        return None

    if x_valid.nunique() < 2:
        return None

    if y_valid.nunique() < 2:
        return None

    x_array = x_valid.to_numpy().reshape(-1, 1)
    y_array = y_valid.to_numpy()

    scaler = StandardScaler()

    x_scaled = scaler.fit_transform(
        x_array
    )

    model = LogisticRegression(
        max_iter=2000,
        random_state=RANDOM_STATE,
    )

    model.fit(
        x_scaled,
        y_array,
    )

    probability = model.predict_proba(
        x_scaled
    )[:, 1]

    coefficient = float(
        model.coef_[0][0]
    )

    odds_ratio = float(
        np.exp(coefficient)
    )

    try:
        auc = float(
            roc_auc_score(
                y_array,
                probability,
            )
        )
    except ValueError:
        auc = np.nan

    try:
        loss = float(
            log_loss(
                y_array,
                probability,
            )
        )
    except ValueError:
        loss = np.nan

    return {
        "n": len(x_valid),
        "coefficient_per_1sd": coefficient,
        "odds_ratio_per_1sd": odds_ratio,
        "auc": auc,
        "log_loss": loss,
    }


def univariate_feature_audit(train, predictors):
    """
    Calculate univariate logistic relationships for every numeric predictor.
    """

    print_section("UNIVARIATE FEATURE AUDIT")

    y = train[TARGET_COLUMN]

    rows = []

    for feature in predictors:

        if not pd.api.types.is_numeric_dtype(
            train[feature]
        ):
            continue

        result = fit_univariate_logistic(
            train[feature],
            y,
        )

        if result is None:
            continue

        missing_pct = (
            train[feature].isna().mean()
        )

        expected_direction = (
            classify_feature_direction(feature)
        )

        observed_direction = (
            1
            if result["coefficient_per_1sd"] > 0
            else -1
            if result["coefficient_per_1sd"] < 0
            else 0
        )

        if expected_direction == 0:
            direction_status = "No expectation"

        elif observed_direction == expected_direction:
            direction_status = "Matches"

        else:
            direction_status = "Opposes"

        rows.append(
            {
                "feature": feature,
                "n": result["n"],
                "expected_direction": direction_label(
                    expected_direction
                ),
                "coefficient_per_1sd": (
                    result["coefficient_per_1sd"]
                ),
                "odds_ratio_per_1sd": (
                    result["odds_ratio_per_1sd"]
                ),
                "auc": result["auc"],
                "log_loss": result["log_loss"],
                "missing_pct": missing_pct,
                "direction_status": direction_status,
            }
        )

    if not rows:
        print("No numeric predictors were available.")
        return pd.DataFrame()

    result = pd.DataFrame(rows)

    result = result.sort_values(
        "auc",
        ascending=False,
    )

    print(
        result.to_string(
            index=False,
            formatters={
                "coefficient_per_1sd": "{:.4f}".format,
                "odds_ratio_per_1sd": "{:.4f}".format,
                "auc": "{:.4f}".format,
                "log_loss": "{:.4f}".format,
                "missing_pct": "{:.4f}".format,
            },
        )
    )

    return result


# ============================================================================
# DIRECTION AUDIT
# ============================================================================

def direction_audit(univariate_results):
    """
    Summarize features whose observed univariate direction opposes the
    expected direction.
    """

    print_section("EXPECTED VS OBSERVED DIRECTION AUDIT")

    if univariate_results.empty:
        print("No univariate results available.")
        return pd.DataFrame()

    comparable = univariate_results[
        univariate_results["direction_status"].isin(
            ["Matches", "Opposes"]
        )
    ].copy()

    if comparable.empty:
        print("No features had a defined expected direction.")
        return pd.DataFrame()

    counts = (
        comparable["direction_status"]
        .value_counts()
    )

    matches = int(
        counts.get("Matches", 0)
    )

    opposes = int(
        counts.get("Opposes", 0)
    )

    print(f"Features with expected direction: {len(comparable):,}")
    print(f"Observed direction matches:       {matches:,}")
    print(f"Observed direction opposes:       {opposes:,}")

    print()

    opposing = comparable[
        comparable["direction_status"] == "Opposes"
    ].copy()

    if opposing.empty:
        print(
            "No univariate features showed a direction "
            "opposite to the expected direction."
        )

        return opposing

    opposing = opposing.sort_values(
        "auc",
        ascending=False,
    )

    print(
        "Features requiring interpretation:"
    )

    print(
        opposing.to_string(
            index=False,
            formatters={
                "coefficient_per_1sd": "{:.4f}".format,
                "odds_ratio_per_1sd": "{:.4f}".format,
                "auc": "{:.4f}".format,
                "log_loss": "{:.4f}".format,
                "missing_pct": "{:.4f}".format,
            },
        )
    )

    print()
    print(
        "IMPORTANT:"
    )
    print(
        "An opposing univariate sign is a diagnostic flag, "
        "not automatically evidence of a data error."
    )
    print(
        "It can result from confounding, correlated predictors, "
        "feature definition, or the distinction between offense "
        "and defensive statistics."
    )

    return opposing


# ============================================================================
# QUANTILE WIN-RATE AUDIT
# ============================================================================

def quantile_win_rate_audit(
    train,
    predictors,
    quantiles=5,
):
    """
    Compare home win rates across feature quantiles.

    This provides a nonlinear/interpretable diagnostic that complements
    the univariate logistic coefficient.
    """

    print_section("FEATURE QUANTILE WIN-RATE AUDIT")

    y = train[TARGET_COLUMN]

    rows = []

    for feature in predictors:

        if not pd.api.types.is_numeric_dtype(
            train[feature]
        ):
            continue

        valid = (
            train[feature].notna()
            & np.isfinite(train[feature])
        )

        if valid.sum() < MIN_UNIVARIATE_N:
            continue

        x = train.loc[valid, feature]
        target = y.loc[valid]

        if x.nunique() < quantiles:
            continue

        try:
            bins = pd.qcut(
                x,
                q=quantiles,
                duplicates="drop",
            )
        except ValueError:
            continue

        grouped = (
            target
            .groupby(
                bins,
                observed=True,
            )
            .mean()
        )

        if len(grouped) < 2:
            continue

        lowest = float(grouped.iloc[0])
        highest = float(grouped.iloc[-1])

        difference = highest - lowest

        rows.append(
            {
                "feature": feature,
                "lowest_quantile_win_rate": lowest,
                "highest_quantile_win_rate": highest,
                "win_rate_difference": difference,
                "abs_difference": abs(difference),
            }
        )

    if not rows:
        print("No quantile relationships could be calculated.")
        return pd.DataFrame()

    result = pd.DataFrame(rows)

    result = result.sort_values(
        "abs_difference",
        ascending=False,
    )

    print(
        result.head(TOP_QUANTILE_RESULTS).to_string(
            index=False,
            formatters={
                "lowest_quantile_win_rate": "{:.4f}".format,
                "highest_quantile_win_rate": "{:.4f}".format,
                "win_rate_difference": "{:.4f}".format,
                "abs_difference": "{:.4f}".format,
            },
        )
    )

    return result


# ============================================================================
# FEATURE DISTRIBUTIONS
# ============================================================================

def feature_distribution_audit(train, predictors):
    """Summarize numeric feature distributions."""

    print_section("FEATURE DISTRIBUTION AUDIT")

    rows = []

    for feature in predictors:

        if not pd.api.types.is_numeric_dtype(
            train[feature]
        ):
            continue

        x = train[feature].dropna()

        if x.empty:
            continue

        x = x[np.isfinite(x)]

        if x.empty:
            continue

        q01 = x.quantile(0.01)
        q25 = x.quantile(0.25)
        median = x.quantile(0.50)
        q75 = x.quantile(0.75)
        q99 = x.quantile(0.99)

        mean = x.mean()
        std = x.std()

        if std == 0 or pd.isna(std):
            cv = np.nan
        else:
            cv = abs(mean / std)

        rows.append(
            {
                "feature": feature,
                "n": len(x),
                "missing_pct": train[feature].isna().mean(),
                "unique_values": x.nunique(),
                "mean": mean,
                "std": std,
                "min": x.min(),
                "q01": q01,
                "q25": q25,
                "median": median,
                "q75": q75,
                "q99": q99,
                "max": x.max(),
                "skew": x.skew(),
                "near_constant": x.nunique() <= 2,
            }
        )

    if not rows:
        print("No numeric predictors available.")
        return pd.DataFrame()

    result = pd.DataFrame(rows)

    print(
        result.to_string(
            index=False,
            formatters={
                "missing_pct": "{:.4f}".format,
                "mean": "{:.4f}".format,
                "std": "{:.4f}".format,
                "min": "{:.4f}".format,
                "q01": "{:.4f}".format,
                "q25": "{:.4f}".format,
                "median": "{:.4f}".format,
                "q75": "{:.4f}".format,
                "q99": "{:.4f}".format,
                "max": "{:.4f}".format,
                "skew": "{:.4f}".format,
            },
        )
    )

    print_subsection("Near-constant features")

    near_constant = result[
        result["near_constant"]
    ]

    if near_constant.empty:
        print("No near-constant numeric features detected.")
    else:
        for feature in near_constant["feature"]:
            print(f"  {feature}")

    return result


# ============================================================================
# HOME / AWAY PAIR IDENTIFICATION
# ============================================================================

def identify_home_away_pairs(predictors):
    """
    Identify numeric home/away feature pairs.

    Example:
        pointsForAvgBefore_home
        pointsForAvgBefore_away

    becomes:

        pointsForAvgBefore
    """

    predictor_set = set(predictors)

    pairs = []

    for feature in predictors:

        if not feature.endswith("_home"):
            continue

        base = feature[:-5]
        away_feature = f"{base}_away"

        if away_feature not in predictor_set:
            continue

        pairs.append(
            (
                base,
                feature,
                away_feature,
            )
        )

    return pairs


# ============================================================================
# HOME-AWAY INTERPRETIVE AUDIT
# ============================================================================

def home_away_direction_audit(train, predictors):
    """
    Construct home-minus-away differences solely for interpretation.

    These are NOT added to the model.

    The purpose is to determine whether the underlying home/away feature
    relationship behaves as expected when expressed in the natural
    matchup direction.
    """

    print_section("HOME VS AWAY FEATURE DIRECTION AUDIT")

    pairs = identify_home_away_pairs(
        predictors
    )

    if not pairs:
        print(
            "No home/away feature pairs were identified."
        )
        return pd.DataFrame()

    y = train[TARGET_COLUMN]

    rows = []

    for base, home_feature, away_feature in pairs:

        if not (
            pd.api.types.is_numeric_dtype(
                train[home_feature]
            )
            and pd.api.types.is_numeric_dtype(
                train[away_feature]
            )
        ):
            continue

        difference = (
            train[home_feature]
            - train[away_feature]
        )

        valid = (
            difference.notna()
            & np.isfinite(difference)
        )

        if valid.sum() < MIN_UNIVARIATE_N:
            continue

        x = difference.loc[valid]
        target = y.loc[valid]

        if x.nunique() < 2:
            continue

        scaler = StandardScaler()

        x_scaled = scaler.fit_transform(
            x.to_numpy().reshape(-1, 1)
        )

        model = LogisticRegression(
            max_iter=2000,
            random_state=RANDOM_STATE,
        )

        model.fit(
            x_scaled,
            target,
        )

        coefficient = float(
            model.coef_[0][0]
        )

        probability = model.predict_proba(
            x_scaled
        )[:, 1]

        auc = roc_auc_score(
            target,
            probability,
        )

        expected = classify_feature_direction(
            home_feature
        )

        observed = (
            1
            if coefficient > 0
            else -1
            if coefficient < 0
            else 0
        )

        if expected == 0:
            status = "No expectation"
        elif observed == expected:
            status = "Matches"
        else:
            status = "Opposes"

        rows.append(
            {
                "feature": base,
                "home_feature": home_feature,
                "away_feature": away_feature,
                "expected_direction": direction_label(
                    expected
                ),
                "coefficient_per_1sd": coefficient,
                "auc": auc,
                "direction_status": status,
                "n": int(valid.sum()),
            }
        )

    if not rows:
        print(
            "No numeric home/away pairs could be evaluated."
        )
        return pd.DataFrame()

    result = pd.DataFrame(rows)

    result = result.sort_values(
        "auc",
        ascending=False,
    )

    print(
        result.to_string(
            index=False,
            formatters={
                "coefficient_per_1sd": "{:.4f}".format,
                "auc": "{:.4f}".format,
            },
        )
    )

    return result


# ============================================================================
# CORRELATION AUDIT
# ============================================================================

def correlation_audit(train, predictors):
    """
    Identify highly correlated numeric predictors.

    Correlation is calculated using pairwise complete observations.
    """

    print_section("FEATURE CORRELATION AUDIT")

    numeric_features = [
        feature
        for feature in predictors
        if pd.api.types.is_numeric_dtype(
            train[feature]
        )
    ]

    if len(numeric_features) < 2:
        print("Not enough numeric predictors.")
        return pd.DataFrame()

    X = train[numeric_features]

    correlation = X.corr(
        method="pearson"
    )

    rows = []

    for i, feature_a in enumerate(
        numeric_features
    ):

        for feature_b in numeric_features[
            i + 1:
        ]:

            value = correlation.loc[
                feature_a,
                feature_b,
            ]

            if pd.isna(value):
                continue

            if abs(value) >= HIGH_CORRELATION_THRESHOLD:

                rows.append(
                    {
                        "feature_a": feature_a,
                        "feature_b": feature_b,
                        "correlation": value,
                        "abs_correlation": abs(value),
                    }
                )

    if not rows:
        print(
            f"No predictor pairs exceeded "
            f"|r| >= {HIGH_CORRELATION_THRESHOLD:.2f}."
        )
        return pd.DataFrame()

    result = pd.DataFrame(rows)

    result = result.sort_values(
        "abs_correlation",
        ascending=False,
    )

    print(
        result.to_string(
            index=False,
            formatters={
                "correlation": "{:.4f}".format,
                "abs_correlation": "{:.4f}".format,
            },
        )
    )

    return result


# ============================================================================
# SEASONAL UNIVARIATE STABILITY
# ============================================================================

def seasonal_univariate_direction_audit(
    train,
    predictors,
):
    """
    Check whether univariate feature direction is broadly stable by season.

    This is an interpretive diagnostic, not a temporal-safety test.
    """

    print_section("SEASONAL UNIVARIATE DIRECTION AUDIT")

    if "season" not in train.columns:
        print(
            "Season column is not available."
        )
        return pd.DataFrame()

    numeric_features = [
        feature
        for feature in predictors
        if pd.api.types.is_numeric_dtype(
            train[feature]
        )
    ]

    rows = []

    for feature in numeric_features:

        expected = classify_feature_direction(
            feature
        )

        if expected == 0:
            continue

        for season in sorted(
            train["season"].dropna().unique()
        ):

            subset = train[
                train["season"] == season
            ]

            result = fit_univariate_logistic(
                subset[feature],
                subset[TARGET_COLUMN],
            )

            if result is None:
                continue

            observed = (
                1
                if result["coefficient_per_1sd"] > 0
                else -1
                if result["coefficient_per_1sd"] < 0
                else 0
            )

            rows.append(
                {
                    "feature": feature,
                    "season": int(season),
                    "expected_direction": expected,
                    "observed_direction": observed,
                    "coefficient": (
                        result["coefficient_per_1sd"]
                    ),
                    "auc": result["auc"],
                }
            )

    if not rows:
        print(
            "No seasonal direction results available."
        )
        return pd.DataFrame()

    result = pd.DataFrame(rows)

    stability_rows = []

    for feature, group in result.groupby(
        "feature"
    ):

        expected = int(
            group["expected_direction"].iloc[0]
        )

        matching = (
            group["observed_direction"]
            == expected
        )

        opposite = (
            group["observed_direction"]
            == -expected
        )

        stability_rows.append(
            {
                "feature": feature,
                "seasons_evaluated": len(group),
                "matching_seasons": int(
                    matching.sum()
                ),
                "opposing_seasons": int(
                    opposite.sum()
                ),
                "direction_stability": (
                    matching.mean()
                ),
            }
        )

    stability = pd.DataFrame(
        stability_rows
    )

    stability = stability.sort_values(
        "direction_stability"
    )

    print(
        stability.to_string(
            index=False,
            formatters={
                "direction_stability": "{:.4f}".format,
            },
        )
    )

    return stability


# ============================================================================
# FINAL DIAGNOSTIC SUMMARY
# ============================================================================

def print_diagnostic_summary(
    univariate_results,
    missingness_results,
    correlation_results,
    direction_results,
    home_away_results,
):
    """Print a concise summary of the audit."""

    print_section("AUDIT SUMMARY")

    # ------------------------------------------------------------------------
    # Univariate
    # ------------------------------------------------------------------------

    if univariate_results.empty:
        print(
            "Univariate relationships: "
            "No numeric results available."
        )
    else:
        strongest = (
            univariate_results
            .sort_values(
                "auc",
                ascending=False,
            )
            .head(10)
        )

        print(
            f"Numeric predictors audited: "
            f"{len(univariate_results):,}"
        )

        print()
        print("Strongest univariate relationships by AUC:")

        for _, row in strongest.iterrows():
            print(
                f"  {row['feature']}: "
                f"AUC={row['auc']:.4f}, "
                f"coef={row['coefficient_per_1sd']:.4f}"
            )

    # ------------------------------------------------------------------------
    # Missingness
    # ------------------------------------------------------------------------

    print()

    if missingness_results.empty:
        print(
            "Missingness: No missing predictor values."
        )
    else:
        print(
            f"Predictors with missing values: "
            f"{len(missingness_results):,}"
        )

        highest_missing = (
            missingness_results
            .sort_values(
                "missing_pct",
                ascending=False,
            )
            .head(5)
        )

        print(
            "Highest missingness:"
        )

        for _, row in highest_missing.iterrows():
            print(
                f"  {row['feature']}: "
                f"{row['missing_pct']:.2%}"
            )

    # ------------------------------------------------------------------------
    # Direction
    # ------------------------------------------------------------------------

    print()

    if direction_results.empty:
        print(
            "Direction audit: "
            "No opposing directions identified."
        )
    else:
        print(
            f"Features with opposing univariate direction: "
            f"{len(direction_results):,}"
        )

    # ------------------------------------------------------------------------
    # Correlation
    # ------------------------------------------------------------------------

    print()

    if correlation_results.empty:
        print(
            "High correlation: "
            f"No pairs with |r| >= "
            f"{HIGH_CORRELATION_THRESHOLD:.2f}."
        )
    else:
        print(
            f"Highly correlated predictor pairs: "
            f"{len(correlation_results):,}"
        )

    # ------------------------------------------------------------------------
    # Home-away
    # ------------------------------------------------------------------------

    print()

    if home_away_results.empty:
        print(
            "Home-away paired diagnostics: "
            "No evaluable pairs."
        )
    else:
        opposing = home_away_results[
            home_away_results[
                "direction_status"
            ]
            == "Opposes"
        ]

        print(
            f"Home-away numeric pairs audited: "
            f"{len(home_away_results):,}"
        )

        print(
            f"Home-away pairs opposing expected direction: "
            f"{len(opposing):,}"
        )

    print()
    print(
        "This audit is diagnostic only."
    )
    print(
        "It does not perform feature selection, "
        "model tuning, leakage validation, "
        "or temporal-safety validation."
    )


# ============================================================================
# MAIN
# ============================================================================

def main():

    print()
    print("=" * 70)
    print("LOGISTIC REGRESSION FEATURE AUDIT")
    print("=" * 70)

    print()
    print(
        "Purpose: statistical feature diagnostics "
        "for the current logistic regression predictors."
    )

    print(
        "Target: win_home "
        "(1 = home win, 0 = away win)"
    )

    print()

    # ========================================================================
    # 1. LOAD DATA
    # ========================================================================

    train, validation, test = (
        load_temporal_data()
    )

    # ========================================================================
    # 2. TARGET
    # ========================================================================

    validate_target(
        train,
        validation,
        test,
    )

    # ========================================================================
    # 3. IDENTIFY ACTUAL PREDICTORS
    # ========================================================================

    print_section(
        "IDENTIFYING ACTUAL MODEL PREDICTORS"
    )

    predictors = identify_predictors(
        train
    )

    print(
        f"Actual predictors used by baseline model: "
        f"{len(predictors):,}"
    )

    audit_predictor_structure(
        train,
        validation,
        test,
        predictors,
    )

    # ========================================================================
    # 4. MISSINGNESS
    # ========================================================================

    missingness_results = (
        missingness_audit(
            train,
            predictors,
        )
    )

    # ========================================================================
    # 5. MISSINGNESS PATTERNS
    # ========================================================================

    missingness_pattern_audit(
        train,
        predictors,
    )

    # ========================================================================
    # 6. UNIVARIATE RELATIONSHIPS
    # ========================================================================

    univariate_results = (
        univariate_feature_audit(
            train,
            predictors,
        )
    )

    # ========================================================================
    # 7. EXPECTED VS OBSERVED DIRECTIONS
    # ========================================================================

    direction_results = (
        direction_audit(
            univariate_results,
        )
    )

    # ========================================================================
    # 8. QUANTILE RELATIONSHIPS
    # ========================================================================

    quantile_win_rate_audit(
        train,
        predictors,
    )

    # ========================================================================
    # 9. DISTRIBUTIONS
    # ========================================================================

    feature_distribution_audit(
        train,
        predictors,
    )

    # ========================================================================
    # 10. HOME / AWAY INTERPRETATION
    # ========================================================================

    home_away_results = (
        home_away_direction_audit(
            train,
            predictors,
        )
    )

    # ========================================================================
    # 11. CORRELATION
    # ========================================================================

    correlation_results = (
        correlation_audit(
            train,
            predictors,
        )
    )

    # ========================================================================
    # 12. SEASONAL DIRECTION STABILITY
    # ========================================================================

    seasonal_univariate_direction_audit(
        train,
        predictors,
    )

    # ========================================================================
    # 13. SUMMARY
    # ========================================================================

    print_diagnostic_summary(
        univariate_results,
        missingness_results,
        correlation_results,
        direction_results,
        home_away_results,
    )

    # ========================================================================
    # COMPLETE
    # ========================================================================

    print()
    print("=" * 70)
    print("LOGISTIC REGRESSION FEATURE AUDIT COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()