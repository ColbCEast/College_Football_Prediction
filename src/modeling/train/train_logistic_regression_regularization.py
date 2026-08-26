"""
train_logistic_regression_regularization.py

Engineered Logistic Regression V5

After handling multicollinearity and missingness,
V5 will experiment with regularization
"""

from pathlib import Path
import json
import warnings

import joblib
import numpy as np
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
    log_loss,
    brier_score_loss,
    precision_score,
    recall_score,
)
from sklearn.preprocessing import StandardScaler

from statsmodels.stats.outliers_influence import variance_inflation_factor


warnings.filterwarnings("ignore")


# ======================================================================
# PATHS
# ======================================================================

PROJECT_ROOT = Path(__file__).resolve().parents[3]

DATA_DIR = PROJECT_ROOT / "data" / "processed" / "modeling"
MODEL_DIR = PROJECT_ROOT / "models" / "logistic_regression"

TRAIN_PATH = DATA_DIR / "logistic_regression_train.csv"
VALIDATION_PATH = DATA_DIR / "logistic_regression_validation.csv"
TEST_PATH = DATA_DIR / "logistic_regression_test.csv"

MODEL_PATH = MODEL_DIR / "logistic_regression_engineered_v5.joblib"
METRICS_PATH = MODEL_DIR / "logistic_regression_engineered_v5_metrics.json"

FEATURE_SELECTION_PATH = (
    MODEL_DIR / "engineered_v5_selected_features.json"
)

VIF_PATH = (
    MODEL_DIR / "engineered_v5_vif_history.csv"
)

MISSINGNESS_PATH = (
    MODEL_DIR / "engineered_v5_missingness.csv"
)


TARGET = "win_home"


# ======================================================================
# V2 CANONICAL FEATURE DEFINITIONS
# ======================================================================

# These are the canonical statistics used to construct the V2 matchup
# features. Each statistic must have a _home and _away version.
#
# The resulting feature is:
#
#     home_value - away_value
#
# This list intentionally mirrors the V2 feature engineering process.

ADVANCED_METRIC_PAIRS = {
    "offenseExplosiveness": (
        "home_pregame_offense_explosiveness",
        "away_pregame_offense_explosiveness",
    ),
    "offensePPA": (
        "home_pregame_offense_ppa",
        "away_pregame_offense_ppa",
    ),
    "offenseSuccessRate": (
        "home_pregame_offense_successRate",
        "away_pregame_offense_successRate",
    ),
    "defenseExplosiveness": (
        "home_pregame_defense_explosiveness",
        "away_pregame_defense_explosiveness",
    ),
    "defensePPA": (
        "home_pregame_defense_ppa",
        "away_pregame_defense_ppa",
    ),
    "defenseSuccessRate": (
        "home_pregame_defense_successRate",
        "away_pregame_defense_successRate",
    ),
}


# Features that should not be eliminated merely because they have
# multicollinearity with other measures of team strength.
#
# These are conceptual anchors rather than arbitrary statistical
# features.
PROTECTED_FEATURES = {
    "matchup_elo",
}


# Thresholds
VIF_THRESHOLD = 5.0
MISSINGNESS_THRESHOLD = 0.25

# Values for Regularization test
REGULARIZATION_C_VALUES = [
    0.01,
    0.03,
    0.10,
    0.30,
    1.00,
    3.00,
    10.00,
]

# ======================================================================
# UTILITY FUNCTIONS
# ======================================================================

def print_section(title):
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)


def load_data():
    print_section("LOADING TEMPORAL SPLITS")

    print(f"Loading Training: {TRAIN_PATH}")
    train = pd.read_csv(TRAIN_PATH)
    print(f"  Shape: {train.shape}")

    print(f"Loading Validation: {VALIDATION_PATH}")
    validation = pd.read_csv(VALIDATION_PATH)
    print(f"  Shape: {validation.shape}")

    print(f"Loading Test: {TEST_PATH}")
    test = pd.read_csv(TEST_PATH)
    print(f"  Shape: {test.shape}")

    return train, validation, test


def validate_targets(train, validation, test):
    print_section("VALIDATING TARGETS")

    for name, df in [
        ("Training", train),
        ("Validation", validation),
        ("Test", test),
    ]:
        if TARGET not in df.columns:
            raise ValueError(
                f"{TARGET} missing from {name} dataset."
            )

        if df[TARGET].isna().any():
            raise ValueError(
                f"{TARGET} contains missing values in {name}."
            )

        unique_values = set(df[TARGET].unique())

        if not unique_values.issubset({0, 1}):
            raise ValueError(
                f"{name} target contains unexpected values: "
                f"{unique_values}"
            )

        print(f"{name}: {df[TARGET].value_counts().to_dict()}")


# ======================================================================
# V2 FEATURE ENGINEERING
# ======================================================================

def create_matchup_features(df):
    """
    Create V2/V3 matchup features.

    For each selected statistic:
        matchup_feature = home_value - away_value

    Advanced pregame metrics use their actual column names because
    they do not follow the standard {stat}_home / {stat}_away naming
    convention.
    """

    print("\n" + "=" * 70)
    print("CREATING ENGINEERED MATCHUP FEATURES")
    print("=" * 70)

    # ------------------------------------------------------------------
    # STANDARD STATISTICS
    # ------------------------------------------------------------------

    canonical_stats = [
        # Passing
        "completionsAvgBefore",
        "netPassingYardsAvgBefore",
        "netPassingYardsAvgLast3",
        "netPassingYardsAvgLast5",
        "passAttemptsAvgBefore",
        "passingTDsAvgBefore",
        "yardsPerPassAttemptBefore",

        # Rushing
        "rushingYardsAvgBefore",
        "rushingYardsAvgLast3",
        "rushingYardsAvgLast5",
        "rushingAttemptsAvgBefore",
        "rushingTDsAvgBefore",
        "yardsPerRushAttemptBefore",

        # Total offense
        "totalYardsAvgBefore",
        "totalYardsAvgLast3",
        "totalYardsAvgLast5",

        # Scoring / results
        "pointsForAvgBefore",
        "pointsForAvgLast3",
        "pointsForAvgLast5",
        "pointsAgainstAvgBefore",
        "pointsAgainstAvgLast3",
        "pointsAgainstAvgLast5",
        "pointDifferentialAvgBefore",
        "pointDifferentialAvgLast3",
        "pointDifferentialAvgLast5",
        "winPctBefore",

        # Turnovers
        "turnoversAvgBefore",
        "turnoversAvgLast3",
        "turnoversAvgLast5",
        "fumblesLostAvgBefore",
        "interceptionsAvgBefore",

        # Defense
        "sacksAvgBefore",
        "sacksAvgLast3",
        "sacksAvgLast5",
        "qbHurriesAvgBefore",
        "qbHurriesAvgLast3",
        "qbHurriesAvgLast5",
        "tacklesForLossAvgBefore",
        "tacklesForLossAvgLast3",
        "tacklesForLossAvgLast5",
        "passesDeflectedAvgBefore",
        "passesDeflectedAvgLast3",
        "passesDeflectedAvgLast5",

        # Situational
        "thirdDownAttemptsAvgBefore",
        "thirdDownConversionsAvgBefore",
        "thirdDownPctBefore",
        "fourthDownAttemptsAvgBefore",
        "fourthDownConversionsAvgBefore",
        "fourthDownPctBefore",

        # Other
        "firstDownsAvgBefore",
        "possessionSecondsAvgBefore",
        "penaltiesAvgBefore",
        "penaltyYardsAvgBefore",

        # Additional passing/rushing volume
        "completionPctBefore",
        "rushingYardsAvgBefore",
        "passAttemptsAvgBefore",
    ]

    # Remove accidental duplicates while preserving order
    canonical_stats = list(dict.fromkeys(canonical_stats))

    # ------------------------------------------------------------------
    # ACTUAL COLUMN NAMES FOR ADVANCED PREGAME METRICS
    #
    # These DO NOT follow the normal {stat}_home / {stat}_away schema.
    # ------------------------------------------------------------------

    advanced_metric_pairs = {
        "offenseExplosiveness": (
            "home_pregame_offense_explosiveness",
            "away_pregame_offense_explosiveness",
        ),
        "offensePPA": (
            "home_pregame_offense_ppa",
            "away_pregame_offense_ppa",
        ),
        "offenseSuccessRate": (
            "home_pregame_offense_successRate",
            "away_pregame_offense_successRate",
        ),
        "defenseExplosiveness": (
            "home_pregame_defense_explosiveness",
            "away_pregame_defense_explosiveness",
        ),
        "defensePPA": (
            "home_pregame_defense_ppa",
            "away_pregame_defense_ppa",
        ),
        "defenseSuccessRate": (
            "home_pregame_defense_successRate",
            "away_pregame_defense_successRate",
        ),
    }

    matchup_features = {}

    # ------------------------------------------------------------------
    # CREATE STANDARD MATCHUP FEATURES
    # ------------------------------------------------------------------

    print(f"\nCanonical statistics selected: {len(canonical_stats)}")

    missing_standard = []

    for stat in canonical_stats:

        home_col = f"{stat}_home"
        away_col = f"{stat}_away"

        if home_col not in df.columns or away_col not in df.columns:
            missing_standard.append(
                (stat, home_col, away_col)
            )
            continue

        matchup_features[f"matchup_{stat}"] = (
            df[home_col] - df[away_col]
        )

    # ------------------------------------------------------------------
    # CREATE ADVANCED-METRIC MATCHUPS
    # ------------------------------------------------------------------

    missing_advanced = []

    for feature_name, (home_col, away_col) in advanced_metric_pairs.items():

        if home_col not in df.columns or away_col not in df.columns:
            missing_advanced.append(
                (feature_name, home_col, away_col)
            )
            continue

        matchup_features[f"matchup_{feature_name}"] = (
            df[home_col] - df[away_col]
        )

    # ------------------------------------------------------------------
    # ADD ELO
    #
    # ELO is deliberately represented as a matchup difference rather
    # than separate home/away absolute features.
    # ------------------------------------------------------------------

    if (
        "homePregameElo" in df.columns
        and "awayPregameElo" in df.columns
    ):
        matchup_features["matchup_elo"] = (
            df["homePregameElo"] - df["awayPregameElo"]
        )
    else:
        raise ValueError(
            "Missing required ELO columns: "
            "homePregameElo and/or awayPregameElo"
        )

    # ------------------------------------------------------------------
    # VALIDATION
    # ------------------------------------------------------------------

    if missing_standard:
        print("\nWARNING: Missing standard feature pairs:")

        for stat, home_col, away_col in missing_standard:
            print(f"  {stat}: {home_col}, {away_col}")

    if missing_advanced:
        print("\nWARNING: Missing advanced feature pairs:")

        for feature_name, home_col, away_col in missing_advanced:
            print(
                f"  {feature_name}: "
                f"{home_col}, {away_col}"
            )

    # The advanced metrics are expected to exist in this dataset.
    if missing_advanced:
        raise ValueError(
            "Required advanced pregame feature pairs are missing."
        )

    result = pd.DataFrame(
        matchup_features,
        index=df.index
    )

    print(f"\nEngineered matchup features: {result.shape[1]}")

    return result


# ======================================================================
# MISSINGNESS ANALYSIS
# ======================================================================

def analyze_missingness(X_train, y_train):
    """
    Analyze missingness using training data only.

    Returns a DataFrame containing:
        - missing count
        - missing percentage
        - missing rate when home wins
        - missing rate when home loses
        - difference between the two
    """

    rows = []

    for column in X_train.columns:

        missing = X_train[column].isna()

        total_missing = int(missing.sum())
        missing_pct = missing.mean()

        if missing.any():

            home_win_missing_rate = missing[
                y_train.values == 1
            ].mean()

            away_win_missing_rate = missing[
                y_train.values == 0
            ].mean()

        else:
            home_win_missing_rate = 0.0
            away_win_missing_rate = 0.0

        rows.append(
            {
                "feature": column,
                "missing_count": total_missing,
                "missing_pct": missing_pct,
                "missing_rate_home_win": home_win_missing_rate,
                "missing_rate_home_loss": away_win_missing_rate,
                "missing_rate_difference": (
                    home_win_missing_rate
                    - away_win_missing_rate
                ),
            }
        )

    result = pd.DataFrame(rows)

    result = result.sort_values(
        "missing_pct",
        ascending=False
    )

    return result


# ======================================================================
# VIF CALCULATION
# ======================================================================

def calculate_vif(X):
    """
    Calculate VIF values.

    Missing values must already be imputed before this function
    is called.

    Standardization is used so that the regression underlying VIF
    is numerically stable.
    """

    if X.shape[1] == 1:
        return pd.DataFrame(
            {
                "feature": X.columns,
                "vif": [1.0],
            }
        )

    scaler = StandardScaler()

    X_scaled = scaler.fit_transform(X)

    vif_rows = []

    for i, column in enumerate(X.columns):

        try:
            vif = variance_inflation_factor(
                X_scaled,
                i
            )
        except Exception:
            vif = np.inf

        if not np.isfinite(vif):
            vif = np.inf

        vif_rows.append(
            {
                "feature": column,
                "vif": vif,
            }
        )

    return (
        pd.DataFrame(vif_rows)
        .sort_values("vif", ascending=False)
        .reset_index(drop=True)
    )


# ======================================================================
# ITERATIVE VIF REDUCTION
# ======================================================================

def iterative_vif_selection(
    X_train,
    protected_features,
    threshold=5.0,
):
    """
    Iteratively remove the highest-VIF feature until all
    unprotected features have VIF < threshold.

    VIF is calculated using training data only.

    Protected features remain regardless of their VIF.
    """

    print_section("ITERATIVE VIF FEATURE REDUCTION")

    print(f"Initial features: {X_train.shape[1]}")
    print(f"VIF threshold   : {threshold}")
    print(
        "Protected features:"
    )

    for feature in protected_features:
        print(f"  {feature}")

    # ------------------------------------------------------------------
    # Impute temporarily for VIF calculation.
    #
    # IMPORTANT:
    # This imputation is only used for feature selection.
    # The actual modeling imputation is performed later.
    # ------------------------------------------------------------------

    X_work = X_train.copy()

    medians = X_work.median(numeric_only=True)

    X_work = X_work.fillna(medians)

    remaining = list(X_work.columns)

    history = []

    iteration = 0

    while len(remaining) > 1:

        iteration += 1

        X_current = X_work[remaining]

        vif_df = calculate_vif(X_current)

        max_vif_row = vif_df.iloc[0]

        max_feature = max_vif_row["feature"]
        max_vif = max_vif_row["vif"]

        # --------------------------------------------------------------
        # Find highest-VIF removable feature.
        # --------------------------------------------------------------

        removable = vif_df[
            ~vif_df["feature"].isin(
                protected_features
            )
        ]

        if removable.empty:
            print(
                "\nNo removable features remain."
            )
            break

        candidate = removable.iloc[0]

        candidate_feature = candidate["feature"]
        candidate_vif = candidate["vif"]

        # --------------------------------------------------------------
        # Stop if highest removable VIF is below threshold.
        # --------------------------------------------------------------

        if candidate_vif < threshold:
            print(
                f"\nStopping VIF reduction."
            )
            print(
                f"Highest remaining removable VIF: "
                f"{candidate_vif:.4f}"
            )
            break

        # --------------------------------------------------------------
        # Remove feature.
        # --------------------------------------------------------------

        remaining.remove(candidate_feature)

        history.append(
            {
                "iteration": iteration,
                "removed_feature": candidate_feature,
                "removed_vif": candidate_vif,
                "max_vif_before_removal": max_vif,
                "features_remaining": len(remaining),
            }
        )

        print(
            f"{iteration:>3}. Removing "
            f"{candidate_feature:<50} "
            f"VIF = {candidate_vif:,.2f}"
        )

    # ------------------------------------------------------------------
    # Final VIF
    # ------------------------------------------------------------------

    final_vif = calculate_vif(
        X_work[remaining]
    )

    print("\nVIF REDUCTION COMPLETE")
    print("-" * 70)
    print(
        f"Starting features : {len(X_work.columns)}"
    )
    print(
        f"Final features    : {len(remaining)}"
    )

    max_final_vif = final_vif["vif"].max()

    print(
        f"Maximum final VIF : {max_final_vif:.4f}"
    )

    protected_remaining = [
        f for f in protected_features
        if f in remaining
    ]

    print(
        f"Protected retained: "
        f"{len(protected_remaining)}"
    )

    return (
        remaining,
        pd.DataFrame(history),
        final_vif,
    )


# ======================================================================
# ADD MISSINGNESS INDICATORS
# ======================================================================

def add_missing_indicators(
    X_train,
    X_validation,
    X_test,
    missingness_threshold=0.25,
):
    """
    Add missingness indicators for features whose TRAINING
    missingness exceeds the specified threshold.

    The threshold is based exclusively on training data.
    """

    print_section("ADDING MISSINGNESS INDICATORS")

    missingness = analyze_missingness(
        X_train,
        pd.Series(
            np.zeros(len(X_train)),
            index=X_train.index
        ),
    )

    selected = missingness[
        missingness["missing_pct"]
        > missingness_threshold
    ]["feature"].tolist()

    print(
        f"Missingness threshold: "
        f">{missingness_threshold:.0%}"
    )

    print(
        f"Features requiring indicators: "
        f"{len(selected)}"
    )

    if selected:
        print("\nSelected features:")
        for feature in selected:
            pct = missingness.loc[
                missingness["feature"] == feature,
                "missing_pct"
            ].iloc[0]

            print(
                f"  {feature:<50} "
                f"{pct:.2%}"
            )

    for feature in selected:

        indicator = f"{feature}_missing"

        X_train[indicator] = (
            X_train[feature]
            .isna()
            .astype(int)
        )

        X_validation[indicator] = (
            X_validation[feature]
            .isna()
            .astype(int)
        )

        X_test[indicator] = (
            X_test[feature]
            .isna()
            .astype(int)
        )

    return (
        X_train,
        X_validation,
        X_test,
        selected,
        missingness,
    )


# ======================================================================
# IMPUTATION
# ======================================================================

def impute_using_training_medians(
    X_train,
    X_validation,
    X_test,
):
    """
    Median-impute continuous features.

    Medians are learned from TRAINING ONLY.
    """

    print_section("TRAINING-ONLY MEDIAN IMPUTATION")

    medians = X_train.median()

    missing_before = (
        X_train.isna().sum().sum(),
        X_validation.isna().sum().sum(),
        X_test.isna().sum().sum(),
    )

    X_train = X_train.fillna(medians)
    X_validation = X_validation.fillna(medians)
    X_test = X_test.fillna(medians)

    missing_after = (
        X_train.isna().sum().sum(),
        X_validation.isna().sum().sum(),
        X_test.isna().sum().sum(),
    )

    print(
        f"Missing values before: "
        f"Train={missing_before[0]}, "
        f"Validation={missing_before[1]}, "
        f"Test={missing_before[2]}"
    )

    print(
        f"Missing values after : "
        f"Train={missing_after[0]}, "
        f"Validation={missing_after[1]}, "
        f"Test={missing_after[2]}"
    )

    if any(value != 0 for value in missing_after):
        raise ValueError(
            "Missing values remain after imputation."
        )

    return X_train, X_validation, X_test, medians


# ======================================================================
# MODEL EVALUATION
# ======================================================================

def evaluate_model(model, X, y, name):
    """
    Generate comprehensive classification and probability metrics.
    """

    print_section(f"{name.upper()} PERFORMANCE")

    predictions = model.predict(X)
    probabilities = model.predict_proba(X)[:, 1]

    metrics = {
        "accuracy": accuracy_score(
            y, predictions
        ),
        "balanced_accuracy": balanced_accuracy_score(
            y, predictions
        ),
        "roc_auc": roc_auc_score(
            y, probabilities
        ),
        "log_loss": log_loss(
            y, probabilities
        ),
        "brier_score": brier_score_loss(
            y, probabilities
        ),
        "precision": precision_score(
            y, predictions
        ),
        "recall": recall_score(
            y, predictions
        ),
    }

    for metric, value in metrics.items():
        print(
            f"{metric:<20}: {value:.4f}"
        )

    print("\nConfusion Matrix:")
    print(
        confusion_matrix(
            y,
            predictions
        )
    )

    print("\nClassification Report:")
    print(
        classification_report(
            y,
            predictions
        )
    )

    return metrics


# ======================================================================
# COEFFICIENT ANALYSIS
# ======================================================================

def get_coefficients(model, feature_names):
    """
    Return coefficient and odds-ratio table.
    """

    coefficients = model.coef_[0]

    result = pd.DataFrame(
        {
            "feature": feature_names,
            "coefficient": coefficients,
            "odds_ratio": np.exp(coefficients),
        }
    )

    result["abs_coefficient"] = (
        result["coefficient"].abs()
    )

    result = result.sort_values(
        "abs_coefficient",
        ascending=False
    )

    return result


# ======================================================================
# MAIN
# ======================================================================

def main():

    print("\n" + "=" * 70)
    print("TRAINING ENGINEERED LOGISTIC REGRESSION V5")
    print("=" * 70)

    # ------------------------------------------------------------------
    # Load data
    # ------------------------------------------------------------------

    train, validation, test = load_data()

    # ------------------------------------------------------------------
    # Validate target
    # ------------------------------------------------------------------

    validate_targets(
        train,
        validation,
        test
    )

    y_train = train[TARGET].copy()
    y_validation = validation[TARGET].copy()
    y_test = test[TARGET].copy()

    # ------------------------------------------------------------------
    # Create V2 engineered features
    # ------------------------------------------------------------------

    print_section("CREATING V2 ENGINEERED FEATURES")

    X_train = create_matchup_features(train)
    X_validation = create_matchup_features(validation)
    X_test = create_matchup_features(test)

    print(
        f"Training features   : {X_train.shape}"
    )
    print(
        f"Validation features : {X_validation.shape}"
    )
    print(
        f"Test features       : {X_test.shape}"
    )

    if list(X_train.columns) != list(X_validation.columns):
        raise ValueError(
            "Training and validation feature alignment failed."
        )

    if list(X_train.columns) != list(X_test.columns):
        raise ValueError(
            "Training and test feature alignment failed."
        )

    print("Feature alignment verified.")

    # ------------------------------------------------------------------
    # Analyze missingness BEFORE VIF selection
    # ------------------------------------------------------------------

    print_section("INITIAL MISSINGNESS ANALYSIS")

    missingness = analyze_missingness(
        X_train,
        y_train
    )

    high_missing = missingness[
        missingness["missing_pct"]
        >= MISSINGNESS_THRESHOLD
    ]

    print(
        f"Features >= {MISSINGNESS_THRESHOLD:.0%} "
        f"missing: {len(high_missing)}"
    )

    if len(high_missing) > 0:

        print(
            "\nTop high-missingness features:"
        )

        print(
            high_missing[
                [
                    "feature",
                    "missing_pct",
                    "missing_rate_home_win",
                    "missing_rate_home_loss",
                    "missing_rate_difference",
                ]
            ].to_string(
                index=False
            )
        )

    # ------------------------------------------------------------------
    # VIF selection
    # ------------------------------------------------------------------

    (
        selected_features,
        vif_history,
        final_vif,
    ) = iterative_vif_selection(
        X_train,
        PROTECTED_FEATURES,
        threshold=VIF_THRESHOLD,
    )

    # ------------------------------------------------------------------
    # Restrict datasets to selected features
    # ------------------------------------------------------------------

    X_train = X_train[selected_features].copy()
    X_validation = X_validation[selected_features].copy()
    X_test = X_test[selected_features].copy()

    # ------------------------------------------------------------------
    # Add missing indicators
    # ------------------------------------------------------------------

    (
        X_train,
        X_validation,
        X_test,
        missing_indicator_features,
        missingness,
    ) = add_missing_indicators(
        X_train,
        X_validation,
        X_test,
        missingness_threshold=MISSINGNESS_THRESHOLD,
    )

    # ------------------------------------------------------------------
    # Impute
    # ------------------------------------------------------------------

    (
        X_train,
        X_validation,
        X_test,
        training_medians,
    ) = impute_using_training_medians(
        X_train,
        X_validation,
        X_test,
    )

    # ------------------------------------------------------------------
    # STANDARDIZE FEATURES FOR REGULARIZED LOGISTIC REGRESSION
    # ------------------------------------------------------------------

    print_section("TRAINING-ONLY FEATURE STANDARDIZATION")

    scaler = StandardScaler()

    X_train_scaled = scaler.fit_transform(X_train)

    X_validation_scaled = scaler.transform(X_validation)

    X_test_scaled = scaler.transform(X_test)

    print("Feature standardization complete.")
    print("Scaler fitted on training data only.")

    # ------------------------------------------------------------------
    # Final feature alignment
    # ------------------------------------------------------------------

    print_section("FINAL FEATURE VALIDATION")

    if list(X_train.columns) != list(X_validation.columns):
        raise ValueError(
            "Final training/validation feature alignment failed."
        )

    if list(X_train.columns) != list(X_test.columns):
        raise ValueError(
            "Final training/test feature alignment failed."
        )

    print(
        f"Final feature count: "
        f"{X_train.shape[1]}"
    )

    print(
        f"Core matchup features: "
        f"{len(selected_features)}"
    )

    print(
        f"Missing indicators: "
        f"{len(missing_indicator_features)}"
    )

    # ------------------------------------------------------------------
    # Train model
    # ------------------------------------------------------------------

    print_section("REGULARIZATION SWEEP")

    regularization_results = []

    for C in REGULARIZATION_C_VALUES:

        print(f"\nTraining C={C}")

        candidate_model = LogisticRegression(
            C=C,
            penalty="l2",
            solver="lbfgs",
            max_iter=5000,
            random_state=42,
        )

        candidate_model.fit(
            X_train_scaled,
            y_train
        )

        validation_probabilities = (
            candidate_model.predict_proba(
                X_validation_scaled
            )[:, 1]
        )

        validation_predictions = (
            candidate_model.predict(
                X_validation_scaled
            )
        )

        result = {
            "C": C,
            "validation_accuracy": accuracy_score(
                y_validation,
                validation_predictions
            ),
            "validation_balanced_accuracy": (
                balanced_accuracy_score(
                    y_validation,
                    validation_predictions
                )
            ),
            "validation_roc_auc": (
                roc_auc_score(
                    y_validation,
                    validation_probabilities
                )
            ),
            "validation_log_loss": (
                log_loss(
                    y_validation,
                    validation_probabilities
                )
            ),
            "validation_brier_score": (
                brier_score_loss(
                    y_validation,
                    validation_probabilities
                )
            ),
        }

        regularization_results.append(result)

    regularization_results = pd.DataFrame(
        regularization_results
    )

    print("\nREGULARIZATION RESULTS")
    print("-" * 70)

    print(
        regularization_results.to_string(
            index=False
        )
    )

    best_result = regularization_results.loc[
        regularization_results[
            "validation_log_loss"
        ].idxmin()
    ]

    best_C = best_result["C"]

    print(
        f"\nSelected C based on validation log loss: "
        f"{best_C}"
    )

    model = LogisticRegression(
        max_iter = 5000,
        solver = "lbfgs",
        random_state = 42,
        C = best_C,
    )

    model.fit(
        X_train,
        y_train
    )

    # ------------------------------------------------------------------
    # Evaluate
    # ------------------------------------------------------------------

    train_metrics = evaluate_model(
        model,
        X_train,
        y_train,
        "Training"
    )

    validation_metrics = evaluate_model(
        model,
        X_validation,
        y_validation,
        "Validation"
    )

    test_metrics = evaluate_model(
        model,
        X_test,
        y_test,
        "Test"
    )

    # ------------------------------------------------------------------
    # Coefficients
    # ------------------------------------------------------------------

    print_section("TOP 20 V5 FEATURES")

    coefficients = get_coefficients(
        model,
        X_train.columns
    )

    print(
        coefficients[
            [
                "feature",
                "coefficient",
                "odds_ratio",
            ]
        ].head(20).to_string(
            index=False
        )
    )

    # ------------------------------------------------------------------
    # Final VIF
    # ------------------------------------------------------------------

    print_section("FINAL VIF SUMMARY")

    print(
        final_vif.head(20).to_string(
            index=False
        )
    )

    print(
        f"\nMaximum final VIF: "
        f"{final_vif['vif'].max():.4f}"
    )

    print(
        f"Features with VIF >= 5: "
        f"{(final_vif['vif'] >= 5).sum()}"
    )

    print(
        f"Features with VIF >= 10: "
        f"{(final_vif['vif'] >= 10).sum()}"
    )

    # ------------------------------------------------------------------
    # Save model
    # ------------------------------------------------------------------

    MODEL_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    model_bundle = {
        "model": model,
        "features": list(X_train.columns),
        "core_features": selected_features,
        "missing_indicator_features": (
            missing_indicator_features
        ),
        "training_medians": (
            training_medians.to_dict()
        ),
        "vif_threshold": VIF_THRESHOLD,
        "missingness_threshold": (
            MISSINGNESS_THRESHOLD
        ),
        "protected_features": list(
            PROTECTED_FEATURES
        ),
        "version": "v5",
    }

    joblib.dump(
        model_bundle,
        MODEL_PATH
    )

    # ------------------------------------------------------------------
    # Save metrics
    # ------------------------------------------------------------------

    metrics_output = {
        "model_version": "engineered_v4",
        "target": TARGET,

        "n_training_rows": len(train),
        "n_validation_rows": len(validation),
        "n_test_rows": len(test),

        "initial_engineered_features": (
            len(ADVANCED_METRIC_PAIRS)
        ),

        "final_core_features": (
            len(selected_features)
        ),

        "missing_indicators": (
            len(missing_indicator_features)
        ),

        "final_total_features": (
            X_train.shape[1]
        ),

        "vif_threshold": VIF_THRESHOLD,

        "missingness_threshold": (
            MISSINGNESS_THRESHOLD
        ),

        "protected_features": list(
            PROTECTED_FEATURES
        ),

        "training": train_metrics,
        "validation": validation_metrics,
        "test": test_metrics,
    }

    with open(
        METRICS_PATH,
        "w",
        encoding="utf-8"
    ) as f:
        json.dump(
            metrics_output,
            f,
            indent=4
        )

    # ------------------------------------------------------------------
    # Save feature selection metadata
    # ------------------------------------------------------------------

    feature_selection_output = {
        "version": "v4",

        "initial_features": list(
            ADVANCED_METRIC_PAIRS
        ),

        "selected_core_features": (
            selected_features
        ),

        "removed_features": [
            feature
            for feature in ADVANCED_METRIC_PAIRS
            if f"matchup_{feature}" not in selected_features
            and not (
                feature == "elo"
                and "matchup_elo"
                in selected_features
            )
        ],

        "missing_indicator_features": (
            missing_indicator_features
        ),

        "protected_features": list(
            PROTECTED_FEATURES
        ),

        "vif_threshold": VIF_THRESHOLD,

        "missingness_threshold": (
            MISSINGNESS_THRESHOLD
        ),
    }

    with open(
        FEATURE_SELECTION_PATH,
        "w",
        encoding="utf-8"
    ) as f:
        json.dump(
            feature_selection_output,
            f,
            indent=4
        )

    # ------------------------------------------------------------------
    # Save VIF history
    # ------------------------------------------------------------------

    if not vif_history.empty:
        vif_history.to_csv(
            VIF_PATH,
            index=False
        )

    final_vif.to_csv(
        MODEL_DIR / "engineered_v4_final_vif.csv",
        index=False
    )

    # ------------------------------------------------------------------
    # Save missingness diagnostics
    # ------------------------------------------------------------------

    missingness.to_csv(
        MISSINGNESS_PATH,
        index=False
    )

    # ------------------------------------------------------------------
    # Final summary
    # ------------------------------------------------------------------

    print_section(
        "ENGINEERED LOGISTIC REGRESSION V5 COMPLETE"
    )

    print(
        f"Training rows     : {len(train):,}"
    )

    print(
        f"Validation rows   : {len(validation):,}"
    )

    print(
        f"Test rows         : {len(test):,}"
    )

    print(
        f"Initial features  : "
        f"{len(ADVANCED_METRIC_PAIRS)}"
    )

    print(
        f"Core features     : "
        f"{len(selected_features)}"
    )

    print(
        f"Missing indicators: "
        f"{len(missing_indicator_features)}"
    )

    print(
        f"Final features    : "
        f"{X_train.shape[1]}"
    )

    print(
        f"Maximum VIF       : "
        f"{final_vif['vif'].max():.4f}"
    )

    print("\nValidation Performance")

    print(
        f"  Accuracy        : "
        f"{validation_metrics['accuracy']:.4f}"
    )

    print(
        f"  ROC-AUC         : "
        f"{validation_metrics['roc_auc']:.4f}"
    )

    print(
        f"  Log Loss        : "
        f"{validation_metrics['log_loss']:.4f}"
    )

    print(
        f"  Brier Score     : "
        f"{validation_metrics['brier_score']:.4f}"
    )

    print("\nTest Performance")

    print(
        f"  Accuracy        : "
        f"{test_metrics['accuracy']:.4f}"
    )

    print(
        f"  ROC-AUC         : "
        f"{test_metrics['roc_auc']:.4f}"
    )

    print(
        f"  Log Loss        : "
        f"{test_metrics['log_loss']:.4f}"
    )

    print(
        f"  Brier Score     : "
        f"{test_metrics['brier_score']:.4f}"
    )

    print("\nModel saved to:")
    print(f"  {MODEL_PATH}")

    print("\nMetrics saved to:")
    print(f"  {METRICS_PATH}")

    print("\nFeature selection saved to:")
    print(f"  {FEATURE_SELECTION_PATH}")

    print("\nVIF history saved to:")
    print(f"  {VIF_PATH}")

    print("\nMissingness analysis saved to:")
    print(f"  {MISSINGNESS_PATH}")


if __name__ == "__main__":
    main()