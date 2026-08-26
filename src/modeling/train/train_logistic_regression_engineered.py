"""
Train V2 Engineered Logistic Regression Model

V2 engineering strategy
------------------------
1. Explicitly select canonical pregame statistics.
2. Create home-minus-away matchup differences.
3. Use home Elo minus away Elo rather than separate Elo features.
4. Include season-long and recent-form statistics where appropriate.
5. Exclude bookkeeping / schedule-position features such as GamesBefore.
6. Avoid semantically duplicated feature representations.
7. Standardize using training data only.
8. Preserve the existing temporal train / validation / test split.

Target
------
win_home
    1 = home team wins
    0 = away team wins
"""

from pathlib import Path
import json
import warnings

import joblib
import numpy as np
import pandas as pd

from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    brier_score_loss,
    classification_report,
    confusion_matrix,
    log_loss,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


warnings.filterwarnings("ignore")


# ============================================================================
# PATHS
# ============================================================================

PROJECT_ROOT = Path(__file__).resolve().parents[3]

DATA_DIR = PROJECT_ROOT / "data" / "processed" / "modeling"

TRAIN_PATH = DATA_DIR / "logistic_regression_train.csv"
VALIDATION_PATH = DATA_DIR / "logistic_regression_validation.csv"
TEST_PATH = DATA_DIR / "logistic_regression_test.csv"

MODEL_DIR = PROJECT_ROOT / "models" / "logistic_regression"

MODEL_PATH = (
    MODEL_DIR /
    "logistic_regression_engineered_v2.joblib"
)

FEATURE_PATH = (
    MODEL_DIR /
    "logistic_regression_engineered_v2_features.csv"
)

METRICS_PATH = (
    MODEL_DIR /
    "logistic_regression_engineered_v2_metrics.json"
)


# ============================================================================
# CONFIGURATION
# ============================================================================

TARGET = "win_home"

RANDOM_STATE = 42


# ============================================================================
# CANONICAL FEATURE SPECIFICATION
# ============================================================================
#
# Each entry represents one underlying football statistic.
#
# The script searches for the canonical home/away columns in priority order.
#
# We deliberately use only ONE representation of each statistic.
#
# We do NOT include:
#   - GamesBefore
#   - WinsBefore
#   - raw cumulative counting statistics where an average exists
#   - duplicate prefixed/non-prefixed representations
#   - raw home/away Elo separately
#
# Instead, most statistics are represented as:
#
#     home_value - away_value
#
# ============================================================================


FEATURE_SPEC = {

    # ------------------------------------------------------------------------
    # ELO / TEAM STRENGTH
    # ------------------------------------------------------------------------

    "elo": {
        "home": ["homePregameElo"],
        "away": ["awayPregameElo"],
        "transformation": "difference",
    },


    # ------------------------------------------------------------------------
    # SCORING
    # ------------------------------------------------------------------------

    "pointsForAvgBefore": {
        "home": ["pointsForAvgBefore_home"],
        "away": ["pointsForAvgBefore_away"],
        "transformation": "difference",
    },

    "pointsForAvgLast3": {
        "home": ["pointsForAvgLast3_home"],
        "away": ["pointsForAvgLast3_away"],
        "transformation": "difference",
    },

    "pointsForAvgLast5": {
        "home": ["pointsForAvgLast5_home"],
        "away": ["pointsForAvgLast5_away"],
        "transformation": "difference",
    },

    "pointsAgainstAvgBefore": {
        "home": ["pointsAgainstAvgBefore_home"],
        "away": ["pointsAgainstAvgBefore_away"],
        "transformation": "difference",
    },

    "pointsAgainstAvgLast3": {
        "home": ["pointsAgainstAvgLast3_home"],
        "away": ["pointsAgainstAvgLast3_away"],
        "transformation": "difference",
    },

    "pointsAgainstAvgLast5": {
        "home": ["pointsAgainstAvgLast5_home"],
        "away": ["pointsAgainstAvgLast5_away"],
        "transformation": "difference",
    },

    "pointDifferentialAvgBefore": {
        "home": ["pointDifferentialAvgBefore_home"],
        "away": ["pointDifferentialAvgBefore_away"],
        "transformation": "difference",
    },

    "pointDifferentialAvgLast3": {
        "home": ["pointDifferentialAvgLast3_home"],
        "away": ["pointDifferentialAvgLast3_away"],
        "transformation": "difference",
    },

    "pointDifferentialAvgLast5": {
        "home": ["pointDifferentialAvgLast5_home"],
        "away": ["pointDifferentialAvgLast5_away"],
        "transformation": "difference",
    },


    # ------------------------------------------------------------------------
    # PASSING
    # ------------------------------------------------------------------------

    "completionPctBefore": {
        "home": ["completionPctBefore_home"],
        "away": ["completionPctBefore_away"],
        "transformation": "difference",
    },

    "completionsAvgBefore": {
        "home": ["completionsAvgBefore_home"],
        "away": ["completionsAvgBefore_away"],
        "transformation": "difference",
    },

    "netPassingYardsAvgBefore": {
        "home": ["netPassingYardsAvgBefore_home"],
        "away": ["netPassingYardsAvgBefore_away"],
        "transformation": "difference",
    },

    "netPassingYardsAvgLast3": {
        "home": ["netPassingYardsAvgLast3_home"],
        "away": ["netPassingYardsAvgLast3_away"],
        "transformation": "difference",
    },

    "netPassingYardsAvgLast5": {
        "home": ["netPassingYardsAvgLast5_home"],
        "away": ["netPassingYardsAvgLast5_away"],
        "transformation": "difference",
    },

    "passAttemptsAvgBefore": {
        "home": ["passAttemptsAvgBefore_home"],
        "away": ["passAttemptsAvgBefore_away"],
        "transformation": "difference",
    },

    "yardsPerPassAttemptBefore": {
        "home": ["yardsPerPassAttemptBefore_home"],
        "away": ["yardsPerPassAttemptBefore_away"],
        "transformation": "difference",
    },

    "passingTDsAvgBefore": {
        "home": ["passingTDsAvgBefore_home"],
        "away": ["passingTDsAvgBefore_away"],
        "transformation": "difference",
    },

    "interceptionsAvgBefore": {
        "home": ["interceptionsAvgBefore_home"],
        "away": ["interceptionsAvgBefore_away"],
        "transformation": "difference",
    },

    "passesDeflectedAvgBefore": {
        "home": ["passesDeflectedAvgBefore_home"],
        "away": ["passesDeflectedAvgBefore_away"],
        "transformation": "difference",
    },

    "passesDeflectedAvgLast3": {
        "home": ["passesDeflectedAvgLast3_home"],
        "away": ["passesDeflectedAvgLast3_away"],
        "transformation": "difference",
    },

    "passesDeflectedAvgLast5": {
        "home": ["passesDeflectedAvgLast5_home"],
        "away": ["passesDeflectedAvgLast5_away"],
        "transformation": "difference",
    },


    # ------------------------------------------------------------------------
    # RUSHING
    # ------------------------------------------------------------------------

    "rushingYardsAvgBefore": {
        "home": ["rushingYardsAvgBefore_home"],
        "away": ["rushingYardsAvgBefore_away"],
        "transformation": "difference",
    },

    "rushingYardsAvgLast3": {
        "home": ["rushingYardsAvgLast3_home"],
        "away": ["rushingYardsAvgLast3_away"],
        "transformation": "difference",
    },

    "rushingYardsAvgLast5": {
        "home": ["rushingYardsAvgLast5_home"],
        "away": ["rushingYardsAvgLast5_away"],
        "transformation": "difference",
    },

    "rushingAttemptsAvgBefore": {
        "home": ["rushingAttemptsAvgBefore_home"],
        "away": ["rushingAttemptsAvgBefore_away"],
        "transformation": "difference",
    },

    "rushingTDsAvgBefore": {
        "home": ["rushingTDsAvgBefore_home"],
        "away": ["rushingTDsAvgBefore_away"],
        "transformation": "difference",
    },

    "yardsPerRushAttemptBefore": {
        "home": ["yardsPerRushAttemptBefore_home"],
        "away": ["yardsPerRushAttemptBefore_away"],
        "transformation": "difference",
    },


    # ------------------------------------------------------------------------
    # TOTAL OFFENSE
    # ------------------------------------------------------------------------

    "totalYardsAvgBefore": {
        "home": ["totalYardsAvgBefore_home"],
        "away": ["totalYardsAvgBefore_away"],
        "transformation": "difference",
    },

    "totalYardsAvgLast3": {
        "home": ["totalYardsAvgLast3_home"],
        "away": ["totalYardsAvgLast3_away"],
        "transformation": "difference",
    },

    "totalYardsAvgLast5": {
        "home": ["totalYardsAvgLast5_home"],
        "away": ["totalYardsAvgLast5_away"],
        "transformation": "difference",
    },

    "firstDownsAvgBefore": {
        "home": ["firstDownsAvgBefore_home"],
        "away": ["firstDownsAvgBefore_away"],
        "transformation": "difference",
    },


    # ------------------------------------------------------------------------
    # DEFENSE
    # ------------------------------------------------------------------------

    "tacklesForLossAvgBefore": {
        "home": ["tacklesForLossAvgBefore_home"],
        "away": ["tacklesForLossAvgBefore_away"],
        "transformation": "difference",
    },

    "tacklesForLossAvgLast3": {
        "home": ["tacklesForLossAvgLast3_home"],
        "away": ["tacklesForLossAvgLast3_away"],
        "transformation": "difference",
    },

    "tacklesForLossAvgLast5": {
        "home": ["tacklesForLossAvgLast5_home"],
        "away": ["tacklesForLossAvgLast5_away"],
        "transformation": "difference",
    },

    "qbHurriesAvgBefore": {
        "home": ["qbHurriesAvgBefore_home"],
        "away": ["qbHurriesAvgBefore_away"],
        "transformation": "difference",
    },

    "qbHurriesAvgLast3": {
        "home": ["qbHurriesAvgLast3_home"],
        "away": ["qbHurriesAvgLast3_away"],
        "transformation": "difference",
    },

    "qbHurriesAvgLast5": {
        "home": ["qbHurriesAvgLast5_home"],
        "away": ["qbHurriesAvgLast5_away"],
        "transformation": "difference",
    },

    "sacksAvgBefore": {
        "home": ["sacksAvgBefore_home"],
        "away": ["sacksAvgBefore_away"],
        "transformation": "difference",
    },

    "sacksAvgLast3": {
        "home": ["sacksAvgLast3_home"],
        "away": ["sacksAvgLast3_away"],
        "transformation": "difference",
    },

    "sacksAvgLast5": {
        "home": ["sacksAvgLast5_home"],
        "away": ["sacksAvgLast5_away"],
        "transformation": "difference",
    },


    # ------------------------------------------------------------------------
    # TURNOVERS / BALL SECURITY
    # ------------------------------------------------------------------------

    "turnoversAvgBefore": {
        "home": ["turnoversAvgBefore_home"],
        "away": ["turnoversAvgBefore_away"],
        "transformation": "difference",
    },

    "turnoversAvgLast3": {
        "home": ["turnoversAvgLast3_home"],
        "away": ["turnoversAvgLast3_away"],
        "transformation": "difference",
    },

    "turnoversAvgLast5": {
        "home": ["turnoversAvgLast5_home"],
        "away": ["turnoversAvgLast5_away"],
        "transformation": "difference",
    },

    "fumblesLostAvgBefore": {
        "home": ["fumblesLostAvgBefore_home"],
        "away": ["fumblesLostAvgBefore_away"],
        "transformation": "difference",
    },


    # ------------------------------------------------------------------------
    # POSSESSION
    # ------------------------------------------------------------------------

    "possessionSecondsAvgBefore": {
        "home": ["possessionSecondsAvgBefore_home"],
        "away": ["possessionSecondsAvgBefore_away"],
        "transformation": "difference",
    },


    # ------------------------------------------------------------------------
    # SITUATIONAL FOOTBALL
    # ------------------------------------------------------------------------

    "thirdDownPctBefore": {
        "home": ["thirdDownPctBefore_home"],
        "away": ["thirdDownPctBefore_away"],
        "transformation": "difference",
    },

    "thirdDownAttemptsAvgBefore": {
        "home": ["thirdDownAttemptsAvgBefore_home"],
        "away": ["thirdDownAttemptsAvgBefore_away"],
        "transformation": "difference",
    },

    "thirdDownConversionsAvgBefore": {
        "home": ["thirdDownConversionsAvgBefore_home"],
        "away": ["thirdDownConversionsAvgBefore_away"],
        "transformation": "difference",
    },

    "fourthDownPctBefore": {
        "home": ["fourthDownPctBefore_home"],
        "away": ["fourthDownPctBefore_away"],
        "transformation": "difference",
    },

    "fourthDownAttemptsAvgBefore": {
        "home": ["fourthDownAttemptsAvgBefore_home"],
        "away": ["fourthDownAttemptsAvgBefore_away"],
        "transformation": "difference",
    },

    "fourthDownConversionsAvgBefore": {
        "home": ["fourthDownConversionsAvgBefore_home"],
        "away": ["fourthDownConversionsAvgBefore_away"],
        "transformation": "difference",
    },


    # ------------------------------------------------------------------------
    # PENALTIES
    # ------------------------------------------------------------------------

    "penaltiesAvgBefore": {
        "home": ["penaltiesAvgBefore_home"],
        "away": ["penaltiesAvgBefore_away"],
        "transformation": "difference",
    },

    "penaltyYardsAvgBefore": {
        "home": ["penaltyYardsAvgBefore_home"],
        "away": ["penaltyYardsAvgBefore_away"],
        "transformation": "difference",
    },


    # ------------------------------------------------------------------------
    # ADVANCED OFFENSE
    # ------------------------------------------------------------------------

    "offenseExplosiveness": {
        "home": ["home_pregame_offense_explosiveness"],
        "away": ["away_pregame_offense_explosiveness"],
        "transformation": "difference",
    },

    "offensePPA": {
        "home": ["home_pregame_offense_ppa"],
        "away": ["away_pregame_offense_ppa"],
        "transformation": "difference",
    },

    "offenseSuccessRate": {
        "home": ["home_pregame_offense_successRate"],
        "away": ["away_pregame_offense_successRate"],
        "transformation": "difference",
    },


    # ------------------------------------------------------------------------
    # ADVANCED DEFENSE
    # ------------------------------------------------------------------------

    "defenseExplosiveness": {
        "home": ["home_pregame_defense_explosiveness"],
        "away": ["away_pregame_defense_explosiveness"],
        "transformation": "difference",
    },

    "defensePPA": {
        "home": ["home_pregame_defense_ppa"],
        "away": ["away_pregame_defense_ppa"],
        "transformation": "difference",
    },

    "defenseSuccessRate": {
        "home": ["home_pregame_defense_successRate"],
        "away": ["away_pregame_defense_successRate"],
        "transformation": "difference",
    },
}


# ============================================================================
# LOGGING
# ============================================================================

def print_header(title):
    print()
    print("=" * 70)
    print(title)
    print("=" * 70)


# ============================================================================
# DATA LOADING
# ============================================================================

def load_split(path, split_name):
    """Load a temporal modeling split."""

    print(f"Loading {split_name}: {path}")

    if not path.exists():
        raise FileNotFoundError(
            f"\nCould not find {split_name} file:\n"
            f"  {path}\n\n"
            f"Check the paths used by split_logistic_regression.py."
        )

    df = pd.read_csv(path)

    print(f"  Shape: {df.shape}")

    return df


# ============================================================================
# COLUMN RESOLUTION
# ============================================================================

def resolve_column(df, candidates):
    """
    Find the first available candidate column.

    Candidates are ordered by preference.
    """

    for column in candidates:
        if column in df.columns:
            return column

    return None


def validate_feature_spec(df):
    """
    Validate that every requested canonical feature can be constructed.
    """

    missing = []

    for feature_name, specification in FEATURE_SPEC.items():

        home_column = resolve_column(
            df,
            specification["home"],
        )

        away_column = resolve_column(
            df,
            specification["away"],
        )

        if home_column is None:
            missing.append(
                f"{feature_name}: HOME missing "
                f"{specification['home']}"
            )

        if away_column is None:
            missing.append(
                f"{feature_name}: AWAY missing "
                f"{specification['away']}"
            )

    if missing:
        print()
        print("MISSING FEATURE COLUMNS")
        print("-" * 70)

        for item in missing:
            print(f"  {item}")

        raise ValueError(
            f"\n{len(missing)} required feature columns could not be resolved."
        )


# ============================================================================
# FEATURE ENGINEERING
# ============================================================================

def engineer_features(df, split_name):
    """
    Construct the explicit V2 matchup feature matrix.
    """

    print_header(
        f"ENGINEERING V2 FEATURES: {split_name.upper()}"
    )

    validate_feature_spec(df)

    engineered = pd.DataFrame(index=df.index)

    metadata = []

    for feature_name, specification in FEATURE_SPEC.items():

        home_column = resolve_column(
            df,
            specification["home"],
        )

        away_column = resolve_column(
            df,
            specification["away"],
        )

        home_values = pd.to_numeric(
            df[home_column],
            errors="coerce",
        )

        away_values = pd.to_numeric(
            df[away_column],
            errors="coerce",
        )

        if specification["transformation"] == "difference":

            engineered_name = (
                f"matchup_{feature_name}"
            )

            engineered[engineered_name] = (
                home_values - away_values
            )

            metadata.append(
                {
                    "feature": engineered_name,
                    "home_source": home_column,
                    "away_source": away_column,
                    "transformation": "home_minus_away",
                }
            )

        else:
            raise ValueError(
                f"Unknown transformation for {feature_name}: "
                f"{specification['transformation']}"
            )

    print(f"Canonical statistics selected: {len(FEATURE_SPEC)}")
    print(f"Engineered matchup features: {engineered.shape[1]}")

    return engineered, metadata


# ============================================================================
# TARGET VALIDATION
# ============================================================================

def validate_target(df, split_name):
    """Validate target column."""

    if TARGET not in df.columns:
        raise ValueError(
            f"{TARGET} not found in {split_name} dataset."
        )

    values = df[TARGET].dropna().unique()

    if not set(values).issubset({0, 1}):
        raise ValueError(
            f"{split_name} target contains values other than 0/1: "
            f"{values}"
        )

    print(
        f"{split_name} target distribution:"
    )

    print(
        df[TARGET]
        .value_counts()
        .sort_index()
        .to_string()
    )


# ============================================================================
# FEATURE VALIDATION
# ============================================================================

def validate_feature_alignment(
    X_train,
    X_validation,
    X_test,
):
    """Ensure temporal splits have identical feature columns."""

    train_columns = list(X_train.columns)
    validation_columns = list(X_validation.columns)
    test_columns = list(X_test.columns)

    if train_columns != validation_columns:
        raise ValueError(
            "Training and validation engineered features do not align."
        )

    if train_columns != test_columns:
        raise ValueError(
            "Training and test engineered features do not align."
        )

    print(
        f"Feature alignment verified: "
        f"{len(train_columns)} features"
    )


# ============================================================================
# MODEL
# ============================================================================

def create_model():
    """
    Create logistic regression pipeline.

    Imputation and scaling are fitted only on training data.
    """

    return Pipeline(
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
# EVALUATION
# ============================================================================

def evaluate_model(
    model,
    X,
    y,
    split_name,
):
    """Evaluate model performance."""

    probabilities = model.predict_proba(X)[:, 1]

    predictions = (
        probabilities >= 0.5
    ).astype(int)

    metrics = {
        "accuracy": accuracy_score(
            y,
            predictions,
        ),

        "balanced_accuracy": balanced_accuracy_score(
            y,
            predictions,
        ),

        "roc_auc": roc_auc_score(
            y,
            probabilities,
        ),

        "log_loss": log_loss(
            y,
            probabilities,
        ),

        "brier_score": brier_score_loss(
            y,
            probabilities,
        ),

        "precision": precision_score(
            y,
            predictions,
            zero_division=0,
        ),

        "recall": recall_score(
            y,
            predictions,
            zero_division=0,
        ),
    }

    print_header(
        f"{split_name.upper()} PERFORMANCE"
    )

    for metric, value in metrics.items():
        print(
            f"{metric:20s}: {value:.4f}"
        )

    print()
    print("Confusion Matrix:")

    print(
        confusion_matrix(
            y,
            predictions,
        )
    )

    print()
    print("Classification Report:")

    print(
        classification_report(
            y,
            predictions,
            digits=4,
            zero_division=0,
        )
    )

    return metrics


# ============================================================================
# COEFFICIENT ANALYSIS
# ============================================================================

def save_feature_coefficients(
    model,
    feature_names,
    metadata,
):
    """
    Save coefficients with source feature information.
    """

    logistic_model = (
        model.named_steps["model"]
    )

    coefficients = (
        logistic_model.coef_[0]
    )

    coefficient_df = pd.DataFrame(
        {
            "feature": feature_names,
            "coefficient": coefficients,
            "absolute_coefficient": np.abs(
                coefficients
            ),
            "odds_ratio": np.exp(
                coefficients
            ),
        }
    )

    metadata_df = pd.DataFrame(
        metadata
    )

    coefficient_df = coefficient_df.merge(
        metadata_df,
        on="feature",
        how="left",
    )

    coefficient_df = coefficient_df.sort_values(
        "absolute_coefficient",
        ascending=False,
    )

    FEATURE_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    coefficient_df.to_csv(
        FEATURE_PATH,
        index=False,
    )

    print()
    print("TOP 20 ENGINEERED FEATURES")
    print("-" * 70)

    print(
        coefficient_df[
            [
                "feature",
                "coefficient",
                "odds_ratio",
                "home_source",
                "away_source",
            ]
        ]
        .head(20)
        .to_string(index=False)
    )

    return coefficient_df


# ============================================================================
# MAIN
# ============================================================================

def main():

    print_header(
        "TRAINING ENGINEERED LOGISTIC REGRESSION V2"
    )

    # ----------------------------------------------------------------------
    # LOAD DATA
    # ----------------------------------------------------------------------

    print_header(
        "LOADING TEMPORAL SPLITS"
    )

    train_df = load_split(
        TRAIN_PATH,
        "Training",
    )

    validation_df = load_split(
        VALIDATION_PATH,
        "Validation",
    )

    test_df = load_split(
        TEST_PATH,
        "Test",
    )

    # ----------------------------------------------------------------------
    # VALIDATE TARGET
    # ----------------------------------------------------------------------

    print_header(
        "VALIDATING TARGETS"
    )

    validate_target(
        train_df,
        "Training",
    )

    validate_target(
        validation_df,
        "Validation",
    )

    validate_target(
        test_df,
        "Test",
    )

    y_train = train_df[TARGET]
    y_validation = validation_df[TARGET]
    y_test = test_df[TARGET]

    # ----------------------------------------------------------------------
    # ENGINEER FEATURES
    # ----------------------------------------------------------------------

    X_train, metadata = engineer_features(
        train_df,
        "training",
    )

    X_validation, validation_metadata = engineer_features(
        validation_df,
        "validation",
    )

    X_test, test_metadata = engineer_features(
        test_df,
        "test",
    )

    # ----------------------------------------------------------------------
    # VALIDATE FEATURE ALIGNMENT
    # ----------------------------------------------------------------------

    print_header(
        "VALIDATING ENGINEERED FEATURES"
    )

    validate_feature_alignment(
        X_train,
        X_validation,
        X_test,
    )

    # ----------------------------------------------------------------------
    # CHECK MISSINGNESS
    # ----------------------------------------------------------------------

    print_header(
        "CHECKING ENGINEERED FEATURE MISSINGNESS"
    )

    missing_summary = (
        X_train.isna()
        .sum()
        .sort_values(
            ascending=False
        )
    )

    missing_features = (
        missing_summary[
            missing_summary > 0
        ]
    )

    if len(missing_features) == 0:

        print(
            "No missing values in engineered "
            "training features."
        )

    else:

        print(
            f"Features containing missing values: "
            f"{len(missing_features)}"
        )

        print(
            missing_features.to_string()
        )

    # ----------------------------------------------------------------------
    # TRAIN MODEL
    # ----------------------------------------------------------------------

    print_header(
        "TRAINING LOGISTIC REGRESSION V2"
    )

    model = create_model()

    model.fit(
        X_train,
        y_train,
    )

    print(
        "Model training complete."
    )

    # ----------------------------------------------------------------------
    # EVALUATE
    # ----------------------------------------------------------------------

    train_metrics = evaluate_model(
        model,
        X_train,
        y_train,
        "Training",
    )

    validation_metrics = evaluate_model(
        model,
        X_validation,
        y_validation,
        "Validation",
    )

    test_metrics = evaluate_model(
        model,
        X_test,
        y_test,
        "Test",
    )

    # ----------------------------------------------------------------------
    # COEFFICIENT ANALYSIS
    # ----------------------------------------------------------------------

    coefficient_df = save_feature_coefficients(
        model,
        list(X_train.columns),
        metadata,
    )

    # ----------------------------------------------------------------------
    # SAVE MODEL
    # ----------------------------------------------------------------------

    MODEL_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    joblib.dump(
        model,
        MODEL_PATH,
    )

    print()
    print("Model saved to:")
    print(
        f"  {MODEL_PATH}"
    )

    # ----------------------------------------------------------------------
    # SAVE METRICS
    # ----------------------------------------------------------------------

    metrics = {
        "model": (
            "engineered_logistic_regression_v2"
        ),

        "target": TARGET,

        "n_training_rows": len(
            train_df
        ),

        "n_validation_rows": len(
            validation_df
        ),

        "n_test_rows": len(
            test_df
        ),

        "n_engineered_features": (
            X_train.shape[1]
        ),

        "feature_categories": {
            "elo": 1,
            "scoring": 9,
            "passing": 11,
            "rushing": 6,
            "total_offense": 4,
            "defense": 9,
            "turnovers": 4,
            "possession": 1,
            "situational": 6,
            "penalties": 2,
            "advanced_offense": 3,
            "advanced_defense": 3,
        },

        "training_metrics": train_metrics,

        "validation_metrics": (
            validation_metrics
        ),

        "test_metrics": test_metrics,

        "engineering": {
            "transformation": (
                "home_minus_away"
            ),

            "uses_absolute_elo": False,

            "uses_elo_difference": True,

            "excludes_games_before": True,

            "uses_explicit_feature_spec": True,
        },
    }

    with open(
        METRICS_PATH,
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            metrics,
            file,
            indent=4,
        )

    print()
    print("Metrics saved to:")
    print(
        f"  {METRICS_PATH}"
    )

    # ----------------------------------------------------------------------
    # FINAL SUMMARY
    # ----------------------------------------------------------------------

    print_header(
        "ENGINEERED LOGISTIC REGRESSION V2 COMPLETE"
    )

    print(
        f"Training rows     : "
        f"{len(train_df):,}"
    )

    print(
        f"Validation rows   : "
        f"{len(validation_df):,}"
    )

    print(
        f"Test rows         : "
        f"{len(test_df):,}"
    )

    print(
        f"Engineered feats  : "
        f"{X_train.shape[1]:,}"
    )

    print()
    print(
        "Validation Performance"
    )

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

    print()
    print(
        "Test Performance"
    )

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


if __name__ == "__main__":
    main()