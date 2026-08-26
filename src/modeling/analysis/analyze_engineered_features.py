"""
analyze_engineered_features.py

Diagnostic analysis for the V2 engineered logistic regression feature set.

This script:
    1. Loads the temporal train/validation/test splits.
    2. Recreates the exact V2 engineered feature set.
    3. Analyzes missingness across all splits.
    4. Analyzes pairwise correlations using TRAINING data only.
    5. Calculates VIF using TRAINING data only.
    6. Groups features into conceptual statistical families.
    7. Identifies highly correlated features and potentially problematic
       multicollinearity.
    8. Saves diagnostic results for later feature-selection decisions.

IMPORTANT:
    This script does NOT modify the modeling datasets.
    It does NOT impute missing values.
    It does NOT train a model.

The purpose is diagnosis before building V3.
"""

from pathlib import Path
import json
import warnings

import numpy as np
import pandas as pd

from sklearn.impute import SimpleImputer
from statsmodels.stats.outliers_influence import variance_inflation_factor

warnings.filterwarnings("ignore")


# ======================================================================
# PATHS
# ======================================================================

PROJECT_ROOT = Path(__file__).resolve().parents[3]

DATA_DIR = PROJECT_ROOT / "data" / "processed" / "modeling"
OUTPUT_DIR = (
    PROJECT_ROOT
    / "models"
    / "logistic_regression"
    / "feature_analysis"
)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


TRAIN_PATH = DATA_DIR / "logistic_regression_train.csv"
VALIDATION_PATH = DATA_DIR / "logistic_regression_validation.csv"
TEST_PATH = DATA_DIR / "logistic_regression_test.csv"


# ======================================================================
# CONFIGURATION
# ======================================================================

CORRELATION_THRESHOLD = 0.80
HIGH_CORRELATION_THRESHOLD = 0.90

VIF_WARNING_THRESHOLD = 5.0
VIF_HIGH_THRESHOLD = 10.0

MISSING_WARNING_THRESHOLD = 0.10
MISSING_HIGH_THRESHOLD = 0.25
MISSING_VERY_HIGH_THRESHOLD = 0.50


# ======================================================================
# V2 CANONICAL STATISTICS
# ======================================================================

V2_FEATURE_MAPPING = {
    # Core offensive production
    "matchup_completionsAvgBefore": (
        "completionsAvgBefore_home",
        "completionsAvgBefore_away",
    ),
    "matchup_passAttemptsAvgBefore": (
        "passAttemptsAvgBefore_home",
        "passAttemptsAvgBefore_away",
    ),
    "matchup_netPassingYardsAvgBefore": (
        "netPassingYardsAvgBefore_home",
        "netPassingYardsAvgBefore_away",
    ),
    "matchup_passingTDsAvgBefore": (
        "passingTDsAvgBefore_home",
        "passingTDsAvgBefore_away",
    ),
    "matchup_yardsPerPassAttemptBefore": (
        "yardsPerPassAttemptBefore_home",
        "yardsPerPassAttemptBefore_away",
    ),
    "matchup_rushingAttemptsAvgBefore": (
        "rushingAttemptsAvgBefore_home",
        "rushingAttemptsAvgBefore_away",
    ),
    "matchup_rushingYardsAvgBefore": (
        "rushingYardsAvgBefore_home",
        "rushingYardsAvgBefore_away",
    ),
    "matchup_rushingTDsAvgBefore": (
        "rushingTDsAvgBefore_home",
        "rushingTDsAvgBefore_away",
    ),
    "matchup_yardsPerRushAttemptBefore": (
        "yardsPerRushAttemptBefore_home",
        "yardsPerRushAttemptBefore_away",
    ),
    "matchup_totalYardsAvgBefore": (
        "totalYardsAvgBefore_home",
        "totalYardsAvgBefore_away",
    ),

    # Scoring / results
    "matchup_pointsForAvgBefore": (
        "pointsForAvgBefore_home",
        "pointsForAvgBefore_away",
    ),
    "matchup_pointsAgainstAvgBefore": (
        "pointsAgainstAvgBefore_home",
        "pointsAgainstAvgBefore_away",
    ),
    "matchup_pointDifferentialAvgBefore": (
        "pointDifferentialAvgBefore_home",
        "pointDifferentialAvgBefore_away",
    ),
    "matchup_winPctBefore": (
        "winPctBefore_home",
        "winPctBefore_away",
    ),

    # Turnovers
    "matchup_turnoversAvgBefore": (
        "turnoversAvgBefore_home",
        "turnoversAvgBefore_away",
    ),
    "matchup_interceptionsAvgBefore": (
        "interceptionsAvgBefore_home",
        "interceptionsAvgBefore_away",
    ),
    "matchup_fumblesLostAvgBefore": (
        "fumblesLostAvgBefore_home",
        "fumblesLostAvgBefore_away",
    ),

    # First downs / possession
    "matchup_firstDownsAvgBefore": (
        "firstDownsAvgBefore_home",
        "firstDownsAvgBefore_away",
    ),
    "matchup_possessionSecondsAvgBefore": (
        "possessionSecondsAvgBefore_home",
        "possessionSecondsAvgBefore_away",
    ),

    # Situational
    "matchup_thirdDownAttemptsAvgBefore": (
        "thirdDownAttemptsAvgBefore_home",
        "thirdDownAttemptsAvgBefore_away",
    ),
    "matchup_thirdDownConversionsAvgBefore": (
        "thirdDownConversionsAvgBefore_home",
        "thirdDownConversionsAvgBefore_away",
    ),
    "matchup_thirdDownPctBefore": (
        "thirdDownPctBefore_home",
        "thirdDownPctBefore_away",
    ),
    "matchup_fourthDownAttemptsAvgBefore": (
        "fourthDownAttemptsAvgBefore_home",
        "fourthDownAttemptsAvgBefore_away",
    ),
    "matchup_fourthDownConversionsAvgBefore": (
        "fourthDownConversionsAvgBefore_home",
        "fourthDownConversionsAvgBefore_away",
    ),
    "matchup_fourthDownPctBefore": (
        "fourthDownPctBefore_home",
        "fourthDownPctBefore_away",
    ),

    # Penalties
    "matchup_penaltiesAvgBefore": (
        "penaltiesAvgBefore_home",
        "penaltiesAvgBefore_away",
    ),
    "matchup_penaltyYardsAvgBefore": (
        "penaltyYardsAvgBefore_home",
        "penaltyYardsAvgBefore_away",
    ),

    # Defense
    "matchup_sacksAvgBefore": (
        "sacksAvgBefore_home",
        "sacksAvgBefore_away",
    ),
    "matchup_tacklesForLossAvgBefore": (
        "tacklesForLossAvgBefore_home",
        "tacklesForLossAvgBefore_away",
    ),
    "matchup_qbHurriesAvgBefore": (
        "qbHurriesAvgBefore_home",
        "qbHurriesAvgBefore_away",
    ),
    "matchup_passesDeflectedAvgBefore": (
        "passesDeflectedAvgBefore_home",
        "passesDeflectedAvgBefore_away",
    ),

    # Recent form
    "matchup_pointsForAvgLast3": (
        "pointsForAvgLast3_home",
        "pointsForAvgLast3_away",
    ),
    "matchup_pointsForAvgLast5": (
        "pointsForAvgLast5_home",
        "pointsForAvgLast5_away",
    ),
    "matchup_pointsAgainstAvgLast3": (
        "pointsAgainstAvgLast3_home",
        "pointsAgainstAvgLast3_away",
    ),
    "matchup_pointsAgainstAvgLast5": (
        "pointsAgainstAvgLast5_home",
        "pointsAgainstAvgLast5_away",
    ),
    "matchup_pointDifferentialAvgLast3": (
        "pointDifferentialAvgLast3_home",
        "pointDifferentialAvgLast3_away",
    ),
    "matchup_pointDifferentialAvgLast5": (
        "pointDifferentialAvgLast5_home",
        "pointDifferentialAvgLast5_away",
    ),
    "matchup_netPassingYardsAvgLast3": (
        "netPassingYardsAvgLast3_home",
        "netPassingYardsAvgLast3_away",
    ),
    "matchup_netPassingYardsAvgLast5": (
        "netPassingYardsAvgLast5_home",
        "netPassingYardsAvgLast5_away",
    ),
    "matchup_rushingYardsAvgLast3": (
        "rushingYardsAvgLast3_home",
        "rushingYardsAvgLast3_away",
    ),
    "matchup_rushingYardsAvgLast5": (
        "rushingYardsAvgLast5_home",
        "rushingYardsAvgLast5_away",
    ),
    "matchup_totalYardsAvgLast3": (
        "totalYardsAvgLast3_home",
        "totalYardsAvgLast3_away",
    ),
    "matchup_totalYardsAvgLast5": (
        "totalYardsAvgLast5_home",
        "totalYardsAvgLast5_away",
    ),
    "matchup_turnoversAvgLast3": (
        "turnoversAvgLast3_home",
        "turnoversAvgLast3_away",
    ),
    "matchup_turnoversAvgLast5": (
        "turnoversAvgLast5_home",
        "turnoversAvgLast5_away",
    ),
    "matchup_sacksAvgLast3": (
        "sacksAvgLast3_home",
        "sacksAvgLast3_away",
    ),
    "matchup_sacksAvgLast5": (
        "sacksAvgLast5_home",
        "sacksAvgLast5_away",
    ),
    "matchup_tacklesForLossAvgLast3": (
        "tacklesForLossAvgLast3_home",
        "tacklesForLossAvgLast3_away",
    ),
    "matchup_tacklesForLossAvgLast5": (
        "tacklesForLossAvgLast5_home",
        "tacklesForLossAvgLast5_away",
    ),
    "matchup_qbHurriesAvgLast3": (
        "qbHurriesAvgLast3_home",
        "qbHurriesAvgLast3_away",
    ),
    "matchup_qbHurriesAvgLast5": (
        "qbHurriesAvgLast5_home",
        "qbHurriesAvgLast5_away",
    ),
    "matchup_passesDeflectedAvgLast3": (
        "passesDeflectedAvgLast3_home",
        "passesDeflectedAvgLast3_away",
    ),
    "matchup_passesDeflectedAvgLast5": (
        "passesDeflectedAvgLast5_home",
        "passesDeflectedAvgLast5_away",
    ),

    # Advanced pregame metrics
    "matchup_offensePPA": (
        "home_pregame_offense_ppa",
        "away_pregame_offense_ppa",
    ),
    "matchup_offenseSuccessRate": (
        "home_pregame_offense_successRate",
        "away_pregame_offense_successRate",
    ),
    "matchup_offenseExplosiveness": (
        "home_pregame_offense_explosiveness",
        "away_pregame_offense_explosiveness",
    ),
    "matchup_defensePPA": (
        "home_pregame_defense_ppa",
        "away_pregame_defense_ppa",
    ),
    "matchup_defenseSuccessRate": (
        "home_pregame_defense_successRate",
        "away_pregame_defense_successRate",
    ),
    "matchup_defenseExplosiveness": (
        "home_pregame_defense_explosiveness",
        "away_pregame_defense_explosiveness",
    ),

    # Elo
    "matchup_elo": (
        "homePregameElo",
        "awayPregameElo",
    ),
}

# ======================================================================
# HELPERS
# ======================================================================

def load_data():
    """Load temporal modeling splits."""

    print("=" * 70)
    print("LOADING TEMPORAL SPLITS")
    print("=" * 70)

    train = pd.read_csv(TRAIN_PATH)
    validation = pd.read_csv(VALIDATION_PATH)
    test = pd.read_csv(TEST_PATH)

    print(f"Training   : {train.shape}")
    print(f"Validation : {validation.shape}")
    print(f"Test       : {test.shape}")

    return train, validation, test


def create_matchup_features(df):
    """
    Recreate the exact 60-feature V2 engineered feature set.

    Every matchup feature is:

        home_value - away_value

    Therefore:
        positive = home-team advantage
        negative = away-team advantage
    """

    features = {}

    for matchup_name, (home_col, away_col) in V2_FEATURE_MAPPING.items():

        # Validate source columns
        if home_col not in df.columns:
            raise ValueError(
                f"Missing home source column for '{matchup_name}': "
                f"{home_col}"
            )

        if away_col not in df.columns:
            raise ValueError(
                f"Missing away source column for '{matchup_name}': "
                f"{away_col}"
            )

        # Home minus away
        features[matchup_name] = (
            df[home_col] - df[away_col]
        )

    return pd.DataFrame(features, index=df.index)


def get_feature_family(feature):
    """Assign each engineered feature to a conceptual family."""

    name = feature.replace("matchup_", "")

    if "PregameElo" in name or name == "elo":
        return "Elo"

    if any(x in name for x in [
        "pointsFor",
        "pointsAgainst",
        "pointDifferential",
        "winPct",
    ]):
        return "Scoring / Results"

    if any(x in name for x in [
        "netPassing",
        "completions",
        "passAttempts",
        "passingTDs",
        "yardsPerPassAttempt",
    ]):
        return "Passing"

    if any(x in name for x in [
        "rushingAttempts",
        "rushingYards",
        "rushingTDs",
        "yardsPerRushAttempt",
    ]):
        return "Rushing"

    if any(x in name for x in [
        "totalYards",
    ]):
        return "Total Offense"

    if any(x in name for x in [
        "turnovers",
        "interceptions",
        "fumblesLost",
    ]):
        return "Turnovers"

    if any(x in name for x in [
        "thirdDown",
        "fourthDown",
    ]):
        return "Situational"

    if any(x in name for x in [
        "penalties",
        "penaltyYards",
    ]):
        return "Penalties"

    if any(x in name for x in [
        "sacks",
        "tacklesForLoss",
        "qbHurries",
        "passesDeflected",
    ]):
        return "Defense"

    if any(x in name for x in [
        "PPA",
        "SuccessRate",
        "Explosiveness",
    ]):
        return "Advanced Metrics"

    if "Last3" in name or "Last5" in name:
        return "Recent Form"

    return "Other"


def calculate_missingness(datasets):
    """Calculate missingness across all datasets."""

    print("\n" + "=" * 70)
    print("MISSINGNESS ANALYSIS")
    print("=" * 70)

    results = []

    for dataset_name, df in datasets.items():

        for feature in df.columns:

            missing_count = df[feature].isna().sum()
            total = len(df)

            results.append({
                "dataset": dataset_name,
                "feature": feature,
                "missing_count": int(missing_count),
                "total_rows": int(total),
                "missing_pct": missing_count / total,
            })

    missingness = pd.DataFrame(results)

    return missingness


def calculate_feature_missingness(train, validation, test):
    """Summarize missingness for each engineered feature."""

    train_pct = train.isna().mean()
    validation_pct = validation.isna().mean()
    test_pct = test.isna().mean()

    summary = pd.DataFrame({
        "feature": train.columns,
        "family": [
            get_feature_family(x)
            for x in train.columns
        ],
        "train_missing_count": train.isna().sum().values,
        "train_missing_pct": train_pct.values,
        "validation_missing_count": validation.isna().sum().values,
        "validation_missing_pct": validation_pct.values,
        "test_missing_count": test.isna().sum().values,
        "test_missing_pct": test_pct.values,
    })

    summary["max_missing_pct"] = summary[
        [
            "train_missing_pct",
            "validation_missing_pct",
            "test_missing_pct",
        ]
    ].max(axis=1)

    summary["missing_flag"] = np.select(
        [
            summary["max_missing_pct"] >= MISSING_VERY_HIGH_THRESHOLD,
            summary["max_missing_pct"] >= MISSING_HIGH_THRESHOLD,
            summary["max_missing_pct"] >= MISSING_WARNING_THRESHOLD,
        ],
        [
            "VERY_HIGH",
            "HIGH",
            "WARNING",
        ],
        default="LOW",
    )

    return summary.sort_values(
        "max_missing_pct",
        ascending=False
    )


def calculate_correlations(X):
    """Calculate pairwise feature correlations."""

    print("\n" + "=" * 70)
    print("PAIRWISE CORRELATION ANALYSIS")
    print("=" * 70)

    # Median imputation is used ONLY so correlation can be calculated.
    # This does not alter the modeling datasets.
    imputer = SimpleImputer(strategy="median")

    X_imputed = pd.DataFrame(
        imputer.fit_transform(X),
        columns=X.columns,
        index=X.index,
    )

    correlation_matrix = X_imputed.corr()

    pairs = []

    features = correlation_matrix.columns

    for i in range(len(features)):
        for j in range(i + 1, len(features)):

            feature_1 = features[i]
            feature_2 = features[j]

            correlation = correlation_matrix.iloc[i, j]

            if abs(correlation) >= CORRELATION_THRESHOLD:

                pairs.append({
                    "feature_1": feature_1,
                    "feature_2": feature_2,
                    "correlation": correlation,
                    "abs_correlation": abs(correlation),
                    "family_1": get_feature_family(feature_1),
                    "family_2": get_feature_family(feature_2),
                })

    pairs_df = pd.DataFrame(pairs)

    if len(pairs_df) > 0:
        pairs_df = pairs_df.sort_values(
            "abs_correlation",
            ascending=False
        )

    return correlation_matrix, pairs_df


def calculate_vif(X):
    """
    Calculate VIF using training data.

    Median imputation is used only for the diagnostic calculation.
    """

    print("\n" + "=" * 70)
    print("VARIANCE INFLATION FACTOR ANALYSIS")
    print("=" * 70)

    imputer = SimpleImputer(strategy="median")

    X_imputed = pd.DataFrame(
        imputer.fit_transform(X),
        columns=X.columns,
    )

    # Remove columns with zero variance.
    variances = X_imputed.var()

    constant_features = variances[
        variances == 0
    ].index.tolist()

    if constant_features:
        print(
            f"Removing {len(constant_features)} "
            "zero-variance features from VIF calculation."
        )

        X_imputed = X_imputed.drop(
            columns=constant_features
        )

    vif_results = []

    values = X_imputed.values

    for i, feature in enumerate(X_imputed.columns):

        try:
            vif = variance_inflation_factor(
                values,
                i,
            )
        except Exception:
            vif = np.inf

        vif_results.append({
            "feature": feature,
            "family": get_feature_family(feature),
            "vif": vif,
        })

    vif_df = pd.DataFrame(vif_results)

    vif_df["vif_flag"] = np.select(
        [
            vif_df["vif"] >= VIF_HIGH_THRESHOLD,
            vif_df["vif"] >= VIF_WARNING_THRESHOLD,
        ],
        [
            "HIGH",
            "WARNING",
        ],
        default="LOW",
    )

    return vif_df.sort_values(
        "vif",
        ascending=False
    )


def create_family_summary(features):
    """Summarize engineered features by family."""

    summary = (
        features
        .assign(
            family=features["feature"].map(
                get_feature_family
            )
        )
        .groupby("family")
        .agg(
            feature_count=("feature", "count")
        )
        .reset_index()
        .sort_values(
            "feature_count",
            ascending=False
        )
    )

    return summary


# ======================================================================
# MAIN
# ======================================================================

def main():

    print("\n")
    print("=" * 70)
    print("ENGINEERED FEATURE ANALYSIS")
    print("=" * 70)
    print("V2 Logistic Regression Feature Diagnostics")
    print("=" * 70)

    # --------------------------------------------------------------
    # Load data
    # --------------------------------------------------------------

    train, validation, test = load_data()

    # --------------------------------------------------------------
    # Validate targets
    # --------------------------------------------------------------

    print("\n" + "=" * 70)
    print("VALIDATING TARGETS")
    print("=" * 70)

    for name, df in [
        ("Training", train),
        ("Validation", validation),
        ("Test", test),
    ]:

        if "win_home" not in df.columns:
            raise ValueError(
                f"win_home missing from {name} dataset."
            )

        print(
            f"{name}: "
            f"{df['win_home'].value_counts().to_dict()}"
        )

    # --------------------------------------------------------------
    # Create V2 features
    # --------------------------------------------------------------

    print("\n" + "=" * 70)
    print("CREATING V2 ENGINEERED FEATURES")
    print("=" * 70)

    X_train = create_matchup_features(train)
    X_validation = create_matchup_features(validation)
    X_test = create_matchup_features(test)

    print(f"Training features   : {X_train.shape}")
    print(f"Validation features : {X_validation.shape}")
    print(f"Test features       : {X_test.shape}")

    if len(X_train.columns) != 60:
        raise ValueError(
            f"Expected 60 engineered features, "
            f"found {len(X_train.columns)}."
        )

    # --------------------------------------------------------------
    # Validate alignment
    # --------------------------------------------------------------

    if list(X_train.columns) != list(X_validation.columns):
        raise ValueError(
            "Training and validation feature columns do not align."
        )

    if list(X_train.columns) != list(X_test.columns):
        raise ValueError(
            "Training and test feature columns do not align."
        )

    print("Feature alignment verified.")

    # --------------------------------------------------------------
    # Feature family summary
    # --------------------------------------------------------------

    feature_metadata = pd.DataFrame({
        "feature": X_train.columns,
    })

    feature_metadata["family"] = feature_metadata[
        "feature"
    ].map(get_feature_family)

    family_summary = create_family_summary(
        feature_metadata
    )

    print("\n" + "=" * 70)
    print("FEATURE FAMILY SUMMARY")
    print("=" * 70)

    print(
        family_summary.to_string(index=False)
    )

    # --------------------------------------------------------------
    # Missingness
    # --------------------------------------------------------------

    missingness = calculate_feature_missingness(
        X_train,
        X_validation,
        X_test,
    )

    print("\n" + "=" * 70)
    print("FEATURES WITH SUBSTANTIAL MISSINGNESS")
    print("=" * 70)

    substantial_missingness = missingness[
        missingness["max_missing_pct"]
        >= MISSING_WARNING_THRESHOLD
    ]

    if len(substantial_missingness) == 0:
        print("No features exceed the missingness threshold.")
    else:
        print(
            substantial_missingness[
                [
                    "feature",
                    "family",
                    "train_missing_pct",
                    "validation_missing_pct",
                    "test_missing_pct",
                    "missing_flag",
                ]
            ]
            .to_string(index=False)
        )

    # --------------------------------------------------------------
    # Correlation
    # --------------------------------------------------------------

    correlation_matrix, correlation_pairs = (
        calculate_correlations(X_train)
    )

    print(
        f"\nFeature pairs with |correlation| >= "
        f"{CORRELATION_THRESHOLD:.2f}: "
        f"{len(correlation_pairs)}"
    )

    if len(correlation_pairs) > 0:

        print("\nTOP 30 CORRELATED PAIRS")
        print("-" * 70)

        print(
            correlation_pairs.head(30).to_string(
                index=False
            )
        )

    high_corr_pairs = correlation_pairs[
        correlation_pairs["abs_correlation"]
        >= HIGH_CORRELATION_THRESHOLD
    ]

    print(
        f"\nFeature pairs with |correlation| >= "
        f"{HIGH_CORRELATION_THRESHOLD:.2f}: "
        f"{len(high_corr_pairs)}"
    )

    # --------------------------------------------------------------
    # VIF
    # --------------------------------------------------------------

    vif_df = calculate_vif(X_train)

    print("\nTOP 30 VIF VALUES")
    print("-" * 70)

    print(
        vif_df.head(30).to_string(
            index=False
        )
    )

    print(
        f"\nFeatures with VIF >= "
        f"{VIF_WARNING_THRESHOLD:.1f}: "
        f"{(vif_df['vif'] >= VIF_WARNING_THRESHOLD).sum()}"
    )

    print(
        f"Features with VIF >= "
        f"{VIF_HIGH_THRESHOLD:.1f}: "
        f"{(vif_df['vif'] >= VIF_HIGH_THRESHOLD).sum()}"
    )

    # --------------------------------------------------------------
    # Merge diagnostic metadata
    # --------------------------------------------------------------

    diagnostics = (
        feature_metadata
        .merge(
            missingness,
            on=["feature", "family"],
            how="left",
        )
        .merge(
            vif_df[
                ["feature", "vif", "vif_flag"]
            ],
            on="feature",
            how="left",
        )
    )

    diagnostics["high_correlation_count"] = (
        diagnostics["feature"]
        .map(
            correlation_pairs[
                "feature_1"
            ].value_counts()
        )
        .fillna(0)
        +
        diagnostics["feature"]
        .map(
            correlation_pairs[
                "feature_2"
            ].value_counts()
        )
        .fillna(0)
    )

    # --------------------------------------------------------------
    # Save results
    # --------------------------------------------------------------

    print("\n" + "=" * 70)
    print("SAVING ANALYSIS RESULTS")
    print("=" * 70)

    correlation_matrix_path = (
        OUTPUT_DIR / "v2_correlation_matrix.csv"
    )

    correlation_pairs_path = (
        OUTPUT_DIR / "v2_high_correlation_pairs.csv"
    )

    vif_path = (
        OUTPUT_DIR / "v2_vif.csv"
    )

    missingness_path = (
        OUTPUT_DIR / "v2_missingness.csv"
    )

    diagnostics_path = (
        OUTPUT_DIR / "v2_feature_diagnostics.csv"
    )

    family_path = (
        OUTPUT_DIR / "v2_feature_families.csv"
    )

    correlation_matrix.to_csv(
        correlation_matrix_path
    )

    correlation_pairs.to_csv(
        correlation_pairs_path,
        index=False,
    )

    vif_df.to_csv(
        vif_path,
        index=False,
    )

    missingness.to_csv(
        missingness_path,
        index=False,
    )

    diagnostics.to_csv(
        diagnostics_path,
        index=False,
    )

    feature_metadata.to_csv(
        family_path,
        index=False,
    )

    # --------------------------------------------------------------
    # Save summary JSON
    # --------------------------------------------------------------

    summary = {
        "engineered_feature_count": len(X_train.columns),

        "correlation_threshold": CORRELATION_THRESHOLD,
        "high_correlation_threshold": HIGH_CORRELATION_THRESHOLD,

        "correlation_pairs_above_threshold": int(
            len(correlation_pairs)
        ),

        "correlation_pairs_above_high_threshold": int(
            len(high_corr_pairs)
        ),

        "vif_warning_threshold": VIF_WARNING_THRESHOLD,
        "vif_high_threshold": VIF_HIGH_THRESHOLD,

        "features_vif_above_warning": int(
            (vif_df["vif"] >= VIF_WARNING_THRESHOLD).sum()
        ),

        "features_vif_above_high": int(
            (vif_df["vif"] >= VIF_HIGH_THRESHOLD).sum()
        ),

        "features_missing_above_10pct": int(
            (
                missingness["max_missing_pct"]
                >= MISSING_WARNING_THRESHOLD
            ).sum()
        ),

        "features_missing_above_25pct": int(
            (
                missingness["max_missing_pct"]
                >= MISSING_HIGH_THRESHOLD
            ).sum()
        ),

        "features_missing_above_50pct": int(
            (
                missingness["max_missing_pct"]
                >= MISSING_VERY_HIGH_THRESHOLD
            ).sum()
        ),
    }

    summary_path = (
        OUTPUT_DIR / "v2_feature_analysis_summary.json"
    )

    with open(summary_path, "w") as f:
        json.dump(
            summary,
            f,
            indent=4,
        )

    print(f"Correlation matrix : {correlation_matrix_path}")
    print(f"Correlation pairs  : {correlation_pairs_path}")
    print(f"VIF results        : {vif_path}")
    print(f"Missingness        : {missingness_path}")
    print(f"Diagnostics        : {diagnostics_path}")
    print(f"Feature families   : {family_path}")
    print(f"Summary            : {summary_path}")

    # --------------------------------------------------------------
    # Final summary
    # --------------------------------------------------------------

    print("\n" + "=" * 70)
    print("ENGINEERED FEATURE ANALYSIS COMPLETE")
    print("=" * 70)

    print(
        f"Engineered features : {len(X_train.columns)}"
    )

    print(
        f"High correlations   : {len(high_corr_pairs)}"
    )

    print(
        f"VIF >= 5            : "
        f"{(vif_df['vif'] >= 5).sum()}"
    )

    print(
        f"VIF >= 10           : "
        f"{(vif_df['vif'] >= 10).sum()}"
    )

    print(
        f"Missing >= 10%      : "
        f"{(missingness['max_missing_pct'] >= 0.10).sum()}"
    )

    print(
        f"Missing >= 25%      : "
        f"{(missingness['max_missing_pct'] >= 0.25).sum()}"
    )

    print(
        f"Missing >= 50%      : "
        f"{(missingness['max_missing_pct'] >= 0.50).sum()}"
    )

    print("=" * 70)


if __name__ == "__main__":
    main()