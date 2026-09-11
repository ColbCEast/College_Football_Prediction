"""
GRADIENT BOOSTING WIN PROBABILITY
RETURNING PRODUCTION - PREDICTION SHIFT ANALYSIS

Purpose
-------
Diagnose HOW the returning-production features change predictions relative
to Gradient Boosting Model 5.

This analysis is diagnostic only. The 2025 test set is used to understand
the behavior of Experiment 1. It must NOT be used to tune a final model.

Comparison
----------
Baseline:
    Gradient Boosting Model 5
    Complete predictive-safe 310-feature space

Experiment:
    Returning Production Experiment Model 1
    Model 5 + 16 raw returning PPA/usage features

Primary questions
-----------------
1. How much do predictions move when returning production is added?
2. Does the prediction shift differ by season phase?
3. Does returning production move predictions toward or away from the
   actual outcome?
4. Does returning production systematically increase or decrease confidence?
5. Which returning-production features are actually being used by the model?

Season phase
------------
Based primarily on the minimum number of games played by either team
before the current game:

    0 games
    1-2 games
    3-5 games
    6+ games

Outputs
-------
models/win_probability/gradient_boosting/
    returning_production/
        model_1/
            prediction_shift_analysis/
                prediction_shift_by_phase.csv
                prediction_shift_overall.csv
                prediction_shift_examples.csv
                returning_feature_importance.csv
                prediction_shift_summary.csv
                prediction_shift_distribution.png
                prediction_shift_by_phase.png

Run from project root:
    python src/modeling/problems/win_probability/gradient_boosting/
        returning_production/analysis/prediction_shift.py
"""

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib

from sklearn.metrics import log_loss


# =============================================================================
# PROJECT PATHS
# =============================================================================

PROJECT_ROOT = Path(__file__).resolve()

while PROJECT_ROOT.name != "College_Football_Prediction":
    if PROJECT_ROOT.parent == PROJECT_ROOT:
        raise RuntimeError(
            "Could not locate project root directory "
            "'College_Football_Prediction'."
        )
    PROJECT_ROOT = PROJECT_ROOT.parent


# Experiment 1
EXPERIMENT_DIR = (
    PROJECT_ROOT
    / "models"
    / "win_probability"
    / "gradient_boosting"
    / "returning_production"
    / "model_1"
)

EXPERIMENT_PREDICTIONS = (
    EXPERIMENT_DIR
    / "test_predictions.csv"
)

EXPERIMENT_MODEL = (
    EXPERIMENT_DIR
    / "model.joblib"
)

EXPERIMENT_FEATURE_LIST = (
    EXPERIMENT_DIR
    / "feature_list.csv"
)


# Model 5
MODEL_5_DIR = (
    PROJECT_ROOT
    / "models"
    / "win_probability"
    / "gradient_boosting"
    / "model_5"
)

MODEL_5_PREDICTIONS = (
    MODEL_5_DIR
    / "test_predictions.csv"
)


# 2025 final features
FINAL_FEATURES = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "features"
    / "final"
    / "final_features_2025.csv"
)


# Output directory
OUTPUT_DIR = (
    EXPERIMENT_DIR
    / "prediction_shift_analysis"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# =============================================================================
# CONFIGURATION
# =============================================================================

SEASON = 2025

EPSILON = 1e-15

PHASE_ORDER = [
    "0 games",
    "1-2 games",
    "3-5 games",
    "6+ games",
]


RETURNING_FEATURES = [
    "home_returning_total_ppa",
    "away_returning_total_ppa",
    "home_returning_passing_ppa",
    "away_returning_passing_ppa",
    "home_returning_receiving_ppa",
    "away_returning_receiving_ppa",
    "home_returning_rushing_ppa",
    "away_returning_rushing_ppa",
    "home_returning_usage",
    "away_returning_usage",
    "home_returning_passing_usage",
    "away_returning_passing_usage",
    "home_returning_receiving_usage",
    "away_returning_receiving_usage",
    "home_returning_rushing_usage",
    "away_returning_rushing_usage",
]


# =============================================================================
# UTILITIES
# =============================================================================

def check_file_exists(
    path: Path,
    description: str,
) -> None:
    """Verify that a required file exists."""
    if not path.exists():
        raise FileNotFoundError(
            f"\nMissing {description}:\n"
            f"  {path}\n"
        )


def safe_probability(
    probabilities: pd.Series,
) -> np.ndarray:
    """Clip probabilities away from 0 and 1."""
    return np.clip(
        probabilities.astype(float).to_numpy(),
        EPSILON,
        1.0 - EPSILON,
    )


def assign_phase(
    value: float,
) -> str:
    """Assign gamesBefore to a season-phase bucket."""
    if value == 0:
        return "0 games"

    if value <= 2:
        return "1-2 games"

    if value <= 5:
        return "3-5 games"

    return "6+ games"


# =============================================================================
# DATA LOADING
# =============================================================================

def load_experiment_predictions() -> pd.DataFrame:
    """Load Experiment 1 predictions."""
    check_file_exists(
        EXPERIMENT_PREDICTIONS,
        "Experiment 1 test predictions",
    )

    df = pd.read_csv(
        EXPERIMENT_PREDICTIONS
    )

    required = {
        "gameId",
        "season",
        "win_home",
        "predicted_probability_home_win",
        "predicted_home_win",
    }

    missing = required - set(df.columns)

    if missing:
        raise ValueError(
            "Experiment 1 prediction file is missing columns:\n"
            f"  {sorted(missing)}"
        )

    df = df[
        [
            "gameId",
            "season",
            "win_home",
            "predicted_probability_home_win",
            "predicted_home_win",
        ]
    ].copy()

    df = df.rename(
        columns={
            "win_home": "actual",
            "predicted_probability_home_win":
                "experiment_probability",
            "predicted_home_win":
                "experiment_prediction",
        }
    )

    return df


def load_model_5_predictions() -> pd.DataFrame:
    """Load Model 5 predictions."""
    check_file_exists(
        MODEL_5_PREDICTIONS,
        "Gradient Boosting Model 5 test predictions",
    )

    df = pd.read_csv(
        MODEL_5_PREDICTIONS
    )

    required = {
        "gameId",
        "season",
        "win_home_actual",
        "win_home_probability",
        "win_home_prediction",
    }

    missing = required - set(df.columns)

    if missing:
        raise ValueError(
            "Model 5 prediction file is missing columns:\n"
            f"  {sorted(missing)}"
        )

    df = df[
        [
            "gameId",
            "season",
            "win_home_actual",
            "win_home_probability",
            "win_home_prediction",
        ]
    ].copy()

    df = df.rename(
        columns={
            "win_home_actual": "actual_model_5",
            "win_home_probability":
                "model_5_probability",
            "win_home_prediction":
                "model_5_prediction",
        }
    )

    return df


def load_games_before() -> pd.DataFrame:
    """Load gamesBefore information from final 2025 features."""
    check_file_exists(
        FINAL_FEATURES,
        "2025 final feature data",
    )

    df = pd.read_csv(
        FINAL_FEATURES
    )

    required = {
        "gameId",
        "season",
        "gamesBefore_home",
        "gamesBefore_away",
    }

    missing = required - set(df.columns)

    if missing:
        raise ValueError(
            "Final feature file is missing columns:\n"
            f"  {sorted(missing)}"
        )

    return df[
        [
            "gameId",
            "season",
            "gamesBefore_home",
            "gamesBefore_away",
        ]
    ].copy()


# =============================================================================
# ALIGN DATA
# =============================================================================

def validate_unique_game_ids(
    df: pd.DataFrame,
    name: str,
) -> None:
    """Verify gameId uniqueness."""
    if df["gameId"].duplicated().any():
        raise ValueError(
            f"{name} contains duplicate gameIds."
        )


def load_and_align() -> pd.DataFrame:
    """Load and align all required datasets."""
    experiment = load_experiment_predictions()
    model_5 = load_model_5_predictions()
    games = load_games_before()

    validate_unique_game_ids(
        experiment,
        "Experiment 1 predictions",
    )

    validate_unique_game_ids(
        model_5,
        "Model 5 predictions",
    )

    validate_unique_game_ids(
        games,
        "Final features",
    )

    print("\n" + "=" * 80)
    print("DATA ALIGNMENT")
    print("=" * 80)

    print(
        f"Experiment 1 rows: {len(experiment):,}"
    )

    print(
        f"Model 5 rows:      {len(model_5):,}"
    )

    print(
        f"Final feature rows:{len(games):,}"
    )

    # -------------------------------------------------------------------------
    # Check seasons
    # -------------------------------------------------------------------------

    for name, data in [
        ("Experiment 1", experiment),
        ("Model 5", model_5),
        ("Final features", games),
    ]:
        seasons = sorted(
            data["season"]
            .dropna()
            .unique()
            .tolist()
        )

        if seasons != [SEASON]:
            raise ValueError(
                f"{name} contains unexpected seasons: "
                f"{seasons}"
            )

    # -------------------------------------------------------------------------
    # Merge
    # -------------------------------------------------------------------------

    df = experiment.merge(
        model_5,
        on=["gameId", "season"],
        how="inner",
        validate="one_to_one",
    )

    df = df.merge(
        games,
        on=["gameId", "season"],
        how="inner",
        validate="one_to_one",
    )

    print(
        f"Aligned rows:       {len(df):,}"
    )

    if len(df) != len(experiment):
        raise ValueError(
            "Not all Experiment 1 prediction rows "
            "were successfully aligned."
        )

    # -------------------------------------------------------------------------
    # Outcome agreement
    # -------------------------------------------------------------------------

    if not (
        df["actual"]
        == df["actual_model_5"]
    ).all():
        raise ValueError(
            "Experiment 1 and Model 5 actual outcomes disagree."
        )

    # -------------------------------------------------------------------------
    # gamesBefore validation
    # -------------------------------------------------------------------------

    if df[
        [
            "gamesBefore_home",
            "gamesBefore_away",
        ]
    ].isna().any().any():
        raise ValueError(
            "Missing gamesBefore values found."
        )

    if (
        df["gamesBefore_home"] < 0
    ).any() or (
        df["gamesBefore_away"] < 0
    ).any():
        raise ValueError(
            "Negative gamesBefore values found."
        )

    # -------------------------------------------------------------------------
    # Create season-phase variables
    # -------------------------------------------------------------------------

    df["gamesBefore_min"] = df[
        [
            "gamesBefore_home",
            "gamesBefore_away",
        ]
    ].min(axis=1)

    df["phase_min"] = (
        df["gamesBefore_min"]
        .apply(assign_phase)
    )

    df["phase_home"] = (
        df["gamesBefore_home"]
        .apply(assign_phase)
    )

    df["phase_away"] = (
        df["gamesBefore_away"]
        .apply(assign_phase)
    )

    return df


# =============================================================================
# PREDICTION SHIFT VARIABLES
# =============================================================================

def create_shift_variables(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """Create variables describing how predictions changed."""
    df = df.copy()

    model_5 = df[
        "model_5_probability"
    ].astype(float)

    experiment = df[
        "experiment_probability"
    ].astype(float)

    # -------------------------------------------------------------------------
    # Raw prediction shift
    # -------------------------------------------------------------------------

    df["prediction_shift"] = (
        experiment - model_5
    )

    df["absolute_prediction_shift"] = (
        df["prediction_shift"].abs()
    )

    # -------------------------------------------------------------------------
    # Probability confidence
    # -------------------------------------------------------------------------

    # Distance from 0.5 represents confidence in either direction.
    df["model_5_confidence"] = (
        (model_5 - 0.5).abs()
    )

    df["experiment_confidence"] = (
        (experiment - 0.5).abs()
    )

    df["confidence_shift"] = (
        df["experiment_confidence"]
        - df["model_5_confidence"]
    )

    # -------------------------------------------------------------------------
    # Correctness of probability direction
    # -------------------------------------------------------------------------

    # For a home-win probability:
    #
    # actual = 1:
    #   higher probability is better
    #
    # actual = 0:
    #   lower probability is better
    #
    # Therefore, multiply probability shift by:
    #
    #   +1 for home wins
    #   -1 for home losses
    #
    # Positive value means Experiment 1 moved toward the actual outcome.
    # Negative value means Experiment 1 moved away from the actual outcome.

    df["outcome_direction"] = (
        2 * df["actual"].astype(int) - 1
    )

    df["directional_shift"] = (
        df["prediction_shift"]
        * df["outcome_direction"]
    )

    df["moved_toward_outcome"] = (
        df["directional_shift"] > 0
    )

    df["moved_away_from_outcome"] = (
        df["directional_shift"] < 0
    )

    df["no_directional_change"] = (
        df["directional_shift"] == 0
    )

    # -------------------------------------------------------------------------
    # Whether each model prediction is correct
    # -------------------------------------------------------------------------

    df["model_5_correct"] = (
        df["model_5_prediction"].astype(int)
        == df["actual"].astype(int)
    )

    df["experiment_correct"] = (
        df["experiment_prediction"].astype(int)
        == df["actual"].astype(int)
    )

    # -------------------------------------------------------------------------
    # Probability movement category
    # -------------------------------------------------------------------------

    df["shift_direction"] = np.select(
        [
            df["prediction_shift"] < 0,
            df["prediction_shift"] > 0,
        ],
        [
            "Lowered probability",
            "Raised probability",
        ],
        default="No change",
    )

    return df


# =============================================================================
# OVERALL METRICS
# =============================================================================

def calculate_overall_shift_metrics(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """Calculate overall prediction-shift statistics."""
    model_5_prob = np.clip(
        df["model_5_probability"]
        .astype(float)
        .to_numpy(),
        EPSILON,
        1 - EPSILON,
    )

    experiment_prob = np.clip(
        df["experiment_probability"]
        .astype(float)
        .to_numpy(),
        EPSILON,
        1 - EPSILON,
    )

    actual = (
        df["actual"]
        .astype(int)
        .to_numpy()
    )

    result = {
        "games": len(df),

        "mean_prediction_shift": df[
            "prediction_shift"
        ].mean(),

        "median_prediction_shift": df[
            "prediction_shift"
        ].median(),

        "mean_absolute_prediction_shift": df[
            "absolute_prediction_shift"
        ].mean(),

        "median_absolute_prediction_shift": df[
            "absolute_prediction_shift"
        ].median(),

        "max_absolute_prediction_shift": df[
            "absolute_prediction_shift"
        ].max(),

        "mean_confidence_shift": df[
            "confidence_shift"
        ].mean(),

        "pct_predictions_raised": (
            df["prediction_shift"] > 0
        ).mean(),

        "pct_predictions_lowered": (
            df["prediction_shift"] < 0
        ).mean(),

        "pct_no_prediction_change": (
            df["prediction_shift"] == 0
        ).mean(),

        "pct_moved_toward_outcome": (
            df["moved_toward_outcome"]
        ).mean(),

        "pct_moved_away_from_outcome": (
            df["moved_away_from_outcome"]
        ).mean(),

        "pct_no_directional_change": (
            df["no_directional_change"]
        ).mean(),

        "model_5_log_loss": log_loss(
            actual,
            model_5_prob,
            labels=[0, 1],
        ),

        "experiment_1_log_loss": log_loss(
            actual,
            experiment_prob,
            labels=[0, 1],
        ),
    }

    result["delta_log_loss"] = (
        result["experiment_1_log_loss"]
        - result["model_5_log_loss"]
    )

    return pd.DataFrame([result])


# =============================================================================
# PHASE ANALYSIS
# =============================================================================

def calculate_phase_shift_metrics(
    df: pd.DataFrame,
    phase_column: str,
) -> pd.DataFrame:
    """Calculate prediction-shift statistics by season phase."""
    results = []

    for phase in PHASE_ORDER:

        subset = df[
            df[phase_column] == phase
        ].copy()

        if subset.empty:
            continue

        model_5_prob = np.clip(
            subset[
                "model_5_probability"
            ].astype(float).to_numpy(),
            EPSILON,
            1 - EPSILON,
        )

        experiment_prob = np.clip(
            subset[
                "experiment_probability"
            ].astype(float).to_numpy(),
            EPSILON,
            1 - EPSILON,
        )

        actual = (
            subset["actual"]
            .astype(int)
            .to_numpy()
        )

        result = {
            "season_phase": phase,
            "games": len(subset),

            "pct_test_games": (
                len(subset) / len(df)
            ),

            "mean_prediction_shift": subset[
                "prediction_shift"
            ].mean(),

            "median_prediction_shift": subset[
                "prediction_shift"
            ].median(),

            "mean_absolute_prediction_shift": subset[
                "absolute_prediction_shift"
            ].mean(),

            "median_absolute_prediction_shift": subset[
                "absolute_prediction_shift"
            ].median(),

            "max_absolute_prediction_shift": subset[
                "absolute_prediction_shift"
            ].max(),

            "mean_confidence_shift": subset[
                "confidence_shift"
            ].mean(),

            "pct_predictions_raised": (
                subset["prediction_shift"] > 0
            ).mean(),

            "pct_predictions_lowered": (
                subset["prediction_shift"] < 0
            ).mean(),

            "pct_moved_toward_outcome": (
                subset["moved_toward_outcome"]
            ).mean(),

            "pct_moved_away_from_outcome": (
                subset["moved_away_from_outcome"]
            ).mean(),

            "model_5_log_loss": log_loss(
                actual,
                model_5_prob,
                labels=[0, 1],
            ),

            "experiment_1_log_loss": log_loss(
                actual,
                experiment_prob,
                labels=[0, 1],
            ),
        }

        result["delta_log_loss"] = (
            result["experiment_1_log_loss"]
            - result["model_5_log_loss"]
        )

        results.append(result)

    result_df = pd.DataFrame(results)

    result_df["season_phase"] = pd.Categorical(
        result_df["season_phase"],
        categories=PHASE_ORDER,
        ordered=True,
    )

    return (
        result_df
        .sort_values("season_phase")
        .reset_index(drop=True)
    )


# =============================================================================
# OUTCOME-SPECIFIC ANALYSIS
# =============================================================================

def calculate_outcome_analysis(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Analyze prediction shifts separately for home wins and home losses.
    """
    results = []

    for actual_value, label in [
        (1, "Home win"),
        (0, "Home loss"),
    ]:

        subset = df[
            df["actual"] == actual_value
        ].copy()

        if subset.empty:
            continue

        results.append(
            {
                "actual_outcome": label,
                "games": len(subset),

                "mean_prediction_shift": subset[
                    "prediction_shift"
                ].mean(),

                "median_prediction_shift": subset[
                    "prediction_shift"
                ].median(),

                "mean_absolute_prediction_shift": subset[
                    "absolute_prediction_shift"
                ].mean(),

                "pct_moved_toward_outcome": subset[
                    "moved_toward_outcome"
                ].mean(),

                "pct_moved_away_from_outcome": subset[
                    "moved_away_from_outcome"
                ].mean(),

                "mean_confidence_shift": subset[
                    "confidence_shift"
                ].mean(),
            }
        )

    return pd.DataFrame(results)


# =============================================================================
# PREDICTION SHIFT EXAMPLES
# =============================================================================

def create_prediction_examples(
    df: pd.DataFrame,
    n: int = 25,
) -> pd.DataFrame:
    """
    Save the largest prediction shifts for manual inspection.
    """
    columns = [
        "gameId",
        "season",
        "actual",
        "gamesBefore_home",
        "gamesBefore_away",
        "gamesBefore_min",
        "phase_min",
        "model_5_probability",
        "experiment_probability",
        "prediction_shift",
        "absolute_prediction_shift",
        "model_5_confidence",
        "experiment_confidence",
        "confidence_shift",
        "directional_shift",
        "moved_toward_outcome",
        "moved_away_from_outcome",
        "model_5_correct",
        "experiment_correct",
    ]

    examples = (
        df.sort_values(
            "absolute_prediction_shift",
            ascending=False,
        )
        .head(n)
        .copy()
    )

    return examples[columns]


# =============================================================================
# FEATURE IMPORTANCE
# =============================================================================

def load_feature_importance() -> pd.DataFrame:
    """
    Load Experiment 1 model and extract feature importance.

    The model is expected to be a sklearn Pipeline containing:
        - an imputation step
        - a GradientBoostingClassifier
    """
    check_file_exists(
        EXPERIMENT_MODEL,
        "Experiment 1 model",
    )

    check_file_exists(
        EXPERIMENT_FEATURE_LIST,
        "Experiment 1 feature list",
    )

    print("\n" + "=" * 80)
    print("RETURNING-PRODUCTION FEATURE IMPORTANCE")
    print("=" * 80)

    model = joblib.load(
        EXPERIMENT_MODEL
    )

    feature_list_df = pd.read_csv(
        EXPERIMENT_FEATURE_LIST
    )

    if "feature" in feature_list_df.columns:
        feature_names = (
            feature_list_df["feature"]
            .astype(str)
            .tolist()
        )
    elif "feature_name" in feature_list_df.columns:
        feature_names = (
            feature_list_df["feature_name"]
            .astype(str)
            .tolist()
        )
    else:
        raise ValueError(
            "Could not identify feature-name column in "
            "Experiment 1 feature_list.csv. Expected "
            "'feature' or 'feature_name'."
        )

    # -------------------------------------------------------------------------
    # Identify GradientBoostingClassifier inside Pipeline
    # -------------------------------------------------------------------------

    estimator = model

    if hasattr(model, "named_steps"):
        classifier = None

        for step_name, step in model.named_steps.items():
            if hasattr(step, "feature_importances_"):
                classifier = step
                break

        if classifier is None:
            raise ValueError(
                "Could not find an estimator with "
                "feature_importances_ inside Experiment 1 Pipeline."
            )

        estimator = classifier

    if not hasattr(
        estimator,
        "feature_importances_",
    ):
        raise ValueError(
            "Experiment 1 model does not expose "
            "feature_importances_."
        )

    importances = np.asarray(
        estimator.feature_importances_
    )

    if len(importances) != len(feature_names):
        raise ValueError(
            "Feature importance length does not match "
            "feature list length.\n"
            f"Importances: {len(importances)}\n"
            f"Features:    {len(feature_names)}"
        )

    importance_df = pd.DataFrame(
        {
            "feature": feature_names,
            "importance": importances,
        }
    )

    importance_df["is_returning_feature"] = (
        importance_df["feature"].isin(
            RETURNING_FEATURES
        )
    )

    importance_df["feature_group"] = np.where(
        importance_df["is_returning_feature"],
        "Returning production",
        "Model 5 baseline",
    )

    importance_df = importance_df.sort_values(
        "importance",
        ascending=False,
    ).reset_index(drop=True)

    importance_df["importance_rank"] = (
        np.arange(len(importance_df)) + 1
    )

    return importance_df


# =============================================================================
# SUMMARY
# =============================================================================

def create_summary(
    overall: pd.DataFrame,
    phase_results: pd.DataFrame,
    feature_importance: pd.DataFrame,
) -> pd.DataFrame:
    """
    Create a compact summary of the major findings.
    """

    returning_importance = feature_importance[
        feature_importance[
            "is_returning_feature"
        ]
    ].copy()

    total_returning_importance = (
        returning_importance[
            "importance"
        ].sum()
    )

    if not returning_importance.empty:
        top_returning = (
            returning_importance.iloc[0]["feature"]
        )

        top_returning_importance = (
            returning_importance.iloc[0]["importance"]
        )

        top_returning_rank = (
            returning_importance.iloc[0][
                "importance_rank"
            ]
        )
    else:
        top_returning = np.nan
        top_returning_importance = np.nan
        top_returning_rank = np.nan

    worst_phase_row = phase_results.loc[
        phase_results["delta_log_loss"].idxmax()
    ]

    best_phase_row = phase_results.loc[
        phase_results["delta_log_loss"].idxmin()
    ]

    return pd.DataFrame(
        [
            {
                "overall_delta_log_loss":
                    overall.iloc[0]["delta_log_loss"],

                "overall_mean_prediction_shift":
                    overall.iloc[0][
                        "mean_prediction_shift"
                    ],

                "overall_mean_absolute_prediction_shift":
                    overall.iloc[0][
                        "mean_absolute_prediction_shift"
                    ],

                "overall_pct_moved_toward_outcome":
                    overall.iloc[0][
                        "pct_moved_toward_outcome"
                    ],

                "overall_pct_moved_away_from_outcome":
                    overall.iloc[0][
                        "pct_moved_away_from_outcome"
                    ],

                "overall_mean_confidence_shift":
                    overall.iloc[0][
                        "mean_confidence_shift"
                    ],

                "worst_phase_by_log_loss":
                    str(
                        worst_phase_row[
                            "season_phase"
                        ]
                    ),

                "worst_phase_delta_log_loss":
                    worst_phase_row[
                        "delta_log_loss"
                    ],

                "best_phase_by_log_loss":
                    str(
                        best_phase_row[
                            "season_phase"
                        ]
                    ),

                "best_phase_delta_log_loss":
                    best_phase_row[
                        "delta_log_loss"
                    ],

                "total_returning_feature_importance":
                    total_returning_importance,

                "top_returning_feature":
                    top_returning,

                "top_returning_feature_importance":
                    top_returning_importance,

                "top_returning_feature_global_rank":
                    top_returning_rank,
            }
        ]
    )


# =============================================================================
# PLOTS
# =============================================================================

def create_shift_distribution_plot(
    df: pd.DataFrame,
) -> Path:
    """
    Plot the distribution of prediction shifts.
    """
    path = (
        OUTPUT_DIR
        / "prediction_shift_distribution.png"
    )

    plt.figure(
        figsize=(9, 6)
    )

    plt.hist(
        df["prediction_shift"],
        bins=40,
        edgecolor="black",
    )

    plt.axvline(
        0,
        linewidth=1,
        linestyle="--",
    )

    plt.xlabel(
        "Prediction Shift "
        "(Experiment 1 − Model 5)"
    )

    plt.ylabel(
        "Number of Games"
    )

    plt.title(
        "Distribution of Returning-Production Prediction Shifts"
    )

    plt.tight_layout()

    plt.savefig(
        path,
        dpi=200,
        bbox_inches="tight",
    )

    plt.close()

    return path


def create_phase_shift_plot(
    phase_results: pd.DataFrame,
) -> Path:
    """
    Plot mean prediction shift by season phase.
    """
    path = (
        OUTPUT_DIR
        / "prediction_shift_by_phase.png"
    )

    plot_df = phase_results.copy()

    x = np.arange(
        len(plot_df)
    )

    plt.figure(
        figsize=(9, 6)
    )

    plt.axhline(
        0,
        linewidth=1,
        linestyle="--",
    )

    plt.plot(
        x,
        plot_df["mean_prediction_shift"],
        marker="o",
        linewidth=2,
    )

    plt.xticks(
        x,
        plot_df["season_phase"].astype(str),
    )

    plt.xlabel(
        "Minimum games played before current game"
    )

    plt.ylabel(
        "Mean Prediction Shift "
        "(Experiment 1 − Model 5)"
    )

    plt.title(
        "Returning-Production Prediction Shift by Season Phase"
    )

    plt.grid(
        axis="y",
        alpha=0.3,
    )

    plt.tight_layout()

    plt.savefig(
        path,
        dpi=200,
        bbox_inches="tight",
    )

    plt.close()

    return path


# =============================================================================
# PRINT RESULTS
# =============================================================================

def print_results(
    overall: pd.DataFrame,
    phase_results: pd.DataFrame,
    outcome_results: pd.DataFrame,
    feature_importance: pd.DataFrame,
) -> None:
    """Print major findings."""

    print("\n" + "=" * 80)
    print("OVERALL PREDICTION SHIFT")
    print("=" * 80)

    print(
        overall.to_string(
            index=False,
            float_format=lambda x: f"{x:.6f}",
        )
    )

    print("\n" + "=" * 80)
    print("PREDICTION SHIFT BY SEASON PHASE")
    print("=" * 80)

    phase_columns = [
        "season_phase",
        "games",
        "pct_test_games",
        "mean_prediction_shift",
        "mean_absolute_prediction_shift",
        "mean_confidence_shift",
        "pct_predictions_raised",
        "pct_predictions_lowered",
        "pct_moved_toward_outcome",
        "pct_moved_away_from_outcome",
        "model_5_log_loss",
        "experiment_1_log_loss",
        "delta_log_loss",
    ]

    print(
        phase_results[
            phase_columns
        ].to_string(
            index=False,
            float_format=lambda x: f"{x:.6f}",
        )
    )

    print("\n" + "=" * 80)
    print("SHIFT BY ACTUAL OUTCOME")
    print("=" * 80)

    print(
        outcome_results.to_string(
            index=False,
            float_format=lambda x: f"{x:.6f}",
        )
    )

    print("\n" + "=" * 80)
    print("RETURNING-PRODUCTION FEATURE IMPORTANCE")
    print("=" * 80)

    returning = feature_importance[
        feature_importance[
            "is_returning_feature"
        ]
    ].copy()

    print(
        returning[
            [
                "importance_rank",
                "feature",
                "importance",
            ]
        ].to_string(
            index=False,
            float_format=lambda x: f"{x:.8f}",
        )
    )


# =============================================================================
# MAIN
# =============================================================================

def main() -> None:
    """Run complete prediction-shift analysis."""

    print("=" * 80)
    print("GRADIENT BOOSTING WIN PROBABILITY")
    print("RETURNING PRODUCTION - PREDICTION SHIFT ANALYSIS")
    print("=" * 80)

    print("\nProject root:")
    print(f"  {PROJECT_ROOT}")

    print("\nOutput directory:")
    print(f"  {OUTPUT_DIR}")

    # -------------------------------------------------------------------------
    # Load and align
    # -------------------------------------------------------------------------

    df = load_and_align()

    # -------------------------------------------------------------------------
    # Create prediction-shift variables
    # -------------------------------------------------------------------------

    df = create_shift_variables(df)

    # -------------------------------------------------------------------------
    # Calculate analyses
    # -------------------------------------------------------------------------

    overall = calculate_overall_shift_metrics(
        df
    )

    phase_results = calculate_phase_shift_metrics(
        df,
        "phase_min",
    )

    outcome_results = calculate_outcome_analysis(
        df
    )

    # -------------------------------------------------------------------------
    # Feature importance
    # -------------------------------------------------------------------------

    feature_importance = load_feature_importance()

    # -------------------------------------------------------------------------
    # Summary
    # -------------------------------------------------------------------------

    summary = create_summary(
        overall,
        phase_results,
        feature_importance,
    )

    # -------------------------------------------------------------------------
    # Save CSVs
    # -------------------------------------------------------------------------

    overall_path = (
        OUTPUT_DIR
        / "prediction_shift_overall.csv"
    )

    phase_path = (
        OUTPUT_DIR
        / "prediction_shift_by_phase.csv"
    )

    outcome_path = (
        OUTPUT_DIR
        / "prediction_shift_by_outcome.csv"
    )

    examples_path = (
        OUTPUT_DIR
        / "prediction_shift_examples.csv"
    )

    importance_path = (
        OUTPUT_DIR
        / "returning_feature_importance.csv"
    )

    summary_path = (
        OUTPUT_DIR
        / "prediction_shift_summary.csv"
    )

    overall.to_csv(
        overall_path,
        index=False,
    )

    phase_results.to_csv(
        phase_path,
        index=False,
    )

    outcome_results.to_csv(
        outcome_path,
        index=False,
    )

    create_prediction_examples(
        df,
        n=25,
    ).to_csv(
        examples_path,
        index=False,
    )

    feature_importance.to_csv(
        importance_path,
        index=False,
    )

    summary.to_csv(
        summary_path,
        index=False,
    )

    # -------------------------------------------------------------------------
    # Plots
    # -------------------------------------------------------------------------

    distribution_plot = (
        create_shift_distribution_plot(df)
    )

    phase_plot = (
        create_phase_shift_plot(
            phase_results
        )
    )

    # -------------------------------------------------------------------------
    # Print results
    # -------------------------------------------------------------------------

    print_results(
        overall,
        phase_results,
        outcome_results,
        feature_importance,
    )

    # -------------------------------------------------------------------------
    # Output summary
    # -------------------------------------------------------------------------

    print("\n" + "=" * 80)
    print("OUTPUTS SAVED")
    print("=" * 80)

    print(f"\n  {overall_path}")
    print(f"  {phase_path}")
    print(f"  {outcome_path}")
    print(f"  {examples_path}")
    print(f"  {importance_path}")
    print(f"  {summary_path}")
    print(f"  {distribution_plot}")
    print(f"  {phase_plot}")

    print("\n" + "=" * 80)
    print("INTERPRETATION GUIDE")
    print("=" * 80)

    print(
        "\nPrediction Shift:"
    )

    print(
        "  Positive -> Experiment 1 increased home-win probability."
    )

    print(
        "  Negative -> Experiment 1 decreased home-win probability."
    )

    print(
        "\nDirectional Shift:"
    )

    print(
        "  Positive -> returning production moved probability "
        "toward the actual outcome."
    )

    print(
        "  Negative -> returning production moved probability "
        "away from the actual outcome."
    )

    print(
        "\nConfidence Shift:"
    )

    print(
        "  Positive -> Experiment 1 became more confident."
    )

    print(
        "  Negative -> Experiment 1 became less confident."
    )

    print(
        "\nFeature importance:"
    )

    print(
        "  Higher values indicate greater contribution to the "
        "Experiment 1 Gradient Boosting model's fitted splits."
    )

    print(
        "\nThis analysis is diagnostic only. "
        "The 2025 test results should not be used "
        "to directly tune Experiment 2."
    )

    print("\nAnalysis complete.")


if __name__ == "__main__":
    main()