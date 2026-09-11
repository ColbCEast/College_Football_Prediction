"""
GRADIENT BOOSTING WIN PROBABILITY
RETURNING PRODUCTION - SEASON PHASE ANALYSIS

Purpose
-------
Analyze whether returning production provides incremental predictive value
at different points in the season.

This is a diagnostic analysis only. The 2025 test set is used to understand
where Experiment 1 differs from the Model 5 baseline. These results should
NOT be used to tune a final model directly, since 2025 remains the final
held-out test season.

Comparison
----------
Baseline:
    Gradient Boosting Model 5
    Complete predictive-safe 310-feature space

Experiment:
    Returning Production Experiment Model 1
    Model 5 + 16 raw returning PPA/usage features

Primary diagnostic:
    Log Loss by season phase

Season phase:
    Based primarily on the minimum number of games played by either team
    before the current game:

        0 games
        1-2 games
        3-5 games
        6+ games

Secondary diagnostics:
    Home-team gamesBefore
    Away-team gamesBefore

Outputs
-------
models/win_probability/gradient_boosting/
    returning_production/
        model_1/
            season_phase_analysis/
                season_phase_comparison.csv
                season_phase_summary.csv
                season_phase_home.csv
                season_phase_away.csv
                season_phase_delta_log_loss.png

Run from project root:
    python src/modeling/problems/win_probability/gradient_boosting/
        returning_production/analysis/season_phase.py
"""

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import log_loss, brier_score_loss, roc_auc_score


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


# Experiment 1 predictions
EXPERIMENT_DIR = (
    PROJECT_ROOT
    / "models"
    / "win_probability"
    / "gradient_boosting"
    / "returning_production"
    / "model_1"
)

EXPERIMENT_PREDICTIONS = EXPERIMENT_DIR / "test_predictions.csv"


# Model 5 predictions
MODEL_5_DIR = (
    PROJECT_ROOT
    / "models"
    / "win_probability"
    / "gradient_boosting"
    / "model_5"
)

MODEL_5_PREDICTIONS = MODEL_5_DIR / "test_predictions.csv"


# 2025 final feature file containing gamesBefore information
FINAL_FEATURES = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "features"
    / "final"
    / "final_features_2025.csv"
)


# Analysis output directory
OUTPUT_DIR = EXPERIMENT_DIR / "season_phase_analysis"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


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


# =============================================================================
# GENERAL UTILITIES
# =============================================================================

def check_file_exists(path: Path, description: str) -> None:
    """Verify that a required file exists."""
    if not path.exists():
        raise FileNotFoundError(
            f"\nMissing {description}:\n"
            f"  {path}\n"
        )


def safe_probability(probabilities: pd.Series) -> np.ndarray:
    """
    Clip probabilities slightly away from 0 and 1 so that log loss remains
    numerically stable.
    """
    return np.clip(
        probabilities.astype(float).to_numpy(),
        EPSILON,
        1.0 - EPSILON,
    )


# =============================================================================
# CALIBRATION METRICS
# =============================================================================

def expected_calibration_error(
    y_true: np.ndarray,
    probabilities: np.ndarray,
    n_bins: int = 10,
) -> float:
    """
    Calculate Expected Calibration Error using equal-width probability bins.
    """
    y_true = np.asarray(y_true)
    probabilities = np.asarray(probabilities)

    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)

    ece = 0.0
    total = len(y_true)

    for i in range(n_bins):
        if i == n_bins - 1:
            mask = (
                (probabilities >= bin_edges[i])
                & (probabilities <= bin_edges[i + 1])
            )
        else:
            mask = (
                (probabilities >= bin_edges[i])
                & (probabilities < bin_edges[i + 1])
            )

        if not np.any(mask):
            continue

        bin_probabilities = probabilities[mask]
        bin_actuals = y_true[mask]

        mean_probability = np.mean(bin_probabilities)
        mean_actual = np.mean(bin_actuals)

        ece += (
            len(bin_actuals)
            / total
        ) * abs(mean_probability - mean_actual)

    return float(ece)


def maximum_calibration_error(
    y_true: np.ndarray,
    probabilities: np.ndarray,
    n_bins: int = 10,
) -> float:
    """
    Calculate Maximum Calibration Error using equal-width probability bins.
    """
    y_true = np.asarray(y_true)
    probabilities = np.asarray(probabilities)

    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)

    calibration_errors = []

    for i in range(n_bins):
        if i == n_bins - 1:
            mask = (
                (probabilities >= bin_edges[i])
                & (probabilities <= bin_edges[i + 1])
            )
        else:
            mask = (
                (probabilities >= bin_edges[i])
                & (probabilities < bin_edges[i + 1])
            )

        if not np.any(mask):
            continue

        mean_probability = np.mean(probabilities[mask])
        mean_actual = np.mean(y_true[mask])

        calibration_errors.append(
            abs(mean_probability - mean_actual)
        )

    if not calibration_errors:
        return np.nan

    return float(max(calibration_errors))


# =============================================================================
# PREDICTION LOADING
# =============================================================================

def load_experiment_predictions() -> pd.DataFrame:
    """
    Load and normalize Experiment 1 predictions.
    """
    check_file_exists(
        EXPERIMENT_PREDICTIONS,
        "Experiment 1 test predictions",
    )

    df = pd.read_csv(EXPERIMENT_PREDICTIONS)

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
            "Experiment 1 prediction file is missing required columns:\n"
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
            "predicted_probability_home_win": "experiment_probability",
            "predicted_home_win": "experiment_prediction",
        }
    )

    return df


def load_model_5_predictions() -> pd.DataFrame:
    """
    Load and normalize Model 5 predictions.
    """
    check_file_exists(
        MODEL_5_PREDICTIONS,
        "Gradient Boosting Model 5 test predictions",
    )

    df = pd.read_csv(MODEL_5_PREDICTIONS)

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
            "Model 5 prediction file is missing required columns:\n"
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
            "win_home_probability": "model_5_probability",
            "win_home_prediction": "model_5_prediction",
        }
    )

    return df


# =============================================================================
# FINAL FEATURE LOADING
# =============================================================================

def load_games_before_features() -> pd.DataFrame:
    """
    Load the 2025 final feature data and extract gamesBefore fields.
    """
    check_file_exists(
        FINAL_FEATURES,
        "2025 final feature data",
    )

    df = pd.read_csv(FINAL_FEATURES)

    required = {
        "gameId",
        "season",
        "gamesBefore_home",
        "gamesBefore_away",
    }

    missing = required - set(df.columns)

    if missing:
        raise ValueError(
            "Final feature file is missing required columns:\n"
            f"  {sorted(missing)}"
        )

    df = df[
        [
            "gameId",
            "season",
            "gamesBefore_home",
            "gamesBefore_away",
        ]
    ].copy()

    return df


# =============================================================================
# DATA ALIGNMENT
# =============================================================================

def validate_unique_ids(df: pd.DataFrame, name: str) -> None:
    """Verify that gameId is unique."""
    duplicates = df["gameId"].duplicated()

    if duplicates.any():
        duplicate_ids = (
            df.loc[duplicates, "gameId"]
            .astype(str)
            .tolist()
        )

        raise ValueError(
            f"{name} contains duplicate gameIds.\n"
            f"Examples: {duplicate_ids[:10]}"
        )


def align_datasets() -> pd.DataFrame:
    """
    Align Experiment 1, Model 5, and final feature data by gameId.
    """
    experiment = load_experiment_predictions()
    model_5 = load_model_5_predictions()
    games = load_games_before_features()

    validate_unique_ids(
        experiment,
        "Experiment 1 predictions",
    )

    validate_unique_ids(
        model_5,
        "Model 5 predictions",
    )

    validate_unique_ids(
        games,
        "Final feature data",
    )

    print("\n" + "=" * 80)
    print("DATA ALIGNMENT")
    print("=" * 80)

    print(f"Experiment 1 rows: {len(experiment):,}")
    print(f"Model 5 rows:      {len(model_5):,}")
    print(f"Final feature rows:{len(games):,}")

    # -------------------------------------------------------------------------
    # Verify seasons
    # -------------------------------------------------------------------------

    for name, df in [
        ("Experiment 1", experiment),
        ("Model 5", model_5),
        ("Final features", games),
    ]:
        seasons = sorted(df["season"].dropna().unique().tolist())

        if seasons != [SEASON]:
            raise ValueError(
                f"{name} contains unexpected seasons: {seasons}"
            )

    # -------------------------------------------------------------------------
    # Merge
    # -------------------------------------------------------------------------

    merged = experiment.merge(
        model_5,
        on=["gameId", "season"],
        how="inner",
        validate="one_to_one",
    )

    merged = merged.merge(
        games,
        on=["gameId", "season"],
        how="inner",
        validate="one_to_one",
    )

    print(f"\nAligned rows:       {len(merged):,}")

    expected_rows = len(experiment)

    if len(merged) != expected_rows:
        raise ValueError(
            "Not all Experiment 1 prediction rows were successfully aligned.\n"
            f"Expected: {expected_rows:,}\n"
            f"Found:    {len(merged):,}"
        )

    # -------------------------------------------------------------------------
    # Verify actual outcomes agree
    # -------------------------------------------------------------------------

    outcome_mismatch = (
        merged["actual"]
        != merged["actual_model_5"]
    )

    if outcome_mismatch.any():
        mismatch_count = int(outcome_mismatch.sum())

        raise ValueError(
            f"Actual outcomes disagree between Experiment 1 and Model 5 "
            f"for {mismatch_count} games."
        )

    merged["actual"] = merged["actual"].astype(int)

    # -------------------------------------------------------------------------
    # Verify gamesBefore values
    # -------------------------------------------------------------------------

    if merged[
        ["gamesBefore_home", "gamesBefore_away"]
    ].isna().any().any():
        raise ValueError(
            "Missing gamesBefore values found after alignment."
        )

    if (
        merged["gamesBefore_home"] < 0
    ).any() or (
        merged["gamesBefore_away"] < 0
    ).any():
        raise ValueError(
            "Negative gamesBefore values found."
        )

    # -------------------------------------------------------------------------
    # Create minimum gamesBefore measure
    # -------------------------------------------------------------------------

    merged["gamesBefore_min"] = merged[
        ["gamesBefore_home", "gamesBefore_away"]
    ].min(axis=1)

    return merged


# =============================================================================
# SEASON PHASE BUCKETS
# =============================================================================

def assign_phase(value: float) -> str:
    """
    Assign a gamesBefore value to a season-phase bucket.
    """
    if value == 0:
        return "0 games"

    if value <= 2:
        return "1-2 games"

    if value <= 5:
        return "3-5 games"

    return "6+ games"


def add_phase_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Create season-phase columns for minimum, home, and away gamesBefore."""
    df = df.copy()

    df["phase_min"] = df["gamesBefore_min"].apply(assign_phase)
    df["phase_home"] = df["gamesBefore_home"].apply(assign_phase)
    df["phase_away"] = df["gamesBefore_away"].apply(assign_phase)

    return df


# =============================================================================
# METRIC CALCULATION
# =============================================================================

def calculate_metrics(
    df: pd.DataFrame,
    phase_column: str,
    phase_name: str,
) -> dict:
    """
    Calculate Model 5 vs Experiment 1 metrics for one season-phase bucket.
    """
    y_true = df["actual"].astype(int).to_numpy()

    model_5_prob = safe_probability(
        df["model_5_probability"]
    )

    experiment_prob = safe_probability(
        df["experiment_probability"]
    )

    # -------------------------------------------------------------------------
    # Log Loss
    # -------------------------------------------------------------------------

    model_5_log_loss = log_loss(
        y_true,
        model_5_prob,
        labels=[0, 1],
    )

    experiment_log_loss = log_loss(
        y_true,
        experiment_prob,
        labels=[0, 1],
    )

    # -------------------------------------------------------------------------
    # Brier Score
    # -------------------------------------------------------------------------

    model_5_brier = brier_score_loss(
        y_true,
        model_5_prob,
    )

    experiment_brier = brier_score_loss(
        y_true,
        experiment_prob,
    )

    # -------------------------------------------------------------------------
    # ROC AUC
    # -------------------------------------------------------------------------

    if len(np.unique(y_true)) == 2:
        model_5_auc = roc_auc_score(
            y_true,
            model_5_prob,
        )

        experiment_auc = roc_auc_score(
            y_true,
            experiment_prob,
        )
    else:
        model_5_auc = np.nan
        experiment_auc = np.nan

    # -------------------------------------------------------------------------
    # Calibration
    # -------------------------------------------------------------------------

    model_5_ece = expected_calibration_error(
        y_true,
        model_5_prob,
    )

    experiment_ece = expected_calibration_error(
        y_true,
        experiment_prob,
    )

    model_5_mce = maximum_calibration_error(
        y_true,
        model_5_prob,
    )

    experiment_mce = maximum_calibration_error(
        y_true,
        experiment_prob,
    )

    # -------------------------------------------------------------------------
    # Return results
    # -------------------------------------------------------------------------

    return {
        "season_phase": phase_name,
        "games": len(df),
        "pct_test_games": len(df) / TOTAL_TEST_GAMES,

        "model_5_log_loss": model_5_log_loss,
        "experiment_1_log_loss": experiment_log_loss,
        "delta_log_loss": (
            experiment_log_loss
            - model_5_log_loss
        ),

        "model_5_brier": model_5_brier,
        "experiment_1_brier": experiment_brier,
        "delta_brier": (
            experiment_brier
            - model_5_brier
        ),

        "model_5_roc_auc": model_5_auc,
        "experiment_1_roc_auc": experiment_auc,
        "delta_roc_auc": (
            experiment_auc
            - model_5_auc
            if not np.isnan(model_5_auc)
            else np.nan
        ),

        "model_5_ece": model_5_ece,
        "experiment_1_ece": experiment_ece,
        "delta_ece": (
            experiment_ece
            - model_5_ece
        ),

        "model_5_mce": model_5_mce,
        "experiment_1_mce": experiment_mce,
        "delta_mce": (
            experiment_mce
            - model_5_mce
        ),
    }


def analyze_phase_dimension(
    df: pd.DataFrame,
    phase_column: str,
) -> pd.DataFrame:
    """
    Calculate metrics for each phase bucket.
    """
    results = []

    for phase in PHASE_ORDER:
        subset = df[df[phase_column] == phase].copy()

        if subset.empty:
            results.append(
                {
                    "season_phase": phase,
                    "games": 0,
                    "pct_test_games": 0.0,
                    "model_5_log_loss": np.nan,
                    "experiment_1_log_loss": np.nan,
                    "delta_log_loss": np.nan,
                    "model_5_brier": np.nan,
                    "experiment_1_brier": np.nan,
                    "delta_brier": np.nan,
                    "model_5_roc_auc": np.nan,
                    "experiment_1_roc_auc": np.nan,
                    "delta_roc_auc": np.nan,
                    "model_5_ece": np.nan,
                    "experiment_1_ece": np.nan,
                    "delta_ece": np.nan,
                    "model_5_mce": np.nan,
                    "experiment_1_mce": np.nan,
                    "delta_mce": np.nan,
                }
            )
            continue

        results.append(
            calculate_metrics(
                subset,
                phase_column,
                phase,
            )
        )

    result_df = pd.DataFrame(results)

    result_df["season_phase"] = pd.Categorical(
        result_df["season_phase"],
        categories=PHASE_ORDER,
        ordered=True,
    )

    result_df = result_df.sort_values(
        "season_phase"
    ).reset_index(drop=True)

    return result_df


# =============================================================================
# OVERALL COMPARISON
# =============================================================================

def calculate_overall_comparison(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Calculate overall Model 5 vs Experiment 1 metrics.
    """
    result = calculate_metrics(
        df,
        "phase_min",
        "Overall",
    )

    return pd.DataFrame([result])


# =============================================================================
# SUMMARY TABLE
# =============================================================================

def create_summary_table(
    min_results: pd.DataFrame,
) -> pd.DataFrame:
    """
    Create a compact primary summary focused on Log Loss and Brier Score.
    """
    summary = min_results[
        [
            "season_phase",
            "games",
            "pct_test_games",
            "model_5_log_loss",
            "experiment_1_log_loss",
            "delta_log_loss",
            "model_5_brier",
            "experiment_1_brier",
            "delta_brier",
        ]
    ].copy()

    summary["season_phase"] = (
        summary["season_phase"]
        .astype(str)
    )

    return summary


# =============================================================================
# PLOT
# =============================================================================

def create_delta_log_loss_plot(
    results: pd.DataFrame,
) -> Path:
    """
    Create a plot of Experiment 1 - Model 5 Log Loss by season phase.

    Interpretation:
        Negative = Experiment 1 improves Log Loss
        Positive = Experiment 1 worsens Log Loss
    """
    plot_path = (
        OUTPUT_DIR
        / "season_phase_delta_log_loss.png"
    )

    plot_df = results.dropna(
        subset=["delta_log_loss"]
    ).copy()

    x = np.arange(len(plot_df))
    y = plot_df["delta_log_loss"].to_numpy()

    plt.figure(figsize=(9, 6))

    plt.axhline(
        0,
        linewidth=1,
        linestyle="--",
    )

    plt.plot(
        x,
        y,
        marker="o",
        linewidth=2,
    )

    plt.xticks(
        x,
        plot_df["season_phase"],
    )

    plt.xlabel(
        "Minimum games played before current game"
    )

    plt.ylabel(
        "Δ Log Loss (Experiment 1 − Model 5)"
    )

    plt.title(
        "Returning Production Incremental Log Loss by Season Phase"
    )

    plt.grid(
        axis="y",
        alpha=0.3,
    )

    plt.tight_layout()

    plt.savefig(
        plot_path,
        dpi=200,
        bbox_inches="tight",
    )

    plt.close()

    return plot_path


# =============================================================================
# OUTPUT FORMATTING
# =============================================================================

def print_results(
    overall: pd.DataFrame,
    minimum: pd.DataFrame,
    home: pd.DataFrame,
    away: pd.DataFrame,
) -> None:
    """Print the most important results to the console."""

    print("\n" + "=" * 80)
    print("OVERALL COMPARISON")
    print("=" * 80)

    print(
        overall[
            [
                "games",
                "model_5_log_loss",
                "experiment_1_log_loss",
                "delta_log_loss",
                "model_5_brier",
                "experiment_1_brier",
                "delta_brier",
                "model_5_roc_auc",
                "experiment_1_roc_auc",
                "delta_roc_auc",
            ]
        ].to_string(
            index=False,
            float_format=lambda x: f"{x:.6f}",
        )
    )

    print("\n" + "=" * 80)
    print("PRIMARY ANALYSIS: MINIMUM GAMES BEFORE CURRENT GAME")
    print("=" * 80)

    print(
        create_summary_table(minimum).to_string(
            index=False,
            float_format=lambda x: f"{x:.6f}",
        )
    )

    print("\n" + "=" * 80)
    print("HOME TEAM GAMES BEFORE CURRENT GAME")
    print("=" * 80)

    print(
        home[
            [
                "season_phase",
                "games",
                "pct_test_games",
                "model_5_log_loss",
                "experiment_1_log_loss",
                "delta_log_loss",
                "model_5_brier",
                "experiment_1_brier",
                "delta_brier",
            ]
        ].to_string(
            index=False,
            float_format=lambda x: f"{x:.6f}",
        )
    )

    print("\n" + "=" * 80)
    print("AWAY TEAM GAMES BEFORE CURRENT GAME")
    print("=" * 80)

    print(
        away[
            [
                "season_phase",
                "games",
                "pct_test_games",
                "model_5_log_loss",
                "experiment_1_log_loss",
                "delta_log_loss",
                "model_5_brier",
                "experiment_1_brier",
                "delta_brier",
            ]
        ].to_string(
            index=False,
            float_format=lambda x: f"{x:.6f}",
        )
    )


# =============================================================================
# MAIN
# =============================================================================

def main() -> None:
    """Run the complete season-phase analysis."""

    global TOTAL_TEST_GAMES

    print("=" * 80)
    print("GRADIENT BOOSTING WIN PROBABILITY - SEASON PHASE ANALYSIS")
    print("RETURNING PRODUCTION EXPERIMENT 1 vs MODEL 5")
    print("=" * 80)

    print("\nProject root:")
    print(f"  {PROJECT_ROOT}")

    print("\nOutput directory:")
    print(f"  {OUTPUT_DIR}")

    # -------------------------------------------------------------------------
    # Load and align data
    # -------------------------------------------------------------------------

    df = align_datasets()

    TOTAL_TEST_GAMES = len(df)

    # -------------------------------------------------------------------------
    # Add phase buckets
    # -------------------------------------------------------------------------

    df = add_phase_columns(df)

    # -------------------------------------------------------------------------
    # Print phase counts
    # -------------------------------------------------------------------------

    print("\n" + "=" * 80)
    print("SEASON PHASE DISTRIBUTION")
    print("=" * 80)

    print("\nMinimum gamesBefore:")
    print(
        df["phase_min"]
        .value_counts()
        .reindex(PHASE_ORDER, fill_value=0)
        .to_string()
    )

    print("\nHome gamesBefore:")
    print(
        df["phase_home"]
        .value_counts()
        .reindex(PHASE_ORDER, fill_value=0)
        .to_string()
    )

    print("\nAway gamesBefore:")
    print(
        df["phase_away"]
        .value_counts()
        .reindex(PHASE_ORDER, fill_value=0)
        .to_string()
    )

    # -------------------------------------------------------------------------
    # Calculate analyses
    # -------------------------------------------------------------------------

    overall = calculate_overall_comparison(df)

    minimum_results = analyze_phase_dimension(
        df,
        "phase_min",
    )

    home_results = analyze_phase_dimension(
        df,
        "phase_home",
    )

    away_results = analyze_phase_dimension(
        df,
        "phase_away",
    )

    # -------------------------------------------------------------------------
    # Save CSVs
    # -------------------------------------------------------------------------

    overall_path = (
        OUTPUT_DIR
        / "overall_comparison.csv"
    )

    comparison_path = (
        OUTPUT_DIR
        / "season_phase_comparison.csv"
    )

    summary_path = (
        OUTPUT_DIR
        / "season_phase_summary.csv"
    )

    home_path = (
        OUTPUT_DIR
        / "season_phase_home.csv"
    )

    away_path = (
        OUTPUT_DIR
        / "season_phase_away.csv"
    )

    overall.to_csv(
        overall_path,
        index=False,
    )

    minimum_results.to_csv(
        comparison_path,
        index=False,
    )

    create_summary_table(
        minimum_results
    ).to_csv(
        summary_path,
        index=False,
    )

    home_results.to_csv(
        home_path,
        index=False,
    )

    away_results.to_csv(
        away_path,
        index=False,
    )

    # -------------------------------------------------------------------------
    # Create plot
    # -------------------------------------------------------------------------

    plot_path = create_delta_log_loss_plot(
        minimum_results
    )

    # -------------------------------------------------------------------------
    # Print results
    # -------------------------------------------------------------------------

    print_results(
        overall,
        minimum_results,
        home_results,
        away_results,
    )

    # -------------------------------------------------------------------------
    # Final output summary
    # -------------------------------------------------------------------------

    print("\n" + "=" * 80)
    print("OUTPUTS SAVED")
    print("=" * 80)

    print(f"\n  {overall_path}")
    print(f"  {comparison_path}")
    print(f"  {summary_path}")
    print(f"  {home_path}")
    print(f"  {away_path}")
    print(f"  {plot_path}")

    print("\n" + "=" * 80)
    print("INTERPRETATION")
    print("=" * 80)

    print(
        "\nΔ Log Loss = Experiment 1 Log Loss − Model 5 Log Loss."
    )

    print(
        "  Negative Δ Log Loss -> returning production improves performance."
    )

    print(
        "  Positive Δ Log Loss -> returning production worsens performance."
    )

    print(
        "\nThis analysis is diagnostic only. Do not use the 2025 results "
        "to directly tune Experiment 2."
    )

    print("\nAnalysis complete.")


if __name__ == "__main__":
    main()