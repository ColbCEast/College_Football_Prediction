"""
GRADIENT BOOSTING WIN PROBABILITY
EXPERIMENT 2 vs EXPERIMENT 3
RELATIVE RETURNING PRODUCTION FEATURE-LEVEL DIAGNOSTIC

Purpose
-------
Compare Experiment 2 (unweighted relative returning production) with
Experiment 3 (phase-weighted relative returning production) to isolate
the effect of phase weighting.

Model definitions
-----------------
Model 5:
    310-feature baseline

Experiment 2:
    Model 5 + 8 unweighted relative returning-production features

Experiment 3:
    Model 5 + 8 phase-weighted relative returning-production features

This script is diagnostic only.
It does NOT tune phase weights or modify any model.

Outputs
-------
models/win_probability/gradient_boosting/returning_production/model_3/analysis/

    relative_vs_phase_importance.csv
    relative_vs_phase_magnitude.csv
    relative_vs_phase_prediction_shift.csv
    relative_vs_phase_prediction_shift_by_phase.csv
    relative_vs_phase_prediction_shift_by_outcome.csv
    relative_vs_phase_performance.csv
    relative_vs_phase_summary.csv
"""

from pathlib import Path
import warnings

import numpy as np
import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    brier_score_loss,
    log_loss,
    roc_auc_score,
)

warnings.filterwarnings("ignore")


# =============================================================================
# PATHS
# =============================================================================

ROOT = Path(__file__).resolve().parents[7]

MODEL_INPUT_DIR = (
    ROOT
    / "data"
    / "processed"
    / "model_inputs"
    / "win_probability"
)

RETURNING_FEATURE_DIR = (
    ROOT
    / "data"
    / "processed"
    / "features"
    / "win_probability"
    / "returning_production"
)

MODEL_2_DIR = (
    ROOT
    / "models"
    / "win_probability"
    / "gradient_boosting"
    / "returning_production"
    / "model_2"
)

MODEL_3_DIR = (
    ROOT
    / "models"
    / "win_probability"
    / "gradient_boosting"
    / "returning_production"
    / "model_3"
)

MODEL_5_DIR = (
    ROOT
    / "models"
    / "win_probability"
    / "gradient_boosting"
    / "model_5"
)

OUTPUT_DIR = MODEL_3_DIR / "analysis"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# =============================================================================
# CONFIGURATION
# =============================================================================

PHASE_ORDER = ["Early", "Mid", "Late"]

PHASE_WEIGHTS = {
    "Early": 1.00,
    "Mid": 0.67,
    "Late": 0.33,
}

RELATIVE_FEATURES = [
    "returning_total_ppa_diff",
    "returning_passing_ppa_diff",
    "returning_receiving_ppa_diff",
    "returning_rushing_ppa_diff",
    "returning_usage_diff",
    "returning_passing_usage_diff",
    "returning_receiving_usage_diff",
    "returning_rushing_usage_diff",
]

PHASE_FEATURES = [
    "returning_total_ppa_diff_phase",
    "returning_passing_ppa_diff_phase",
    "returning_receiving_ppa_diff_phase",
    "returning_rushing_ppa_diff_phase",
    "returning_usage_diff_phase",
    "returning_passing_usage_diff_phase",
    "returning_receiving_usage_diff_phase",
    "returning_rushing_usage_diff_phase",
]

FEATURE_LABELS = {
    "returning_total_ppa_diff": "Total PPA",
    "returning_passing_ppa_diff": "Passing PPA",
    "returning_receiving_ppa_diff": "Receiving PPA",
    "returning_rushing_ppa_diff": "Rushing PPA",
    "returning_usage_diff": "Total Usage",
    "returning_passing_usage_diff": "Passing Usage",
    "returning_receiving_usage_diff": "Receiving Usage",
    "returning_rushing_usage_diff": "Rushing Usage",
}


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def print_header(title):
    """Print a formatted section header."""
    print()
    print("=" * 88)
    print(title)
    print("=" * 88)


def load_csv(path, description):
    """Load a CSV and fail with a useful message if unavailable."""
    if not path.exists():
        raise FileNotFoundError(
            f"{description} not found:\n{path}"
        )

    df = pd.read_csv(path)

    print(f"Loaded {description}: {len(df):,} rows")
    return df


def find_prediction_probability_column(df, candidates, description):
    """Find the first available probability column."""
    for column in candidates:
        if column in df.columns:
            return column

    raise ValueError(
        f"Could not find probability column for {description}.\n"
        f"Expected one of: {candidates}\n"
        f"Available columns: {list(df.columns)}"
    )


def find_prediction_class_column(df, candidates, description):
    """Find the first available predicted-class column."""
    for column in candidates:
        if column in df.columns:
            return column

    raise ValueError(
        f"Could not find prediction column for {description}.\n"
        f"Expected one of: {candidates}\n"
        f"Available columns: {list(df.columns)}"
    )


def calculate_ece(y_true, probabilities, n_bins=10):
    """Calculate Expected Calibration Error."""
    y_true = np.asarray(y_true)
    probabilities = np.asarray(probabilities)

    bins = np.linspace(0.0, 1.0, n_bins + 1)

    ece = 0.0

    for i in range(n_bins):
        if i == n_bins - 1:
            mask = (
                (probabilities >= bins[i])
                & (probabilities <= bins[i + 1])
            )
        else:
            mask = (
                (probabilities >= bins[i])
                & (probabilities < bins[i + 1])
            )

        if not np.any(mask):
            continue

        bin_accuracy = np.mean(y_true[mask])
        bin_confidence = np.mean(probabilities[mask])
        bin_weight = np.mean(mask)

        ece += bin_weight * abs(bin_accuracy - bin_confidence)

    return ece


def calculate_mce(y_true, probabilities, n_bins=10):
    """Calculate Maximum Calibration Error."""
    y_true = np.asarray(y_true)
    probabilities = np.asarray(probabilities)

    bins = np.linspace(0.0, 1.0, n_bins + 1)

    errors = []

    for i in range(n_bins):
        if i == n_bins - 1:
            mask = (
                (probabilities >= bins[i])
                & (probabilities <= bins[i + 1])
            )
        else:
            mask = (
                (probabilities >= bins[i])
                & (probabilities < bins[i + 1])
            )

        if not np.any(mask):
            continue

        bin_accuracy = np.mean(y_true[mask])
        bin_confidence = np.mean(probabilities[mask])

        errors.append(abs(bin_accuracy - bin_confidence))

    return max(errors) if errors else np.nan


def calculate_metrics(y_true, probabilities):
    """Calculate the standard model evaluation metrics."""
    probabilities = np.clip(
        np.asarray(probabilities),
        1e-15,
        1 - 1e-15,
    )

    y_true = np.asarray(y_true)
    predictions = (probabilities >= 0.5).astype(int)

    metrics = {
        "N": len(y_true),
        "Log Loss": log_loss(y_true, probabilities),
        "Brier": brier_score_loss(y_true, probabilities),
        "ROC AUC": roc_auc_score(y_true, probabilities),
        "Accuracy": accuracy_score(y_true, predictions),
        "ECE": calculate_ece(y_true, probabilities),
        "MCE": calculate_mce(y_true, probabilities),
    }

    return metrics


def get_phase(games_before_min):
    """Assign season phase based on gamesBefore_min."""
    if games_before_min <= 2:
        return "Early"
    elif games_before_min <= 5:
        return "Mid"
    else:
        return "Late"


def standardized_mean_difference(values, outcomes):
    """
    Calculate standardized difference between home wins and home losses.

    Effect:
        mean(feature | home win) - mean(feature | home loss)
        divided by pooled standard deviation.
    """
    values = np.asarray(values, dtype=float)
    outcomes = np.asarray(outcomes)

    win_values = values[outcomes == 1]
    loss_values = values[outcomes == 0]

    if len(win_values) < 2 or len(loss_values) < 2:
        return np.nan

    win_std = np.std(win_values, ddof=1)
    loss_std = np.std(loss_values, ddof=1)

    pooled_variance = (
        ((len(win_values) - 1) * win_std ** 2)
        + ((len(loss_values) - 1) * loss_std ** 2)
    ) / (
        len(win_values) + len(loss_values) - 2
    )

    pooled_std = np.sqrt(pooled_variance)

    if pooled_std == 0:
        return np.nan

    return (
        np.mean(win_values) - np.mean(loss_values)
    ) / pooled_std


# =============================================================================
# LOAD FEATURE IMPORTANCE
# =============================================================================

print_header("GRADIENT BOOSTING WIN PROBABILITY")
print("EXPERIMENT 2 vs EXPERIMENT 3 FEATURE-LEVEL DIAGNOSTIC")
print("=" * 88)

print(f"Project root: {ROOT}")
print(f"Output directory: {OUTPUT_DIR}")

print_header("1. LOAD FEATURE IMPORTANCE")

model_2_importance = load_csv(
    MODEL_2_DIR / "feature_importance.csv",
    "Experiment 2 feature importance",
)

model_3_importance = load_csv(
    MODEL_3_DIR / "feature_importance.csv",
    "Experiment 3 feature importance",
)

required_importance_columns = {"feature", "importance"}

for name, df in [
    ("Experiment 2", model_2_importance),
    ("Experiment 3", model_3_importance),
]:
    missing = required_importance_columns - set(df.columns)

    if missing:
        raise ValueError(
            f"{name} feature importance is missing columns: {missing}"
        )


# =============================================================================
# FEATURE IMPORTANCE COMPARISON
# =============================================================================

print_header("2. RELATIVE VS PHASE-WEIGHTED FEATURE IMPORTANCE")

importance_2 = (
    model_2_importance
    .set_index("feature")["importance"]
)

importance_3 = (
    model_3_importance
    .set_index("feature")["importance"]
)

importance_rows = []

for relative_feature, phase_feature in zip(
    RELATIVE_FEATURES,
    PHASE_FEATURES,
):
    if relative_feature not in importance_2.index:
        raise ValueError(
            f"Experiment 2 is missing expected feature: "
            f"{relative_feature}"
        )

    if phase_feature not in importance_3.index:
        raise ValueError(
            f"Experiment 3 is missing expected feature: "
            f"{phase_feature}"
        )

    imp_2 = float(importance_2[relative_feature])
    imp_3 = float(importance_3[phase_feature])

    rank_2 = int(
        model_2_importance.index[
            model_2_importance["feature"] == relative_feature
        ][0]
    ) + 1

    rank_3 = int(
        model_3_importance.index[
            model_3_importance["feature"] == phase_feature
        ][0]
    ) + 1

    importance_rows.append(
        {
            "feature_group": FEATURE_LABELS[relative_feature],
            "experiment_2_feature": relative_feature,
            "experiment_3_feature": phase_feature,
            "experiment_2_importance": imp_2,
            "experiment_3_importance": imp_3,
            "importance_change": imp_3 - imp_2,
            "importance_change_percent": (
                ((imp_3 - imp_2) / imp_2) * 100
                if imp_2 != 0
                else np.nan
            ),
            "experiment_2_rank": rank_2,
            "experiment_3_rank": rank_3,
            "rank_change": rank_3 - rank_2,
        }
    )

importance_comparison = pd.DataFrame(importance_rows)

total_importance_2 = importance_comparison[
    "experiment_2_importance"
].sum()

total_importance_3 = importance_comparison[
    "experiment_3_importance"
].sum()

importance_comparison["experiment_2_share_of_model"] = (
    importance_comparison["experiment_2_importance"]
)

importance_comparison["experiment_3_share_of_model"] = (
    importance_comparison["experiment_3_importance"]
)

importance_comparison.to_csv(
    OUTPUT_DIR / "relative_vs_phase_importance.csv",
    index=False,
)

print()
print(
    f"Experiment 2 returning-feature importance: "
    f"{total_importance_2:.6f} "
    f"({total_importance_2 * 100:.4f}%)"
)

print(
    f"Experiment 3 returning-feature importance: "
    f"{total_importance_3:.6f} "
    f"({total_importance_3 * 100:.4f}%)"
)

print()
print(
    importance_comparison[
        [
            "feature_group",
            "experiment_2_importance",
            "experiment_3_importance",
            "importance_change",
            "experiment_2_rank",
            "experiment_3_rank",
            "rank_change",
        ]
    ].to_string(index=False)
)


# =============================================================================
# LOAD GAME-LEVEL RETURNING FEATURES
# =============================================================================

print_header("3. LOAD RETURNING-PRODUCTION FEATURES")

split_files = {
    "train": MODEL_INPUT_DIR / "train.csv",
    "validation": MODEL_INPUT_DIR / "validation.csv",
    "test": MODEL_INPUT_DIR / "test.csv",
}

split_data = {}

for split, path in split_files.items():
    split_data[split] = load_csv(
        path,
        f"{split.title()} model inputs",
    )

    required_columns = {
        "gameId",
        "season",
        "win_home",
        "gamesBefore_home",
        "gamesBefore_away",
    }

    missing = required_columns - set(split_data[split].columns)

    if missing:
        raise ValueError(
            f"{split.title()} model inputs are missing: {missing}"
        )


def load_returning_for_split(split_df, split_name):
    """Load and combine returning features for all seasons in a split."""
    frames = []

    for season in sorted(split_df["season"].unique()):
        season = int(season)

        path = RETURNING_FEATURE_DIR / (
            f"returning_features_{season}.csv"
        )

        season_features = load_csv(
            path,
            f"{split_name.title()} returning features {season}",
        )

        if "gameId" not in season_features.columns:
            raise ValueError(
                f"{path} does not contain gameId."
            )

        season_features = season_features.copy()

        if season_features["gameId"].duplicated().any():
            raise ValueError(
                f"Duplicate gameId values found in "
                f"{path}."
            )

        frames.append(season_features)

    if not frames:
        raise ValueError(
            f"No returning feature files loaded for {split_name}."
        )

    combined = pd.concat(
        frames,
        ignore_index=True,
    )

    if combined["gameId"].duplicated().any():
        raise ValueError(
            f"Duplicate gameId values found after combining "
            f"{split_name} returning features."
        )

    return combined


returning_by_split = {}

for split in split_files:
    returning_by_split[split] = load_returning_for_split(
        split_data[split],
        split,
    )


# =============================================================================
# RECONSTRUCT RELATIVE FEATURES AND PHASE
# =============================================================================

print_header("4. RECONSTRUCT RELATIVE FEATURES AND PHASE")

all_feature_rows = []

for split in ["train", "validation", "test"]:

    base = split_data[split].copy()
    returning = returning_by_split[split].copy()

    merged = base.merge(
        returning,
        on="gameId",
        how="left",
        suffixes=("", "_returning"),
        validate="one_to_one",
    )

    if len(merged) != len(base):
        raise ValueError(
            f"{split.title()} merge changed row count: "
            f"{len(base):,} -> {len(merged):,}"
        )

    if merged["gameId"].isna().any():
        raise ValueError(
            f"Missing gameId values after {split} merge."
        )

    merged["gamesBefore_min"] = merged[
        ["gamesBefore_home", "gamesBefore_away"]
    ].min(axis=1)

    merged["season_phase"] = merged[
        "gamesBefore_min"
    ].apply(get_phase)

    merged["phase_weight"] = merged[
        "season_phase"
    ].map(PHASE_WEIGHTS)

    # Required home/away returning fields.
    required_returning = [
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

    missing = [
        column
        for column in required_returning
        if column not in merged.columns
    ]

    if missing:
        raise ValueError(
            f"{split.title()} returning data missing expected "
            f"columns: {missing}"
        )

    # Reconstruct the eight relative features.
    merged["returning_total_ppa_diff"] = (
        merged["home_returning_total_ppa"]
        - merged["away_returning_total_ppa"]
    )

    merged["returning_passing_ppa_diff"] = (
        merged["home_returning_passing_ppa"]
        - merged["away_returning_passing_ppa"]
    )

    merged["returning_receiving_ppa_diff"] = (
        merged["home_returning_receiving_ppa"]
        - merged["away_returning_receiving_ppa"]
    )

    merged["returning_rushing_ppa_diff"] = (
        merged["home_returning_rushing_ppa"]
        - merged["away_returning_rushing_ppa"]
    )

    merged["returning_usage_diff"] = (
        merged["home_returning_usage"]
        - merged["away_returning_usage"]
    )

    merged["returning_passing_usage_diff"] = (
        merged["home_returning_passing_usage"]
        - merged["away_returning_passing_usage"]
    )

    merged["returning_receiving_usage_diff"] = (
        merged["home_returning_receiving_usage"]
        - merged["away_returning_receiving_usage"]
    )

    merged["returning_rushing_usage_diff"] = (
        merged["home_returning_rushing_usage"]
        - merged["away_returning_rushing_usage"]
    )

    # Construct the eight phase-aware features.
    for relative_feature, phase_feature in zip(
        RELATIVE_FEATURES,
        PHASE_FEATURES,
    ):
        merged[phase_feature] = (
            merged[relative_feature]
            * merged["phase_weight"]
        )

    # Validate phase weighting exactly.
    for relative_feature, phase_feature in zip(
        RELATIVE_FEATURES,
        PHASE_FEATURES,
    ):
        expected = (
            merged[relative_feature]
            * merged["phase_weight"]
        )

        actual = merged[phase_feature]

        valid = (
            np.isclose(
                actual,
                expected,
                equal_nan=True,
                atol=1e-10,
                rtol=1e-10,
            )
        )

        if not valid.all():
            raise ValueError(
                f"Phase arithmetic failed for {phase_feature} "
                f"in {split}."
            )

    print(
        f"{split.title():12s}: "
        f"{len(merged):,} rows | "
        f"Early={sum(merged['season_phase'] == 'Early'):,} | "
        f"Mid={sum(merged['season_phase'] == 'Mid'):,} | "
        f"Late={sum(merged['season_phase'] == 'Late'):,}"
    )

    all_feature_rows.append(merged)


feature_data = pd.concat(
    all_feature_rows,
    ignore_index=True,
)


# =============================================================================
# FEATURE MAGNITUDE ANALYSIS
# =============================================================================

print_header("5. FEATURE MAGNITUDE: EXPERIMENT 2 VS EXPERIMENT 3")

magnitude_rows = []

for split in ["train", "validation", "test"]:

    split_df = feature_data[
        (
            feature_data["season"].isin(
                split_data[split]["season"].unique()
            )
        )
    ].copy()

    # The season sets are non-overlapping in this project, but retain an
    # explicit split label rather than relying on season alone in outputs.
    split_df["split"] = split

    for phase in PHASE_ORDER:

        phase_df = split_df[
            split_df["season_phase"] == phase
        ]

        for relative_feature, phase_feature in zip(
            RELATIVE_FEATURES,
            PHASE_FEATURES,
        ):

            raw = phase_df[relative_feature]
            weighted = phase_df[phase_feature]

            magnitude_rows.append(
                {
                    "split": split,
                    "season_phase": phase,
                    "feature_group": FEATURE_LABELS[
                        relative_feature
                    ],
                    "relative_feature": relative_feature,
                    "phase_feature": phase_feature,
                    "N": len(phase_df),
                    "phase_weight": PHASE_WEIGHTS[phase],
                    "raw_mean": raw.mean(),
                    "raw_mean_abs": raw.abs().mean(),
                    "raw_median_abs": raw.abs().median(),
                    "raw_std": raw.std(),
                    "phase_mean": weighted.mean(),
                    "phase_mean_abs": weighted.abs().mean(),
                    "phase_median_abs": weighted.abs().median(),
                    "phase_std": weighted.std(),
                    "mean_abs_attenuation": (
                        weighted.abs().mean()
                        / raw.abs().mean()
                        if raw.abs().mean() != 0
                        else np.nan
                    ),
                }
            )

magnitude_comparison = pd.DataFrame(magnitude_rows)

magnitude_comparison.to_csv(
    OUTPUT_DIR / "relative_vs_phase_magnitude.csv",
    index=False,
)

print(
    magnitude_comparison[
        [
            "split",
            "season_phase",
            "feature_group",
            "N",
            "phase_weight",
            "raw_mean_abs",
            "phase_mean_abs",
            "mean_abs_attenuation",
        ]
    ].to_string(index=False)
)


# =============================================================================
# OUTCOME SEPARATION
# =============================================================================

print_header("6. OUTCOME SEPARATION BY PHASE")

outcome_rows = []

for split in ["train", "validation", "test"]:

    split_df = feature_data[
        feature_data["season"].isin(
            split_data[split]["season"].unique()
        )
    ].copy()

    for phase in PHASE_ORDER:

        phase_df = split_df[
            split_df["season_phase"] == phase
        ]

        for relative_feature, phase_feature in zip(
            RELATIVE_FEATURES,
            PHASE_FEATURES,
        ):

            raw_effect = standardized_mean_difference(
                phase_df[relative_feature],
                phase_df["win_home"],
            )

            phase_effect = standardized_mean_difference(
                phase_df[phase_feature],
                phase_df["win_home"],
            )

            outcome_rows.append(
                {
                    "split": split,
                    "season_phase": phase,
                    "feature_group": FEATURE_LABELS[
                        relative_feature
                    ],
                    "relative_feature": relative_feature,
                    "phase_feature": phase_feature,
                    "N": len(phase_df),
                    "raw_standardized_difference": raw_effect,
                    "phase_standardized_difference": phase_effect,
                    "effect_change": phase_effect - raw_effect,
                }
            )

outcome_separation = pd.DataFrame(outcome_rows)

outcome_separation.to_csv(
    OUTPUT_DIR / "relative_vs_phase_outcome_separation.csv",
    index=False,
)

print(
    outcome_separation[
        [
            "split",
            "season_phase",
            "feature_group",
            "raw_standardized_difference",
            "phase_standardized_difference",
            "effect_change",
        ]
    ].to_string(index=False)
)


# =============================================================================
# LOAD PREDICTIONS
# =============================================================================

print_header("7. LOAD MODEL PREDICTIONS")

model_2_val = load_csv(
    MODEL_2_DIR / "validation_predictions.csv",
    "Experiment 2 validation predictions",
)

model_2_test = load_csv(
    MODEL_2_DIR / "test_predictions.csv",
    "Experiment 2 test predictions",
)

model_3_val = load_csv(
    MODEL_3_DIR / "validation_predictions.csv",
    "Experiment 3 validation predictions",
)

model_3_test = load_csv(
    MODEL_3_DIR / "test_predictions.csv",
    "Experiment 3 test predictions",
)

model_5_val = load_csv(
    MODEL_5_DIR / "validation_predictions.csv",
    "Model 5 validation predictions",
)

model_5_test = load_csv(
    MODEL_5_DIR / "test_predictions.csv",
    "Model 5 test predictions",
)


def standardize_predictions(
    df,
    model_name,
    probability_candidates,
    prediction_candidates,
):
    """Standardize prediction-file schema."""
    probability_column = find_prediction_probability_column(
        df,
        probability_candidates,
        model_name,
    )

    prediction_column = find_prediction_class_column(
        df,
        prediction_candidates,
        model_name,
    )

    required = {"gameId", "season"}

    missing = required - set(df.columns)

    if missing:
        raise ValueError(
            f"{model_name} missing required columns: {missing}"
        )

    result = df[
        ["gameId", "season", probability_column, prediction_column]
    ].copy()

    result = result.rename(
        columns={
            probability_column: "probability",
            prediction_column: "prediction",
        }
    )

    result["model"] = model_name

    if result["gameId"].duplicated().any():
        raise ValueError(
            f"{model_name} contains duplicate gameId values."
        )

    return result


# Experiment 2
exp2_val = standardize_predictions(
    model_2_val,
    "Experiment 2",
    [
        "predicted_probability_home_win",
        "predicted_probability",
    ],
    [
        "predicted_home_win",
        "predicted_class",
    ],
)

exp2_test = standardize_predictions(
    model_2_test,
    "Experiment 2",
    [
        "predicted_probability_home_win",
        "predicted_probability",
    ],
    [
        "predicted_home_win",
        "predicted_class",
    ],
)

# Experiment 3
exp3_val = standardize_predictions(
    model_3_val,
    "Experiment 3",
    [
        "predicted_probability",
        "predicted_probability_home_win",
    ],
    [
        "predicted_class",
        "predicted_home_win",
    ],
)

exp3_test = standardize_predictions(
    model_3_test,
    "Experiment 3",
    [
        "predicted_probability",
        "predicted_probability_home_win",
    ],
    [
        "predicted_class",
        "predicted_home_win",
    ],
)

# Model 5
model5_val = standardize_predictions(
    model_5_val,
    "Model 5",
    [
        "win_home_probability",
        "predicted_probability",
    ],
    [
        "win_home_prediction",
        "predicted_class",
    ],
)

model5_test = standardize_predictions(
    model_5_test,
    "Model 5",
    [
        "win_home_probability",
        "predicted_probability",
    ],
    [
        "win_home_prediction",
        "predicted_class",
    ],
)


# =============================================================================
# ALIGN PREDICTIONS
# =============================================================================

print_header("8. ALIGN EXPERIMENT 2 AND EXPERIMENT 3 PREDICTIONS")


def align_predictions(exp2, exp3, split):
    """Align Exp2 and Exp3 predictions on gameId."""
    if set(exp2["gameId"]) != set(exp3["gameId"]):
        missing_exp2 = set(exp3["gameId"]) - set(exp2["gameId"])
        missing_exp3 = set(exp2["gameId"]) - set(exp3["gameId"])

        raise ValueError(
            f"{split.title()} gameId mismatch.\n"
            f"Missing from Exp2: {len(missing_exp2)}\n"
            f"Missing from Exp3: {len(missing_exp3)}"
        )

    merged = exp2.merge(
        exp3,
        on=["gameId", "season"],
        how="inner",
        suffixes=("_exp2", "_exp3"),
        validate="one_to_one",
    )

    if len(merged) != len(exp2):
        raise ValueError(
            f"{split.title()} alignment changed row count: "
            f"{len(exp2):,} -> {len(merged):,}"
        )

    return merged


val_predictions = align_predictions(
    exp2_val,
    exp3_val,
    "validation",
)

test_predictions = align_predictions(
    exp2_test,
    exp3_test,
    "test",
)

print(
    f"Validation aligned: {len(val_predictions):,} games"
)

print(
    f"Test aligned:       {len(test_predictions):,} games"
)


# =============================================================================
# ATTACH PHASE AND OUTCOME
# =============================================================================

def attach_phase_and_outcome(predictions, feature_df):
    """Attach gamesBefore phase and actual outcome."""
    metadata = feature_df[
        [
            "gameId",
            "season",
            "win_home",
            "gamesBefore_min",
            "season_phase",
            "phase_weight",
        ]
    ].drop_duplicates(
        subset=["gameId"]
    )

    merged = predictions.merge(
        metadata,
        on=["gameId", "season"],
        how="left",
        validate="one_to_one",
    )

    if merged["season_phase"].isna().any():
        raise ValueError(
            "Missing phase after attaching feature metadata."
        )

    return merged


val_predictions = attach_phase_and_outcome(
    val_predictions,
    feature_data[
        feature_data["season"].isin(
            split_data["validation"]["season"].unique()
        )
    ],
)

test_predictions = attach_phase_and_outcome(
    test_predictions,
    feature_data[
        feature_data["season"].isin(
            split_data["test"]["season"].unique()
        )
    ],
)


# =============================================================================
# PREDICTION SHIFT: OVERALL
# =============================================================================

print_header("9. PREDICTION SHIFT: EXPERIMENT 2 -> EXPERIMENT 3")


def prediction_shift_summary(predictions, split):
    """Calculate overall probability-shift diagnostics."""
    shift = (
        predictions["probability_exp3"]
        - predictions["probability_exp2"]
    )

    abs_shift = shift.abs()

    result = {
        "split": split,
        "N": len(predictions),
        "mean_probability_exp2": (
            predictions["probability_exp2"].mean()
        ),
        "mean_probability_exp3": (
            predictions["probability_exp3"].mean()
        ),
        "mean_probability_shift": shift.mean(),
        "mean_absolute_probability_shift": abs_shift.mean(),
        "median_absolute_probability_shift": abs_shift.median(),
        "std_probability_shift": shift.std(),
        "min_probability_shift": shift.min(),
        "max_probability_shift": shift.max(),
        "p05_probability_shift": shift.quantile(0.05),
        "p95_probability_shift": shift.quantile(0.95),
    }

    return result


prediction_shift_rows = [
    prediction_shift_summary(
        val_predictions,
        "validation",
    ),
    prediction_shift_summary(
        test_predictions,
        "test",
    ),
]

prediction_shift = pd.DataFrame(
    prediction_shift_rows
)

prediction_shift.to_csv(
    OUTPUT_DIR / "relative_vs_phase_prediction_shift.csv",
    index=False,
)

print(
    prediction_shift.to_string(index=False)
)


# =============================================================================
# PREDICTION SHIFT BY PHASE
# =============================================================================

print_header("10. PREDICTION SHIFT BY PHASE")

phase_shift_rows = []

for split, predictions in [
    ("validation", val_predictions),
    ("test", test_predictions),
]:

    for phase in PHASE_ORDER:

        subset = predictions[
            predictions["season_phase"] == phase
        ]

        shift = (
            subset["probability_exp3"]
            - subset["probability_exp2"]
        )

        phase_shift_rows.append(
            {
                "split": split,
                "season_phase": phase,
                "N": len(subset),
                "phase_weight": PHASE_WEIGHTS[phase],
                "mean_probability_exp2": (
                    subset["probability_exp2"].mean()
                ),
                "mean_probability_exp3": (
                    subset["probability_exp3"].mean()
                ),
                "mean_probability_shift": shift.mean(),
                "mean_absolute_probability_shift": shift.abs().mean(),
                "median_absolute_probability_shift": shift.abs().median(),
                "std_probability_shift": shift.std(),
                "min_probability_shift": shift.min(),
                "max_probability_shift": shift.max(),
                "p05_probability_shift": shift.quantile(0.05),
                "p95_probability_shift": shift.quantile(0.95),
            }
        )

phase_prediction_shift = pd.DataFrame(
    phase_shift_rows
)

phase_prediction_shift.to_csv(
    OUTPUT_DIR / "relative_vs_phase_prediction_shift_by_phase.csv",
    index=False,
)

print(
    phase_prediction_shift.to_string(index=False)
)


# =============================================================================
# PREDICTION SHIFT BY OUTCOME
# =============================================================================

print_header("11. PREDICTION SHIFT BY OUTCOME")

outcome_shift_rows = []

for split, predictions in [
    ("validation", val_predictions),
    ("test", test_predictions),
]:

    for outcome, outcome_label in [
        (1, "Home Win"),
        (0, "Home Loss"),
    ]:

        subset = predictions[
            predictions["win_home"] == outcome
        ]

        shift = (
            subset["probability_exp3"]
            - subset["probability_exp2"]
        )

        outcome_shift_rows.append(
            {
                "split": split,
                "outcome": outcome_label,
                "N": len(subset),
                "mean_probability_exp2": (
                    subset["probability_exp2"].mean()
                ),
                "mean_probability_exp3": (
                    subset["probability_exp3"].mean()
                ),
                "mean_probability_shift": shift.mean(),
                "mean_absolute_probability_shift": shift.abs().mean(),
                "median_absolute_probability_shift": shift.abs().median(),
                "std_probability_shift": shift.std(),
            }
        )

outcome_prediction_shift = pd.DataFrame(
    outcome_shift_rows
)

outcome_prediction_shift.to_csv(
    OUTPUT_DIR / "relative_vs_phase_prediction_shift_by_outcome.csv",
    index=False,
)

print(
    outcome_prediction_shift.to_string(index=False)
)


# =============================================================================
# PERFORMANCE COMPARISON
# =============================================================================

print_header("12. PHASE-SPECIFIC PERFORMANCE")

performance_rows = []


def add_performance_rows(
    predictions,
    split,
    model_name,
    probability_column,
):
    """
    Add overall and phase-specific performance rows.
    """
    for phase in ["Overall"] + PHASE_ORDER:

        if phase == "Overall":
            subset = predictions.copy()
        else:
            subset = predictions[
                predictions["season_phase"] == phase
            ].copy()

        y_true = subset["win_home"]
        probabilities = subset[probability_column]

        metrics = calculate_metrics(
            y_true,
            probabilities,
        )

        performance_rows.append(
            {
                "split": split,
                "model": model_name,
                "season_phase": phase,
                **metrics,
            }
        )


# Experiment 2
add_performance_rows(
    val_predictions,
    "validation",
    "Experiment 2",
    "probability_exp2",
)

add_performance_rows(
    test_predictions,
    "test",
    "Experiment 2",
    "probability_exp2",
)


# Experiment 3
add_performance_rows(
    val_predictions,
    "validation",
    "Experiment 3",
    "probability_exp3",
)

add_performance_rows(
    test_predictions,
    "test",
    "Experiment 3",
    "probability_exp3",
)


# Model 5
def attach_model5_phase(
    predictions,
    feature_df,
):
    """Attach phase and outcome metadata to Model 5 predictions."""
    return attach_phase_and_outcome(
        predictions,
        feature_df,
    )


model5_val_phase = attach_model5_phase(
    model5_val,
    feature_data[
        feature_data["season"].isin(
            split_data["validation"]["season"].unique()
        )
    ],
)

model5_test_phase = attach_model5_phase(
    model5_test,
    feature_data[
        feature_data["season"].isin(
            split_data["test"]["season"].unique()
        )
    ],
)

add_performance_rows(
    model5_val_phase,
    "validation",
    "Model 5",
    "probability",
)

add_performance_rows(
    model5_test_phase,
    "test",
    "Model 5",
    "probability",
)


performance = pd.DataFrame(
    performance_rows
)

# Put models in logical order.
model_order = {
    "Model 5": 0,
    "Experiment 2": 1,
    "Experiment 3": 2,
}

performance["_model_order"] = performance[
    "model"
].map(model_order)

phase_order = {
    "Overall": 0,
    "Early": 1,
    "Mid": 2,
    "Late": 3,
}

performance["_phase_order"] = performance[
    "season_phase"
].map(phase_order)

performance = (
    performance
    .sort_values(
        ["split", "_phase_order", "_model_order"]
    )
    .drop(
        columns=["_model_order", "_phase_order"]
    )
    .reset_index(drop=True)
)

performance.to_csv(
    OUTPUT_DIR / "relative_vs_phase_performance.csv",
    index=False,
)

print(
    performance.to_string(index=False)
)


# =============================================================================
# EXPERIMENT 3 VS EXPERIMENT 2 PERFORMANCE DELTAS
# =============================================================================

print_header("13. EXPERIMENT 3 VS EXPERIMENT 2 PERFORMANCE DELTAS")

delta_rows = []

for split in ["validation", "test"]:
    for phase in ["Overall"] + PHASE_ORDER:

        exp2_row = performance[
            (
                performance["split"] == split
            )
            & (
                performance["model"] == "Experiment 2"
            )
            & (
                performance["season_phase"] == phase
            )
        ]

        exp3_row = performance[
            (
                performance["split"] == split
            )
            & (
                performance["model"] == "Experiment 3"
            )
            & (
                performance["season_phase"] == phase
            )
        ]

        if len(exp2_row) != 1 or len(exp3_row) != 1:
            raise ValueError(
                f"Could not uniquely locate "
                f"{split} / {phase} performance rows."
            )

        exp2_row = exp2_row.iloc[0]
        exp3_row = exp3_row.iloc[0]

        delta_rows.append(
            {
                "split": split,
                "season_phase": phase,
                "N": int(exp3_row["N"]),
                "log_loss_delta_exp3_minus_exp2": (
                    exp3_row["Log Loss"]
                    - exp2_row["Log Loss"]
                ),
                "brier_delta_exp3_minus_exp2": (
                    exp3_row["Brier"]
                    - exp2_row["Brier"]
                ),
                "roc_auc_delta_exp3_minus_exp2": (
                    exp3_row["ROC AUC"]
                    - exp2_row["ROC AUC"]
                ),
                "accuracy_delta_exp3_minus_exp2": (
                    exp3_row["Accuracy"]
                    - exp2_row["Accuracy"]
                ),
                "ece_delta_exp3_minus_exp2": (
                    exp3_row["ECE"]
                    - exp2_row["ECE"]
                ),
                "mce_delta_exp3_minus_exp2": (
                    exp3_row["MCE"]
                    - exp2_row["MCE"]
                ),
            }
        )

performance_deltas = pd.DataFrame(delta_rows)


# =============================================================================
# SUMMARY
# =============================================================================

print_header("14. DIAGNOSTIC SUMMARY")

test_overall = performance_deltas[
    (
        performance_deltas["split"] == "test"
    )
    & (
        performance_deltas["season_phase"] == "Overall"
    )
].iloc[0]

val_overall = performance_deltas[
    (
        performance_deltas["split"] == "validation"
    )
    & (
        performance_deltas["season_phase"] == "Overall"
    )
].iloc[0]

summary_rows = [
    {
        "finding": "Exp2 returning-feature total importance",
        "value": total_importance_2,
        "unit": "fraction of total importance",
    },
    {
        "finding": "Exp3 returning-feature total importance",
        "value": total_importance_3,
        "unit": "fraction of total importance",
    },
    {
        "finding": "Exp3 - Exp2 test Log Loss",
        "value": test_overall[
            "log_loss_delta_exp3_minus_exp2"
        ],
        "unit": "lower is better",
    },
    {
        "finding": "Exp3 - Exp2 test Brier",
        "value": test_overall[
            "brier_delta_exp3_minus_exp2"
        ],
        "unit": "lower is better",
    },
    {
        "finding": "Exp3 - Exp2 test ROC AUC",
        "value": test_overall[
            "roc_auc_delta_exp3_minus_exp2"
        ],
        "unit": "higher is better",
    },
    {
        "finding": "Exp3 - Exp2 validation Log Loss",
        "value": val_overall[
            "log_loss_delta_exp3_minus_exp2"
        ],
        "unit": "lower is better",
    },
    {
        "finding": "Exp3 - Exp2 validation Brier",
        "value": val_overall[
            "brier_delta_exp3_minus_exp2"
        ],
        "unit": "lower is better",
    },
]

summary = pd.DataFrame(summary_rows)

summary.to_csv(
    OUTPUT_DIR / "relative_vs_phase_summary.csv",
    index=False,
)

print(
    summary.to_string(index=False)
)


# =============================================================================
# PRINT FEATURE IMPORTANCE CHANGE
# =============================================================================

print_header("15. FEATURE IMPORTANCE CHANGE")

importance_display = importance_comparison[
    [
        "feature_group",
        "experiment_2_importance",
        "experiment_3_importance",
        "importance_change",
        "importance_change_percent",
        "experiment_2_rank",
        "experiment_3_rank",
        "rank_change",
    ]
].copy()

print(
    importance_display.to_string(index=False)
)


# =============================================================================
# PRINT PHASE PERFORMANCE DELTAS
# =============================================================================

print_header("16. EXPERIMENT 3 VS EXPERIMENT 2 BY PHASE")

print(
    performance_deltas.to_string(index=False)
)


# =============================================================================
# PRINT FINAL INTERPRETATION CUES
# =============================================================================

print_header("17. INTERPRETATION CUES")

print(
    """
Important interpretation rules:

1. A negative Log Loss/Brier delta means Experiment 3 improved over
   Experiment 2 for that metric.

2. A positive ROC AUC delta means Experiment 3 improved over
   Experiment 2.

3. Feature importance changes describe how the fitted model distributed
   split importance between the corresponding returning-production
   representations. They do NOT establish causal importance.

4. Phase-weighted feature magnitude should decline mechanically according
   to the 1.00 / 0.67 / 0.33 weights. This is expected.

5. If prediction shifts are small in Early but larger in Late, that does
   not by itself mean late-season returning production is intrinsically
   more predictive. It only shows that the phase-weighted representation
   changes model predictions more in those observations.

6. Model 5 remains the untouched 310-feature benchmark. Experiment 3
   should be evaluated against both Experiment 2 and Model 5.

7. Do not use the 2025 test results to select or tune new phase weights.
"""
)

print()
print("=" * 88)
print("ANALYSIS COMPLETE")
print("=" * 88)
print(f"Outputs written to:")
print(OUTPUT_DIR)
print()