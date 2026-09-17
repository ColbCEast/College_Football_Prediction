"""
Gradient Boosting Win Probability
Model 3 - Returning Production Feature Analysis

Purpose
-------
Diagnose the eight phase-aware returning-production features used by
Returning Production Experiment 3.

This script does NOT:
- retrain the model
- tune hyperparameters
- modify Model 3
- optimize phase weights
- use the 2025 test set to make modeling decisions

It analyzes:
1. Global feature importance
2. Feature magnitude by season phase
3. Raw relative vs phase-aware feature magnitude
4. Outcome separation by phase
5. Validation and test phase distributions
6. Feature behavior across all available seasons

Model 3 phase definition:
    Early: gamesBefore_min 0-2  -> weight 1.00
    Mid:   gamesBefore_min 3-5  -> weight 0.67
    Late:  gamesBefore_min 6+   -> weight 0.33

Project structure expected:
    src/modeling/problems/win_probability/gradient_boosting/
        returning_production/
            analysis/
                feature_analysis.py

    data/processed/
        model_inputs/win_probability/
            train.csv
            validation.csv
            test.csv

        features/win_probability/returning_production/
            returning_features_2015.csv
            ...
            returning_features_2025.csv

    models/win_probability/gradient_boosting/
        returning_production/model_3/
            feature_importance.csv

Outputs:
    models/win_probability/gradient_boosting/
        returning_production/model_3/analysis/
            model3_returning_feature_importance.csv
            model3_returning_feature_magnitude_by_phase.csv
            model3_returning_raw_vs_phase_magnitude.csv
            model3_returning_outcome_separation.csv
            model3_returning_phase_summary.csv
"""

from pathlib import Path

import numpy as np
import pandas as pd


# =============================================================================
# CONFIGURATION
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

MODEL3_DIR = (
    ROOT
    / "models"
    / "win_probability"
    / "gradient_boosting"
    / "returning_production"
    / "model_3"
)

OUTPUT_DIR = MODEL3_DIR / "analysis"


# =============================================================================
# FEATURE DEFINITIONS
# =============================================================================

RETURNING_FEATURES = [
    "total_ppa",
    "passing_ppa",
    "receiving_ppa",
    "rushing_ppa",
    "usage",
    "passing_usage",
    "receiving_usage",
    "rushing_usage",
]

PHASE_FEATURES = [
    f"returning_{feature}_diff_phase"
    for feature in RETURNING_FEATURES
]

RELATIVE_FEATURES = [
    f"returning_{feature}_diff"
    for feature in RETURNING_FEATURES
]

HOME_FEATURES = [
    f"home_returning_{feature}"
    for feature in RETURNING_FEATURES
]

AWAY_FEATURES = [
    f"away_returning_{feature}"
    for feature in RETURNING_FEATURES
]

PHASE_WEIGHTS = {
    "Early": 1.00,
    "Mid": 0.67,
    "Late": 0.33,
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


def require_columns(df, columns, dataset_name):
    """Raise a clear error if required columns are missing."""
    missing = [col for col in columns if col not in df.columns]

    if missing:
        raise ValueError(
            f"{dataset_name} is missing required columns:\n"
            + "\n".join(f"  - {col}" for col in missing)
        )


def load_model_inputs():
    """
    Load train, validation, and test model-input datasets.

    These are used for:
    - game IDs
    - season
    - target
    - gamesBefore_home
    - gamesBefore_away
    """
    paths = {
        "Train": MODEL_INPUT_DIR / "train.csv",
        "Validation": MODEL_INPUT_DIR / "validation.csv",
        "Test": MODEL_INPUT_DIR / "test.csv",
    }

    frames = []

    for split, path in paths.items():
        if not path.exists():
            raise FileNotFoundError(f"Missing model input file: {path}")

        df = pd.read_csv(path)

        required = [
            "gameId",
            "season",
            "win_home",
            "gamesBefore_home",
            "gamesBefore_away",
        ]

        require_columns(df, required, f"{split} model input")

        df = df[
            [
                "gameId",
                "season",
                "win_home",
                "gamesBefore_home",
                "gamesBefore_away",
            ]
        ].copy()

        df["split"] = split

        frames.append(df)

    combined = pd.concat(frames, ignore_index=True)

    if combined["gameId"].duplicated().any():
        duplicates = combined.loc[
            combined["gameId"].duplicated(keep=False),
            "gameId",
        ].unique()

        raise ValueError(
            "Duplicate game IDs found across model-input splits. "
            f"Examples: {duplicates[:10].tolist()}"
        )

    return combined


def load_returning_features(seasons):
    """
    Load returning-production game-level features for all requested seasons.
    """
    frames = []

    required = (
        ["gameId", "season"]
        + HOME_FEATURES
        + AWAY_FEATURES
    )

    for season in sorted(seasons):
        path = RETURNING_FEATURE_DIR / f"returning_features_{season}.csv"

        if not path.exists():
            raise FileNotFoundError(
                f"Missing returning-production feature file:\n{path}"
            )

        df = pd.read_csv(path)

        require_columns(
            df,
            required,
            f"Returning features {season}",
        )

        df = df[required].copy()

        frames.append(df)

    combined = pd.concat(frames, ignore_index=True)

    if combined["gameId"].duplicated().any():
        duplicates = combined.loc[
            combined["gameId"].duplicated(keep=False),
            "gameId",
        ].unique()

        raise ValueError(
            "Duplicate game IDs found in returning-production features. "
            f"Examples: {duplicates[:10].tolist()}"
        )

    return combined


def assign_phase(games_before_min):
    """
    Assign Model 3 season phase.
    """
    conditions = [
        games_before_min <= 2,
        games_before_min <= 5,
        games_before_min >= 6,
    ]

    choices = [
        "Early",
        "Mid",
        "Late",
    ]

    phase = np.select(
        conditions,
        choices,
        default="Unknown",
    )

    return pd.Series(
        phase,
        index=games_before_min.index,
        name="season_phase",
    )


def assign_phase_weight(phase):
    """
    Assign the exact Model 3 phase weight.
    """
    return phase.map(PHASE_WEIGHTS)


def safe_mean(series):
    """Return mean or NaN when no valid observations exist."""
    values = pd.to_numeric(series, errors="coerce").dropna()

    if len(values) == 0:
        return np.nan

    return values.mean()


def safe_std(series):
    """Return sample standard deviation or NaN."""
    values = pd.to_numeric(series, errors="coerce").dropna()

    if len(values) < 2:
        return np.nan

    return values.std(ddof=1)


def safe_median(series):
    """Return median or NaN."""
    values = pd.to_numeric(series, errors="coerce").dropna()

    if len(values) == 0:
        return np.nan

    return values.median()


def safe_abs_mean(series):
    """Return mean absolute value."""
    values = pd.to_numeric(series, errors="coerce").dropna()

    if len(values) == 0:
        return np.nan

    return np.abs(values).mean()


def safe_min(series):
    """Return minimum or NaN."""
    values = pd.to_numeric(series, errors="coerce").dropna()

    if len(values) == 0:
        return np.nan

    return values.min()


def safe_max(series):
    """Return maximum or NaN."""
    values = pd.to_numeric(series, errors="coerce").dropna()

    if len(values) == 0:
        return np.nan

    return values.max()


def pooled_std(x1, x2):
    """
    Calculate pooled sample standard deviation.
    """
    x1 = pd.to_numeric(x1, errors="coerce").dropna().to_numpy()
    x2 = pd.to_numeric(x2, errors="coerce").dropna().to_numpy()

    n1 = len(x1)
    n2 = len(x2)

    if n1 < 2 or n2 < 2:
        return np.nan

    s1 = np.std(x1, ddof=1)
    s2 = np.std(x2, ddof=1)

    numerator = (
        ((n1 - 1) * s1 ** 2)
        + ((n2 - 1) * s2 ** 2)
    )

    denominator = n1 + n2 - 2

    if denominator <= 0:
        return np.nan

    return np.sqrt(numerator / denominator)


def standardized_difference(wins, losses):
    """
    Calculate standardized difference:

        (mean_wins - mean_losses) / pooled_std

    Positive values indicate larger feature values among home wins.
    """
    win_values = pd.to_numeric(wins, errors="coerce").dropna()
    loss_values = pd.to_numeric(losses, errors="coerce").dropna()

    if len(win_values) < 2 or len(loss_values) < 2:
        return np.nan

    pooled = pooled_std(win_values, loss_values)

    if pd.isna(pooled) or pooled == 0:
        return np.nan

    return (
        win_values.mean() - loss_values.mean()
    ) / pooled


def safe_correlation(x, y, method="pearson"):
    """
    Calculate correlation while handling missing values.
    """
    pair = pd.DataFrame(
        {
            "x": pd.to_numeric(x, errors="coerce"),
            "y": pd.to_numeric(y, errors="coerce"),
        }
    ).dropna()

    if len(pair) < 3:
        return np.nan

    if pair["x"].nunique() < 2:
        return np.nan

    if pair["y"].nunique() < 2:
        return np.nan

    return pair["x"].corr(pair["y"], method=method)


# =============================================================================
# LOAD DATA
# =============================================================================

print_header("MODEL 3 - RETURNING PRODUCTION FEATURE ANALYSIS")

print(f"Project root: {ROOT}")
print(f"Model 3 directory: {MODEL3_DIR}")
print(f"Output directory: {OUTPUT_DIR}")

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

print()
print("Loading model inputs...")

model_inputs = load_model_inputs()

print(f"  Total games: {len(model_inputs):,}")
print(
    f"  Seasons: "
    f"{model_inputs['season'].min()}-{model_inputs['season'].max()}"
)
print(
    "  Splits: "
    + ", ".join(
        f"{split}={count:,}"
        for split, count in model_inputs["split"].value_counts().sort_index().items()
    )
)

seasons = sorted(model_inputs["season"].unique())

print()
print("Loading returning-production features...")

returning = load_returning_features(seasons)

print(f"  Total returning-feature rows: {len(returning):,}")
print(
    f"  Seasons: "
    f"{returning['season'].min()}-{returning['season'].max()}"
)

# =============================================================================
# MERGE
# =============================================================================

print()
print("Merging model inputs with returning-production features...")

data = model_inputs.merge(
    returning,
    on=["gameId", "season"],
    how="left",
    validate="one_to_one",
)

if len(data) != len(model_inputs):
    raise ValueError(
        "Merge changed the number of model-input rows."
    )

for col in HOME_FEATURES + AWAY_FEATURES:
    if data[col].isna().all():
        raise ValueError(
            f"Returning feature is completely missing after merge: {col}"
        )

print(f"  Merged rows: {len(data):,}")

# =============================================================================
# PHASE RECONSTRUCTION
# =============================================================================

data["gamesBefore_min"] = data[
    ["gamesBefore_home", "gamesBefore_away"]
].min(axis=1)

data["season_phase"] = assign_phase(data["gamesBefore_min"])

data["phase_weight"] = assign_phase_weight(
    data["season_phase"]
)

if data["season_phase"].eq("Unknown").any():
    unknown = data.loc[
        data["season_phase"].eq("Unknown"),
        "gamesBefore_min",
    ].unique()

    raise ValueError(
        f"Unknown phase values encountered: {unknown}"
    )

if data["phase_weight"].isna().any():
    raise ValueError(
        "Missing phase weights encountered."
    )

# =============================================================================
# RECONSTRUCT RELATIVE AND PHASE-AWARE FEATURES
# =============================================================================

print()
print("Reconstructing relative and phase-aware returning features...")

for feature in RETURNING_FEATURES:
    home_col = f"home_returning_{feature}"
    away_col = f"away_returning_{feature}"

    relative_col = f"returning_{feature}_diff"
    phase_col = f"returning_{feature}_diff_phase"

    data[relative_col] = (
        data[home_col] - data[away_col]
    )

    data[phase_col] = (
        data[relative_col] * data["phase_weight"]
    )

# =============================================================================
# VALIDATE RECONSTRUCTION
# =============================================================================

print()
print("Validating reconstructed phase-aware features...")

for feature in RETURNING_FEATURES:
    relative_col = f"returning_{feature}_diff"
    phase_col = f"returning_{feature}_diff_phase"

    expected = (
        data[relative_col] * data["phase_weight"]
    )

    difference = (
        data[phase_col] - expected
    ).abs()

    max_error = difference.max()

    if pd.isna(max_error):
        continue

    if max_error > 1e-10:
        raise ValueError(
            f"Phase feature reconstruction failed for {phase_col}. "
            f"Maximum error: {max_error}"
        )

print("  All 8 phase-aware features reconstructed successfully.")


# =============================================================================
# [1] GLOBAL FEATURE IMPORTANCE
# =============================================================================

print_header("[1] GLOBAL FEATURE IMPORTANCE")

importance_path = MODEL3_DIR / "feature_importance.csv"

if not importance_path.exists():
    raise FileNotFoundError(
        f"Missing Model 3 feature importance file:\n{importance_path}"
    )

importance = pd.read_csv(importance_path)

required_importance_columns = [
    "feature",
    "importance",
]

require_columns(
    importance,
    required_importance_columns,
    "Model 3 feature importance",
)

importance = importance[
    ["feature", "importance"]
].copy()

importance["importance"] = pd.to_numeric(
    importance["importance"],
    errors="coerce",
)

importance = importance.dropna(
    subset=["importance"]
)

importance["rank"] = (
    importance["importance"]
    .rank(method="min", ascending=False)
    .astype(int)
)

total_importance = importance["importance"].sum()

if total_importance > 0:
    importance["importance_pct"] = (
        importance["importance"] / total_importance * 100
    )
else:
    importance["importance_pct"] = np.nan

returning_importance = (
    importance[
        importance["feature"].isin(PHASE_FEATURES)
    ]
    .copy()
)

returning_importance["returning_feature"] = (
    returning_importance["feature"]
    .str.replace(
        "returning_",
        "",
        regex=False,
    )
    .str.replace(
        "_diff_phase",
        "",
        regex=False,
    )
)

returning_importance["returning_rank"] = (
    returning_importance["importance"]
    .rank(method="min", ascending=False)
    .astype(int)
)

returning_importance = returning_importance.sort_values(
    "importance",
    ascending=False,
).reset_index(drop=True)

importance_output = (
    returning_importance[
        [
            "returning_rank",
            "feature",
            "importance",
            "importance_pct",
            "rank",
        ]
    ]
    .copy()
)

importance_output.columns = [
    "returning_rank",
    "phase_feature",
    "importance",
    "importance_pct",
    "global_rank",
]

importance_output.to_csv(
    OUTPUT_DIR / "model3_returning_feature_importance.csv",
    index=False,
)

print()
print(
    importance_output.to_string(
        index=False,
        formatters={
            "importance": "{:.8f}".format,
            "importance_pct": "{:.4f}%".format,
        },
    )
)

print()
print(
    f"Total importance of 8 returning features: "
    f"{importance_output['importance_pct'].sum():.4f}%"
)


# =============================================================================
# [2] FEATURE MAGNITUDE BY PHASE
# =============================================================================

print_header("[2] FEATURE MAGNITUDE BY PHASE")

magnitude_rows = []

phase_order = ["Early", "Mid", "Late"]

for feature in RETURNING_FEATURES:
    phase_feature = f"returning_{feature}_diff_phase"
    relative_feature = f"returning_{feature}_diff"

    for phase in phase_order:
        subset = data[
            data["season_phase"] == phase
        ]

        raw_values = subset[relative_feature]
        phase_values = subset[phase_feature]

        magnitude_rows.append(
            {
                "feature": feature,
                "phase": phase,
                "phase_weight": PHASE_WEIGHTS[phase],
                "n": len(subset),
                "raw_mean": safe_mean(raw_values),
                "raw_std": safe_std(raw_values),
                "raw_median": safe_median(raw_values),
                "raw_mean_abs": safe_abs_mean(raw_values),
                "raw_min": safe_min(raw_values),
                "raw_max": safe_max(raw_values),
                "phase_mean": safe_mean(phase_values),
                "phase_std": safe_std(phase_values),
                "phase_median": safe_median(phase_values),
                "phase_mean_abs": safe_abs_mean(phase_values),
                "phase_min": safe_min(phase_values),
                "phase_max": safe_max(phase_values),
            }
        )

magnitude = pd.DataFrame(magnitude_rows)

magnitude.to_csv(
    OUTPUT_DIR / "model3_returning_feature_magnitude_by_phase.csv",
    index=False,
)

print()

for feature in RETURNING_FEATURES:
    display = magnitude[
        magnitude["feature"] == feature
    ].copy()

    print(f"{feature}:")
    print(
        display[
            [
                "phase",
                "n",
                "raw_mean",
                "raw_mean_abs",
                "phase_mean",
                "phase_mean_abs",
            ]
        ].to_string(
            index=False,
            formatters={
                "raw_mean": "{:.5f}".format,
                "raw_mean_abs": "{:.5f}".format,
                "phase_mean": "{:.5f}".format,
                "phase_mean_abs": "{:.5f}".format,
            },
        )
    )
    print()


# =============================================================================
# [3] RAW RELATIVE VS PHASE-AWARE MAGNITUDE
# =============================================================================

print_header("[3] RAW RELATIVE VS PHASE-AWARE MAGNITUDE")

comparison_rows = []

for feature in RETURNING_FEATURES:
    relative_feature = f"returning_{feature}_diff"
    phase_feature = f"returning_{feature}_diff_phase"

    for phase in phase_order:
        subset = data[
            data["season_phase"] == phase
        ]

        raw_abs = subset[relative_feature].abs()
        phase_abs = subset[phase_feature].abs()

        raw_mean_abs = safe_mean(raw_abs)
        phase_mean_abs = safe_mean(phase_abs)

        if (
            pd.notna(raw_mean_abs)
            and raw_mean_abs != 0
        ):
            observed_ratio = (
                phase_mean_abs / raw_mean_abs
            )
        else:
            observed_ratio = np.nan

        comparison_rows.append(
            {
                "feature": feature,
                "phase": phase,
                "phase_weight": PHASE_WEIGHTS[phase],
                "n": len(subset),
                "raw_mean_abs": raw_mean_abs,
                "phase_mean_abs": phase_mean_abs,
                "observed_phase_to_raw_ratio": observed_ratio,
                "expected_weight_ratio": PHASE_WEIGHTS[phase],
            }
        )

raw_vs_phase = pd.DataFrame(comparison_rows)

raw_vs_phase.to_csv(
    OUTPUT_DIR / "model3_returning_raw_vs_phase_magnitude.csv",
    index=False,
)

print(
    raw_vs_phase.to_string(
        index=False,
        formatters={
            "raw_mean_abs": "{:.5f}".format,
            "phase_mean_abs": "{:.5f}".format,
            "observed_phase_to_raw_ratio": "{:.5f}".format,
            "expected_weight_ratio": "{:.2f}".format,
        },
    )
)


# =============================================================================
# [4] OUTCOME SEPARATION BY PHASE
# =============================================================================

print_header("[4] OUTCOME SEPARATION BY PHASE")

outcome_rows = []

for feature in RETURNING_FEATURES:
    phase_feature = f"returning_{feature}_diff_phase"

    for phase in phase_order:
        subset = data[
            data["season_phase"] == phase
        ].copy()

        wins = subset.loc[
            subset["win_home"] == 1,
            phase_feature,
        ]

        losses = subset.loc[
            subset["win_home"] == 0,
            phase_feature,
        ]

        outcome_rows.append(
            {
                "feature": feature,
                "phase": phase,
                "n": len(subset),
                "home_wins_n": wins.notna().sum(),
                "home_losses_n": losses.notna().sum(),
                "home_win_mean": safe_mean(wins),
                "home_loss_mean": safe_mean(losses),
                "home_win_median": safe_median(wins),
                "home_loss_median": safe_median(losses),
                "home_win_std": safe_std(wins),
                "home_loss_std": safe_std(losses),
                "standardized_difference": standardized_difference(
                    wins,
                    losses,
                ),
                "pearson_correlation": safe_correlation(
                    subset[phase_feature],
                    subset["win_home"],
                    method="pearson",
                ),
                "spearman_correlation": safe_correlation(
                    subset[phase_feature],
                    subset["win_home"],
                    method="spearman",
                ),
            }
        )

outcome_separation = pd.DataFrame(outcome_rows)

outcome_separation.to_csv(
    OUTPUT_DIR / "model3_returning_outcome_separation.csv",
    index=False,
)

print()

for feature in RETURNING_FEATURES:
    display = outcome_separation[
        outcome_separation["feature"] == feature
    ].copy()

    print(f"{feature}:")
    print(
        display[
            [
                "phase",
                "home_win_mean",
                "home_loss_mean",
                "standardized_difference",
                "pearson_correlation",
                "spearman_correlation",
            ]
        ].to_string(
            index=False,
            formatters={
                "home_win_mean": "{:.5f}".format,
                "home_loss_mean": "{:.5f}".format,
                "standardized_difference": "{:.5f}".format,
                "pearson_correlation": "{:.5f}".format,
                "spearman_correlation": "{:.5f}".format,
            },
        )
    )
    print()


# =============================================================================
# [5] VALIDATION / TEST PHASE SUMMARY
# =============================================================================

print_header("[5] VALIDATION / TEST PHASE SUMMARY")

phase_summary_rows = []

for split in ["Train", "Validation", "Test"]:
    split_data = data[
        data["split"] == split
    ].copy()

    for phase in phase_order:
        subset = split_data[
            split_data["season_phase"] == phase
        ]

        phase_summary_rows.append(
            {
                "split": split,
                "phase": phase,
                "n": len(subset),
                "pct_of_split": (
                    len(subset) / len(split_data) * 100
                    if len(split_data) > 0
                    else np.nan
                ),
                "gamesBefore_min_mean": safe_mean(
                    subset["gamesBefore_min"]
                ),
                "gamesBefore_min_median": safe_median(
                    subset["gamesBefore_min"]
                ),
                "phase_weight": PHASE_WEIGHTS[phase],
            }
        )

phase_summary = pd.DataFrame(
    phase_summary_rows
)

phase_summary.to_csv(
    OUTPUT_DIR / "model3_returning_phase_summary.csv",
    index=False,
)

print(
    phase_summary.to_string(
        index=False,
        formatters={
            "pct_of_split": "{:.2f}%".format,
            "gamesBefore_min_mean": "{:.3f}".format,
            "gamesBefore_min_median": "{:.1f}".format,
            "phase_weight": "{:.2f}".format,
        },
    )
)


# =============================================================================
# [6] TEST-SET-SPECIFIC FEATURE DIAGNOSTIC
# =============================================================================

print_header("[6] 2025 TEST-SET FEATURE DIAGNOSTIC")

test_data = data[
    data["split"] == "Test"
].copy()

test_rows = []

for feature in RETURNING_FEATURES:
    phase_feature = f"returning_{feature}_diff_phase"
    relative_feature = f"returning_{feature}_diff"

    for phase in phase_order:
        subset = test_data[
            test_data["season_phase"] == phase
        ]

        wins = subset.loc[
            subset["win_home"] == 1,
            phase_feature,
        ]

        losses = subset.loc[
            subset["win_home"] == 0,
            phase_feature,
        ]

        test_rows.append(
            {
                "feature": feature,
                "phase": phase,
                "n": len(subset),
                "raw_mean_abs": safe_abs_mean(
                    subset[relative_feature]
                ),
                "phase_mean_abs": safe_abs_mean(
                    subset[phase_feature]
                ),
                "home_win_mean": safe_mean(wins),
                "home_loss_mean": safe_mean(losses),
                "standardized_difference": standardized_difference(
                    wins,
                    losses,
                ),
                "pearson_correlation": safe_correlation(
                    subset[phase_feature],
                    subset["win_home"],
                    method="pearson",
                ),
            }
        )

test_diagnostic = pd.DataFrame(
    test_rows
)

print(
    test_diagnostic.to_string(
        index=False,
        formatters={
            "raw_mean_abs": "{:.5f}".format,
            "phase_mean_abs": "{:.5f}".format,
            "home_win_mean": "{:.5f}".format,
            "home_loss_mean": "{:.5f}".format,
            "standardized_difference": "{:.5f}".format,
            "pearson_correlation": "{:.5f}".format,
        },
    )
)


# =============================================================================
# [7] COMBINED DIAGNOSTIC TABLE
# =============================================================================

print_header("[7] COMBINED FEATURE DIAGNOSTIC")

combined = (
    importance_output[
        [
            "phase_feature",
            "importance",
            "importance_pct",
            "global_rank",
        ]
    ]
    .copy()
)

combined["feature"] = (
    combined["phase_feature"]
    .str.replace(
        "returning_",
        "",
        regex=False,
    )
    .str.replace(
        "_diff_phase",
        "",
        regex=False,
    )
)

# Average absolute magnitude across all observations by phase.
magnitude_pivot = (
    magnitude
    .pivot(
        index="feature",
        columns="phase",
        values="phase_mean_abs",
    )
    .reset_index()
)

magnitude_pivot.columns.name = None

magnitude_pivot = magnitude_pivot.rename(
    columns={
        "Early": "early_mean_abs",
        "Mid": "mid_mean_abs",
        "Late": "late_mean_abs",
    }
)

# Test-set standardized differences by phase.
test_effect_pivot = (
    test_diagnostic
    .pivot(
        index="feature",
        columns="phase",
        values="standardized_difference",
    )
    .reset_index()
)

test_effect_pivot.columns.name = None

test_effect_pivot = test_effect_pivot.rename(
    columns={
        "Early": "test_early_effect",
        "Mid": "test_mid_effect",
        "Late": "test_late_effect",
    }
)

combined = combined.merge(
    magnitude_pivot,
    on="feature",
    how="left",
)

combined = combined.merge(
    test_effect_pivot,
    on="feature",
    how="left",
)

combined = combined.sort_values(
    "importance",
    ascending=False,
).reset_index(drop=True)

combined.to_csv(
    OUTPUT_DIR / "model3_returning_combined_feature_diagnostic.csv",
    index=False,
)

print(
    combined.to_string(
        index=False,
        formatters={
            "importance": "{:.8f}".format,
            "importance_pct": "{:.4f}%".format,
            "early_mean_abs": "{:.5f}".format,
            "mid_mean_abs": "{:.5f}".format,
            "late_mean_abs": "{:.5f}".format,
            "test_early_effect": "{:.5f}".format,
            "test_mid_effect": "{:.5f}".format,
            "test_late_effect": "{:.5f}".format,
        },
    )
)


# =============================================================================
# [8] HIGH-LEVEL DIAGNOSTIC CHECKS
# =============================================================================

print_header("[8] HIGH-LEVEL DIAGNOSTIC CHECKS")

print()
print("Global importance ranking:")
for _, row in importance_output.iterrows():
    print(
        f"  {int(row['returning_rank'])}. "
        f"{row['phase_feature']}: "
        f"{row['importance']:.8f} "
        f"(global rank {int(row['global_rank'])})"
    )

print()
print("Phase magnitude comparison:")

for feature in RETURNING_FEATURES:
    row = magnitude_pivot[
        magnitude_pivot["feature"] == feature
    ]

    if row.empty:
        continue

    row = row.iloc[0]

    early = row["early_mean_abs"]
    mid = row["mid_mean_abs"]
    late = row["late_mean_abs"]

    print(
        f"  {feature}: "
        f"Early={early:.5f}, "
        f"Mid={mid:.5f}, "
        f"Late={late:.5f}"
    )

print()
print("2025 test outcome separation:")

for feature in RETURNING_FEATURES:
    row = test_effect_pivot[
        test_effect_pivot["feature"] == feature
    ]

    if row.empty:
        continue

    row = row.iloc[0]

    print(
        f"  {feature}: "
        f"Early={row['test_early_effect']:.5f}, "
        f"Mid={row['test_mid_effect']:.5f}, "
        f"Late={row['test_late_effect']:.5f}"
    )


# =============================================================================
# FINAL SUMMARY
# =============================================================================

print_header("ANALYSIS COMPLETE")

print("Generated files:")

output_files = sorted(
    OUTPUT_DIR.glob("model3_returning_*.csv")
)

for path in output_files:
    print(f"  {path.name}")

print()
print("No model training or hyperparameter tuning was performed.")
print("No test-set information was used to alter the model.")
print()
print("The analysis is intended to determine:")
print("  1. Which returning-production features Model 3 actually uses.")
print("  2. Whether their magnitude changes across season phases.")
print("  3. Whether phase weighting materially changes their magnitude.")
print("  4. Whether the features separate home wins from home losses.")
print("  5. Whether the 2025 late-season improvement is concentrated")
print("     in particular returning-production feature families.")
print()
print("Next step: inspect the output before deciding whether to modify")
print("the returning-production representation or phase weighting.")
print("=" * 88)