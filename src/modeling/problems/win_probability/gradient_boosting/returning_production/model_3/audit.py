"""
Gradient Boosting Win Probability - Model 3
Phase-Aware Relative Returning Production Audit

Model 3:
    Baseline: exact Model 5 310-feature space
    Added: 8 phase-aware relative returning-production features
    Total predictors: 318

Phase definition:
    0-2 games   -> weight 1.00
    3-5 games   -> weight 0.67
    6+ games    -> weight 0.33

This audit verifies:
    1. Required artifacts
    2. Feature-list integrity
    3. Model hyperparameters
    4. Training summary
    5. Prediction files
    6. Model 5 alignment
    7. Phase reconstruction
    8. Phase-aware feature arithmetic
    9. Feature importance
    10. Calibration
    11. Model 5 / Experiment 2 / Model 3 comparison
"""

from pathlib import Path
import sys
import warnings

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    brier_score_loss,
    log_loss,
    precision_score,
    recall_score,
    roc_auc_score,
)

warnings.filterwarnings("ignore")


# =============================================================================
# PATHS
# =============================================================================

ROOT = Path(__file__).resolve().parents[7]

MODEL_DIR = (
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

MODEL_2_DIR = (
    ROOT
    / "models"
    / "win_probability"
    / "gradient_boosting"
    / "returning_production"
    / "model_2"
)

INPUT_DIR = (
    ROOT
    / "data"
    / "processed"
    / "model_inputs"
    / "win_probability"
)

RETURNING_DIR = (
    ROOT
    / "data"
    / "processed"
    / "features"
    / "win_probability"
    / "returning_production"
)


# =============================================================================
# EXPECTED MODEL DEFINITION
# =============================================================================

EXPECTED_BASELINE_FEATURE_COUNT = 310
EXPECTED_TOTAL_FEATURE_COUNT = 318

EXPECTED_PARAMS = {
    "n_estimators": 200,
    "learning_rate": 0.03,
    "max_depth": 4,
    "min_samples_leaf": 10,
    "subsample": 0.75,
    "random_state": 42,
}

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

RELATIVE_SOURCE_FEATURES = {
    "returning_total_ppa_diff_phase": (
        "home_returning_total_ppa",
        "away_returning_total_ppa",
    ),
    "returning_passing_ppa_diff_phase": (
        "home_returning_passing_ppa",
        "away_returning_passing_ppa",
    ),
    "returning_receiving_ppa_diff_phase": (
        "home_returning_receiving_ppa",
        "away_returning_receiving_ppa",
    ),
    "returning_rushing_ppa_diff_phase": (
        "home_returning_rushing_ppa",
        "away_returning_rushing_ppa",
    ),
    "returning_usage_diff_phase": (
        "home_returning_usage",
        "away_returning_usage",
    ),
    "returning_passing_usage_diff_phase": (
        "home_returning_passing_usage",
        "away_returning_passing_usage",
    ),
    "returning_receiving_usage_diff_phase": (
        "home_returning_receiving_usage",
        "away_returning_receiving_usage",
    ),
    "returning_rushing_usage_diff_phase": (
        "home_returning_rushing_usage",
        "away_returning_rushing_usage",
    ),
}

EXPECTED_PHASE_COUNTS = {
    "train": {
        "Early": 2095,
        "Mid": 1551,
        "Late": 2786,
    },
    "validation": {
        "Early": 522,
        "Mid": 414,
        "Late": 805,
    },
    "test": {
        "Early": 264,
        "Mid": 212,
        "Late": 412,
    },
}


# =============================================================================
# HELPERS
# =============================================================================

PASS_COUNT = 0
FAIL_COUNT = 0
WARN_COUNT = 0


def check(condition, message, detail=None):
    global PASS_COUNT, FAIL_COUNT

    if condition:
        print(f"  [PASS] {message}")
        PASS_COUNT += 1
    else:
        print(f"  [FAIL] {message}")
        if detail:
            print(f"         {detail}")
        FAIL_COUNT += 1


def warning(message):
    global WARN_COUNT
    print(f"  [WARN] {message}")
    WARN_COUNT += 1


def section(title):
    print()
    print("=" * 88)
    print(title)
    print("=" * 88)


def load_csv(path):
    if not path.exists():
        raise FileNotFoundError(path)
    return pd.read_csv(path)


def first_existing(columns, candidates):
    """
    Return the first candidate that exists in columns.
    """
    for candidate in candidates:
        if candidate in columns:
            return candidate
    return None


def find_estimator(obj):
    """
    Robustly locate a GradientBoostingClassifier or estimator exposing
    get_params(), whether joblib saved the estimator directly or inside
    a sklearn Pipeline.
    """
    # Direct estimator
    if hasattr(obj, "get_params"):
        params = obj.get_params(deep=False)
        if "n_estimators" in params and "learning_rate" in params:
            return obj

    # Pipeline
    if hasattr(obj, "steps"):
        for _, step in reversed(obj.steps):
            estimator = find_estimator(step)
            if estimator is not None:
                return estimator

    # Generic nested object
    if hasattr(obj, "__dict__"):
        for value in obj.__dict__.values():
            if value is obj:
                continue

            if hasattr(value, "get_params"):
                try:
                    params = value.get_params(deep=False)
                    if (
                        "n_estimators" in params
                        and "learning_rate" in params
                    ):
                        return value
                except Exception:
                    pass

    return None


def calibration_metrics(y_true, probabilities, n_bins=10):
    """
    Calculate Expected Calibration Error (ECE) and Maximum Calibration Error
    (MCE) using equal-width bins.
    """
    y_true = np.asarray(y_true)
    probabilities = np.asarray(probabilities)

    bins = np.linspace(0.0, 1.0, n_bins + 1)

    ece = 0.0
    mce = 0.0

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

        if not mask.any():
            continue

        confidence = probabilities[mask].mean()
        accuracy = y_true[mask].mean()

        gap = abs(confidence - accuracy)

        ece += mask.mean() * gap
        mce = max(mce, gap)

    return ece, mce


def identify_prediction_columns(df):
    """
    Identify target, predicted class, and probability columns using the
    actual prediction-file schema.

    The audit accepts several common naming conventions so it does not
    depend on one exact implementation detail in train.py.
    """

    target_col = first_existing(
        df.columns,
        [
            "target",
            "actual",
            "y_true",
            "win_home",
            "actual_target",
        ],
    )

    prediction_col = first_existing(
        df.columns,
        [
            "prediction",
            "predicted_class",
            "y_pred",
            "predicted",
            "class_prediction",
        ],
    )

    probability_col = first_existing(
        df.columns,
        [
            "win_probability",
            "predicted_probability",
            "probability",
            "predicted_proba",
            "home_win_probability",
            "proba",
        ],
    )

    return target_col, prediction_col, probability_col


def identify_id_columns(df):
    game_id = first_existing(
        df.columns,
        ["gameId", "game_id", "id"],
    )

    season = first_existing(
        df.columns,
        ["season", "year"],
    )

    return game_id, season


def load_split_prediction(name):
    path = MODEL_DIR / f"{name}_predictions.csv"
    df = load_csv(path)

    target_col, prediction_col, probability_col = identify_prediction_columns(df)
    game_id_col, season_col = identify_id_columns(df)

    return {
        "df": df,
        "target_col": target_col,
        "prediction_col": prediction_col,
        "probability_col": probability_col,
        "game_id_col": game_id_col,
        "season_col": season_col,
    }


def get_phase(games_before_min):
    if games_before_min <= 2:
        return "Early"
    elif games_before_min <= 5:
        return "Mid"
    return "Late"


def get_phase_weight(games_before_min):
    if games_before_min <= 2:
        return 1.00
    elif games_before_min <= 5:
        return 0.67
    return 0.33


# =============================================================================
# HEADER
# =============================================================================

print("=" * 88)
print("GRADIENT BOOSTING WIN PROBABILITY - MODEL 3")
print("PHASE-AWARE RELATIVE RETURNING PRODUCTION AUDIT")
print("=" * 88)

print()
print("Model 3 definition:")
print("  Baseline: exact Model 5 310-feature space")
print("  Added: 8 phase-aware relative returning-production features")
print("  Total predictors: 318")
print("  Phase 1: 0-2 games -> weight 1.00")
print("  Phase 2: 3-5 games -> weight 0.67")
print("  Phase 3: 6+ games -> weight 0.33")
print("  Hyperparameters: exact Model 5 configuration")


# =============================================================================
# 1. REQUIRED FILES
# =============================================================================

section("1. REQUIRED FILES")

required_files = [
    MODEL_DIR / "model.joblib",
    MODEL_DIR / "feature_list.csv",
    MODEL_DIR / "training_summary.csv",
    MODEL_DIR / "train_predictions.csv",
    MODEL_DIR / "validation_predictions.csv",
    MODEL_DIR / "test_predictions.csv",
    MODEL_DIR / "feature_importance.csv",
    MODEL_5_DIR / "feature_list.csv",
    MODEL_5_DIR / "validation_predictions.csv",
    MODEL_5_DIR / "test_predictions.csv",
    MODEL_2_DIR / "validation_predictions.csv",
    MODEL_2_DIR / "test_predictions.csv",
]

for path in required_files:
    check(
        path.exists(),
        f"Found {path.relative_to(ROOT)}",
        f"Missing: {path}",
    )


# =============================================================================
# 2. FEATURE LIST AUDIT
# =============================================================================

section("2. FEATURE LIST AUDIT")

feature_df = load_csv(MODEL_DIR / "feature_list.csv")
model5_feature_df = load_csv(MODEL_5_DIR / "feature_list.csv")

feature_list = feature_df.iloc[:, 0].astype(str).tolist()
model5_features = model5_feature_df.iloc[:, 0].astype(str).tolist()

feature_set = set(feature_list)
model5_set = set(model5_features)
phase_set = set(PHASE_FEATURES)

print(f"  Feature count: {len(feature_list)}")

check(
    len(feature_list) == EXPECTED_TOTAL_FEATURE_COUNT,
    f"Exactly {EXPECTED_TOTAL_FEATURE_COUNT} predictors",
)

check(
    len(feature_set) == len(feature_list),
    "No duplicate feature names",
)

check(
    len(model5_features) == EXPECTED_BASELINE_FEATURE_COUNT,
    f"Model 5 contains exactly {EXPECTED_BASELINE_FEATURE_COUNT} predictors",
)

missing_baseline = sorted(model5_set - feature_set)

check(
    not missing_baseline,
    "All 310 Model 5 baseline features are present",
    f"Missing baseline features: {missing_baseline}",
)

# Correct definition of "unexpected":
# Anything in Model 3 that is NOT in Model 5 AND NOT one of the 8
# intentionally added phase-aware features.
unexpected_features = sorted(
    feature_set - model5_set - phase_set
)

check(
    not unexpected_features,
    "No unexpected predictors relative to Model 5",
    f"Unexpected predictors: {unexpected_features}",
)

missing_phase_features = sorted(
    phase_set - feature_set
)

check(
    not missing_phase_features,
    "All 8 expected phase-aware features are present",
    f"Missing phase features: {missing_phase_features}",
)

unexpected_phase_features = sorted(
    [
        feature
        for feature in feature_set
        if "returning_" in feature
        and "_phase" in feature
        and feature not in phase_set
    ]
)

check(
    not unexpected_phase_features,
    "No unexpected phase-aware features",
    f"Unexpected phase features: {unexpected_phase_features}",
)

print()
print("  Phase-aware features:")

for feature in PHASE_FEATURES:
    print(f"    - {feature}")


# =============================================================================
# 3. MODEL CONFIGURATION
# =============================================================================

section("3. MODEL CONFIGURATION")

model_object = joblib.load(MODEL_DIR / "model.joblib")
estimator = find_estimator(model_object)

check(
    estimator is not None,
    "Gradient boosting estimator located in model.joblib",
)

if estimator is not None:
    params = estimator.get_params(deep=False)

    print(f"  Estimator type: {type(estimator).__name__}")

    for parameter, expected in EXPECTED_PARAMS.items():
        actual = params.get(parameter)

        if isinstance(expected, float):
            matches = (
                actual is not None
                and np.isclose(float(actual), expected, atol=1e-12)
            )
        else:
            matches = actual == expected

        check(
            matches,
            f"{parameter} = {expected}",
            f"Found: {actual}",
        )


# =============================================================================
# 4. TRAINING SUMMARY
# =============================================================================

section("4. TRAINING SUMMARY")

summary = load_csv(MODEL_DIR / "training_summary.csv")

print(summary.to_string(index=False))

expected_summary = {
    "train": {
        "seasons": "2015-2022",
        "rows": 6432,
        "features": 318,
    },
    "validation": {
        "seasons": "2023-2024",
        "rows": 1741,
        "features": 318,
    },
    "test": {
        "seasons": "2025",
        "rows": 888,
        "features": 318,
    },
}

for split, expected in expected_summary.items():
    rows = summary.loc[
        summary["split"].astype(str).str.lower() == split
    ]

    check(
        len(rows) == 1,
        f"Training summary contains exactly one {split} row",
    )

    if len(rows) == 1:
        row = rows.iloc[0]

        check(
            int(row["rows"]) == expected["rows"],
            f"{split.capitalize()} summary rows = {expected['rows']}",
            f"Found: {row['rows']}",
        )

        check(
            int(row["features"]) == expected["features"],
            f"{split.capitalize()} summary features = {expected['features']}",
            f"Found: {row['features']}",
        )


# =============================================================================
# 5. PREDICTION FILE AUDIT
# =============================================================================

section("5. PREDICTION FILE AUDIT")

splits = {
    "train": 6432,
    "validation": 1741,
    "test": 888,
}

prediction_data = {}

for split, expected_rows in splits.items():

    data = load_split_prediction(split)
    df = data["df"]

    prediction_data[split] = data

    print()
    print(f"  {split.capitalize()} columns:")
    print(f"    {list(df.columns)}")

    print(f"  {split.capitalize()} rows: {len(df):,}")

    check(
        len(df) == expected_rows,
        f"{split.capitalize()} prediction row count = {expected_rows:,}",
    )

    check(
        data["target_col"] is not None,
        f"{split.capitalize()} target column identified",
        f"Columns: {list(df.columns)}",
    )

    check(
        data["prediction_col"] is not None,
        f"{split.capitalize()} prediction column identified",
        f"Columns: {list(df.columns)}",
    )

    check(
        data["probability_col"] is not None,
        f"{split.capitalize()} probability column identified",
        f"Columns: {list(df.columns)}",
    )

    if data["prediction_col"] is not None:
        prediction_values = df[data["prediction_col"]].dropna().unique()

        check(
            set(prediction_values).issubset({0, 1}),
            f"{split.capitalize()} predicted classes are binary",
            f"Values: {prediction_values}",
        )

    if data["target_col"] is not None:
        target_values = df[data["target_col"]].dropna().unique()

        check(
            set(target_values).issubset({0, 1}),
            f"{split.capitalize()} targets are binary",
            f"Values: {target_values}",
        )

    if data["probability_col"] is not None:
        probabilities = pd.to_numeric(
            df[data["probability_col"]],
            errors="coerce",
        )

        check(
            probabilities.notna().all(),
            f"{split.capitalize()} probabilities contain no non-numeric values",
        )

        if probabilities.notna().all():
            check(
                ((probabilities >= 0) & (probabilities <= 1)).all(),
                f"{split.capitalize()} probabilities are in [0, 1]",
                (
                    f"Min={probabilities.min():.6f}, "
                    f"Max={probabilities.max():.6f}"
                ),
            )


# =============================================================================
# 6. MODEL 5 ALIGNMENT AUDIT
# =============================================================================

section("6. MODEL 5 ALIGNMENT AUDIT")

model5_validation = load_csv(
    MODEL_5_DIR / "validation_predictions.csv"
)

model5_test = load_csv(
    MODEL_5_DIR / "test_predictions.csv"
)

for split, model3_data, model5_df in [
    (
        "validation",
        prediction_data["validation"],
        model5_validation,
    ),
    (
        "test",
        prediction_data["test"],
        model5_test,
    ),
]:

    model3_df = model3_data["df"]

    model3_game_id = model3_data["game_id_col"]
    model5_game_id, model5_season = identify_id_columns(model5_df)

    model3_season = model3_data["season_col"]

    check(
        model3_game_id is not None and model5_game_id is not None,
        f"{split.capitalize()} game ID columns identified",
    )

    if model3_game_id is not None and model5_game_id is not None:

        model3_ids = set(model3_df[model3_game_id].astype(str))
        model5_ids = set(model5_df[model5_game_id].astype(str))

        check(
            model3_ids == model5_ids,
            f"{split.capitalize()} game IDs exactly match Model 5",
            (
                f"Model 3 only: {len(model3_ids - model5_ids)}; "
                f"Model 5 only: {len(model5_ids - model3_ids)}"
            ),
        )

    if model3_season is not None and model5_season is not None:

        model3_sorted = model3_df.sort_values(
            model3_game_id
        ).reset_index(drop=True)

        model5_sorted = model5_df.sort_values(
            model5_game_id
        ).reset_index(drop=True)

        check(
            model3_sorted[model3_season].reset_index(drop=True).equals(
                model5_sorted[model5_season].reset_index(drop=True)
            ),
            f"{split.capitalize()} seasons exactly match Model 5",
        )

    model3_target = model3_data["target_col"]
    model5_target, _, _ = identify_prediction_columns(model5_df)

    if model3_target is not None and model5_target is not None:

        model3_sorted = model3_df.sort_values(
            model3_game_id
        ).reset_index(drop=True)

        model5_sorted = model5_df.sort_values(
            model5_game_id
        ).reset_index(drop=True)

        check(
            model3_sorted[model3_target].reset_index(drop=True).equals(
                model5_sorted[model5_target].reset_index(drop=True)
            ),
            f"{split.capitalize()} targets exactly match Model 5",
        )


# =============================================================================
# 7. PHASE RECONSTRUCTION
# =============================================================================

section("7. PHASE RECONSTRUCTION")

input_splits = {
    "train": INPUT_DIR / "train.csv",
    "validation": INPUT_DIR / "validation.csv",
    "test": INPUT_DIR / "test.csv",
}

phase_frames = {}

for split, path in input_splits.items():

    df = load_csv(path)

    required_columns = [
        "gameId",
        "season",
        "gamesBefore_home",
        "gamesBefore_away",
    ]

    missing = [
        column
        for column in required_columns
        if column not in df.columns
    ]

    check(
        not missing,
        f"{split.capitalize()} contains required phase columns",
        f"Missing: {missing}",
    )

    if missing:
        continue

    phase_df = df[
        [
            "gameId",
            "season",
            "gamesBefore_home",
            "gamesBefore_away",
        ]
    ].copy()

    phase_df["gamesBefore_min"] = phase_df[
        ["gamesBefore_home", "gamesBefore_away"]
    ].min(axis=1)

    phase_df["season_phase"] = phase_df["gamesBefore_min"].apply(
        get_phase
    )

    phase_df["phase_weight"] = phase_df["gamesBefore_min"].apply(
        get_phase_weight
    )

    phase_frames[split] = phase_df

    phase_counts = (
        phase_df["season_phase"]
        .value_counts()
        .reindex(["Early", "Mid", "Late"], fill_value=0)
    )

    print()
    print(f"  {split.capitalize()} phase distribution:")
    print(f"    Early (0-2): {phase_counts['Early']:,}")
    print(f"    Mid   (3-5): {phase_counts['Mid']:,}")
    print(f"    Late   (6+): {phase_counts['Late']:,}")

    expected = EXPECTED_PHASE_COUNTS[split]

    for phase in ["Early", "Mid", "Late"]:
        check(
            int(phase_counts[phase]) == expected[phase],
            (
                f"{split.capitalize()} {phase} phase count = "
                f"{expected[phase]:,}"
            ),
            f"Found: {phase_counts[phase]:,}",
        )

    check(
        phase_df["phase_weight"].isin([1.00, 0.67, 0.33]).all(),
        f"{split.capitalize()} phase weights are valid",
    )

    early_weights = phase_df.loc[
        phase_df["season_phase"] == "Early",
        "phase_weight",
    ]

    mid_weights = phase_df.loc[
        phase_df["season_phase"] == "Mid",
        "phase_weight",
    ]

    late_weights = phase_df.loc[
        phase_df["season_phase"] == "Late",
        "phase_weight",
    ]

    check(
        (early_weights == 1.00).all(),
        f"{split.capitalize()} Early weight = 1.00",
    )

    check(
        (mid_weights == 0.67).all(),
        f"{split.capitalize()} Mid weight = 0.67",
    )

    check(
        (late_weights == 0.33).all(),
        f"{split.capitalize()} Late weight = 0.33",
    )


# =============================================================================
# 8. PHASE-AWARE FEATURE ARITHMETIC
# =============================================================================

section("8. PHASE-AWARE FEATURE ARITHMETIC")

print(
    "  Reconstructing expected phase-aware features independently from "
    "returning-production game features."
)
print(
    "  Note: train.py does not persist the transformed 318-feature matrix, "
    "so this validates the construction formula rather than comparing "
    "against a saved transformed matrix."
)

arithmetic_failures = []

for split, phase_df in phase_frames.items():

    seasons = sorted(phase_df["season"].unique())

    for season in seasons:

        returning_path = (
            RETURNING_DIR
            / f"returning_features_{season}.csv"
        )

        if not returning_path.exists():
            arithmetic_failures.append(
                f"{split} {season}: missing {returning_path}"
            )
            continue

        returning = load_csv(returning_path)

        merged = phase_df[
            phase_df["season"] == season
        ].merge(
            returning,
            on="gameId",
            how="left",
            validate="one_to_one",
        )

        if len(merged) != len(
            phase_df[phase_df["season"] == season]
        ):
            arithmetic_failures.append(
                f"{split} {season}: row count changed during merge"
            )
            continue

        for output_feature, (home_col, away_col) in (
            RELATIVE_SOURCE_FEATURES.items()
        ):

            if (
                home_col not in merged.columns
                or away_col not in merged.columns
            ):
                arithmetic_failures.append(
                    f"{split} {season}: missing source columns "
                    f"for {output_feature}"
                )
                continue

            expected = (
                pd.to_numeric(
                    merged[home_col],
                    errors="coerce",
                )
                - pd.to_numeric(
                    merged[away_col],
                    errors="coerce",
                )
            ) * merged["phase_weight"]

            if expected.isna().all():
                continue

            # Validate the mathematical construction independently.
            # Because the transformed matrix is not persisted, we test
            # that the expected formula produces finite values wherever
            # both source values are available.
            source_available = (
                merged[home_col].notna()
                & merged[away_col].notna()
            )

            if source_available.any():
                finite = np.isfinite(
                    expected[source_available].astype(float)
                )

                if not finite.all():
                    arithmetic_failures.append(
                        f"{split} {season}: non-finite values "
                        f"constructed for {output_feature}"
                    )

if not arithmetic_failures:
    check(
        True,
        "All 8 phase-aware feature formulas reconstruct successfully "
        "across train/validation/test seasons",
    )
else:
    check(
        False,
        "Phase-aware feature reconstruction failures detected",
        "; ".join(arithmetic_failures[:10]),
    )


# =============================================================================
# 9. FEATURE IMPORTANCE
# =============================================================================

section("9. FEATURE IMPORTANCE")

importance = load_csv(
    MODEL_DIR / "feature_importance.csv"
)

importance_feature_col = first_existing(
    importance.columns,
    ["feature", "feature_name"],
)

importance_value_col = first_existing(
    importance.columns,
    ["importance", "feature_importance"],
)

check(
    importance_feature_col is not None,
    "Feature importance contains a feature-name column",
)

check(
    importance_value_col is not None,
    "Feature importance contains an importance column",
)

if importance_feature_col is not None:

    importance_features = (
        importance[importance_feature_col]
        .astype(str)
        .tolist()
    )

    check(
        len(importance_features) == EXPECTED_TOTAL_FEATURE_COUNT,
        f"Feature importance contains exactly {EXPECTED_TOTAL_FEATURE_COUNT} entries",
        f"Found: {len(importance_features)}",
    )

    check(
        set(importance_features) == feature_set,
        "Feature importance features exactly match feature list",
    )


# =============================================================================
# 10. CALIBRATION
# =============================================================================

section("10. CALIBRATION")

for split in ["validation", "test"]:

    data = prediction_data[split]

    target_col = data["target_col"]
    probability_col = data["probability_col"]

    if target_col is None or probability_col is None:
        warning(
            f"Skipping {split} calibration because required columns "
            "could not be identified."
        )
        continue

    y_true = data["df"][target_col].astype(int)
    probabilities = data["df"][probability_col].astype(float)

    ece, mce = calibration_metrics(
        y_true,
        probabilities,
    )

    print()
    print(f"  {split.capitalize()}:")
    print(f"    ECE: {ece:.6f}")
    print(f"    MCE: {mce:.6f}")

    check(
        0 <= ece <= 1,
        f"{split.capitalize()} ECE is valid",
    )

    check(
        0 <= mce <= 1,
        f"{split.capitalize()} MCE is valid",
    )


# =============================================================================
# 11. MODEL 5 / EXPERIMENT 2 / MODEL 3 COMPARISON
# =============================================================================

section("11. MODEL 5 / EXPERIMENT 2 / MODEL 3 COMPARISON")


def load_metrics_from_prediction_file(
    path,
    target_column,
    probability_column,
    prediction_column,
):
    """
    Load a prediction file using its known schema and calculate
    evaluation metrics.
    """

    df = pd.read_csv(path)

    required_columns = [
        target_column,
        probability_column,
        prediction_column,
    ]

    missing = [
        column
        for column in required_columns
        if column not in df.columns
    ]

    if missing:

        print(
            f"  [WARN] {path.name} missing required columns: {missing}"
        )

        print(
            f"         Available columns: {list(df.columns)}"
        )

        return None

    y_true = df[target_column].astype(int)

    probabilities = (
        df[probability_column]
        .astype(float)
    )

    predictions = (
        df[prediction_column]
        .astype(int)
    )

    ece, mce = calibration_metrics(
        y_true,
        probabilities,
    )

    return {
        "accuracy": accuracy_score(
            y_true,
            predictions,
        ),
        "balanced_accuracy": balanced_accuracy_score(
            y_true,
            predictions,
        ),
        "precision": precision_score(
            y_true,
            predictions,
            zero_division=0,
        ),
        "recall": recall_score(
            y_true,
            predictions,
            zero_division=0,
        ),
        "roc_auc": roc_auc_score(
            y_true,
            probabilities,
        ),
        "log_loss": log_loss(
            y_true,
            probabilities,
        ),
        "brier": brier_score_loss(
            y_true,
            probabilities,
        ),
        "ece": ece,
        "mce": mce,
    }


model3_metrics = {}
model5_metrics = {}
model2_metrics = {}


for split in ["validation", "test"]:

    # -------------------------------------------------------------------------
    # Model 3
    # -------------------------------------------------------------------------

    model3_metrics[split] = load_metrics_from_prediction_file(
        MODEL_DIR / f"{split}_predictions.csv",
        target_column="win_home",
        probability_column="predicted_probability",
        prediction_column="predicted_class",
    )

    # -------------------------------------------------------------------------
    # Model 5
    # -------------------------------------------------------------------------

    model5_metrics[split] = load_metrics_from_prediction_file(
        MODEL_5_DIR / f"{split}_predictions.csv",
        target_column="win_home_actual",
        probability_column="win_home_probability",
        prediction_column="win_home_prediction",
    )

    # -------------------------------------------------------------------------
    # Experiment 2
    # -------------------------------------------------------------------------

    model2_metrics[split] = load_metrics_from_prediction_file(
        MODEL_2_DIR / f"{split}_predictions.csv",
        target_column="win_home",
        probability_column="predicted_probability_home_win",
        prediction_column="predicted_home_win",
    )


for split in ["validation", "test"]:

    print()
    print(f"  {split.upper()}")

    comparison_rows = []

    for model_name, metrics in [
        ("Model 5", model5_metrics[split]),
        ("Experiment 2", model2_metrics[split]),
        ("Model 3", model3_metrics[split]),
    ]:

        if metrics is None:
            continue

        comparison_rows.append(
            {
                "Model": model_name,
                "Log Loss": metrics["log_loss"],
                "Brier": metrics["brier"],
                "ROC AUC": metrics["roc_auc"],
                "Accuracy": metrics["accuracy"],
                "ECE": metrics["ece"],
                "MCE": metrics["mce"],
            }
        )

    comparison_df = pd.DataFrame(comparison_rows)

    if not comparison_df.empty:

        print(
            comparison_df.to_string(
                index=False,
                formatters={
                    "Log Loss": "{:.6f}".format,
                    "Brier": "{:.6f}".format,
                    "ROC AUC": "{:.6f}".format,
                    "Accuracy": "{:.6f}".format,
                    "ECE": "{:.6f}".format,
                    "MCE": "{:.6f}".format,
                },
            )
        )


# =============================================================================
# 12. MODEL 3 VS MODEL 5 DELTAS
# =============================================================================

section("12. MODEL 3 VS MODEL 5 DELTAS")


for split in ["validation", "test"]:

    m3 = model3_metrics[split]
    m5 = model5_metrics[split]

    if m3 is None or m5 is None:

        warning(
            f"Could not calculate {split} Model 3 vs Model 5 deltas."
        )

        continue

    print()
    print(f"  {split.capitalize()}")

    metrics_to_compare = [
        ("log_loss", "Log Loss"),
        ("brier", "Brier"),
        ("roc_auc", "ROC AUC"),
        ("accuracy", "Accuracy"),
        ("ece", "ECE"),
        ("mce", "MCE"),
    ]

    for metric_key, metric_label in metrics_to_compare:

        delta = (
            m3[metric_key]
            - m5[metric_key]
        )

        print(
            f"    {metric_label:12s}: "
            f"Model 5={m5[metric_key]:.6f} | "
            f"Model 3={m3[metric_key]:.6f} | "
            f"Delta={delta:+.6f}"
        )



# =============================================================================
# 13. FINAL ASSESSMENT
# =============================================================================

section("13. FINAL AUDIT SUMMARY")

print(f"  Passed checks:   {PASS_COUNT}")
print(f"  Failed checks:   {FAIL_COUNT}")
print(f"  Warnings:        {WARN_COUNT}")

print()

if FAIL_COUNT == 0:
    print("  [PASS] MODEL 3 AUDIT PASSED")
else:
    print("  [FAIL] MODEL 3 AUDIT HAS FAILURES")

print()
print("  Important interpretation:")
print("    - Model 3 uses the exact Model 5 310-feature baseline.")
print("    - Exactly 8 phase-aware relative returning features are added.")
print("    - Total feature count is 318.")
print("    - The phase variable is used only to construct the new features.")
print("    - gamesBefore_min, season_phase, and phase_weight are NOT predictors.")
print("    - Test Log Loss remains the primary comparison metric.")
print("    - A lower Log Loss is better.")
print("    - Model 3 should not be declared the final champion based on")
print("      test performance alone; validation behavior and calibration")
print("      should be considered together with the primary test metric.")

print()
print("=" * 88)
print("AUDIT COMPLETE")
print("=" * 88)