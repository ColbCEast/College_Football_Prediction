# =============================================================================
# PHASE-SPECIFIC MODEL COMPARISON
# Gradient Boosting Win Probability
#
# Compares:
#   - Model 5: Complete 310-feature baseline
#   - Experiment 2: Relative returning production
#   - Model 3: Phase-aware relative returning production
#
# Splits:
#   - Validation: 2023-2024
#   - Test: 2025
#
# Phases:
#   - Early: 0-2 games
#   - Mid:   3-5 games
#   - Late:  6+ games
# =============================================================================


from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    brier_score_loss,
    log_loss,
    roc_auc_score,
)


# =============================================================================
# PATHS
# =============================================================================

ROOT = Path(__file__).resolve().parents[7]

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

MODEL_3_DIR = (
    ROOT
    / "models"
    / "win_probability"
    / "gradient_boosting"
    / "returning_production"
    / "model_3"
)

MODEL_INPUT_DIR = (
    ROOT
    / "data"
    / "processed"
    / "model_inputs"
    / "win_probability"
)


# =============================================================================
# SETTINGS
# =============================================================================

SPLITS = [
    "validation",
    "test",
]

PHASE_ORDER = [
    "Early",
    "Mid",
    "Late",
]


# =============================================================================
# DISPLAY HELPERS
# =============================================================================

def section(title):
    print()
    print("=" * 88)
    print(title)
    print("=" * 88)


def print_metric_table(df):
    if df.empty:
        print("  No results.")
        return

    print(
        df.to_string(
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
# CALIBRATION
# =============================================================================

def calibration_metrics(y_true, probabilities, n_bins=10):
    """
    Calculate Expected Calibration Error (ECE) and
    Maximum Calibration Error (MCE) using equal-width bins.
    """

    y_true = np.asarray(y_true)
    probabilities = np.asarray(probabilities)

    bin_edges = np.linspace(
        0.0,
        1.0,
        n_bins + 1,
    )

    ece = 0.0
    mce = 0.0

    for i in range(n_bins):

        lower = bin_edges[i]
        upper = bin_edges[i + 1]

        if i == n_bins - 1:

            mask = (
                (probabilities >= lower)
                & (probabilities <= upper)
            )

        else:

            mask = (
                (probabilities >= lower)
                & (probabilities < upper)
            )

        if not np.any(mask):
            continue

        bin_true = y_true[mask]
        bin_prob = probabilities[mask]

        observed_rate = np.mean(bin_true)
        mean_probability = np.mean(bin_prob)

        error = abs(
            observed_rate
            - mean_probability
        )

        proportion = len(bin_true) / len(y_true)

        ece += proportion * error
        mce = max(
            mce,
            error,
        )

    return ece, mce


# =============================================================================
# PHASE CONSTRUCTION
# =============================================================================

def assign_phase(games_before_min):
    """
    Assign season phase using the same definitions used by Model 3.
    """

    if games_before_min <= 2:
        return "Early"

    if games_before_min <= 5:
        return "Mid"

    return "Late"


def add_phase_to_prediction_file(
    prediction_df,
    model_name,
    split,
):
    """
    Add gamesBefore_min and season_phase.

    Model 3 already stores these values in its prediction file.

    Model 5 and Experiment 2 prediction files do not necessarily store
    them, so they are reconstructed from the corresponding model-input
    file using gameId.
    """

    df = prediction_df.copy()

    # -------------------------------------------------------------------------
    # Model 3 already contains phase information
    # -------------------------------------------------------------------------

    if (
        model_name == "Model 3"
        and "season_phase" in df.columns
        and "gamesBefore_min" in df.columns
    ):

        return df

    # -------------------------------------------------------------------------
    # Load model-input data
    # -------------------------------------------------------------------------

    input_path = (
        MODEL_INPUT_DIR
        / f"{split}.csv"
    )

    input_df = pd.read_csv(input_path)

    required_columns = [
        "gameId",
        "gamesBefore_home",
        "gamesBefore_away",
    ]

    missing = [
        column
        for column in required_columns
        if column not in input_df.columns
    ]

    if missing:

        raise ValueError(
            f"{input_path} is missing required columns: {missing}"
        )

    phase_df = input_df[
        [
            "gameId",
            "gamesBefore_home",
            "gamesBefore_away",
        ]
    ].copy()

    phase_df["gamesBefore_min"] = phase_df[
        [
            "gamesBefore_home",
            "gamesBefore_away",
        ]
    ].min(axis=1)

    phase_df["season_phase"] = (
        phase_df["gamesBefore_min"]
        .apply(assign_phase)
    )

    # -------------------------------------------------------------------------
    # Validate game IDs before merging
    # -------------------------------------------------------------------------

    if df["gameId"].duplicated().any():

        raise ValueError(
            f"{model_name} {split} prediction file contains duplicate gameIds."
        )

    if phase_df["gameId"].duplicated().any():

        raise ValueError(
            f"{split} model-input file contains duplicate gameIds."
        )

    # -------------------------------------------------------------------------
    # Merge phase information
    # -------------------------------------------------------------------------

    df = df.merge(
        phase_df[
            [
                "gameId",
                "gamesBefore_min",
                "season_phase",
            ]
        ],
        on="gameId",
        how="left",
        validate="one_to_one",
    )

    # -------------------------------------------------------------------------
    # Validate merge
    # -------------------------------------------------------------------------

    if df["gamesBefore_min"].isna().any():

        missing_count = (
            df["gamesBefore_min"]
            .isna()
            .sum()
        )

        raise ValueError(
            f"{model_name} {split}: "
            f"{missing_count} predictions could not be matched "
            f"to gamesBefore values."
        )

    if df["season_phase"].isna().any():

        raise ValueError(
            f"{model_name} {split}: "
            "some predictions have no assigned season phase."
        )

    return df


# =============================================================================
# MODEL PREDICTION FILE LOADING
# =============================================================================

def load_model_predictions(
    model_name,
    split,
):
    """
    Load a model's prediction file and standardize the relevant columns.
    """

    if model_name == "Model 5":

        path = (
            MODEL_5_DIR
            / f"{split}_predictions.csv"
        )

        target_column = "win_home_actual"
        probability_column = "win_home_probability"
        prediction_column = "win_home_prediction"

    elif model_name == "Experiment 2":

        path = (
            MODEL_2_DIR
            / f"{split}_predictions.csv"
        )

        target_column = "win_home"
        probability_column = "predicted_probability_home_win"
        prediction_column = "predicted_home_win"

    elif model_name == "Model 3":

        path = (
            MODEL_3_DIR
            / f"{split}_predictions.csv"
        )

        target_column = "win_home"
        probability_column = "predicted_probability"
        prediction_column = "predicted_class"

    else:

        raise ValueError(
            f"Unknown model: {model_name}"
        )

    if not path.exists():

        raise FileNotFoundError(
            f"Prediction file not found: {path}"
        )

    df = pd.read_csv(path)

    required_columns = [
        "gameId",
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

        raise ValueError(
            f"{model_name} {split} is missing columns: {missing}"
        )

    df = df[
        [
            "gameId",
            target_column,
            probability_column,
            prediction_column,
        ]
    ].copy()

    df = df.rename(
        columns={
            target_column: "win_home",
            probability_column: "predicted_probability",
            prediction_column: "predicted_class",
        }
    )

    return df


# =============================================================================
# PHASE-SPECIFIC METRICS
# =============================================================================

def calculate_phase_metrics(
    df,
    model_name,
    split,
    phase,
):
    """
    Calculate evaluation metrics for one model / split / phase.
    """

    phase_df = df[
        df["season_phase"] == phase
    ].copy()

    if phase_df.empty:

        return None

    y_true = phase_df["win_home"].astype(int)

    probabilities = (
        phase_df["predicted_probability"]
        .astype(float)
    )

    predictions = (
        phase_df["predicted_class"]
        .astype(int)
    )

    # -------------------------------------------------------------------------
    # ROC AUC
    # -------------------------------------------------------------------------

    if y_true.nunique() < 2:

        roc_auc = np.nan

    else:

        roc_auc = roc_auc_score(
            y_true,
            probabilities,
        )

    # -------------------------------------------------------------------------
    # Calibration
    # -------------------------------------------------------------------------

    ece, mce = calibration_metrics(
        y_true,
        probabilities,
    )

    return {
        "Split": split,
        "Phase": phase,
        "Model": model_name,
        "N": len(phase_df),
        "Log Loss": log_loss(
            y_true,
            probabilities,
        ),
        "Brier": brier_score_loss(
            y_true,
            probabilities,
        ),
        "ROC AUC": roc_auc,
        "Accuracy": accuracy_score(
            y_true,
            predictions,
        ),
        "ECE": ece,
        "MCE": mce,
    }


# =============================================================================
# LOAD AND PREPARE ALL DATA
# =============================================================================

section("PHASE-SPECIFIC MODEL COMPARISON")

print()
print("Models:")
print("  Model 5       = Complete 310-feature baseline")
print("  Experiment 2  = Relative returning production")
print("  Model 3       = Phase-aware relative returning production")

print()
print("Phases:")
print("  Early = 0-2 games")
print("  Mid   = 3-5 games")
print("  Late  = 6+ games")

print()
print("Evaluation priority:")
print("  1. Log Loss")
print("  2. Brier")
print("  3. ROC AUC")
print("  4. Accuracy")
print("  5. Calibration")


models = [
    "Model 5",
    "Experiment 2",
    "Model 3",
]


all_results = []


for split in SPLITS:

    section(f"{split.upper()}")

    prepared_predictions = {}

    for model_name in models:

        df = load_model_predictions(
            model_name,
            split,
        )

        df = add_phase_to_prediction_file(
            df,
            model_name,
            split,
        )

        prepared_predictions[model_name] = df

        print(
            f"  {model_name}: "
            f"{len(df):,} predictions loaded"
        )

    # -------------------------------------------------------------------------
    # Validate game ID alignment across models
    # -------------------------------------------------------------------------

    reference_ids = set(
        prepared_predictions["Model 5"]["gameId"]
    )

    for model_name in [
        "Experiment 2",
        "Model 3",
    ]:

        current_ids = set(
            prepared_predictions[model_name]["gameId"]
        )

        if current_ids != reference_ids:

            raise ValueError(
                f"{split}: game IDs do not match between "
                f"Model 5 and {model_name}."
            )

    print("  [PASS] Game IDs exactly align across all three models")

    # -------------------------------------------------------------------------
    # Validate phase alignment
    # -------------------------------------------------------------------------

    reference_phase = (
        prepared_predictions["Model 5"]
        .set_index("gameId")["season_phase"]
    )

    for model_name in [
        "Experiment 2",
        "Model 3",
    ]:

        current_phase = (
            prepared_predictions[model_name]
            .set_index("gameId")["season_phase"]
        )

        current_phase = current_phase.reindex(
            reference_phase.index
        )

        if not current_phase.equals(reference_phase):

            raise ValueError(
                f"{split}: phase assignments do not match between "
                f"Model 5 and {model_name}."
            )

    print("  [PASS] Phase assignments exactly align across all three models")

    # -------------------------------------------------------------------------
    # Calculate metrics
    # -------------------------------------------------------------------------

    for model_name in models:

        df = prepared_predictions[model_name]

        for phase in PHASE_ORDER:

            result = calculate_phase_metrics(
                df,
                model_name,
                split,
                phase,
            )

            if result is not None:

                all_results.append(result)


results_df = pd.DataFrame(all_results)


# =============================================================================
# COMPLETE PHASE RESULTS
# =============================================================================

section("1. PHASE-SPECIFIC RESULTS")

for split in SPLITS:

    print()
    print(split.upper())

    split_df = results_df[
        results_df["Split"] == split
    ].copy()

    for phase in PHASE_ORDER:

        print()
        print(f"  {phase}")

        phase_df = split_df[
            split_df["Phase"] == phase
        ].copy()

        phase_df = phase_df[
            [
                "Model",
                "N",
                "Log Loss",
                "Brier",
                "ROC AUC",
                "Accuracy",
                "ECE",
                "MCE",
            ]
        ]

        print_metric_table(
            phase_df
        )


# =============================================================================
# MODEL 3 VS MODEL 5 DELTAS
# =============================================================================

section("2. MODEL 3 VS MODEL 5 PHASE DELTAS")

print()
print("Negative Log Loss / Brier delta = Model 3 improvement")
print("Positive ROC AUC delta = Model 3 improvement")
print("Negative ECE / MCE delta = Model 3 improvement")


delta_rows = []


for split in SPLITS:

    for phase in PHASE_ORDER:

        m5 = results_df[
            (results_df["Split"] == split)
            & (results_df["Phase"] == phase)
            & (results_df["Model"] == "Model 5")
        ]

        m3 = results_df[
            (results_df["Split"] == split)
            & (results_df["Phase"] == phase)
            & (results_df["Model"] == "Model 3")
        ]

        if m5.empty or m3.empty:
            continue

        m5 = m5.iloc[0]
        m3 = m3.iloc[0]

        delta_rows.append(
            {
                "Split": split,
                "Phase": phase,
                "N": int(m3["N"]),
                "Log Loss Delta": (
                    m3["Log Loss"]
                    - m5["Log Loss"]
                ),
                "Brier Delta": (
                    m3["Brier"]
                    - m5["Brier"]
                ),
                "ROC AUC Delta": (
                    m3["ROC AUC"]
                    - m5["ROC AUC"]
                ),
                "Accuracy Delta": (
                    m3["Accuracy"]
                    - m5["Accuracy"]
                ),
                "ECE Delta": (
                    m3["ECE"]
                    - m5["ECE"]
                ),
                "MCE Delta": (
                    m3["MCE"]
                    - m5["MCE"]
                ),
            }
        )


delta_df = pd.DataFrame(
    delta_rows
)


for split in SPLITS:

    print()
    print(split.upper())

    split_delta = delta_df[
        delta_df["Split"] == split
    ].copy()

    if split_delta.empty:
        continue

    print(
        split_delta.to_string(
            index=False,
            formatters={
                "Log Loss Delta": "{:+.6f}".format,
                "Brier Delta": "{:+.6f}".format,
                "ROC AUC Delta": "{:+.6f}".format,
                "Accuracy Delta": "{:+.6f}".format,
                "ECE Delta": "{:+.6f}".format,
                "MCE Delta": "{:+.6f}".format,
            },
        )
    )


# =============================================================================
# PRIMARY METRIC FOCUS
# =============================================================================

section("3. PRIMARY METRIC — LOG LOSS")

print()
print(
    "Model 3 vs Model 5 by season phase:"
)

for split in SPLITS:

    print()
    print(f"  {split.upper()}")

    split_delta = delta_df[
        delta_df["Split"] == split
    ].copy()

    for _, row in split_delta.iterrows():

        delta = row["Log Loss Delta"]

        if delta < 0:

            interpretation = "Model 3 lower"

        elif delta > 0:

            interpretation = "Model 5 lower"

        else:

            interpretation = "Equal"

        print(
            f"    {row['Phase']:5s}: "
            f"Delta={delta:+.6f} "
            f"({interpretation})"
        )


# =============================================================================
# PHASE DISTRIBUTIONS
# =============================================================================

section("4. SAMPLE SIZE BY PHASE")

phase_counts = (
    results_df[
        results_df["Model"] == "Model 5"
    ]
    [
        [
            "Split",
            "Phase",
            "N",
        ]
    ]
    .copy()
)

print(
    phase_counts.to_string(
        index=False
    )
)


# =============================================================================
# FINAL INTERPRETATION
# =============================================================================

section("5. INTERPRETATION GUIDE")

print()
print("The key question is whether Model 3 improves performance where")
print("the phase-aware returning-production features are intended to matter most.")

print()
print("Evidence supporting the phase-aware hypothesis would include:")

print("  - Model 3 having lower Early-season Log Loss than Model 5.")
print("  - The Early improvement being larger than Mid/Late improvements.")
print("  - Model 3's advantage diminishing as gamesBefore_min increases.")
print("  - Similar behavior appearing in validation and test.")

print()
print("Evidence against the hypothesis would include:")

print("  - Model 3 being worse during the Early phase.")
print("  - Improvements occurring primarily in Late-season games.")
print("  - Large differences between validation and test phase behavior.")
print("  - No consistent phase-related pattern.")

print()
print("Important:")
print("  - Test performance should remain the primary final evaluation.")
print("  - Validation behavior is important for judging robustness.")
print("  - Phase-specific results are diagnostic and should not be used")
print("    to selectively choose whichever phase produces the best result.")