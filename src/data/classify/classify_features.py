import pandas as pd
from pathlib import Path


"""
Classify every column in the final master game dataset.

This classifier is designed around the CURRENT final feature schema,
including home/away game-level features.

Classification categories:

    identifier
    metadata
    static_pregame
    pregame
    rolling_pregame
    current_game
    target_candidate
    unknown

Predictive-safe means the feature can legitimately be used as an
input to a pregame prediction model without leaking information
from the current game.
"""


# ============================================================================
# PATHS
# ============================================================================

MASTER_PATH = "data/master/master_game_data.csv"
OUTPUT_PATH = "data/metadata/feature_classification.csv"


# ============================================================================
# LOAD DATA
# ============================================================================

df = pd.read_csv(MASTER_PATH)

print("=" * 70)
print("CLASSIFYING FINAL FEATURE SET")
print("=" * 70)

print(f"\nLoaded dataset:")
print(f"  Rows:    {df.shape[0]:,}")
print(f"  Columns: {df.shape[1]:,}")


# ============================================================================
# IDENTIFIER COLUMNS
# ============================================================================

IDENTIFIER_COLUMNS = {
    "Unnamed: 0",
    "id",
    "gameId",
    "teamId",

    # Game-level IDs
    "homeId",
    "awayId",
    "venueId",
}


# ============================================================================
# METADATA COLUMNS
# ============================================================================

METADATA_COLUMNS = {
    "season",
    "week",
    "conference",
    "team",
    "homeTeam",
    "awayTeam",
    "opponent",
    "completed",

    # Non-predictive game description
    "venue",
    "highlights",
    "notes",
}


# ============================================================================
# STATIC PREGAME COLUMNS
# ============================================================================

# Information that is available before kickoff and does not depend
# on the outcome/statistics of the current game.

STATIC_PREGAME_COLUMNS = {
    "isHome",
    "isAway",
    "homeAway",
    "seasonType",
    "startDate",
    "startTimeTBD",

    "neutralSite",
    "conferenceGame",

    "homeClassification",
    "awayClassification",

    "homeConference",
    "awayConference",
}


# ============================================================================
# TARGET CANDIDATES
# ============================================================================

# These represent current-game outcomes that could be targets.

TARGET_BASE_COLUMNS = {
    "points",
    "homePoints",
    "awayPoints",
    "pointsFor",
    "pointsAgainst",
    "pointDifferential",
    "win",
}


# ============================================================================
# CURRENT-GAME STATISTICS
# ============================================================================

CURRENT_GAME_BASE_COLUMNS = {
    # Scoring
    "points",
    "homePoints",
    "awayPoints",
    "pointsFor",
    "pointsAgainst",
    "pointDifferential",

    "rushingTDs",
    "passingTDs",
    "puntReturnTDs",
    "kickReturnTDs",
    "interceptionTDs",
    "defensiveTDs",
    "kickingPoints",

    # Rushing
    "rushingAttempts",
    "rushingYards",
    "yardsPerRushAttempt",

    # Passing
    "passAttempts",
    "completions",
    "completionAttempts",
    "completionPct",
    "netPassingYards",
    "yardsPerPass",

    # First / third / fourth down
    "firstDowns",

    "thirdDownConversions",
    "thirdDownAttempts",
    "thirdDownPct",
    "thirdDownEff",

    "fourthDownConversions",
    "fourthDownAttempts",
    "fourthDownPct",
    "fourthDownEff",

    # Turnovers / fumbles
    "fumblesRecovered",
    "fumblesLost",
    "totalFumbles",
    "interceptions",
    "passesIntercepted",
    "turnovers",

    # Returns
    "puntReturnYards",
    "puntReturns",
    "kickReturnYards",
    "kickReturns",
    "interceptionYards",

    # Penalties
    "penalties",
    "penaltyYards",
    "totalPenaltiesYards",

    # Possession
    "possessionTime",
    "possessionSeconds",

    # Defense
    "tacklesForLoss",
    "tackles",
    "sacks",
    "qbHurries",
    "passesDeflected",

    # General
    "totalYards",

    # Other current-game information
    "attendance",
    "excitementIndex",
    "homeLineScores",
    "awayLineScores",
    "highlights",
    "notes",
    "playoff",

    # Postgame information
    "homePostgameWinProbability",
    "awayPostgameWinProbability",
    "homePostgameElo",
    "awayPostgameElo",
}


# ============================================================================
# ROLLING PREGAME FEATURES
# ============================================================================

# These are historical statistics calculated from games BEFORE
# the current game.

ROLLING_MARKERS = (
    "AvgLast3",
    "AvgLast5",
    "Last3",
    "Last5",
    "winPctLast3",
    "winPctLast5",
)


# ============================================================================
# EXPLICIT PREGAME MARKERS
# ============================================================================

PREGAME_MARKERS = (
    "Before",
    "Pregame",
)


# ============================================================================
# HELPERS
# ============================================================================

def split_location_suffix(column):
    """
    Separate a final home/away suffix from the base feature name.

    Examples
    --------
    points_home
        -> ("points", "home")

    rushingYards_away
        -> ("rushingYards", "away")

    home_pregame_offense_ppa
        -> ("home_pregame_offense_ppa", None)

    gameId
        -> ("gameId", None)
    """

    if column.endswith("_home"):
        return column[:-5], "home"

    if column.endswith("_away"):
        return column[:-5], "away"

    return column, None


def is_rolling_pregame(base_column):
    """
    Determine whether a feature is a rolling historical feature.
    """

    return any(marker in base_column for marker in ROLLING_MARKERS)


def is_explicit_pregame(base_column):
    """
    Determine whether a feature explicitly represents pregame data.
    """

    lowered = base_column.lower()

    return (
        "pregame" in lowered
        or "before" in lowered
    )


# ============================================================================
# CLASSIFICATION FUNCTION
# ============================================================================

def classify_feature(column):
    """
    Classify a single feature.

    Returns
    -------
    category
    predictive_safe
    target_candidate
    leakage_risk
    reason
    """

    base_column, location = split_location_suffix(column)

    # ------------------------------------------------------------------------
    # 1. IDENTIFIER
    # ------------------------------------------------------------------------

    if column in IDENTIFIER_COLUMNS:
        return (
            "identifier",
            False,
            False,
            "low",
            "Identifier used to identify a game, team, venue, or row"
        )

    # ------------------------------------------------------------------------
    # 2. METADATA
    # ------------------------------------------------------------------------

    if column in METADATA_COLUMNS:
        return (
            "metadata",
            False,
            False,
            "low",
            "Descriptive metadata rather than a predictive feature"
        )

    # ------------------------------------------------------------------------
    # 3. STATIC PREGAME
    # ------------------------------------------------------------------------

    if column in STATIC_PREGAME_COLUMNS:
        return (
            "static_pregame",
            True,
            False,
            "low",
            "Known before kickoff and does not depend on the current-game outcome"
        )

    # ------------------------------------------------------------------------
    # 4. ROLLING PREGAME
    # ------------------------------------------------------------------------

    if is_rolling_pregame(base_column):

        return (
            "rolling_pregame",
            True,
            False,
            "low",
            "Historical rolling statistic calculated from previous games"
        )

    # ------------------------------------------------------------------------
    # 5. EXPLICIT PREGAME
    # ------------------------------------------------------------------------

    if is_explicit_pregame(base_column):

        return (
            "pregame",
            True,
            False,
            "low",
            "Historical statistic explicitly calculated before the current game"
        )

    # ------------------------------------------------------------------------
    # 6. TARGET CANDIDATES
    # ------------------------------------------------------------------------

    if base_column in TARGET_BASE_COLUMNS:

        return (
            "target_candidate",
            False,
            True,
            "high",
            "Current-game outcome that may be used as a model target"
        )

    # ------------------------------------------------------------------------
    # 7. CURRENT-GAME FEATURES
    # ------------------------------------------------------------------------

    if base_column in CURRENT_GAME_BASE_COLUMNS:

        return (
            "current_game",
            False,
            False,
            "high",
            "Statistic or information generated during or after the current game"
        )

    # ------------------------------------------------------------------------
    # 8. UNKNOWN
    # ------------------------------------------------------------------------

    return (
        "unknown",
        False,
        False,
        "review",
        "Could not be confidently classified automatically"
    )


# ============================================================================
# BUILD CLASSIFICATION TABLE
# ============================================================================

records = []

for column in df.columns:

    (
        category,
        predictive_safe,
        target_candidate,
        leakage_risk,
        reason
    ) = classify_feature(column)

    base_column, location = split_location_suffix(column)

    records.append({
        "column": column,
        "base_column": base_column,
        "location": location,
        "category": category,
        "predictive_safe": predictive_safe,
        "target_candidate": target_candidate,
        "leakage_risk": leakage_risk,
        "reason": reason,
        "notes": "",
    })


feature_df = pd.DataFrame(records)


# ============================================================================
# SAVE CLASSIFICATION
# ============================================================================

output_path = Path(OUTPUT_PATH)
output_path.parent.mkdir(parents=True, exist_ok=True)

feature_df.to_csv(output_path, index=False)


# ============================================================================
# SUMMARY
# ============================================================================

print("\n" + "=" * 70)
print("FEATURE CLASSIFICATION COMPLETE")
print("=" * 70)

print(f"\nSaved classification:")
print(f"  {OUTPUT_PATH}")

print(f"\nTotal columns classified:")
print(f"  {len(feature_df)}")


# ============================================================================
# CATEGORY COUNTS
# ============================================================================

print("\n" + "-" * 70)
print("CATEGORY COUNTS")
print("-" * 70)

category_counts = (
    feature_df["category"]
    .value_counts()
    .sort_index()
)

for category, count in category_counts.items():
    print(f"  {category:<20} {count:>5}")


# ============================================================================
# PREDICTIVE SAFETY
# ============================================================================

print("\n" + "-" * 70)
print("PREGAME PREDICTIVE SAFETY")
print("-" * 70)

safe_count = int(feature_df["predictive_safe"].sum())
unsafe_count = int((~feature_df["predictive_safe"]).sum())

print(f"Predictive-safe:      {safe_count}")
print(f"Not predictive-safe:  {unsafe_count}")


# ============================================================================
# TARGET CANDIDATES
# ============================================================================

print("\n" + "-" * 70)
print("TARGET CANDIDATES")
print("-" * 70)

target_candidates = feature_df[
    feature_df["target_candidate"]
]["column"]

if len(target_candidates) == 0:
    print("None")

else:
    for column in target_candidates:
        print(f"  {column}")


# ============================================================================
# UNKNOWN FEATURES
# ============================================================================

print("\n" + "-" * 70)
print("UNKNOWN FEATURES REQUIRING REVIEW")
print("-" * 70)

unknown = feature_df[
    feature_df["category"] == "unknown"
]

if len(unknown) == 0:

    print("None")

else:

    print(f"\n{len(unknown)} unknown columns:\n")

    for _, row in unknown.iterrows():
        print(
            f"  {row['column']}"
        )


# ============================================================================
# PREGAME FEATURES
# ============================================================================

print("\n" + "-" * 70)
print("PREDICTIVE-SAFE FEATURE COUNTS BY CATEGORY")
print("-" * 70)

safe_by_category = (
    feature_df[
        feature_df["predictive_safe"]
    ]
    ["category"]
    .value_counts()
    .sort_index()
)

for category, count in safe_by_category.items():
    print(f"  {category:<20} {count:>5}")


# ============================================================================
# LEAKAGE RISK COUNTS
# ============================================================================

print("\n" + "-" * 70)
print("LEAKAGE RISK COUNTS")
print("-" * 70)

risk_counts = (
    feature_df["leakage_risk"]
    .value_counts()
    .sort_index()
)

for risk, count in risk_counts.items():
    print(f"  {risk:<20} {count:>5}")


# ============================================================================
# FINAL RESULT
# ============================================================================

print("\n" + "=" * 70)
print("END")
print("=" * 70)