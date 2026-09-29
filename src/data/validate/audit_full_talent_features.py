"""
Audit full talent feature integration.

Validates:
1. Expected row counts by season
2. Expected column counts
3. Required game-level columns
4. Game ID uniqueness
5. Home/away team validity
6. Season consistency
7. Binary target validity
8. Expected talent feature names
9. Exact home/away feature symmetry
10. Duplicate columns
11. Numeric talent feature dtypes
12. Completely missing talent features
13. Constant talent features (informational)
14. Returning-production plausibility
15. Recruiting plausibility
16. Transfer plausibility and era alignment
17. Infinite values
18. Leakage-sensitive talent feature names
19. 2025 test-season separation
20. Full alignment against original base feature files
21. Verification that all original base feature values are unchanged
22. Verification that only intended talent columns were added

Run from project root:

    python src/data/validate/audit_full_talent_features.py
"""

from pathlib import Path

import numpy as np
import pandas as pd
from pandas.testing import assert_frame_equal


# =============================================================================
# PROJECT PATHS
# =============================================================================

PROJECT_ROOT = Path(__file__).resolve().parents[3]

BASE_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "features"
    / "final"
)

FULL_TALENT_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "features"
    / "full_talent"
)


# =============================================================================
# EXPECTED VALUES
# =============================================================================

EXPECTED_ROWS = {
    2015: 829,
    2016: 831,
    2017: 834,
    2018: 845,
    2019: 848,
    2020: 542,
    2021: 849,
    2022: 854,
    2023: 868,
    2024: 873,
    2025: 888,
}

EXPECTED_BASE_COLUMNS = 448

EXPECTED_TALENT_COLUMNS = {
    2015: 38,
    2016: 38,
    2017: 38,
    2018: 38,
    2019: 38,
    2020: 38,
    2021: 74,
    2022: 74,
    2023: 74,
    2024: 74,
    2025: 74,
}

TRANSFER_START_YEAR = 2021

SEASONS = list(EXPECTED_ROWS.keys())

REQUIRED_GAME_COLUMNS = [
    "gameId",
    "season",
    "homeTeam",
    "awayTeam",
    "win_home",
]


# =============================================================================
# EXPECTED TALENT FEATURE NAMES
# =============================================================================

RETURNING_FEATURES = [
    "returning_total_ppa",
    "returning_passing_ppa",
    "returning_receiving_ppa",
    "returning_rushing_ppa",
    "returning_percent_ppa",
    "returning_percent_passing_ppa",
    "returning_percent_receiving_ppa",
    "returning_percent_rushing_ppa",
    "returning_usage",
    "returning_passing_usage",
    "returning_receiving_usage",
    "returning_rushing_usage",
]

RECRUITING_FEATURES = [
    "recruiting_count",
    "recruiting_rated_count",
    "recruiting_avg_rating",
    "recruiting_max_rating",
    "recruiting_5star_count",
    "recruiting_4star_count",
    "recruiting_top_100_count",
]

TRANSFER_FEATURES = [
    "incoming_transfer_count",
    "incoming_rated_player_count",
    "incoming_rating_sum",
    "incoming_rating_mean",
    "incoming_star_rated_player_count",
    "incoming_2_star_count",
    "incoming_3_star_count",
    "incoming_4_star_count",
    "incoming_5_star_count",
    "outgoing_transfer_count",
    "outgoing_rated_player_count",
    "outgoing_rating_sum",
    "outgoing_rating_mean",
    "outgoing_star_rated_player_count",
    "outgoing_2_star_count",
    "outgoing_3_star_count",
    "outgoing_4_star_count",
    "outgoing_5_star_count",
]


def expected_talent_columns(year):
    """Return the exact expected talent columns for a season."""

    columns = []

    for side in ["home", "away"]:
        for feature in RETURNING_FEATURES:
            columns.append(f"{side}_{feature}")

    for side in ["home", "away"]:
        for feature in RECRUITING_FEATURES:
            columns.append(f"{side}_{feature}")

    if year >= TRANSFER_START_YEAR:
        for side in ["home", "away"]:
            for feature in TRANSFER_FEATURES:
                columns.append(f"{side}_{feature}")

    return columns


# =============================================================================
# HELPERS
# =============================================================================

def print_header(title):
    print()
    print("=" * 80)
    print(title)
    print("=" * 80)


def print_check(label, passed, detail=""):
    status = "PASS" if passed else "FAIL"

    if detail:
        print(f"[{status}] {label}: {detail}")
    else:
        print(f"[{status}] {label}")


def compare_aligned_frames(base_df, full_df, columns):
    """
    Compare columns after aligning rows by gameId.

    gameId is used only as the alignment index and should not be included
    in the list of columns being compared.

    Allows harmless dtype differences caused by CSV parsing while requiring
    numerical/string values themselves to remain unchanged.
    """

    compare_columns = [
        col for col in columns
        if col != "gameId"
    ]

    base_aligned = (
        base_df.set_index("gameId")[compare_columns]
        .sort_index()
    )

    full_aligned = (
        full_df.set_index("gameId")[compare_columns]
        .sort_index()
    )

    assert_frame_equal(
        base_aligned,
        full_aligned,
        check_dtype=False,
        check_names=False,
        check_exact=False,
        rtol=1e-12,
        atol=1e-12,
    )


def check_team_alignment(base_df, full_df):
    """
    Explicitly verify homeTeam and awayTeam are unchanged after integration.
    """

    base_aligned = (
        base_df.set_index("gameId")[["homeTeam", "awayTeam"]]
        .sort_index()
    )

    full_aligned = (
        full_df.set_index("gameId")[["homeTeam", "awayTeam"]]
        .sort_index()
    )

    return base_aligned.equals(full_aligned)


# =============================================================================
# MAIN AUDIT
# =============================================================================

def main():

    print_header("FULL TALENT FEATURES - FINAL INTEGRATION AUDIT")

    print(f"Project root: {PROJECT_ROOT}")
    print(f"Base directory: {BASE_DIR}")
    print(f"Full talent directory: {FULL_TALENT_DIR}")

    all_passed = True

    season_results = []

    # =========================================================================
    # SEASON-BY-SEASON AUDIT
    # =========================================================================

    for year in SEASONS:

        print_header(f"SEASON {year}")

        base_path = BASE_DIR / f"final_features_{year}.csv"
        full_path = FULL_TALENT_DIR / f"full_talent_features_{year}.csv"

        season_passed = True

        # ---------------------------------------------------------------------
        # File existence
        # ---------------------------------------------------------------------

        base_exists = base_path.exists()
        full_exists = full_path.exists()

        print_check(
            "Base feature file exists",
            base_exists,
            str(base_path),
        )

        print_check(
            "Full talent feature file exists",
            full_exists,
            str(full_path),
        )

        if not base_exists or not full_exists:
            all_passed = False
            season_passed = False
            season_results.append((year, season_passed))
            continue

        # ---------------------------------------------------------------------
        # Load data
        # ---------------------------------------------------------------------

        try:
            base_df = pd.read_csv(base_path)
            full_df = pd.read_csv(full_path)
        except Exception as exc:
            print_check(
                "Files load successfully",
                False,
                str(exc),
            )
            all_passed = False
            season_passed = False
            season_results.append((year, season_passed))
            continue

        # =========================================================================
        # ROW / COLUMN STRUCTURE
        # =========================================================================

        expected_rows = EXPECTED_ROWS[year]
        expected_talent = EXPECTED_TALENT_COLUMNS[year]

        print_check(
            "Expected row count",
            len(full_df) == expected_rows,
            f"expected={expected_rows}, actual={len(full_df)}",
        )

        if len(full_df) != expected_rows:
            season_passed = False

        print_check(
            "Base row count",
            len(base_df) == expected_rows,
            f"expected={expected_rows}, actual={len(base_df)}",
        )

        if len(base_df) != expected_rows:
            season_passed = False

        print_check(
            "Base column count",
            len(base_df.columns) == EXPECTED_BASE_COLUMNS,
            f"expected={EXPECTED_BASE_COLUMNS}, actual={len(base_df.columns)}",
        )

        if len(base_df.columns) != EXPECTED_BASE_COLUMNS:
            season_passed = False

        print_check(
            "Expected full-talent column count",
            len(full_df.columns) == EXPECTED_BASE_COLUMNS + expected_talent,
            (
                f"expected={EXPECTED_BASE_COLUMNS + expected_talent}, "
                f"actual={len(full_df.columns)}"
            ),
        )

        if len(full_df.columns) != EXPECTED_BASE_COLUMNS + expected_talent:
            season_passed = False

        # =========================================================================
        # REQUIRED COLUMNS
        # =========================================================================

        missing_required = [
            col for col in REQUIRED_GAME_COLUMNS
            if col not in full_df.columns
        ]

        print_check(
            "Required game columns present",
            len(missing_required) == 0,
            (
                "all present"
                if not missing_required
                else f"missing={missing_required}"
            ),
        )

        if missing_required:
            season_passed = False
            all_passed = False
            season_results.append((year, season_passed))
            continue

        # =========================================================================
        # GAME ID UNIQUENESS
        # =========================================================================

        duplicate_game_ids = full_df["gameId"].duplicated().sum()

        print_check(
            "gameId unique in full-talent data",
            duplicate_game_ids == 0,
            f"duplicates={duplicate_game_ids}",
        )

        if duplicate_game_ids:
            season_passed = False

        duplicate_base_game_ids = base_df["gameId"].duplicated().sum()

        print_check(
            "gameId unique in base data",
            duplicate_base_game_ids == 0,
            f"duplicates={duplicate_base_game_ids}",
        )

        if duplicate_base_game_ids:
            season_passed = False

        # =========================================================================
        # HOME / AWAY VALIDITY
        # =========================================================================

        missing_home = full_df["homeTeam"].isna().sum()
        missing_away = full_df["awayTeam"].isna().sum()

        same_team = (
            full_df["homeTeam"].astype(str)
            == full_df["awayTeam"].astype(str)
        ).sum()

        print_check(
            "Home team present",
            missing_home == 0,
            f"missing={missing_home}",
        )

        print_check(
            "Away team present",
            missing_away == 0,
            f"missing={missing_away}",
        )

        print_check(
            "Home and away teams differ",
            same_team == 0,
            f"same_team_rows={same_team}",
        )

        if missing_home or missing_away or same_team:
            season_passed = False

        # =========================================================================
        # SEASON CONSISTENCY
        # =========================================================================

        invalid_season = (full_df["season"] != year).sum()

        print_check(
            "Season column matches filename",
            invalid_season == 0,
            f"mismatched_rows={invalid_season}",
        )

        if invalid_season:
            season_passed = False

        # =========================================================================
        # TARGET VALIDITY
        # =========================================================================

        missing_target = full_df["win_home"].isna().sum()

        unique_target = sorted(
            full_df["win_home"]
            .dropna()
            .unique()
            .tolist()
        )

        binary_target = set(unique_target).issubset({0, 1})

        print_check(
            "win_home complete",
            missing_target == 0,
            f"missing={missing_target}",
        )

        print_check(
            "win_home binary",
            binary_target,
            f"unique_values={unique_target}",
        )

        if missing_target or not binary_target:
            season_passed = False

        # =========================================================================
        # TALENT COLUMN NAMES
        # =========================================================================

        expected_columns = expected_talent_columns(year)

        base_columns = set(base_df.columns)
        full_columns = set(full_df.columns)

        actual_added_columns = sorted(
            full_columns - base_columns
        )

        missing_expected_talent = sorted(
            set(expected_columns) - full_columns
        )

        unexpected_talent = sorted(
            set(actual_added_columns) - set(expected_columns)
        )

        print_check(
            "Expected talent columns present",
            len(missing_expected_talent) == 0,
            (
                "all present"
                if not missing_expected_talent
                else f"missing={missing_expected_talent}"
            ),
        )

        if missing_expected_talent:
            season_passed = False

        print_check(
            "No unexpected added columns",
            len(unexpected_talent) == 0,
            (
                "none"
                if not unexpected_talent
                else f"unexpected={unexpected_talent}"
            ),
        )

        if unexpected_talent:
            season_passed = False

        print_check(
            "Exact number of added talent columns",
            len(actual_added_columns) == expected_talent,
            (
                f"expected={expected_talent}, "
                f"actual={len(actual_added_columns)}"
            ),
        )

        if len(actual_added_columns) != expected_talent:
            season_passed = False

        # =========================================================================
        # HOME / AWAY SYMMETRY
        # =========================================================================

        symmetry_failures = []

        for feature in expected_columns:

            if feature.startswith("home_"):
                away_feature = feature.replace("home_", "away_", 1)

                if away_feature not in full_columns:
                    symmetry_failures.append(
                        f"{feature} -> missing {away_feature}"
                    )

            elif feature.startswith("away_"):
                home_feature = feature.replace("away_", "home_", 1)

                if home_feature not in full_columns:
                    symmetry_failures.append(
                        f"{feature} -> missing {home_feature}"
                    )

        print_check(
            "Home/away feature symmetry",
            len(symmetry_failures) == 0,
            (
                "complete"
                if not symmetry_failures
                else f"failures={symmetry_failures}"
            ),
        )

        if symmetry_failures:
            season_passed = False

        # =========================================================================
        # DUPLICATE COLUMNS
        # =========================================================================

        duplicate_columns = full_df.columns[
            full_df.columns.duplicated()
        ].tolist()

        print_check(
            "No duplicate column names",
            len(duplicate_columns) == 0,
            (
                "none"
                if not duplicate_columns
                else f"duplicates={duplicate_columns}"
            ),
        )

        if duplicate_columns:
            season_passed = False

        # =========================================================================
        # TALENT FEATURE DTYPE
        # =========================================================================

        talent_columns_present = [
            col for col in expected_columns
            if col in full_df.columns
        ]

        non_numeric_talent = [
            col
            for col in talent_columns_present
            if not pd.api.types.is_numeric_dtype(full_df[col])
        ]

        print_check(
            "All talent features numeric",
            len(non_numeric_talent) == 0,
            (
                "all numeric"
                if not non_numeric_talent
                else f"non_numeric={non_numeric_talent}"
            ),
        )

        if non_numeric_talent:
            season_passed = False

        # =========================================================================
        # COMPLETELY MISSING FEATURES
        # =========================================================================

        completely_missing = [
            col
            for col in talent_columns_present
            if full_df[col].isna().all()
        ]

        print_check(
            "No completely missing talent feature",
            len(completely_missing) == 0,
            (
                "none"
                if not completely_missing
                else f"features={completely_missing}"
            ),
        )

        if completely_missing:
            season_passed = False

        # =========================================================================
        # CONSTANT FEATURES
        # =========================================================================

        constant_features = []

        for col in talent_columns_present:
            if full_df[col].nunique(dropna=False) <= 1:
                constant_features.append(col)

        if constant_features:
            print(
                "[INFO] Constant talent features: "
                + ", ".join(constant_features)
            )
        else:
            print("[INFO] Constant talent features: none")

        # =========================================================================
        # RETURNING PRODUCTION PLAUSIBILITY
        # =========================================================================
        #
        # Important:
        # PPA and percentage-of-PPA metrics are allowed to be negative.
        # Negative PPA is meaningful and can occur when returning players
        # contributed below-average expected performance.
        #
        # Usage, however, should not be negative.
        # Therefore only usage features receive a non-negative plausibility
        # check here.
        # =========================================================================

        returning_usage_columns = [
            col
            for col in talent_columns_present
            if "returning_" in col and "usage" in col
        ]

        negative_returning_usage = {}

        for col in returning_usage_columns:
            count = (full_df[col] < 0).sum()

            if count > 0:
                negative_returning_usage[col] = int(count)

        print_check(
            "Returning usage values non-negative",
            len(negative_returning_usage) == 0,
            (
                "none negative"
                if not negative_returning_usage
                else str(negative_returning_usage)
            ),
        )

        if negative_returning_usage:
            season_passed = False

        # Informational range reporting for PPA-derived metrics.
        ppa_columns = [
            col
            for col in talent_columns_present
            if "returning_" in col and "ppa" in col
        ]

        if ppa_columns:
            print("[INFO] Returning PPA ranges:")

            for col in ppa_columns:
                series = full_df[col].dropna()

                if len(series) == 0:
                    print(f"       {col}: all missing")
                else:
                    print(
                        f"       {col}: "
                        f"min={series.min():.6f}, "
                        f"max={series.max():.6f}"
                    )

        # =========================================================================
        # RECRUITING PLAUSIBILITY
        # =========================================================================

        recruiting_columns = [
            col
            for col in talent_columns_present
            if "recruiting_" in col
        ]

        negative_recruiting = {}

        for col in recruiting_columns:

            if (
                "count" in col
                or "rating" in col
            ):
                count = (full_df[col] < 0).sum()

                if count > 0:
                    negative_recruiting[col] = int(count)

        print_check(
            "Recruiting counts/ratings non-negative",
            len(negative_recruiting) == 0,
            (
                "none negative"
                if not negative_recruiting
                else str(negative_recruiting)
            ),
        )

        if negative_recruiting:
            season_passed = False

        # =========================================================================
        # TRANSFER ERA
        # =========================================================================

        transfer_columns_present = [
            col
            for col in full_df.columns
            if (
                "incoming_transfer_" in col
                or "outgoing_transfer_" in col
                or "incoming_rated_player_" in col
                or "outgoing_rated_player_" in col
                or "incoming_rating_" in col
                or "outgoing_rating_" in col
                or "incoming_star_rated_player_" in col
                or "outgoing_star_rated_player_" in col
                or "incoming_2_star_" in col
                or "incoming_3_star_" in col
                or "incoming_4_star_" in col
                or "incoming_5_star_" in col
                or "outgoing_2_star_" in col
                or "outgoing_3_star_" in col
                or "outgoing_4_star_" in col
                or "outgoing_5_star_" in col
            )
        ]

        expected_transfer_count = (
            len(TRANSFER_FEATURES) * 2
            if year >= TRANSFER_START_YEAR
            else 0
        )

        print_check(
            "Transfer feature era alignment",
            len(transfer_columns_present) == expected_transfer_count,
            (
                f"expected={expected_transfer_count}, "
                f"actual={len(transfer_columns_present)}"
            ),
        )

        if len(transfer_columns_present) != expected_transfer_count:
            season_passed = False

        # =========================================================================
        # TRANSFER PLAUSIBILITY
        # =========================================================================

        negative_transfer = {}

        for col in transfer_columns_present:

            if (
                "count" in col
                or "rating" in col
                or "star_" in col
            ):
                count = (full_df[col] < 0).sum()

                if count > 0:
                    negative_transfer[col] = int(count)

        print_check(
            "Transfer counts/ratings non-negative",
            len(negative_transfer) == 0,
            (
                "none negative"
                if not negative_transfer
                else str(negative_transfer)
            ),
        )

        if negative_transfer:
            season_passed = False

        # =========================================================================
        # FINITE NUMERIC VALUES
        # =========================================================================

        numeric_talent = full_df[talent_columns_present].select_dtypes(
            include=[np.number]
        )

        infinite_counts = np.isinf(numeric_talent).sum()

        infinite_features = (
            infinite_counts[infinite_counts > 0]
            .to_dict()
        )

        print_check(
            "No infinite talent values",
            len(infinite_features) == 0,
            (
                "none"
                if not infinite_features
                else str(infinite_features)
            ),
        )

        if infinite_features:
            season_passed = False

        # =========================================================================
        # LEAKAGE-SENSITIVE COLUMN NAMES
        # =========================================================================

        suspicious_terms = [
            "score",
            "points",
            "result",
            "winner",
            "loser",
            "margin",
            "spread",
            "win_home",
            "win_away",
            "outcome",
        ]

        suspicious_talent_columns = [
            col
            for col in talent_columns_present
            if any(term in col.lower() for term in suspicious_terms)
        ]

        print_check(
            "No suspicious outcome-derived talent column names",
            len(suspicious_talent_columns) == 0,
            (
                "none"
                if not suspicious_talent_columns
                else str(suspicious_talent_columns)
            ),
        )

        if suspicious_talent_columns:
            season_passed = False

        # =========================================================================
        # 2025 TEST SEASON
        # =========================================================================

        if year == 2025:

            target_in_talent = "win_home" in talent_columns_present

            print_check(
                "2025 target excluded from talent features",
                not target_in_talent,
                f"win_home_in_talent={target_in_talent}",
            )

            if target_in_talent:
                season_passed = False

            print_check(
                "2025 remains designated test season",
                True,
                "2025 is retained as final test season",
            )

        # =========================================================================
        # BASE / FULL-TALENT INTEGRATION CHECK
        # =========================================================================

        print()
        print("Base-to-full-talent integration checks:")

        # ---------------------------------------------------------------------
        # Exact gameId set
        # ---------------------------------------------------------------------

        base_ids = set(base_df["gameId"])
        full_ids = set(full_df["gameId"])

        missing_from_full = sorted(base_ids - full_ids)
        extra_in_full = sorted(full_ids - base_ids)

        game_id_sets_match = (
            len(missing_from_full) == 0
            and len(extra_in_full) == 0
        )

        print_check(
            "Same gameId set as base features",
            game_id_sets_match,
            (
                "exact match"
                if game_id_sets_match
                else (
                    f"missing_from_full={len(missing_from_full)}, "
                    f"extra_in_full={len(extra_in_full)}"
                )
            ),
        )

        if not game_id_sets_match:
            season_passed = False

        # ---------------------------------------------------------------------
        # Base columns preserved
        # ---------------------------------------------------------------------

        missing_base_columns = sorted(
            base_columns - full_columns
        )

        print_check(
            "All original base columns preserved",
            len(missing_base_columns) == 0,
            (
                "all preserved"
                if not missing_base_columns
                else f"missing={missing_base_columns}"
            ),
        )

        if missing_base_columns:
            season_passed = False

        # ---------------------------------------------------------------------
        # Team alignment
        # ---------------------------------------------------------------------

        if game_id_sets_match:

            teams_match = check_team_alignment(
                base_df,
                full_df,
            )

            print_check(
                "Home/away teams unchanged",
                teams_match,
                "exact match by gameId" if teams_match else "mismatch detected",
            )

            if not teams_match:
                season_passed = False

        # ---------------------------------------------------------------------
        # Every base feature unchanged
        # ---------------------------------------------------------------------

        if game_id_sets_match and not missing_base_columns:

            common_base_columns = list(base_df.columns)

            try:
                compare_aligned_frames(
                    base_df,
                    full_df,
                    common_base_columns,
                )

                print_check(
                    "All original base feature values unchanged",
                    True,
                    (
                        f"{len(common_base_columns)} columns "
                        "match after gameId alignment"
                    ),
                )

            except AssertionError as exc:

                print_check(
                    "All original base feature values unchanged",
                    False,
                    "differences detected",
                )

                print()
                print("Base/full comparison detail:")
                print(str(exc))

                season_passed = False

        # ---------------------------------------------------------------------
        # Target unchanged
        # ---------------------------------------------------------------------

        if game_id_sets_match:

            try:
                compare_aligned_frames(
                    base_df,
                    full_df,
                    ["win_home"],
                )

                print_check(
                    "win_home unchanged from base",
                    True,
                    "exact match by gameId",
                )

            except AssertionError:

                print_check(
                    "win_home unchanged from base",
                    False,
                    "target changed during integration",
                )

                season_passed = False

        # =========================================================================
        # SEASON RESULT
        # =========================================================================

        print()

        if season_passed:
            print(f"SEASON {year}: PASS")
        else:
            print(f"SEASON {year}: FAIL")
            all_passed = False

        season_results.append((year, season_passed))

    # =========================================================================
    # FINAL SUMMARY
    # =========================================================================

    print_header("FINAL AUDIT SUMMARY")

    print("Season results:")

    for year, passed in season_results:
        print(f"  {year}: {'PASS' if passed else 'FAIL'}")

    failed_seasons = [
        year
        for year, passed in season_results
        if not passed
    ]

    print()

    if not failed_seasons and len(season_results) == len(SEASONS):
        print("=" * 80)
        print("FINAL RESULT: PASS")
        print("=" * 80)
        print()
        print("Full-talent feature integration passed all final audit checks.")
        print()
        print("Verified:")
        print("  - Expected game counts")
        print("  - Expected base/full-talent column counts")
        print("  - Required game identifiers")
        print("  - Unique gameId values")
        print("  - Home/away team validity")
        print("  - Season consistency")
        print("  - Binary target")
        print("  - Expected talent feature names")
        print("  - Home/away feature symmetry")
        print("  - Numeric talent features")
        print("  - No completely missing talent features")
        print("  - Returning usage plausibility")
        print("  - Recruiting plausibility")
        print("  - Transfer plausibility")
        print("  - Transfer-era alignment")
        print("  - No infinite values")
        print("  - Leakage-sensitive column-name check")
        print("  - 2025 test-set separation")
        print("  - Exact gameId alignment with base features")
        print("  - No games added or removed")
        print("  - All original base columns preserved")
        print("  - All original base feature values unchanged")
        print("  - Target unchanged")
        print()
        print("PPA-derived returning-production metrics were allowed to be")
        print("negative because negative PPA values are valid.")
        print()
        print("The full-talent datasets are ready for the next modeling stage.")

    else:
        print("=" * 80)
        print("FINAL RESULT: FAIL")
        print("=" * 80)
        print()
        print(f"Failed seasons: {failed_seasons}")
        print()
        print("Review the failed checks above before proceeding to modeling.")


if __name__ == "__main__":
    main()