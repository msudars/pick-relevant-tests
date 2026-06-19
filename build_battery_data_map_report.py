#!/usr/bin/env python3
"""Build a battery data map report from a PEC metadata workbook."""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

try:
    from openpyxl.drawing.image import Image as OpenPyxlImage
except ImportError:  # pragma: no cover - optional dependency at runtime
    OpenPyxlImage = None


DEFAULT_OUTPUT_DIR = Path("outputs/battery_data_map")
DEFAULT_METADATA_PATH = Path("data/2026_PEC_log.xlsx")
DEFAULT_CSV_DIR = Path("data/timeseries")
REQUIRED_CELL_COLUMNS = [
    "cell_id",
    "lot_number",
    "chemistry",
    "manufacturer",
    "capacity",
]
REQUIRED_TEST_COLUMNS = [
    "test_id",
    "cell_id",
    "regime",
    "temperature",
    "c_rate",
    "status",
    "timestamp",
]
NUMERIC_COLUMNS = [
    "test_id",
    "temperature",
    "capacity",
    "start_ocv",
    "version_no",
    "rack_no",
    "position_in_rack",
    "pec_shelf",
    "pec_position_no",
    "cell_thickness_mm",
    "last_test",
]
TEXT_COLUMNS = [
    "cell_id",
    "lot_number",
    "regime",
    "status",
    "chemistry",
    "manufacturer",
    "comment",
    "comment_2",
    "other_condition",
    "test_program",
    "c_rate",
    "temperature_raw",
]
CANONICAL_COLUMN_ALIASES = {
    "timestamp": [
        "date_and_time",
        "datetime",
        "date_time",
    ],
    "operator": ["operator"],
    "rack_no": ["rack_no", "rack_number"],
    "position_in_rack": ["position_in_rack"],
    "temp_chamber": ["temp_chamber", "temperature_chamber"],
    "pec_tester": ["pec_tester"],
    "pec_shelf": ["pec_shelf_a_b_1_4", "pec_shelf", "pec_shelf_ab_1_4"],
    "pec_position_no": ["pec_position_no_1_20", "pec_position_no", "pec_position"],
    "last_test": ["last_test"],
    "cell_id": ["cell_serial_no", "cell_id", "serial_number"],
    "lot_number": ["lot", "lot_number", "batch", "batch_number"],
    "test_id": ["test_no", "test_id"],
    "regime": ["test_regime", "regime"],
    "version_no": ["ver_no", "version_no"],
    "comment": ["comment"],
    "test_program": ["test_program"],
    "other_condition": ["other_condition"],
    "test_soc": ["test_soc"],
    "c_rate": ["test_rate_cha_dch", "c_rate", "test_rate"],
    "temperature_raw": ["test_temp_c", "temperature_raw", "test_temperature"],
    "cell_thickness_mm": ["cell_thickness_mm"],
    "cell_type": ["cell_type"],
    "chemistry": ["cell_chem_istry", "chemistry", "cell_chemistry"],
    "cell_type_id": ["cell_type_id"],
    "manufacturer": ["cell_manu_facturer", "manufacturer", "cell_manufacturer"],
    "capacity": ["cell_capacity_ah", "capacity", "capacity_ah"],
    "status": ["status"],
    "comment_2": ["comment_1", "comment_2"],
    "start_ocv": ["start_ocv"],
}
FORMATION_KEYWORDS = [
    "formation",
    "ffi",
]
AGING_KEYWORDS = [
    "aging",
    "ageing",
    "cyc",
    "cycle",
    "calendar",
    "storage",
]
RPT_KEYWORDS = [
    "rpt",
    "charact",
    "character",
    "hppc",
    "ocv",
    "pulse test",
    "pulse",
    "c20",
    "c10",
    "c/20",
]
FAILED_KEYWORDS = [
    "failed",
    "failure",
    "abort",
    "aborted",
    "error",
    "problem",
    "stop",
    "stopped",
]


@dataclass(frozen=True)
class ReportArtifacts:
    cleaned_metadata: pd.DataFrame
    cells_df: pd.DataFrame
    tests_df: pd.DataFrame
    lifecycle_df: pd.DataFrame
    summary_tables: Dict[str, pd.DataFrame]
    quality_tables: Dict[str, pd.DataFrame]
    ml_tables: Dict[str, pd.DataFrame]
    figures: Dict[str, Path]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a Battery Data Map Report from a PEC metadata workbook."
    )
    parser.add_argument(
        "--metadata-path",
        type=Path,
        default=DEFAULT_METADATA_PATH,
        help="Path to the PEC metadata Excel workbook.",
    )
    parser.add_argument(
        "--sheet-name",
        default=0,
        help="Sheet name or zero-based sheet index to read from the workbook.",
    )
    parser.add_argument(
        "--csv-dir",
        type=Path,
        default=DEFAULT_CSV_DIR,
        help="Directory containing TestXXXX.csv files.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory where report files will be written.",
    )
    parser.add_argument(
        "--min-tests-for-ml",
        type=int,
        default=3,
        help="Minimum number of tests per cell for aging-ready ML candidate selection.",
    )
    parser.add_argument(
        "--regime-map",
        type=Path,
        default=None,
        help="Optional JSON file with explicit regime classifications.",
    )
    parser.add_argument(
        "--interactive-regime-review",
        action="store_true",
        help="Prompt for classification of uncategorized regimes.",
    )
    return parser.parse_args()


def standardize_column_name(column: Any) -> str:
    value = str(column).strip().lower()
    value = value.replace("°", "deg")
    value = value.replace("/", "_")
    value = re.sub(r"[^a-z0-9]+", "_", value)
    value = re.sub(r"_+", "_", value).strip("_")
    return value


def normalize_text(value: Any) -> Optional[str]:
    if pd.isna(value):
        return None
    text = str(value).strip()
    if not text or text.lower() in {"nan", "none", "null"}:
        return None
    return text


def excel_sheet_arg(sheet_arg: Any) -> Any:
    if isinstance(sheet_arg, str) and sheet_arg.isdigit():
        return int(sheet_arg)
    return sheet_arg


def ensure_output_dirs(output_dir: Path) -> Dict[str, Path]:
    paths = {
        "root": output_dir,
        "tables": output_dir / "tables",
        "figures": output_dir / "figures",
        "logs": output_dir / "logs",
    }
    for path in paths.values():
        path.mkdir(parents=True, exist_ok=True)
    return paths


def read_metadata(metadata_path: Path, sheet_name: Any) -> pd.DataFrame:
    if not metadata_path.exists():
        raise FileNotFoundError(f"Metadata workbook not found: {metadata_path}")
    df = pd.read_excel(metadata_path, sheet_name=excel_sheet_arg(sheet_name))
    if df.empty:
        raise ValueError(f"Metadata workbook is empty: {metadata_path}")
    df = df.rename(columns={column: standardize_column_name(column) for column in df.columns})
    return df


def resolve_columns(df: pd.DataFrame) -> pd.DataFrame:
    rename_map: Dict[str, str] = {}
    used_targets = set()
    for target, aliases in CANONICAL_COLUMN_ALIASES.items():
        for alias in aliases:
            if alias in df.columns and target not in used_targets:
                rename_map[alias] = target
                used_targets.add(target)
                break
    df = df.rename(columns=rename_map).copy()
    for target in CANONICAL_COLUMN_ALIASES:
        if target not in df.columns:
            df[target] = pd.NA
    if "comment" in df.columns and "comment_2" in df.columns:
        overlap = df["comment"].notna() & df["comment_2"].notna()
        if overlap.any():
            df.loc[overlap, "comment_2"] = (
                df.loc[overlap, "comment"].astype(str).str.strip()
                + " | "
                + df.loc[overlap, "comment_2"].astype(str).str.strip()
            )
    return df


def convert_excel_serial_to_timestamp(series: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(series, errors="coerce")
    converted = pd.to_datetime(numeric, unit="D", origin="1899-12-30", errors="coerce")
    parsed = pd.to_datetime(series, errors="coerce")
    return converted.fillna(parsed)


def clean_metadata(df: pd.DataFrame) -> pd.DataFrame:
    df = resolve_columns(df)
    df["timestamp"] = convert_excel_serial_to_timestamp(df["timestamp"])

    for column in NUMERIC_COLUMNS:
        if column in df.columns:
            df[column] = pd.to_numeric(df[column], errors="coerce")

    for column in TEXT_COLUMNS:
        if column in df.columns:
            df[column] = df[column].map(normalize_text)

    df["cell_id"] = df["cell_id"].astype("string").str.strip()
    df["lot_number"] = df["lot_number"].astype("string").str.strip()
    df["regime"] = df["regime"].astype("string").str.strip()
    df["status"] = df["status"].astype("string").str.strip()
    df["chemistry"] = df["chemistry"].astype("string").str.strip()
    df["manufacturer"] = df["manufacturer"].astype("string").str.strip()
    df["c_rate"] = df["c_rate"].astype("string").str.strip()
    df["temperature_display"] = (
        pd.to_numeric(df["temperature_raw"], errors="coerce")
        .round(3)
        .astype("string")
        .where(pd.to_numeric(df["temperature_raw"], errors="coerce").notna(), df["temperature_raw"])
    )
    df["temperature"] = pd.to_numeric(df["temperature_raw"], errors="coerce")
    df["test_id_text"] = df["test_id"].map(
        lambda value: str(int(value)) if pd.notna(value) else pd.NA
    ).astype("string")
    df["csv_filename"] = df["test_id_text"].map(
        lambda value: f"Test{value}.csv" if pd.notna(value) else pd.NA
    )
    df["regime_normalized"] = df["regime"].map(
        lambda value: standardize_column_name(value) if pd.notna(value) else None
    )
    return df


def build_cells_df(df: pd.DataFrame) -> pd.DataFrame:
    cells = (
        df[REQUIRED_CELL_COLUMNS]
        .dropna(subset=["cell_id"])
        .sort_values(["cell_id", "lot_number"], na_position="last")
        .drop_duplicates(subset=["cell_id"], keep="first")
        .reset_index(drop=True)
    )
    return cells


def build_tests_df(df: pd.DataFrame) -> pd.DataFrame:
    columns = REQUIRED_TEST_COLUMNS + [
        "temperature_display",
        "csv_filename",
        "comment",
        "comment_2",
        "other_condition",
        "test_program",
        "lot_number",
        "chemistry",
        "manufacturer",
        "capacity",
    ]
    tests = df[columns].copy()
    return tests.sort_values(["timestamp", "test_id"], na_position="last").reset_index(drop=True)


def load_regime_map(path: Optional[Path]) -> Dict[str, Dict[str, bool]]:
    if path is None or not path.exists():
        return {}
    data = json.loads(path.read_text(encoding="utf-8"))
    normalized: Dict[str, Dict[str, bool]] = {}
    for regime, flags in data.items():
        normalized[regime.lower()] = {
            "is_formation": bool(flags.get("is_formation", False)),
            "is_aging": bool(flags.get("is_aging", False)),
            "is_rpt": bool(flags.get("is_rpt", False)),
        }
    return normalized


def classify_regime(
    regime: Optional[str],
    explicit_map: Dict[str, Dict[str, bool]],
) -> Dict[str, bool]:
    if pd.isna(regime):
        return {"is_formation": False, "is_aging": False, "is_rpt": False}

    key = str(regime).lower().strip()
    if not key:
        return {"is_formation": False, "is_aging": False, "is_rpt": False}
    if key in explicit_map:
        return explicit_map[key]

    lowered = key
    flags = {
        "is_formation": any(keyword in lowered for keyword in FORMATION_KEYWORDS),
        "is_aging": any(keyword in lowered for keyword in AGING_KEYWORDS),
        "is_rpt": any(keyword in lowered for keyword in RPT_KEYWORDS),
    }

    if "char" in lowered and not flags["is_aging"]:
        flags["is_rpt"] = True
    if "hppc" in lowered:
        flags["is_rpt"] = True
    if "cyc" in lowered and not flags["is_rpt"]:
        flags["is_aging"] = True
    return flags


def interactive_regime_review(
    df: pd.DataFrame,
    explicit_map: Dict[str, Dict[str, bool]],
    regime_map_path: Optional[Path],
) -> Dict[str, Dict[str, bool]]:
    unknown_regimes = sorted(
        {
            regime
            for regime, is_unknown in zip(df["regime"], df["is_unknown_regime"])
            if is_unknown and pd.notna(regime)
        }
    )
    if not unknown_regimes:
        return explicit_map

    print("\nUncategorized regimes detected. Enter comma-separated flags from {formation,aging,rpt}.")
    print("Press Enter to keep a regime uncategorized.\n")
    for regime in unknown_regimes:
        response = input(f"{regime}: ").strip().lower()
        labels = {token.strip() for token in response.split(",") if token.strip()}
        explicit_map[regime.lower()] = {
            "is_formation": "formation" in labels,
            "is_aging": "aging" in labels,
            "is_rpt": "rpt" in labels,
        }

    if regime_map_path is not None:
        serializable = {key: value for key, value in sorted(explicit_map.items())}
        regime_map_path.parent.mkdir(parents=True, exist_ok=True)
        regime_map_path.write_text(json.dumps(serializable, indent=2), encoding="utf-8")
    return explicit_map


def apply_regime_classification(
    df: pd.DataFrame,
    explicit_map: Dict[str, Dict[str, bool]],
) -> pd.DataFrame:
    df = df.copy()
    classified = df["regime"].map(lambda value: classify_regime(value, explicit_map))
    flags_df = pd.DataFrame(classified.tolist(), index=df.index)
    for column in ["is_formation", "is_aging", "is_rpt"]:
        df[column] = flags_df[column].fillna(False).astype(bool)
    df["is_unknown_regime"] = (
        df["regime"].notna()
        & ~(df["is_formation"] | df["is_aging"] | df["is_rpt"])
    )
    return df


def format_distribution(series: pd.Series, name: str) -> pd.DataFrame:
    dist = series.fillna("missing").astype("string").value_counts(dropna=False).rename_axis(name)
    return dist.reset_index(name="count")


def build_global_summary(cells_df: pd.DataFrame, tests_df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    overview = pd.DataFrame(
        {
            "metric": ["number_of_cells", "number_of_tests", "number_of_lots"],
            "value": [
                int(cells_df["cell_id"].nunique(dropna=True)),
                int(tests_df.shape[0]),
                int(cells_df["lot_number"].nunique(dropna=True)),
            ],
        }
    )
    return {
        "global_overview": overview,
        "chemistry_distribution": format_distribution(cells_df["chemistry"], "chemistry"),
        "temperature_distribution": format_distribution(
            tests_df["temperature_display"], "temperature"
        ),
        "c_rate_distribution": format_distribution(tests_df["c_rate"], "c_rate"),
    }


def build_test_coverage_summary(tests_df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    tests_per_cell = (
        tests_df.dropna(subset=["cell_id"])
        .groupby("cell_id", dropna=True)
        .size()
        .rename("test_count")
        .reset_index()
        .sort_values("test_count", ascending=False)
    )
    distribution = (
        tests_per_cell["test_count"]
        .value_counts()
        .sort_index()
        .rename_axis("tests_per_cell")
        .reset_index(name="cell_count")
    )
    tests_per_lot = (
        tests_df.groupby("lot_number", dropna=False)
        .size()
        .rename("test_count")
        .reset_index()
        .sort_values("test_count", ascending=False)
    )
    temp_vs_rate = pd.crosstab(
        tests_df["temperature_display"].fillna("missing"),
        tests_df["c_rate"].fillna("missing"),
    )
    temp_vs_rate.index.name = "temperature"
    temp_vs_rate.columns.name = "c_rate"
    return {
        "tests_per_cell": tests_per_cell,
        "tests_per_cell_distribution": distribution,
        "tests_per_lot": tests_per_lot,
        "temperature_vs_c_rate_counts": temp_vs_rate.reset_index(),
    }


def build_regime_summary(df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    counts = pd.DataFrame(
        {
            "regime_flag": ["is_formation", "is_aging", "is_rpt", "is_unknown_regime"],
            "count": [
                int(df["is_formation"].sum()),
                int(df["is_aging"].sum()),
                int(df["is_rpt"].sum()),
                int(df["is_unknown_regime"].sum()),
            ],
        }
    )

    combo_counts = (
        df.assign(
            regime_combination=df.apply(
                lambda row: "+".join(
                    [
                        label
                        for label, column in [
                            ("formation", "is_formation"),
                            ("aging", "is_aging"),
                            ("rpt", "is_rpt"),
                        ]
                        if bool(row[column])
                    ]
                )
                or "unclassified",
                axis=1,
            )
        )["regime_combination"]
        .value_counts()
        .rename_axis("regime_combination")
        .reset_index(name="count")
    )

    unknown_regimes = (
        df.loc[df["is_unknown_regime"], "regime"]
        .dropna()
        .value_counts()
        .rename_axis("regime")
        .reset_index(name="count")
    )
    return {
        "regime_flag_counts": counts,
        "regime_combinations": combo_counts,
        "unknown_regimes": unknown_regimes,
    }


def classify_lifecycle(row: pd.Series) -> str:
    formation = bool(row["has_formation"])
    aging = bool(row["has_aging"])
    rpt = bool(row["has_rpt"])
    if formation and aging and rpt:
        return "full_lifecycle"
    if aging and not formation and not rpt:
        return "aging_only"
    if formation and not aging and not rpt:
        return "formation_only"
    if formation or aging or rpt:
        return "partial"
    return "unknown"


def build_lifecycle_summary(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    grouped = (
        df.dropna(subset=["cell_id"])
        .groupby("cell_id", dropna=True)
        .agg(
            lot_number=("lot_number", "first"),
            chemistry=("chemistry", "first"),
            manufacturer=("manufacturer", "first"),
            capacity=("capacity", "first"),
            test_count=("test_id", "size"),
            has_formation=("is_formation", "max"),
            has_aging=("is_aging", "max"),
            has_rpt=("is_rpt", "max"),
            regimes=("regime", lambda values: " | ".join(sorted({str(v) for v in values if pd.notna(v)}))),
        )
        .reset_index()
    )
    grouped["lifecycle_category"] = grouped.apply(classify_lifecycle, axis=1)
    summary = (
        grouped["lifecycle_category"]
        .value_counts()
        .rename_axis("lifecycle_category")
        .reset_index(name="cell_count")
    )
    return grouped, summary


def find_csv_path(csv_dir: Path, test_id: Any) -> Optional[Path]:
    if pd.isna(test_id):
        return None
    try:
        test_number = int(test_id)
    except (TypeError, ValueError):
        return None
    candidate = csv_dir / f"Test{test_number}.csv"
    return candidate if candidate.exists() else None


def build_quality_checks(df: pd.DataFrame, csv_dir: Path) -> Dict[str, pd.DataFrame]:
    duplicate_mask = df["test_id"].notna() & df["test_id"].duplicated(keep=False)
    csv_lookup = df["test_id"].map(lambda value: find_csv_path(csv_dir, value))
    missing_csv_mask = df["test_id"].notna() & csv_lookup.isna()
    failure_text = (
        df[["status", "comment", "comment_2", "other_condition"]]
        .fillna("")
        .astype(str)
        .agg(" | ".join, axis=1)
        .str.lower()
    )
    failed_mask = failure_text.map(lambda value: any(keyword in value for keyword in FAILED_KEYWORDS))

    return {
        "missing_test_id": df.loc[df["test_id"].isna(), REQUIRED_TEST_COLUMNS + ["lot_number", "regime"]],
        "duplicate_test_id": df.loc[duplicate_mask, REQUIRED_TEST_COLUMNS + ["lot_number", "regime"]],
        "missing_cell_id": df.loc[df["cell_id"].isna(), REQUIRED_TEST_COLUMNS + ["lot_number", "regime"]],
        "tests_without_csv": df.loc[missing_csv_mask, REQUIRED_TEST_COLUMNS + ["csv_filename", "lot_number"]],
        "failed_or_aborted_tests": df.loc[
            failed_mask, REQUIRED_TEST_COLUMNS + ["comment", "comment_2", "other_condition"]
        ],
    }


def build_ml_readiness_tables(df: pd.DataFrame, min_tests: int) -> Dict[str, pd.DataFrame]:
    per_cell = (
        df.dropna(subset=["cell_id"])
        .groupby("cell_id", dropna=True)
        .agg(
            lot_number=("lot_number", "first"),
            chemistry=("chemistry", "first"),
            aging_test_count=("is_aging", "sum"),
            rpt_test_count=("is_rpt", "sum"),
            formation_test_count=("is_formation", "sum"),
            total_test_count=("test_id", "size"),
            valid_temperature_count=("temperature", lambda values: values.notna().sum()),
            valid_c_rate_count=("c_rate", lambda values: values.notna().sum()),
        )
        .reset_index()
    )

    aging_ready = per_cell.loc[
        (per_cell["aging_test_count"] > 0)
        & (per_cell["total_test_count"] >= min_tests)
        & (per_cell["valid_temperature_count"] > 0)
        & (per_cell["valid_c_rate_count"] > 0)
    ].sort_values("total_test_count", ascending=False)

    rpt_ready = per_cell.loc[per_cell["rpt_test_count"] > 0].sort_values(
        "rpt_test_count", ascending=False
    )
    formation_aging_ready = per_cell.loc[
        (per_cell["formation_test_count"] > 0) & (per_cell["aging_test_count"] > 0)
    ].sort_values(["formation_test_count", "aging_test_count"], ascending=False)

    counts = pd.DataFrame(
        {
            "candidate_group": [
                "aging_with_min_tests_and_valid_temp_rate",
                "cells_with_rpt_tests",
                "cells_with_formation_and_aging",
            ],
            "cell_count": [
                int(aging_ready["cell_id"].nunique()),
                int(rpt_ready["cell_id"].nunique()),
                int(formation_aging_ready["cell_id"].nunique()),
            ],
        }
    )

    return {
        "ml_candidate_counts": counts,
        "aging_ready_cells": aging_ready,
        "rpt_ready_cells": rpt_ready,
        "formation_aging_cells": formation_aging_ready,
    }


def save_table(df: pd.DataFrame, path: Path) -> None:
    if path.suffix == ".csv":
        df.to_csv(path, index=False)
    elif path.suffix == ".parquet":
        parquet_df = prepare_dataframe_for_parquet(df)
        parquet_df.to_parquet(path, index=False)
    else:
        raise ValueError(f"Unsupported table extension: {path.suffix}")


def prepare_dataframe_for_parquet(df: pd.DataFrame) -> pd.DataFrame:
    parquet_df = df.copy()
    for column in parquet_df.columns:
        series = parquet_df[column]
        if pd.api.types.is_object_dtype(series):
            non_null = series.dropna()
            python_types = {type(value) for value in non_null}
            if len(python_types) > 1:
                parquet_df[column] = series.map(
                    lambda value: str(value) if pd.notna(value) else pd.NA
                ).astype("string")
    return parquet_df


def create_visualizations(
    tests_df: pd.DataFrame,
    cells_df: pd.DataFrame,
    coverage_tables: Dict[str, pd.DataFrame],
    lifecycle_summary: pd.DataFrame,
    figure_dir: Path,
) -> Dict[str, Path]:
    sns.set_theme(style="whitegrid")
    figure_paths: Dict[str, Path] = {}

    tests_per_cell = coverage_tables["tests_per_cell"]
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.hist(tests_per_cell["test_count"], bins=min(40, max(10, tests_per_cell["test_count"].nunique())))
    ax.set_title("Tests per Cell")
    ax.set_xlabel("Tests per Cell")
    ax.set_ylabel("Cell Count")
    fig.tight_layout()
    figure_paths["tests_per_cell_histogram"] = figure_dir / "tests_per_cell_histogram.png"
    fig.savefig(figure_paths["tests_per_cell_histogram"], dpi=150)
    plt.close(fig)

    chemistry_dist = (
        cells_df["chemistry"]
        .fillna("missing")
        .value_counts()
        .sort_values(ascending=False)
        .head(20)
    )
    fig, ax = plt.subplots(figsize=(12, 6))
    chemistry_dist.plot(kind="bar", ax=ax, color="#3B6FB6")
    ax.set_title("Chemistry Distribution")
    ax.set_xlabel("Chemistry")
    ax.set_ylabel("Cell Count")
    fig.tight_layout()
    figure_paths["chemistry_distribution"] = figure_dir / "chemistry_distribution.png"
    fig.savefig(figure_paths["chemistry_distribution"], dpi=150)
    plt.close(fig)

    heatmap_frame = pd.crosstab(
        tests_df["temperature_display"].fillna("missing"),
        tests_df["c_rate"].fillna("missing"),
    )
    fig, ax = plt.subplots(figsize=(14, 8))
    sns.heatmap(heatmap_frame, cmap="YlGnBu", ax=ax)
    ax.set_title("Temperature vs C-rate")
    fig.tight_layout()
    figure_paths["temperature_vs_c_rate_heatmap"] = figure_dir / "temperature_vs_c_rate_heatmap.png"
    fig.savefig(figure_paths["temperature_vs_c_rate_heatmap"], dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 6))
    lifecycle_summary.set_index("lifecycle_category")["cell_count"].plot(
        kind="bar",
        ax=ax,
        color="#C26D3F",
    )
    ax.set_title("Lifecycle Category Distribution")
    ax.set_xlabel("Lifecycle Category")
    ax.set_ylabel("Cell Count")
    fig.tight_layout()
    figure_paths["lifecycle_category_distribution"] = (
        figure_dir / "lifecycle_category_distribution.png"
    )
    fig.savefig(figure_paths["lifecycle_category_distribution"], dpi=150)
    plt.close(fig)

    return figure_paths


def write_excel_report(
    report_path: Path,
    artifacts: ReportArtifacts,
    coverage_tables: Dict[str, pd.DataFrame],
    lifecycle_summary: pd.DataFrame,
) -> None:
    with pd.ExcelWriter(report_path, engine="openpyxl") as writer:
        artifacts.summary_tables["global_overview"].to_excel(
            writer, sheet_name="global_summary", index=False
        )
        artifacts.summary_tables["chemistry_distribution"].to_excel(
            writer, sheet_name="global_summary", index=False, startrow=6
        )
        artifacts.summary_tables["temperature_distribution"].to_excel(
            writer, sheet_name="global_summary", index=False, startrow=6, startcol=4
        )
        artifacts.summary_tables["c_rate_distribution"].to_excel(
            writer, sheet_name="global_summary", index=False, startrow=6, startcol=8
        )
        coverage_tables["tests_per_cell"].to_excel(writer, sheet_name="test_coverage", index=False)
        coverage_tables["tests_per_lot"].to_excel(
            writer, sheet_name="test_coverage", index=False, startcol=4
        )
        coverage_tables["temperature_vs_c_rate_counts"].to_excel(
            writer, sheet_name="temp_vs_rate", index=False
        )
        artifacts.summary_tables["regime_flag_counts"].to_excel(
            writer, sheet_name="regime_summary", index=False
        )
        artifacts.summary_tables["regime_combinations"].to_excel(
            writer, sheet_name="regime_summary", index=False, startcol=4
        )
        artifacts.lifecycle_df.to_excel(writer, sheet_name="lifecycle", index=False)
        lifecycle_summary.to_excel(writer, sheet_name="lifecycle", index=False, startcol=10)
        artifacts.ml_tables["ml_candidate_counts"].to_excel(
            writer, sheet_name="ml_readiness", index=False
        )
        artifacts.ml_tables["aging_ready_cells"].to_excel(
            writer, sheet_name="ml_readiness", index=False, startrow=6
        )
        artifacts.quality_tables["tests_without_csv"].to_excel(
            writer, sheet_name="quality_checks", index=False
        )
        artifacts.quality_tables["failed_or_aborted_tests"].to_excel(
            writer, sheet_name="quality_checks", index=False, startrow=6, startcol=8
        )
        workbook = writer.book
        if OpenPyxlImage is not None:
            visual_sheet = workbook.create_sheet("visualizations")
            anchors = ["A1", "A28", "J1", "J28"]
            for anchor, image_path in zip(anchors, artifacts.figures.values()):
                visual_sheet.add_image(OpenPyxlImage(str(image_path)), anchor)


def print_report(
    summary_tables: Dict[str, pd.DataFrame],
    coverage_tables: Dict[str, pd.DataFrame],
    regime_tables: Dict[str, pd.DataFrame],
    lifecycle_summary: pd.DataFrame,
    quality_tables: Dict[str, pd.DataFrame],
    ml_tables: Dict[str, pd.DataFrame],
) -> None:
    print("\nGLOBAL SUMMARY")
    print(summary_tables["global_overview"].to_string(index=False))
    print("\nChemistry distribution")
    print(summary_tables["chemistry_distribution"].head(20).to_string(index=False))
    print("\nTemperature distribution")
    print(summary_tables["temperature_distribution"].head(20).to_string(index=False))
    print("\nC-rate distribution")
    print(summary_tables["c_rate_distribution"].head(20).to_string(index=False))

    print("\nTEST COVERAGE")
    print(coverage_tables["tests_per_cell_distribution"].head(20).to_string(index=False))
    print("\nTests per lot")
    print(coverage_tables["tests_per_lot"].head(20).to_string(index=False))

    print("\nREGIME CLASSIFICATION")
    print(regime_tables["regime_flag_counts"].to_string(index=False))
    print("\nRegime combinations")
    print(regime_tables["regime_combinations"].to_string(index=False))

    print("\nLIFECYCLE SUMMARY")
    print(lifecycle_summary.to_string(index=False))

    print("\nDATA QUALITY")
    quality_counts = pd.DataFrame(
        {
            "check": list(quality_tables.keys()),
            "count": [len(table) for table in quality_tables.values()],
        }
    )
    print(quality_counts.to_string(index=False))

    print("\nML READINESS")
    print(ml_tables["ml_candidate_counts"].to_string(index=False))


def main() -> None:
    args = parse_args()
    paths = ensure_output_dirs(args.output_dir)

    metadata = read_metadata(args.metadata_path, args.sheet_name)
    cleaned_metadata = clean_metadata(metadata)

    explicit_map = load_regime_map(args.regime_map)
    classified = apply_regime_classification(cleaned_metadata, explicit_map)
    if args.interactive_regime_review:
        explicit_map = interactive_regime_review(classified, explicit_map, args.regime_map)
        classified = apply_regime_classification(cleaned_metadata, explicit_map)

    cells_df = build_cells_df(classified)
    tests_df = build_tests_df(classified)

    global_summary = build_global_summary(cells_df, tests_df)
    coverage_tables = build_test_coverage_summary(tests_df)
    regime_tables = build_regime_summary(classified)
    lifecycle_df, lifecycle_summary = build_lifecycle_summary(classified)
    quality_tables = build_quality_checks(classified, args.csv_dir)
    ml_tables = build_ml_readiness_tables(classified, args.min_tests_for_ml)
    figures = create_visualizations(
        tests_df=tests_df,
        cells_df=cells_df,
        coverage_tables=coverage_tables,
        lifecycle_summary=lifecycle_summary,
        figure_dir=paths["figures"],
    )

    all_summary_tables = {
        **global_summary,
        **coverage_tables,
        **regime_tables,
        "lifecycle_summary": lifecycle_summary,
    }
    artifacts = ReportArtifacts(
        cleaned_metadata=classified,
        cells_df=cells_df,
        tests_df=tests_df,
        lifecycle_df=lifecycle_df,
        summary_tables=all_summary_tables,
        quality_tables=quality_tables,
        ml_tables=ml_tables,
        figures=figures,
    )

    save_table(artifacts.cleaned_metadata, paths["tables"] / "cleaned_metadata.parquet")
    save_table(artifacts.cells_df, paths["tables"] / "cells_df.parquet")
    save_table(artifacts.tests_df, paths["tables"] / "tests_df.parquet")
    save_table(artifacts.lifecycle_df, paths["tables"] / "lifecycle_classification.parquet")

    for name, table in all_summary_tables.items():
        save_table(table, paths["tables"] / f"{name}.csv")
    for name, table in quality_tables.items():
        save_table(table, paths["tables"] / f"{name}.csv")
    for name, table in ml_tables.items():
        save_table(table, paths["tables"] / f"{name}.csv")

    report_path = paths["root"] / "battery_data_map_report.xlsx"
    write_excel_report(report_path, artifacts, coverage_tables, lifecycle_summary)
    print_report(
        summary_tables=global_summary,
        coverage_tables=coverage_tables,
        regime_tables=regime_tables,
        lifecycle_summary=lifecycle_summary,
        quality_tables=quality_tables,
        ml_tables=ml_tables,
    )
    print(f"\nReport written to: {report_path}")


if __name__ == "__main__":
    main()
