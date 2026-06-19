from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import pandas as pd


def standardize_column_name(column: Any) -> str:
    text = str(column).strip().lower()
    text = text.replace("°", "deg").replace("/", "_")
    text = re.sub(r"[^a-z0-9]+", "_", text)
    return re.sub(r"_+", "_", text).strip("_")


def normalize_text(value: Any, uppercase: bool = False) -> str | None:
    if pd.isna(value):
        return None
    text = str(value).strip()
    if not text or text.lower() in {"nan", "none", "null"}:
        return None
    text = re.sub(r"\s+", " ", text)
    return text.upper() if uppercase else text


def clean_rate(value: Any) -> str | None:
    text = normalize_text(value)
    if text is None:
        return None
    text = text.upper().replace(" ", "")
    text = text.replace("CHARGE", "").replace("DISCHARGE", "")
    text = text.replace("RATE", "")
    return text or None


def convert_timestamp(series: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(series, errors="coerce")
    excel_time = pd.to_datetime(numeric, unit="D", origin="1899-12-30", errors="coerce")
    parsed = pd.to_datetime(series, errors="coerce")
    return excel_time.fillna(parsed)


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def resolve_metadata_path(path: Path) -> Path:
    candidates = [path]
    if path.suffix == "":
        candidates.extend([path.with_suffix(".xlsx"), path.with_suffix(".xls")])
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"Metadata workbook not found for base path: {path}")


def resolve_csv_dir(path: Path) -> Path:
    if path.exists() and any(path.rglob("Test*.csv")):
        return path
    candidates = [path / "timeseries", path / "csv", path / "test_csvs"]
    for candidate in candidates:
        if candidate.exists() and any(candidate.rglob("Test*.csv")):
            return candidate
    return path


def import_faiss() -> Any:
    try:
        import faiss
    except ImportError as exc:
        raise RuntimeError(
            "faiss-cpu is required for semantic search. Install project dependencies first."
        ) from exc
    return faiss


def import_sentence_transformer() -> Any:
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as exc:
        raise RuntimeError(
            "sentence-transformers is required for semantic search. Install project dependencies first."
        ) from exc
    return SentenceTransformer
