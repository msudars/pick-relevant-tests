from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable
from urllib import error, request

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from .config import (
    AGING_KEYWORDS,
    CANONICAL_ALIASES,
    DEFAULT_CSV_DIR,
    DEFAULT_METADATA_PATH,
    DEFAULT_MODEL_NAME,
    DEFAULT_OUTPUT_DIR,
    DEFAULT_SHEET_NAME,
    FORMATION_KEYWORDS,
    RPT_KEYWORDS,
    TEXT_COLUMNS,
)
from .utils import (
    clean_rate,
    convert_timestamp,
    ensure_dir,
    import_faiss,
    import_sentence_transformer,
    normalize_text,
    resolve_csv_dir,
    resolve_metadata_path,
    standardize_column_name,
)


@dataclass
class SemanticArtifacts:
    index: Any
    texts: list[str]
    test_ids: list[int]
    model_name: str


@dataclass
class BatteryDataSystem:
    metadata_path: Path = DEFAULT_METADATA_PATH
    csv_dir: Path = DEFAULT_CSV_DIR
    output_dir: Path = DEFAULT_OUTPUT_DIR
    sheet_name: str | int = DEFAULT_SHEET_NAME
    model_name: str = DEFAULT_MODEL_NAME
    cleaned_df: pd.DataFrame | None = None
    cells_df: pd.DataFrame | None = None
    tests_df: pd.DataFrame | None = None
    lifecycle_df: pd.DataFrame | None = None
    semantic: SemanticArtifacts | None = None
    _model: Any = field(default=None, init=False, repr=False)
    _csv_lookup: dict[int, Path] | None = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        self.metadata_path = Path(self.metadata_path)
        self.csv_dir = Path(self.csv_dir)
        self.output_dir = Path(self.output_dir)

    def load_and_clean_data(self) -> pd.DataFrame:
        metadata_path = resolve_metadata_path(self.metadata_path)
        raw_df = pd.read_excel(metadata_path, sheet_name=self.sheet_name)
        if raw_df.empty:
            raise ValueError(f"Metadata workbook is empty: {metadata_path}")

        raw_df = raw_df.rename(columns={c: standardize_column_name(c) for c in raw_df.columns})
        df = raw_df.copy()

        rename_map: dict[str, str] = {}
        used_source_columns: set[str] = set()
        for target, aliases in CANONICAL_ALIASES.items():
            for alias in aliases:
                alias_name = standardize_column_name(alias)
                if alias_name in df.columns and alias_name not in used_source_columns:
                    rename_map[alias_name] = target
                    used_source_columns.add(alias_name)
                    break
        df = df.rename(columns=rename_map)

        for target in CANONICAL_ALIASES:
            if target not in df.columns:
                df[target] = pd.NA

        comment_like_columns = [
            column
            for column in raw_df.columns
            if column.startswith("comment") and column not in {"comments"}
        ]
        if comment_like_columns:
            df["comments"] = (
                raw_df[comment_like_columns]
                .apply(lambda column: column.map(normalize_text))
                .apply(
                    lambda row: " | ".join(
                        [str(value) for value in row if pd.notna(value) and value is not None]
                    ),
                    axis=1,
                )
                .replace("", pd.NA)
            )

        df["test_id"] = pd.to_numeric(df["test_id"], errors="coerce").astype("Int64")
        df["temperature"] = pd.to_numeric(df["temperature"], errors="coerce")
        df["capacity"] = pd.to_numeric(df["capacity"], errors="coerce")
        df["timestamp"] = convert_timestamp(df["timestamp"])

        for column in TEXT_COLUMNS:
            if column not in df.columns:
                continue
            upper = column in {"cell_id", "lot_number", "chemistry", "manufacturer", "status"}
            df[column] = df[column].map(lambda value: normalize_text(value, uppercase=upper))

        df["c_rate"] = df["c_rate"].map(clean_rate)
        df["regime"] = df["regime"].map(normalize_text)
        df["other_condition"] = df["other_condition"].map(normalize_text)
        df["comments"] = df["comments"].map(normalize_text)
        df["combined_text"] = (
            df[["regime", "comments", "other_condition", "status", "test_program"]]
            .fillna("")
            .astype(str)
            .agg(" | ".join, axis=1)
            .str.replace(r"(?:\s*\|\s*)+", " | ", regex=True)
            .str.strip(" |")
        ).replace("", pd.NA)
        df["csv_filename"] = df["test_id"].map(
            lambda value: f"Test{int(value)}.csv" if pd.notna(value) else pd.NA
        )
        df["csv_path"] = df["test_id"].map(self.find_csv_path)

        flags = df["regime"].map(self.classify_regime).apply(pd.Series)
        df[["is_formation", "is_aging", "is_rpt"]] = flags.fillna(False).astype(bool)
        self.cleaned_df = df.sort_values(["cell_id", "test_id"], na_position="last").reset_index(
            drop=True
        )
        return self.cleaned_df

    def build_relational_tables(self) -> tuple[pd.DataFrame, pd.DataFrame]:
        if self.cleaned_df is None:
            self.load_and_clean_data()

        assert self.cleaned_df is not None
        self.cells_df = (
            self.cleaned_df[
                ["cell_id", "lot_number", "chemistry", "manufacturer", "capacity", "cell_type_id"]
            ]
            .dropna(subset=["cell_id"])
            .sort_values(["cell_id", "lot_number"], na_position="last")
            .drop_duplicates(subset=["cell_id"], keep="first")
            .reset_index(drop=True)
        )
        self.tests_df = (
            self.cleaned_df[
                [
                    "test_id",
                    "cell_id",
                    "lot_number",
                    "regime",
                    "temperature",
                    "c_rate",
                    "status",
                    "comments",
                    "timestamp",
                    "chemistry",
                    "manufacturer",
                    "capacity",
                    "test_program",
                    "other_condition",
                    "test_rate",
                    "csv_filename",
                    "csv_path",
                    "combined_text",
                    "is_formation",
                    "is_aging",
                    "is_rpt",
                ]
            ]
            .sort_values(["cell_id", "test_id", "timestamp"], na_position="last")
            .reset_index(drop=True)
        )
        return self.cells_df, self.tests_df

    def build_data_map_report(self) -> dict[str, pd.DataFrame]:
        if self.tests_df is None or self.cells_df is None:
            self.build_relational_tables()

        assert self.cleaned_df is not None
        assert self.cells_df is not None
        assert self.tests_df is not None

        tests_per_cell = (
            self.tests_df.dropna(subset=["cell_id"])
            .groupby("cell_id", dropna=True)
            .size()
            .rename("test_count")
            .reset_index()
        )
        self.lifecycle_df = (
            self.cleaned_df.dropna(subset=["cell_id"])
            .groupby("cell_id", dropna=True)
            .agg(
                lot_number=("lot_number", "first"),
                chemistry=("chemistry", "first"),
                manufacturer=("manufacturer", "first"),
                capacity=("capacity", "first"),
                test_count=("test_id", "size"),
                regimes=(
                    "regime",
                    lambda values: " | ".join(sorted({str(v) for v in values if pd.notna(v)})),
                ),
                has_formation=("is_formation", "max"),
                has_aging=("is_aging", "max"),
                has_rpt=("is_rpt", "max"),
            )
            .reset_index()
        )
        self.lifecycle_df["lifecycle_category"] = self.lifecycle_df.apply(
            self.classify_lifecycle,
            axis=1,
        )

        return {
            "global_summary": pd.DataFrame(
                {
                    "metric": ["number_of_cells", "number_of_tests", "number_of_lots"],
                    "value": [
                        int(self.cells_df["cell_id"].nunique(dropna=True)),
                        int(self.tests_df.shape[0]),
                        int(self.cells_df["lot_number"].nunique(dropna=True)),
                    ],
                }
            ),
            "chemistry_distribution": self.cells_df["chemistry"]
            .fillna("MISSING")
            .value_counts(dropna=False)
            .rename_axis("chemistry")
            .reset_index(name="count"),
            "temperature_distribution": self.tests_df["temperature"]
            .round(3)
            .astype("string")
            .fillna("MISSING")
            .value_counts(dropna=False)
            .rename_axis("temperature")
            .reset_index(name="count"),
            "c_rate_distribution": self.tests_df["c_rate"]
            .fillna("MISSING")
            .value_counts(dropna=False)
            .rename_axis("c_rate")
            .reset_index(name="count"),
            "tests_per_cell": tests_per_cell.sort_values("test_count", ascending=False),
            "regime_summary": pd.DataFrame(
                {
                    "regime_flag": ["is_formation", "is_aging", "is_rpt"],
                    "count": [
                        int(self.tests_df["is_formation"].sum()),
                        int(self.tests_df["is_aging"].sum()),
                        int(self.tests_df["is_rpt"].sum()),
                    ],
                }
            ),
            "lifecycle_df": self.lifecycle_df,
            "lifecycle_distribution": self.lifecycle_df["lifecycle_category"]
            .value_counts(dropna=False)
            .rename_axis("lifecycle_category")
            .reset_index(name="cell_count"),
            "relevant_cells": pd.DataFrame(
                {
                    "metric": [
                        "cells_with_ge_3_tests",
                        "cells_with_aging_tests",
                        "cells_with_rpt_tests",
                        "cells_with_formation_and_aging",
                    ],
                    "count": [
                        int((tests_per_cell["test_count"] >= 3).sum()),
                        int(self.lifecycle_df["has_aging"].sum()),
                        int(self.lifecycle_df["has_rpt"].sum()),
                        int(
                            (self.lifecycle_df["has_formation"] & self.lifecycle_df["has_aging"]).sum()
                        ),
                    ],
                }
            ),
            "temperature_vs_c_rate": pd.crosstab(
                self.tests_df["temperature"].round(3).astype("string").fillna("MISSING"),
                self.tests_df["c_rate"].fillna("MISSING"),
            ).reset_index(),
        }

    def save_outputs(self) -> dict[str, Path]:
        if self.tests_df is None or self.cells_df is None:
            self.build_relational_tables()
        report = self.build_data_map_report()

        tables_dir = ensure_dir(self.output_dir / "tables")
        figures_dir = ensure_dir(self.output_dir / "figures")
        semantic_dir = ensure_dir(self.output_dir / "semantic")

        assert self.cleaned_df is not None
        assert self.cells_df is not None
        assert self.tests_df is not None
        assert self.lifecycle_df is not None

        self.prepare_dataframe_for_parquet(self.cleaned_df).to_parquet(
            tables_dir / "cleaned_metadata.parquet",
            index=False,
        )
        self.prepare_dataframe_for_parquet(self.cells_df).to_parquet(
            tables_dir / "cells_df.parquet",
            index=False,
        )
        self.prepare_dataframe_for_parquet(self.tests_df).to_parquet(
            tables_dir / "tests_df.parquet",
            index=False,
        )
        self.prepare_dataframe_for_parquet(self.lifecycle_df).to_parquet(
            tables_dir / "lifecycle_df.parquet",
            index=False,
        )

        for name, table in report.items():
            if isinstance(table, pd.DataFrame):
                table.to_csv(tables_dir / f"{name}.csv", index=False)

        figure_paths = self.create_visualizations(figures_dir)
        if self.semantic is not None:
            faiss = import_faiss()
            faiss.write_index(self.semantic.index, str(semantic_dir / "tests.faiss"))
            pd.DataFrame(
                {
                    "test_id": self.semantic.test_ids,
                    "combined_text": self.semantic.texts,
                }
            ).to_parquet(semantic_dir / "semantic_metadata.parquet", index=False)
            (semantic_dir / "model_name.txt").write_text(self.semantic.model_name, encoding="utf-8")

        return {**figure_paths, "tables_dir": tables_dir, "semantic_dir": semantic_dir}

    def create_visualizations(self, figures_dir: Path) -> dict[str, Path]:
        if self.tests_df is None or self.cells_df is None or self.lifecycle_df is None:
            self.build_data_map_report()

        assert self.tests_df is not None
        assert self.cells_df is not None
        assert self.lifecycle_df is not None

        sns.set_theme(style="whitegrid")
        outputs: dict[str, Path] = {}

        tests_per_cell = (
            self.tests_df.dropna(subset=["cell_id"])
            .groupby("cell_id")
            .size()
            .rename("test_count")
        )
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.hist(tests_per_cell.values, bins=min(40, max(10, tests_per_cell.nunique() or 10)))
        ax.set_title("Tests per Cell")
        ax.set_xlabel("Tests")
        ax.set_ylabel("Cells")
        fig.tight_layout()
        outputs["tests_per_cell_histogram"] = figures_dir / "tests_per_cell_histogram.png"
        fig.savefig(outputs["tests_per_cell_histogram"], dpi=150)
        plt.close(fig)

        chemistry_counts = self.cells_df["chemistry"].fillna("MISSING").value_counts().head(20)
        fig, ax = plt.subplots(figsize=(12, 6))
        chemistry_counts.plot(kind="bar", ax=ax, color="#2E6F95")
        ax.set_title("Chemistry Distribution")
        ax.set_xlabel("Chemistry")
        ax.set_ylabel("Cell Count")
        fig.tight_layout()
        outputs["chemistry_bar_chart"] = figures_dir / "chemistry_bar_chart.png"
        fig.savefig(outputs["chemistry_bar_chart"], dpi=150)
        plt.close(fig)

        heatmap = pd.crosstab(
            self.tests_df["temperature"].round(3).astype("string").fillna("MISSING"),
            self.tests_df["c_rate"].fillna("MISSING"),
        )
        fig, ax = plt.subplots(figsize=(14, 8))
        sns.heatmap(heatmap, cmap="YlGnBu", ax=ax)
        ax.set_title("Temperature vs C-rate")
        fig.tight_layout()
        outputs["temperature_c_rate_heatmap"] = figures_dir / "temperature_c_rate_heatmap.png"
        fig.savefig(outputs["temperature_c_rate_heatmap"], dpi=150)
        plt.close(fig)

        lifecycle_counts = self.lifecycle_df["lifecycle_category"].value_counts()
        fig, ax = plt.subplots(figsize=(10, 6))
        lifecycle_counts.plot(kind="bar", ax=ax, color="#D17A22")
        ax.set_title("Lifecycle Distribution")
        ax.set_xlabel("Lifecycle")
        ax.set_ylabel("Cell Count")
        fig.tight_layout()
        outputs["lifecycle_distribution"] = figures_dir / "lifecycle_distribution.png"
        fig.savefig(outputs["lifecycle_distribution"], dpi=150)
        plt.close(fig)

        return outputs

    def prepare_dataframe_for_parquet(self, df: pd.DataFrame) -> pd.DataFrame:
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

    def query_tests(
        self,
        chemistry: str | None = None,
        temperature: float | int | None = None,
        c_rate: str | None = None,
        regime: str | None = None,
        lot_number: str | None = None,
        status: str | None = None,
    ) -> pd.DataFrame:
        if self.tests_df is None:
            self.build_relational_tables()

        assert self.tests_df is not None
        df = self.tests_df.copy()

        if chemistry is not None:
            df = df[df["chemistry"].fillna("").str.upper() == normalize_text(chemistry, uppercase=True)]
        if temperature is not None:
            df = df[df["temperature"].round(6) == float(temperature)]
        if c_rate is not None:
            df = df[df["c_rate"].fillna("") == clean_rate(c_rate)]
        if lot_number is not None:
            df = df[
                df["lot_number"].fillna("").str.upper()
                == normalize_text(lot_number, uppercase=True)
            ]
        if status is not None:
            df = df[df["status"].fillna("").str.upper() == normalize_text(status, uppercase=True)]
        if regime is not None:
            pattern = re.escape(normalize_text(regime) or "")
            df = df[df["regime"].fillna("").str.contains(pattern, case=False, regex=True)]
        return df.reset_index(drop=True)

    def get_csv_paths(self, df: pd.DataFrame, base_path: Path | None = None) -> list[Path]:
        root = resolve_csv_dir(base_path or self.csv_dir)
        paths: list[Path] = []
        for test_id in df["test_id"].dropna().tolist():
            path = self.find_csv_path(test_id, base_path=root)
            if path is not None:
                paths.append(path)
        return paths

    def build_semantic_index(self) -> SemanticArtifacts:
        if self.tests_df is None:
            self.build_relational_tables()

        assert self.tests_df is not None
        semantic_df = self.tests_df.dropna(subset=["test_id"]).copy()
        semantic_df["combined_text"] = semantic_df["combined_text"].fillna("")
        if semantic_df.empty:
            raise ValueError("No tests with valid test_id available to build semantic index.")

        SentenceTransformer = import_sentence_transformer()
        faiss = import_faiss()
        if self._model is None:
            self._model = SentenceTransformer(self.model_name)

        texts = semantic_df["combined_text"].tolist()
        vectors = self._model.encode(
            texts,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        ).astype("float32")
        index = faiss.IndexFlatIP(vectors.shape[1])
        index.add(vectors)

        self.semantic = SemanticArtifacts(
            index=index,
            texts=texts,
            test_ids=semantic_df["test_id"].astype(int).tolist(),
            model_name=self.model_name,
        )
        return self.semantic

    def semantic_search(self, query: str, k: int = 10) -> pd.DataFrame:
        if not query.strip():
            return pd.DataFrame(columns=["test_id", "score"])
        if self.semantic is None:
            self.build_semantic_index()

        assert self.semantic is not None
        assert self._model is not None
        query_vector = self._model.encode(
            [query],
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        ).astype("float32")
        scores, indices = self.semantic.index.search(query_vector, k)

        rows = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0:
                continue
            rows.append(
                {
                    "test_id": self.semantic.test_ids[idx],
                    "score": float(score),
                    "combined_text": self.semantic.texts[idx],
                }
            )
        return pd.DataFrame(rows).sort_values("score", ascending=False).reset_index(drop=True)

    def hybrid_query(self, user_query: str, k: int = 10) -> pd.DataFrame:
        parsed = self.parse_query_heuristic(user_query)
        structured_df = self.query_tests(
            chemistry=parsed.get("chemistry"),
            temperature=parsed.get("temperature"),
            c_rate=parsed.get("c_rate"),
            regime=parsed.get("regime"),
            lot_number=parsed.get("lot_number"),
            status=parsed.get("status"),
        )

        semantic_text = parsed.get("semantic") or user_query
        semantic_df = self.semantic_search(semantic_text, k=max(k * 5, 20))

        if structured_df.empty and parsed.get("has_structured_filters"):
            return structured_df.assign(explanation=pd.Series(dtype="string"))

        if not semantic_df.empty:
            if not structured_df.empty:
                merged = structured_df.merge(semantic_df, on="test_id", how="inner")
            else:
                assert self.tests_df is not None
                merged = self.tests_df.merge(semantic_df, on="test_id", how="inner")
        else:
            merged = structured_df.copy()
            merged["score"] = pd.NA

        if merged.empty:
            return merged

        merged["explanation"] = merged.apply(
            lambda row: self.build_explanation(row=row, parsed=parsed, semantic_used=not semantic_df.empty),
            axis=1,
        )
        sort_cols = ["score", "timestamp"] if "score" in merged.columns else ["timestamp"]
        ascending = [False, True] if "score" in merged.columns else [True]
        return merged.sort_values(sort_cols, ascending=ascending, na_position="last").head(k)

    def parse_query_with_llm(self, user_query: str, model: str = "llama3") -> dict[str, Any]:
        prompt = (
            "Return only compact JSON with keys chemistry, temperature, c_rate, regime, "
            "lot_number, status, semantic. Use null for missing values.\n"
            f"Query: {user_query}"
        )
        payload = json.dumps({"model": model, "prompt": prompt, "stream": False}).encode("utf-8")
        req = request.Request(
            "http://127.0.0.1:11434/api/generate",
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with request.urlopen(req, timeout=120) as response:
                body = json.loads(response.read().decode("utf-8"))
        except (error.URLError, TimeoutError) as exc:
            raise RuntimeError("Ollama is not reachable on http://127.0.0.1:11434.") from exc

        text = body.get("response", "").strip()
        match = re.search(r"\{.*\}", text, flags=re.DOTALL)
        if not match:
            raise ValueError("Ollama response did not contain JSON.")
        parsed = json.loads(match.group(0))
        return {
            "chemistry": parsed.get("chemistry"),
            "temperature": parsed.get("temperature"),
            "c_rate": parsed.get("c_rate"),
            "regime": parsed.get("regime"),
            "lot_number": parsed.get("lot_number"),
            "status": parsed.get("status"),
            "semantic": parsed.get("semantic"),
        }

    def run(self, build_semantic: bool = True) -> None:
        self.load_and_clean_data()
        self.build_relational_tables()
        report = self.build_data_map_report()
        if build_semantic:
            self.build_semantic_index()
        self.save_outputs()
        self.print_report(report)

    def classify_regime(self, regime: Any) -> dict[str, bool]:
        text = normalize_text(regime)
        lowered = text.lower() if text else ""
        return {
            "is_formation": any(keyword in lowered for keyword in FORMATION_KEYWORDS),
            "is_aging": any(keyword in lowered for keyword in AGING_KEYWORDS),
            "is_rpt": any(keyword in lowered for keyword in RPT_KEYWORDS),
        }

    def classify_lifecycle(self, row: pd.Series) -> str:
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

    def find_csv_path(self, test_id: Any, base_path: Path | None = None) -> Path | None:
        if pd.isna(test_id):
            return None
        try:
            test_num = int(test_id)
        except (TypeError, ValueError):
            return None

        csv_root = resolve_csv_dir(base_path or self.csv_dir)
        lookup = self._build_csv_lookup(csv_root)
        return lookup.get(test_num)

    def _build_csv_lookup(self, csv_root: Path) -> dict[int, Path]:
        if self._csv_lookup is not None:
            return self._csv_lookup

        lookup: dict[int, Path] = {}
        if not csv_root.exists():
            self._csv_lookup = lookup
            return lookup

        for path in csv_root.rglob("Test*.csv"):
            match = re.search(r"Test(\d+)", path.name, flags=re.IGNORECASE)
            if not match:
                continue
            lookup.setdefault(int(match.group(1)), path.resolve())

        self._csv_lookup = lookup
        return lookup

    def parse_query_heuristic(self, user_query: str) -> dict[str, Any]:
        parsed: dict[str, Any] = {
            "chemistry": None,
            "temperature": None,
            "c_rate": None,
            "regime": None,
            "lot_number": None,
            "status": None,
            "semantic": None,
            "has_structured_filters": False,
        }
        working = user_query

        temp_match = re.search(
            r"(-?\d+(?:\.\d+)?)\s*(?:°\s*)?[Cc]\b|(-?\d+(?:\.\d+)?)\s*celsius",
            working,
        )
        if temp_match:
            parsed["temperature"] = float(next(group for group in temp_match.groups() if group is not None))

        rate_match = re.search(r"\b(\d+(?:\.\d+)?\s*C|C/\d+(?:\.\d+)?)\b", working, flags=re.IGNORECASE)
        if rate_match:
            parsed["c_rate"] = clean_rate(rate_match.group(1))

        lot_match = re.search(r"\blot[_\s-]*([A-Za-z0-9._-]+)\b", working, flags=re.IGNORECASE)
        if lot_match:
            parsed["lot_number"] = lot_match.group(1).upper()

        status_match = re.search(
            r"\bstatus[_\s-]*([A-Za-z0-9._-]+)\b",
            working,
            flags=re.IGNORECASE,
        )
        if status_match:
            parsed["status"] = status_match.group(1).upper()

        if self.tests_df is not None:
            chemistries = [value for value in self.tests_df["chemistry"].dropna().unique().tolist() if value]
            for chemistry in sorted(chemistries, key=len, reverse=True):
                if chemistry.lower() in working.lower():
                    parsed["chemistry"] = chemistry
                    break

            statuses = [value for value in self.tests_df["status"].dropna().unique().tolist() if value]
            for status in sorted(statuses, key=len, reverse=True):
                if status.lower() in working.lower():
                    parsed["status"] = status
                    break

        regime_match = re.search(
            r"\b(formation|aging|ageing|rpt|hppc|cycle|cycling|calendar|storage)\b",
            working,
            flags=re.IGNORECASE,
        )
        if regime_match:
            parsed["regime"] = regime_match.group(1)

        parsed["semantic"] = self.strip_structured_terms(working, parsed).strip() or None
        parsed["has_structured_filters"] = any(
            parsed[key] is not None
            for key in ("chemistry", "temperature", "c_rate", "regime", "lot_number", "status")
        )
        return parsed

    def strip_structured_terms(self, text: str, parsed: dict[str, Any]) -> str:
        cleaned = text
        replacements: Iterable[str | float | int | None] = (
            parsed.get("chemistry"),
            parsed.get("c_rate"),
            parsed.get("regime"),
            parsed.get("lot_number"),
            parsed.get("status"),
        )
        for value in replacements:
            if value is None:
                continue
            cleaned = re.sub(re.escape(str(value)), " ", cleaned, flags=re.IGNORECASE)
        if parsed.get("temperature") is not None:
            cleaned = re.sub(r"(-?\d+(?:\.\d+)?)\s*(?:°\s*)?[Cc]\b", " ", cleaned, flags=re.IGNORECASE)
            cleaned = re.sub(r"(-?\d+(?:\.\d+)?)\s*celsius", " ", cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(r"\blot[_\s-]*[A-Za-z0-9._-]+\b", " ", cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(r"\bstatus[_\s-]*[A-Za-z0-9._-]+\b", " ", cleaned, flags=re.IGNORECASE)
        return re.sub(r"\s+", " ", cleaned)

    def build_explanation(self, row: pd.Series, parsed: dict[str, Any], semantic_used: bool) -> str:
        reasons: list[str] = []
        for key in ("chemistry", "temperature", "c_rate", "regime", "lot_number", "status"):
            if parsed.get(key) is not None:
                reasons.append(f"{key} matched")
        if semantic_used and pd.notna(row.get("score")):
            reasons.append(f"semantic score={row['score']:.4f}")
        return "; ".join(reasons) if reasons else "matched semantic search"

    def print_report(self, report: dict[str, pd.DataFrame]) -> None:
        print("\nGLOBAL SUMMARY")
        print(report["global_summary"].to_string(index=False))
        print("\nCHEMISTRY DISTRIBUTION")
        print(report["chemistry_distribution"].to_string(index=False))
        print("\nTEMPERATURE DISTRIBUTION")
        print(report["temperature_distribution"].to_string(index=False))
        print("\nC_RATE DISTRIBUTION")
        print(report["c_rate_distribution"].to_string(index=False))
        print("\nREGIME SUMMARY")
        print(report["regime_summary"].to_string(index=False))
        print("\nLIFECYCLE DISTRIBUTION")
        print(report["lifecycle_distribution"].to_string(index=False))
        print("\nCELL TESTS RELEVANT")
        print(report["relevant_cells"].to_string(index=False))
