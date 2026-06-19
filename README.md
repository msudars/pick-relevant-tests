# Pick Relevant Tests

Local Python tooling for PEC battery test metadata cleaning, dataset mapping, semantic retrieval, and hybrid test selection.

## What It Does

- Loads and cleans the PEC Excel log
- Standardizes core metadata fields
- Builds relational battery metadata tables
- Saves cleaned metadata to parquet
- Generates a battery data map report
- Builds a local semantic index with `sentence-transformers` and `FAISS`
- Supports structured filtering and hybrid free-text querying
- Optionally uses a local Ollama model to parse natural-language queries

## Repository Layout

```text
pick-relevant-tests/
├── data/
│   ├── 2026_PEC_log.xlsx
│   └── TestXXXXX.csv or nested folders containing Test*.csv
├── outputs/
│   └── battery_data_system/
├── src/
│   └── pick_relevant_tests/
│       ├── cli.py
│       ├── config.py
│       ├── system.py
│       └── utils.py
├── main.py
├── pyproject.toml
└── README.md
```

## Requirements

- Python 3.12+
- Local filesystem access to the Excel metadata log and CSV test files
- Enough disk space for parquet outputs, plots, and the FAISS index

Optional:

- Ollama running locally on `http://127.0.0.1:11434`
- A pulled model such as `llama3`

## Installation

### Option 1: `uv`

```bash
uv sync
```

Run the tool:

```bash
uv run pick-relevant-tests
```

### Option 2: `venv` + `pip`

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -e .
```

Run the tool:

```bash
pick-relevant-tests
```

If editable install is not desired:

```bash
pip install .
python3 main.py
```

## Data Expectations

### Metadata workbook

Default input path:

```text
data/2026_PEC_log.xlsx
```

The loader accepts a base path of `data/2026_PEC_log` and automatically resolves `.xlsx` or `.xls`.

The code attempts to normalize and map source columns into canonical fields such as:

- `cell_id`
- `lot_number`
- `test_id`
- `regime`
- `temperature`
- `c_rate`
- `status`
- `comments`
- `chemistry`
- `manufacturer`
- `capacity`
- `timestamp`

### CSV test files

The tool searches for `Test*.csv` under the configured CSV directory.

Default CSV search root:

```text
data/
```

It also checks common nested locations such as:

- `data/timeseries`
- `data/csv`
- `data/test_csvs`

## Basic Usage

Run the full local pipeline:

```bash
pick-relevant-tests
```

Equivalent:

```bash
python3 main.py
```

Use explicit paths:

```bash
pick-relevant-tests \
  --metadata-path data/2026_PEC_log.xlsx \
  --csv-dir data/timeseries \
  --output-dir outputs/battery_data_system
```

Use a non-default sheet:

```bash
pick-relevant-tests --sheet-name 0
```

Skip semantic indexing:

```bash
pick-relevant-tests --skip-semantic
```

## Querying

Run a hybrid query directly from the CLI:

```bash
pick-relevant-tests --query "NMC aging tests at 25 C with fast charge"
```

Limit results:

```bash
pick-relevant-tests --query "formation tests lot ABC123" --top-k 5
```

Use Ollama only for query parsing:

```bash
pick-relevant-tests \
  --query "find NMC tests at 25 C related to fast charge" \
  --use-ollama \
  --ollama-model llama3
```

## Output Files

Default output directory:

```text
outputs/battery_data_system/
```

Generated artifacts:

- `tables/cleaned_metadata.parquet`
- `tables/cells_df.parquet`
- `tables/tests_df.parquet`
- `tables/lifecycle_df.parquet`
- CSV summary tables for report sections
- `figures/tests_per_cell_histogram.png`
- `figures/chemistry_bar_chart.png`
- `figures/temperature_c_rate_heatmap.png`
- `figures/lifecycle_distribution.png`
- `semantic/tests.faiss`
- `semantic/semantic_metadata.parquet`
- `semantic/model_name.txt`

## Programmatic Use

```python
from pathlib import Path

from pick_relevant_tests import BatteryDataSystem

system = BatteryDataSystem(
    metadata_path=Path("data/2026_PEC_log.xlsx"),
    csv_dir=Path("data"),
    output_dir=Path("outputs/battery_data_system"),
)

system.load_and_clean_data()
system.build_relational_tables()
system.build_data_map_report()
system.build_semantic_index()

filtered = system.query_tests(
    chemistry="NMC",
    temperature=25,
    regime="aging",
)

semantic_hits = system.semantic_search("fast charge swelling", k=10)
hybrid_hits = system.hybrid_query("NMC aging tests at 25 C with fast charge", k=10)

csv_paths = system.get_csv_paths(filtered)
print(filtered[["test_id", "cell_id", "regime"]].head())
print(csv_paths[:5])
```

## Main CLI Arguments

- `--metadata-path`: Excel log path or base path without extension
- `--csv-dir`: root directory containing `Test*.csv`
- `--output-dir`: where tables, figures, and semantic artifacts are written
- `--sheet-name`: Excel sheet name or index
- `--model-name`: sentence-transformers model name
- `--query`: run a hybrid search after pipeline build
- `--top-k`: number of query results to return
- `--skip-semantic`: disable embedding generation and FAISS index creation
- `--use-ollama`: use Ollama to parse the query into structured fields
- `--ollama-model`: Ollama model name, for example `llama3`

## Ollama Setup

Install Ollama and start the local service, then pull a model:

```bash
ollama pull llama3
```

Verify the service is running:

```bash
curl http://127.0.0.1:11434/api/tags
```

Ollama is optional. The system works locally without it. Ollama is only used to parse natural-language queries into structured filters.

## Notes

- Semantic search is fully local and does not require cloud APIs
- The first sentence-transformers run may download the model weights locally
- If your environment is offline, pre-download the embedding model before running semantic indexing
- CSV files are resolved by `test_id` using names like `Test123.csv`, `Test00123.csv`, or recursive matches under the configured CSV root

## Development

Syntax check:

```bash
python3 -m py_compile main.py src/pick_relevant_tests/*.py
```

Run with local data:

```bash
pick-relevant-tests --metadata-path data/2026_PEC_log.xlsx --csv-dir data
```
