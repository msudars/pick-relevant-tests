from pathlib import Path


DEFAULT_METADATA_PATH = Path("data/2026_PEC_log")
DEFAULT_CSV_DIR = Path("data")
DEFAULT_OUTPUT_DIR = Path("outputs/battery_data_system")
DEFAULT_MODEL_NAME = "all-MiniLM-L6-v2"
DEFAULT_SHEET_NAME = "PEC_log"

CANONICAL_ALIASES = {
    "cell_id": ["cell_id", "cell_serial_no", "serial_number", "cell_serial_number"],
    "lot_number": ["lot_number", "lot", "batch", "batch_number"],
    "test_id": ["test_id", "test_no", "test_number"],
    "regime": ["test_regime", "regime", "test_type", "protocol"],
    "test_program": ["test_program", "program"],
    "other_condition": ["other_condition", "other_conditions", "condition"],
    "temperature": ["test_temp", "test_temp_c", "test_temp_degc", "temperature", "test_temperature"],
    "c_rate": ["c_rate", "test_rate", "test_rate_cha_dch"],
    "status": ["status"],
    "manufacturer": ["cell_manufacturer", "manufacturer", "cell_manu_facturer"],
    "chemistry": ["cell_chemistry", "chemistry", "cell_chem_istry"],
    "cell_type_id": ["cell_type_id"],
    "comments": ["comments", "comment", "comment_1", "comment_2", "comment_1_1", "notes"],
    "timestamp": ["timestamp", "date_and_time", "datetime", "date_time"],
    "capacity": ["cell_capacity", "capacity", "cell_capacity_ah", "capacity_ah"],
    "test_rate": ["test_rate_cha_dch", "test_rate"],
}

TEXT_COLUMNS = [
    "cell_id",
    "lot_number",
    "regime",
    "test_program",
    "other_condition",
    "c_rate",
    "status",
    "manufacturer",
    "chemistry",
    "cell_type_id",
    "comments",
    "test_rate",
]

FORMATION_KEYWORDS = ("formation", "ffi", "init")
AGING_KEYWORDS = ("aging", "ageing", "cycle", "cycling", "cyc", "calendar", "storage")
RPT_KEYWORDS = ("rpt", "character", "charact", "hppc", "pulse", "ocv", "reference")
