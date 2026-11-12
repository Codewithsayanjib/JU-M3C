"""Load notes and select records per task/split."""
import json
from functools import lru_cache
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
NOTES_PATH = ROOT / "data" / "notes.jsonl"

TASKS = ("T1", "T2", "T3", "T4")
SPLITS = ("train", "val", "test")

# Fixed order used in the T2 prompt.
T2_LABELS = [
    "Operative note",
    "Consult / H&P",
    "Imaging report",
    "Progress note",
    "Discharge summary",
    "ER report",
]


@lru_cache(maxsize=1)
def load_notes(path: Path = NOTES_PATH) -> tuple:
    with open(path) as f:
        return tuple(json.loads(line) for line in f if line.strip())


def _in_task(note: dict, task: str) -> bool:
    if task == "T1":
        return note["t1_specialty"] is not None
    if task == "T2":
        return note["t2_doc_type"] is not None
    if task == "T3":
        return note["t3_summary"] is not None
    if task == "T4":
        return bool(note["t4_has_keywords"])
    raise ValueError(f"unknown task {task}")


def task_records(task: str, split: str) -> list[dict]:
    if split not in SPLITS:
        raise ValueError(f"unknown split {split}")
    return [n for n in load_notes() if n["split"] == split and _in_task(n, task)]


LABELS_PATH = ROOT / "configs" / "specialty_labels.json"


@lru_cache(maxsize=1)
def t1_labels() -> tuple:
    """Sorted unique specialty labels from TRAIN (stored in configs/specialty_labels.json)."""
    if not NOTES_PATH.exists():
        return tuple(json.load(open(LABELS_PATH)))
    return tuple(sorted({n["t1_specialty"] for n in task_records("T1", "train")}))


def labels_for(task: str) -> list[str] | None:
    if task == "T1":
        return list(t1_labels())
    if task == "T2":
        return list(T2_LABELS)
    return None


def gold(note: dict, task: str):
    return {
        "T1": note["t1_specialty"],
        "T2": note["t2_doc_type"],
        "T3": note["t3_summary"],
        "T4": note["keywords"],
    }[task]
