# Build the word-level table: one row per word per recording.

import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw"
OUTPUT_PATH = ROOT / "data" / "processed" / "word_table.csv"

# fields from labels.jsonl, copied onto every word of the recording
RECORDING_FIELDS = [
    "reciter", "surah", "ayah", "n_words", "orig_duration_ms",
    "new_duration_ms", "stretch_rate", "error_type", "tier", "split",
]
# per-word alignment features from scores_full.jsonl
WORD_FIELDS = [
    "test_frac", "ref_frac", "dtw_cost", "width", "dur_ratio", "dur_z",
    "cost_z", "collapsed", "log_dev", "suspicion",
]


def read_jsonl(path):
    records = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            # skip blank lines, such as a trailing newline at the end of the file
            if line.strip() == "":
                continue
            records.append(json.loads(line))
    return records


def build():
    label_list = read_jsonl(RAW_DIR / "labels.jsonl")
    score_list = read_jsonl(RAW_DIR / "scores_full.jsonl")

    # index the labels by recording id so each score can find its label directly
    labels_by_id = {}
    for label in label_list:
        labels_by_id[label["id"]] = label

    rows = []
    for recording in score_list:
        label = labels_by_id[recording["id"]]
        words = recording["words"]

        # stop early if the two files disagree about a recording
        if len(words) != label["n_words"]:
            raise ValueError(f"recording {recording['id']}: word count mismatch")

        for word in words:
            row = {}
            row["recording_id"] = recording["id"]
            row["word_idx"] = word["word_idx"]

            # .get because stretch_rate exists only for madd errors (missing becomes NaN)
            for field in RECORDING_FIELDS:
                row[field] = label.get(field)

            for field in WORD_FIELDS:
                row[field] = word[field]

            # only the one word the error was injected into is labelled 1
            recording_has_error = label["is_error"] == 1
            is_the_error_word = word["word_idx"] == label["word_idx"]
            row["is_error_word"] = int(recording_has_error and is_the_error_word)

            rows.append(row)

    return pd.DataFrame(rows)


if __name__ == "__main__":
    table = build()

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(OUTPUT_PATH, index=False)

    n_rows, n_cols = table.shape
    n_positive = table["is_error_word"].sum()
    positive_rate = table["is_error_word"].mean()
    print(f"wrote {OUTPUT_PATH.relative_to(ROOT)}: {n_rows} rows x {n_cols} columns")
    print(f"positives: {n_positive} ({positive_rate:.1%})")
