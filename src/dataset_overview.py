# Dataset overview for the README: recording-type figure, length statistics, balance.
# Writes figures/overview_*.png and data/processed/dataset_overview.txt.

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

ROOT = Path(__file__).resolve().parents[1]
TABLE_PATH = ROOT / "data" / "processed" / "word_table.csv"
FIG = ROOT / "figures"
SUMMARY_PATH = ROOT / "data" / "processed" / "dataset_overview.txt"

TARGET = "is_error_word"
# the order and labels of the README table
TYPE_LABELS = {
    "clean": "Clean (untouched)",
    "control_seam": "Control seam",
    "word_deletion": "Word deletion",
    "madd_shortening": "Vowel shortening",
    "madd_lengthening": "Vowel lengthening",
    "word_substitution": "Word substitution",
    "near_miss_substitution": "Near-miss substitution",
    "word_repetition": "Word repetition",
    "truncation": "Truncation",
}
CORRECT_COLOR = "#4c72b0"
ERROR_COLOR = "#c44e52"


def save(fig, name):
    fig.savefig(FIG / f"{name}.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def recording_type_table(df, recordings):
    table = pd.DataFrame({
        "recordings": recordings.groupby("error_type").size(),
        "words": df.groupby("error_type").size(),
        "error_words": df.groupby("error_type")[TARGET].sum(),
        "mean_seconds": recordings.groupby("error_type")["new_duration_ms"].mean() / 1000,
    })
    return table.loc[list(TYPE_LABELS)]


# One bar per recording type, coloured by whether it carries an error.
def figure_recording_types(table):
    labels = [TYPE_LABELS[name] for name in table.index]
    colors = [ERROR_COLOR if name not in ("clean", "control_seam") else CORRECT_COLOR
              for name in table.index]

    fig, (left, right) = plt.subplots(1, 2, figsize=(11, 4.2))
    left.barh(labels, table["recordings"], color=colors)
    for position, count in enumerate(table["recordings"]):
        left.text(count, position, f" {count}", va="center")
    left.set_xlabel("recordings")
    left.set_title("Recordings by type")
    left.invert_yaxis()

    right.barh(labels, table["words"], color=colors)
    for position, count in enumerate(table["words"]):
        right.text(count, position, f" {count:,}", va="center")
    right.set_xlabel("words (rows)")
    right.set_title("Words by recording type")
    right.invert_yaxis()
    right.set_yticklabels([])

    # a legend built from two proxy patches, because the bars share one colour per type
    handles = [plt.Rectangle((0, 0), 1, 1, color=CORRECT_COLOR),
               plt.Rectangle((0, 0), 1, 1, color=ERROR_COLOR)]
    fig.legend(handles, ["no error injected", "one error word injected"],
               loc="lower center", ncol=2, bbox_to_anchor=(0.5, -0.06), frameon=False)
    left.set_xlim(0, table["recordings"].max() * 1.15)
    right.set_xlim(0, table["words"].max() * 1.18)
    fig.suptitle("Recording types in the Suwayhib word table", y=1.02)
    save(fig, "overview_recording_types")


def figure_lengths(recordings):
    seconds = recordings["new_duration_ms"] / 1000
    fig, (left, right) = plt.subplots(1, 2, figsize=(11, 3.8))
    left.hist(seconds, bins=40, color=CORRECT_COLOR)
    left.axvline(seconds.mean(), color="black", linestyle="--", label=f"mean {seconds.mean():.1f} s")
    left.axvline(seconds.median(), color="gray", linestyle=":", label=f"median {seconds.median():.1f} s")
    left.set_xlabel("recording length (s)")
    left.set_ylabel("recordings")
    left.set_title("Recording length")
    left.legend()

    right.hist(recordings["n_words"], bins=range(3, 23), color=CORRECT_COLOR)
    right.set_xlabel("words per recording")
    right.set_ylabel("recordings")
    right.set_title("Words per recording")
    save(fig, "overview_recording_lengths")


def figure_balance(df, recordings):
    fig, (left, right) = plt.subplots(1, 2, figsize=(11, 3.8))
    counts = df[TARGET].value_counts().sort_index()
    left.bar(["correct word (0)", "error word (1)"], counts.values, color=[CORRECT_COLOR, ERROR_COLOR])
    for position, count in enumerate(counts.values):
        left.text(position, count, f"{count:,}\n({count / len(df):.1%})", ha="center", va="bottom")
    left.set_ylim(0, counts.max() * 1.2)
    left.set_title(f"Words: {counts[0] / counts[1]:.2f} : 1 imbalance")

    rate = df.groupby("reciter")[TARGET].mean().sort_values()
    right.barh([name.split("_")[0] for name in rate.index], rate.values, color=ERROR_COLOR)
    right.axvline(df[TARGET].mean(), color="black", linestyle="--")
    right.set_xlabel("share of words that are errors")
    right.set_title("Error rate by reciter")
    fig.subplots_adjust(wspace=0.4)
    save(fig, "overview_class_balance")


def length_statistics(recordings):
    rows = {}
    rows["learner-side length (s)"] = recordings["new_duration_ms"] / 1000
    rows["reference length (s)"] = recordings["orig_duration_ms"] / 1000
    rows["words per recording"] = recordings["n_words"]
    stats_table = pd.DataFrame({name: values.describe() for name, values in rows.items()}).T
    return stats_table[["mean", "std", "min", "25%", "50%", "75%", "max"]]


# new / orig recording length per recording type: the clean recordings sit at exactly 1.0
def rec_len_ratio_by_type(recordings):
    ratio = recordings["new_duration_ms"] / recordings["orig_duration_ms"]
    table = ratio.groupby(recordings["error_type"]).agg(["mean", "std", "min", "max"])
    return table.loc[list(TYPE_LABELS)]


def main():
    df = pd.read_csv(TABLE_PATH)
    recordings = df.drop_duplicates("recording_id")

    table = recording_type_table(df, recordings)
    figure_recording_types(table)
    figure_lengths(recordings)
    figure_balance(df, recordings)

    lines = []
    lines.append("RECORDING TYPES")
    lines.append(table.round(2).to_string())
    lines.append("")
    lines.append(f"recordings {len(recordings)}, words {len(df)}, reciters {df['reciter'].nunique()}")
    lines.append("")
    lines.append("LENGTH STATISTICS (one row per recording)")
    lines.append(length_statistics(recordings).round(2).to_string())
    lines.append("")
    n_pos = int(df[TARGET].sum())
    n_neg = len(df) - n_pos
    lines.append("REC_LEN_RATIO (new / orig duration) BY RECORDING TYPE")
    lines.append(rec_len_ratio_by_type(recordings).round(3).to_string())
    lines.append("")
    lines.append("BALANCE")
    lines.append(f"words: {n_neg} correct, {n_pos} error ({n_pos / len(df):.1%}), {n_neg / n_pos:.2f} : 1")
    n_error_recordings = int((recordings["is_error"] if "is_error" in recordings else
                              ~recordings["error_type"].isin(["clean", "control_seam"])).sum())
    lines.append(f"recordings: {len(recordings) - n_error_recordings} without an error, "
                 f"{n_error_recordings} with one ({n_error_recordings / len(recordings):.0%}); "
                 "each error recording has exactly one error word")
    lines.append("by reciter (error rate): " + ", ".join(
        f"{name.split('_')[0]} {rate:.1%}" for name, rate in df.groupby("reciter")[TARGET].mean().items()))
    SUMMARY_PATH.write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
