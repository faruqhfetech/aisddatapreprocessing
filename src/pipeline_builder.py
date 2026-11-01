# Column lists and Pipeline definitions, fixed by the Phase 1 plan.
from sklearn.compose import ColumnTransformer
from lightgbm import LGBMClassifier
from sklearn.ensemble import (ExtraTreesClassifier, HistGradientBoostingClassifier,
                              RandomForestClassifier)
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.transformers import (ColumnDropper, FeatureEngineer, IQRWinsorizer,
                              Log1pTransformer, NeighborFeatures)

TARGET = "is_error_word"
GROUP = "reciter"
# the two reciters that testset_meta.json lists as the holdout; the Phase 1 audit still ran on all rows
TEST_RECITERS = ["Husary_64kbps", "Minshawy_Murattal_128kbps"]

# columns dropped before modelling (Phase 1, Sections 4.2 and 4.3)
LEAKAGE = ["error_type", "tier", "stretch_rate", "split", "word_idx"]
IDENTIFIERS = ["recording_id", "reciter", "surah", "ayah"]
LOW_VARIANCE = ["collapsed"]
COLLINEAR = ["test_frac", "dur_z", "orig_duration_ms", "new_duration_ms"]
DROP = LEAKAGE + IDENTIFIERS + LOW_VARIANCE + COLLINEAR

# the 9 model inputs left after the drops
NUMERIC_FEATURES = [
    "ref_frac", "dtw_cost", "width", "dur_ratio", "cost_z", "log_dev",
    "suspicion", "n_words", "rec_len_ratio",
]
# positive, right-skewed features that get log1p
LOG_COLS = ["width", "dur_ratio", "suspicion", "n_words"]

# lag/lead features: dtw_cost and dur_ratio of the previous and next word
NEIGHBOR_BASES = ["dtw_cost", "dur_ratio"]
NEIGHBOR_FEATURES = ["dtw_cost_prev", "dtw_cost_next", "dur_ratio_prev", "dur_ratio_next"]
NEIGHBOR_LOG_COLS = ["dur_ratio_prev", "dur_ratio_next"]


# Impute -> log -> cap -> scale, applied to the numeric features.
def build_numeric_pipeline(cap_k=3.0, scale=True, neighbors=False):
    # median, not mean, because most features are right-skewed
    imputer = SimpleImputer(strategy="median")
    log_columns = LOG_COLS + NEIGHBOR_LOG_COLS if neighbors else LOG_COLS
    log_step = Log1pTransformer(columns=log_columns)
    capper = IQRWinsorizer(k=cap_k)

    # order matters: cap after the log, so the fences are computed on the log scale
    steps = [
        ("impute", imputer),
        ("log", log_step),
        ("cap", capper),
    ]
    # scale=False stops after capping, leaving values in their original units
    if scale:
        steps.append(("scale", StandardScaler()))
    return Pipeline(steps)


# Takes the raw word table and returns the 9 cleaned, scaled features.
# neighbors=True adds the lag/lead features (the 9 features become 13).
# add_features / remove_features are for ablations: add_features puts a Phase 1 dropped
# column back into the model, remove_features takes one of the 9 out.
def build_preprocessor(cap_k=3.0, scale=True, neighbors=False,
                       add_features=(), remove_features=()):
    numeric_pipeline = build_numeric_pipeline(cap_k, scale, neighbors)
    numeric_features = NUMERIC_FEATURES + NEIGHBOR_FEATURES if neighbors else NUMERIC_FEATURES
    numeric_features = [f for f in numeric_features if f not in remove_features]
    numeric_features = numeric_features + list(add_features)
    dropped = [column for column in DROP if column not in add_features]

    # remainder="drop" removes any column not in NUMERIC_FEATURES
    columns = ColumnTransformer(
        [("num", numeric_pipeline, numeric_features)],
        remainder="drop",
        verbose_feature_names_out=False,
    )

    # rec_len_ratio is added before the drops, since it needs the two duration columns
    # neighbors needs recording_id and word_idx, so it runs before the drops
    steps = [("engineer", FeatureEngineer())]
    if neighbors:
        steps.append(("neighbors", NeighborFeatures(columns=NEIGHBOR_BASES)))
    steps += [
        ("drop", ColumnDropper(columns=dropped)),
        ("columns", columns),
    ]
    preprocessor = Pipeline(steps)
    # keep DataFrames between steps so the transformers can find columns by name
    preprocessor.set_output(transform="pandas")
    return preprocessor


def build_classifier(model, C=1.0, depth=None, leaf=5):
    # class_weight="balanced" handles the 5.49 : 1 imbalance without resampling
    if model == "logreg":
        return LogisticRegression(C=C, class_weight="balanced", max_iter=1000)
    # gradient boosting won in cross-validation; random_state keeps runs repeatable
    if model == "hgb":
        return HistGradientBoostingClassifier(
            class_weight="balanced",
            max_iter=300,
            learning_rate=0.05,
            random_state=0,
        )
    # random forest and extra trees: balanced_subsample reweights inside each bootstrap
    if model in ("rf", "et"):
        forest = RandomForestClassifier if model == "rf" else ExtraTreesClassifier
        return forest(n_estimators=300, max_depth=depth, min_samples_leaf=leaf,
                      class_weight="balanced_subsample", n_jobs=-1, random_state=0)
    # scale_pos_weight is the negatives-per-positive ratio of the training split (5.70)
    if model == "lgbm":
        return LGBMClassifier(n_estimators=300, learning_rate=0.05, num_leaves=15,
                              min_child_samples=leaf, scale_pos_weight=5.70,
                              random_state=0, n_jobs=-1, verbose=-1)
    raise ValueError(f"unknown model: {model}")


# Preprocessor + classifier in one Pipeline, so cross-validation refits everything per fold.
def build_model_pipeline(model="logreg", cap_k=3.0, C=1.0, scale=True,
                         neighbors=False, depth=None, leaf=5,
                         add_features=(), remove_features=()):
    preprocessor = build_preprocessor(cap_k, scale, neighbors,
                                      add_features, remove_features)
    classifier = build_classifier(model, C, depth, leaf)
    return Pipeline([
        ("prep", preprocessor),
        ("model", classifier),
    ])
