"""Feature lists and preprocessing for Model A (UCI clinical data)."""
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

NUMERIC = ["age", "trestbps", "chol", "thalach", "oldpeak", "ca"]
CATEGORICAL = ["sex", "cp", "fbs", "restecg", "exang", "slope", "thal"]
FEATURES = NUMERIC + CATEGORICAL

# Which columns each variant of Model A is allowed to use.
FEATURE_SETS = {
    # everything in the UCI files
    "full13": FEATURES,
    # drops the two specialist-test columns (fluoroscopy, thallium scan)
    "no_ca_thal": [f for f in FEATURES if f not in ("ca", "thal")],
    # things a person could plausibly know from a checkup
    "checkup7": ["age", "sex", "cp", "trestbps", "chol", "fbs", "exang"],
}


def build_preprocessor(scale=False, flags=True, features=None):
    """Impute missing values, optionally flag them, encode categories.

    flags=True adds a 'was this value missing?' column per feature.
    features restricts the columns used (default: all 13).
    """
    features = FEATURES if features is None else features
    numeric = [c for c in NUMERIC if c in features]
    categorical = [c for c in CATEGORICAL if c in features]

    num_steps = [("impute", SimpleImputer(strategy="median", add_indicator=flags))]
    if scale:
        num_steps.append(("scale", StandardScaler()))
    cat_steps = [
        ("impute", SimpleImputer(strategy="most_frequent", add_indicator=flags)),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ]
    return ColumnTransformer([
        ("num", Pipeline(num_steps), numeric),
        ("cat", Pipeline(cat_steps), categorical),
    ])
