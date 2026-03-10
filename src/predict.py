from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))
MODEL_PATH = PROJECT_ROOT / "models" / "rdkit_polymer_rf.joblib"

from src.preprocessing import calc_rdkit_descriptors, polymer_aware_features
import joblib
import pandas as pd

bundle = joblib.load(MODEL_PATH)
model = bundle["model"]
expected_cols = bundle["feature_columns"]
model_version = bundle.get("model_version", "unknown")


def compute_features(smiles):
    feature_dict = {}
    feature_dict.update(calc_rdkit_descriptors(smiles))
    feature_dict.update(polymer_aware_features(smiles))
    df = pd.DataFrame([feature_dict])
    return df

def predict_tg(smiles: str) -> float:
    df = compute_features(smiles)
    df = df.reindex(columns=expected_cols)
    return float(model.predict(df)[0])

    



