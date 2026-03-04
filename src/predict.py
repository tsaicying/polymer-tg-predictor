from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))
MODEL_PATH = PROJECT_ROOT / "models" / "rdkit_polymer_rf.joblib"

from preprocessing import calc_rdkit_descriptors, polymer_aware_features
import joblib
import pandas as pd

bundle = joblib.load(MODEL_PATH)
model = bundle["model"]
expected_cols = bundle["feature_columns"]

model.named_steps["rf"].set_params(n_jobs=1)

def compute_features(smiles):
    feature_dict = {}
    feature_dict.update(calc_rdkit_descriptors(smiles))
    feature_dict.update(polymer_aware_features(smiles))
    df = pd.DataFrame([feature_dict])
    return df

if __name__ == "__main__":
    test_smiles = "*/C=C/CCC*"
    df = compute_features(test_smiles)
    df = df.reindex(columns=expected_cols)
    print(model.predict(df))



