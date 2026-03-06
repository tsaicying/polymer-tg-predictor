from fastapi import FastAPI, HTTPException
from schemas import PredictRequest, PredictResponse
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))

from src.predict import predict_tg, model_version

app = FastAPI(title="Polymer Tg Predictor API", version="1.0")

@app.post("/predict", response_model=PredictResponse)
def predict(request: PredictRequest) -> PredictResponse:
    try:
        tg = predict_tg(request.smiles)
        return PredictResponse(
            input_smiles=request.smiles,
            predicted_tg=tg,
            model_version=model_version
        )
    except ValueError as e:
        raise HTTPException(
            status_code=400, 
            detail=f"{str(e)}: {request.smiles}"
            )
    except Exception as e:
        raise HTTPException(
            status_code=500, 
            detail="Internal server error"
            )