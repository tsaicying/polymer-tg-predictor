from fastapi import FastAPI
from schemas import PredictRequest, PredictResponse
from model_loader import predict_tg

app = FastAPI(title="Polymer Tg Predictor API", version="1.0")

@app.post("/predict", response_model=PredictResponse)
def predict(request: PredictRequest) -> PredictResponse:
    tg = predict_tg(request.smiles)
    return PredictResponse(predicted_tg=tg, model_version="1.0")