from pydantic import BaseModel

class PredictRequest(BaseModel):
    smiles: str

class PredictResponse(BaseModel):
    predicted_tg: float
    model_version: str