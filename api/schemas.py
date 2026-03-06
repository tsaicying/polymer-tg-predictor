from pydantic import BaseModel

class PredictRequest(BaseModel):
    smiles: str

class PredictResponse(BaseModel):
    input_smiles: str
    predicted_tg: float
    model_version: str