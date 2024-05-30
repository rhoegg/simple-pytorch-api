from model import *
from fastapi import FastAPI, Request, Response, status
from typing import List, Optional
from pydantic import BaseModel
import io
import uuid
import torch

class Sample(BaseModel):
	features: list[float]
	predictions: Optional[list[float]] = None

device = (
	"cuda"
	if torch.cuda.is_available()
	else "mps"
	if torch.backends.mps.is_available()
	else "cpu"
)
print(f"Using {device} device")

app = FastAPI()
models = {}

@app.get("/")
async def root():
	return {}

@app.post("/models/", status_code=status.HTTP_201_CREATED)
async def create_model(spec: NeuralNetSpec):
	model_id = str(uuid.uuid4())
	models[model_id] = NeuralNet(spec).to(device)
	return { "id": model_id }

@app.post("/models/{model_id}/predict")
async def predict(model_id: str, sample: Sample) -> Sample:
	if model_id not in models:
		return Response(status_code = 404)
	m = models.get(model_id)
	nn_output = m.forward(torch.FloatTensor(sample.features).to(device))
	return Sample(features=sample.features, predictions=nn_output.tolist())

@app.post("/models/{model_id}/train")
async def train(model_id: str, samples: List[Sample]):
	if model_id not in models:
		return Response(status_code = 404)
	train_model(models.get(model_id), 
			list(map(lambda s: s.features, samples)),
			list(map(lambda s: s.predictions, samples)),
			device)


@app.get("/models/{model_id}", response_class=Response)
async def get_model(model_id: str):
	if model_id not in models:
		return Response(status_code=404)
	m = models.get(model_id)
	return Response(status_code=200, content=m.export(), media_type="application/octet-stream")

@app.put("/models/{model_id}", openapi_extra={
	"requestBody": {
		"content": {
			"application/octet-stream": {
				"schema": {
					"type": "string",
					"format": "binary"
				}
			}
		}
	}
})
async def replace_model(model_id: str, request: Request):
	with io.BytesIO(await request.body()) as buf:
		models[model_id] = NeuralNet.from_saved(buf).to(device)
