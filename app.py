from fastapi import FastAPI
import torch

app = FastAPI()

@app.post("/models")
async def root():
	return str(torch.rand(5, 3))
